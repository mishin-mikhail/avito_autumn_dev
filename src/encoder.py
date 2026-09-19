"""
Двухбашенный энкодер (bi-encoder) для генерации кандидатов по смыслу.

Зачем: 62% запросов бенчмарка имеют текст, не встречавшийся в train, и лексический
поиск их не находит («перевозка груза» → «грузоперевозки», «замена гидроаккумулятора»
→ «ремонт скважин»). Эмбеддинги дают отдельный список кандидатов и признак близости.

Обучение — InfoNCE: в батче каждый запрос сближается со «своим» объявлением и
отталкивается от всех остальных объявлений батча и от трудных негативов
(похожие объявления из корпуса, которые пользователь не выбрал).
Чем больше батч, тем больше негативов видит модель, поэтому размер батча подбирается
под свободную память GPU (probe_batch_size).

Тексты моделей семейства e5 обязательно идут с префиксами «query: » и «passage: ».

Воспроизводимость.
  * Обучение на GPU бит-в-бит повторяется только на том же железе, поэтому результат
    обучения — артефакт: вектора объявлений И вектора всех запросов, которые нужны
    ноутбуку 04. Сам ноутбук 04 нейросеть не запускает.
  * Вектора квантуются в целые числа (QUANT_SCALE). Скалярное произведение целых чисел,
    посчитанное во float64, точное при любом порядке сложения, поэтому близость
    запрос–объявление не зависит от BLAS и числа потоков на машине проверяющего.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .params import service_text
from .repro import file_md5

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
QUANT_SCALE = 2 ** 14      # |x| ≤ 1 → |q| ≤ 2^14; сумма 768 произведений < 2^38 — точно во float64


# ─────────────────────────── тексты ───────────────────────────

def query_texts(queries: pd.DataFrame) -> list:
    """Запрос + фильтры: фильтр — часть намерения пользователя."""
    return [f"{QUERY_PREFIX}{q}" + (f" | {f}" if f else "")
            for q, f in zip(queries["search_query"], queries["search_infm_params_text"])]


def item_texts(items: pd.DataFrame, desc_chars: int) -> list:
    """Заголовок + вид/тип услуги + начало описания (адрес не берём: локацию учитывают другие признаки)."""
    return [f"{PASSAGE_PREFIX}{t} | {service_text(p)} | {d[:desc_chars]}"
            for t, p, d in zip(items["item_title_raw"], items["item_infm_params_text"],
                               items["item_description_raw"])]


# ─────────────────────────── устройство ───────────────────────────

def device_info() -> dict:
    """Что за вычислитель доступен: имя, память, поддержка bfloat16."""
    import torch
    if not torch.cuda.is_available():
        return {"device": "cpu", "name": "CPU", "memory_gb": 0.0, "bf16": False}
    props = torch.cuda.get_device_properties(0)
    return {"device": "cuda", "name": props.name, "memory_gb": round(props.total_memory / 2 ** 30, 1),
            "bf16": bool(torch.cuda.is_bf16_supported())}


@dataclass(frozen=True)
class Amp:
    """Настройки смешанной точности. bf16 устойчивее fp16 и не требует GradScaler."""
    enabled: bool
    name: str               # "bf16" | "fp16" | "off"

    @property
    def dtype(self):
        import torch
        return torch.bfloat16 if self.name == "bf16" else torch.float16

    @property
    def needs_scaler(self) -> bool:
        return self.name == "fp16"


def choose_amp(amp_dtype: str, info: dict) -> Amp:
    """amp_dtype: "auto" (bf16, если GPU умеет, иначе fp16) | "bf16" | "fp16" | "off"."""
    if info["device"] != "cuda" or amp_dtype == "off":
        return Amp(False, "off")
    if amp_dtype == "auto":
        amp_dtype = "bf16" if info["bf16"] else "fp16"
    return Amp(True, amp_dtype)


# ─────────────────────────── скачивание модели ───────────────────────────

def fetch_model(source) -> Path:
    """
    Локальная папка с моделью. source — путь к уже скачанной папке или имя на Hugging Face.
    Скачиваются только нужные файлы (конфиг, токенизатор, веса в одном формате), без ONNX
    и дублей весов. Зеркало задаётся переменной окружения HF_ENDPOINT, кэш — HF_HOME.
    """
    path = Path(str(source)).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import HfApi, snapshot_download
    try:
        files = HfApi().list_repo_files(str(source))
    except Exception as error:     # сеть, прокси, неверное имя
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        raise RuntimeError(
            f"Не удалось получить модель {source} с {endpoint}: {type(error).__name__}: {error}\n"
            "Варианты: задать зеркало в настройке HF_ENDPOINT (например, https://hf-mirror.com) "
            "или скачать модель заранее и указать путь к её папке в MODEL_CANDIDATES.") from error
    weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    patterns = ["*.json", "*.model", "*.txt", weights]
    return Path(snapshot_download(str(source), allow_patterns=patterns))


# ─────────────────────────── энкодер ───────────────────────────

class BiEncoder:
    def __init__(self, model, tokenizer, device: str = "cpu", amp: Amp = Amp(False, "off")):
        self.model, self.tokenizer, self.device, self.amp = model.to(device), tokenizer, device, amp

    @classmethod
    def from_pretrained(cls, source, device: str = "cpu", amp: Amp = Amp(False, "off")) -> "BiEncoder":
        """source — имя модели на Hugging Face или путь к скачанной папке."""
        from transformers import AutoModel, AutoTokenizer
        source = str(source)
        return cls(AutoModel.from_pretrained(source), AutoTokenizer.from_pretrained(source), device, amp)

    load = from_pretrained      # загрузка сохранённой дообученной модели — то же самое

    def save(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

    def _forward(self, texts: list, max_len: int, pad_to_max: bool = False):
        """Среднее по токенам с учётом маски + L2-нормировка (как рекомендуют авторы e5)."""
        import torch
        batch = self.tokenizer(texts, padding="max_length" if pad_to_max else True, truncation=True,
                               max_length=max_len, return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=self.amp.dtype, enabled=self.amp.enabled):
            out = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(emb.float(), p=2, dim=1)

    def encode(self, texts: list, max_len: int, batch_size: int = 256, log_every: int = 0) -> np.ndarray:
        """Вектора текстов (float32). Тексты кодируются в порядке длины — так в батче меньше
        паддинга и кодирование заметно быстрее; на выходе порядок исходный."""
        import torch
        self.model.eval()
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for n, start in enumerate(range(0, len(texts), batch_size)):
                idx = order[start:start + batch_size]
                out[idx] = self._forward([texts[i] for i in idx], max_len).cpu().numpy()
                if log_every and n % log_every == 0:
                    print(f"  закодировано {min(start + batch_size, len(texts)):,} из {len(texts):,}")
        return out


# ─────────────────────────── поиск ближайших ───────────────────────────

def _sim_topk(query_emb: np.ndarray, item_emb: np.ndarray, k: int, device: str = "cpu",
              batch: int = 512) -> np.ndarray:
    """Индексы k ближайших объявлений для каждого запроса (косинус = скалярное произведение
    нормированных векторов). Батчами на torch: матрица «все ко всем» не помещается в память,
    а на GPU это на порядки быстрее. Используется только при обучении и его оценке."""
    import torch
    items = torch.from_numpy(np.ascontiguousarray(item_emb)).to(device)
    k = min(k, items.shape[0])
    out = np.empty((len(query_emb), k), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(query_emb), batch):
            q = torch.from_numpy(np.ascontiguousarray(query_emb[start:start + batch])).to(device)
            out[start:start + batch] = torch.topk(q @ items.T, k=k, dim=1).indices.cpu().numpy()
    return out


def dense_recall(query_emb: np.ndarray, item_emb: np.ndarray, positive_idx: np.ndarray,
                 k: int, device: str = "cpu") -> float:
    """Доля запросов, у которых нужное объявление попало в топ-k по близости векторов."""
    top = _sim_topk(query_emb, item_emb, k, device)
    return float((top == positive_idx[:, None]).any(axis=1).mean())


def mine_hard_negatives(query_emb: np.ndarray, item_emb: np.ndarray, positive_idx: np.ndarray,
                        n_neg: int, skip_top: int, depth: int, seed: int, device: str = "cpu") -> np.ndarray:
    """
    Трудные негативы: случайные объявления из окна рангов [skip_top, depth) по близости к запросу
    (собственный позитив исключается).
    Самый верх не берём намеренно: ближайшие объявления часто тоже подходят запросу — просто
    пользователь выбрал другое. Учить модель отталкивать их — значит портить полноту.
    """
    top = _sim_topk(query_emb, item_emb, depth + 1, device)
    rng = np.random.default_rng(seed)
    out = np.empty((len(top), n_neg), dtype=np.int64)
    for row in range(len(top)):
        window = [i for i in top[row] if i != positive_idx[row]][skip_top:depth]
        out[row] = rng.choice(window, size=n_neg, replace=len(window) < n_neg)
    return out


# ─────────────────────────── обучение ───────────────────────────

@dataclass
class TrainPairs:
    queries: list          # тексты запросов (с префиксом)
    positives: list        # тексты выбранных объявлений (с префиксом)
    negatives: list        # списки текстов трудных негативов (может быть пустым)


def _step_texts(pairs: TrainPairs, idx) -> tuple:
    docs = [pairs.positives[i] for i in idx]
    if pairs.negatives:
        docs += [text for i in idx for text in pairs.negatives[i]]
    return [pairs.queries[i] for i in idx], docs


def _loss(encoder: BiEncoder, queries: list, docs: list, cfg, pad_to_max: bool = False):
    """InfoNCE: i-й запрос должен быть ближе всего к i-му документу среди всех документов батча."""
    import torch
    q = encoder._forward(queries, cfg.max_len_query, pad_to_max)
    d = encoder._forward(docs, cfg.max_len_item, pad_to_max)
    logits = (q @ d.T) / cfg.temperature
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(queries), device=encoder.device))


def probe_batch_size(encoder: BiEncoder, cfg, candidates=(512, 384, 320, 256, 192, 160, 128, 96, 64, 48, 32, 16),
                     headroom: float = 0.9) -> int:
    """
    Наибольший батч из candidates, для которого шаг обучения помещается в память GPU
    с запасом (headroom). Проверка — на худшем случае: все тексты дополнены до максимальной
    длины, а память под состояние оптимизатора (2 копии весов в fp32) заранее занята.
    """
    import torch
    if encoder.device != "cuda":
        return candidates[-1]
    model = encoder.model
    total = torch.cuda.get_device_properties(0).total_memory
    n_params = sum(p.numel() for p in model.parameters())
    dummy = TrainPairs(queries=["query: " + "слово " * 200] * max(candidates),
                       positives=["passage: " + "слово " * 400] * max(candidates),
                       negatives=[["passage: " + "слово " * 400] * cfg.hard_negatives] * max(candidates))
    model.train()
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    chosen = None
    for size in candidates:
        reserve = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            reserve = torch.empty(2 * n_params, dtype=torch.float32, device="cuda")   # под Adam
            loss = _loss(encoder, *_step_texts(dummy, range(size)), cfg, pad_to_max=True)
            loss.backward()
            peak = torch.cuda.max_memory_allocated()
            ok = peak <= headroom * total
            print(f"  батч {size}: пик памяти {peak / 2 ** 30:.1f} из {total / 2 ** 30:.1f} ГБ"
                  f"{'' if ok else ' — без запаса'}")
        except torch.cuda.OutOfMemoryError:
            ok = False
            print(f"  батч {size}: не помещается")
        finally:
            model.zero_grad(set_to_none=True)
            del reserve
            loss = None
            torch.cuda.empty_cache()
        if ok:
            chosen = size
            break
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_disable()
    model.eval()
    if chosen is None:
        raise RuntimeError("Даже минимальный батч не помещается в память GPU. "
                           "Включите grad_checkpointing или уменьшите max_len_item.")
    return chosen


def train_biencoder(encoder: BiEncoder, pairs: TrainPairs, cfg, seed: int, log_every: int = 100):
    """
    Дообучение InfoNCE: для каждого запроса батча «свой» документ — позитив, а негативы —
    позитивы остальных запросов батча плюс все трудные негативы батча.
    """
    import torch
    torch.manual_seed(seed)
    model = encoder.model
    model.train()
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()

    n = len(pairs.queries)
    steps = max((n // cfg.batch_size) * cfg.epochs, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=steps, pct_start=cfg.warmup_frac, anneal_strategy="linear")
    scaler = torch.amp.GradScaler(enabled=encoder.amp.needs_scaler)     # нужен только для fp16
    rng = np.random.default_rng(seed)
    history, step = [], 0

    for _ in range(cfg.epochs):
        order = rng.permutation(n)
        for start in range(0, n - cfg.batch_size + 1, cfg.batch_size):
            loss = _loss(encoder, *_step_texts(pairs, order[start:start + cfg.batch_size]), cfg)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % log_every == 0 or step == 1:
                history.append({"step": step, "loss": float(loss.item())})
                print(f"  шаг {step}/{steps}: loss {loss.item():.4f}")
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_disable()
    model.eval()
    return encoder, pd.DataFrame(history)


# ─────────────────────────── артефакт ───────────────────────────
#
# Папка артефакта:
#   model/                    дообученная модель (нужна только для дообучения/перекодирования)
#   items.parquet             item_id в порядке строк item_embeddings.npy
#   item_embeddings.npy       вектора объявлений, float16
#   queries.parquet           тексты запросов в порядке строк query_embeddings.npy
#   query_embeddings.npy      вектора запросов (бенчмарк, валидация, фолды), float16
#   manifest.json             параметры, метрики и md5 всех файлов

def save_artifact(directory, item_ids, item_emb: np.ndarray, texts, query_emb: np.ndarray, meta: dict) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "item_embeddings.npy", item_emb.astype(np.float16))
    np.save(directory / "query_embeddings.npy", query_emb.astype(np.float16))
    pd.DataFrame({"item_id": list(item_ids)}).to_parquet(directory / "items.parquet", index=False)
    pd.DataFrame({"text": list(texts)}).to_parquet(directory / "queries.parquet", index=False)
    files = ["item_embeddings.npy", "query_embeddings.npy", "items.parquet", "queries.parquet"]
    manifest = dict(meta, n_items=len(item_ids), n_queries=len(texts), dim=int(item_emb.shape[1]),
                    md5={f: file_md5(directory / f) for f in files})
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
    return manifest


def quantize(x: np.ndarray) -> np.ndarray:
    """Вектора → целые числа (хранятся во float32 точно, т.к. |q| ≤ 2^14 < 2^24)."""
    return np.rint(x.astype(np.float32) * QUANT_SCALE).astype(np.float32)


def exact_dot(query_q: np.ndarray, item_q: np.ndarray, chunk: int = 16384) -> np.ndarray:
    """
    Косинусная близость квантованных векторов (запросы × объявления), float64.
    Произведения и суммы целых чисел < 2^53 во float64 точны, поэтому результат одинаков
    при любом порядке сложения внутри BLAS — на любой машине и при любом числе потоков.
    """
    q64 = query_q.astype(np.float64)
    out = np.empty((len(query_q), len(item_q)), dtype=np.float64)
    for start in range(0, len(item_q), chunk):
        out[:, start:start + chunk] = q64 @ item_q[start:start + chunk].astype(np.float64).T
    out /= float(QUANT_SCALE) ** 2
    return out


def _lookup(keys, saved_keys, matrix: np.ndarray, what: str):
    position = {v: i for i, v in enumerate(saved_keys)}
    rows = np.array([position.get(v, -1) for v in keys], dtype=np.int64)
    found = rows >= 0
    out = np.zeros((len(keys), matrix.shape[1]), dtype=np.float32)
    out[found] = matrix[rows[found]]
    if not found.all():
        print(f"[warn] нет векторов для {int((~found).sum()):,} {what} из {len(keys):,}")
    return out, found


def load_item_embeddings(directory, item_ids) -> np.ndarray:
    """Квантованные вектора в порядке строк корпуса. Нет вектора — нулевой (близость 0)."""
    directory = Path(directory)
    saved = pd.read_parquet(directory / "items.parquet")["item_id"].tolist()
    emb, _ = _lookup(list(item_ids), saved, np.load(directory / "item_embeddings.npy"), "объявлений")
    return quantize(emb)


def load_query_embeddings(directory, texts: list, encode_missing=None) -> np.ndarray:
    """
    Квантованные вектора запросов по их текстам. Если каких-то текстов в артефакте нет
    (например, артефакт собран в режиме DRY_RUN), их можно докодировать функцией
    encode_missing(texts) -> np.ndarray — но тогда результат будет зависеть от железа.
    """
    directory = Path(directory)
    saved = pd.read_parquet(directory / "queries.parquet")["text"].tolist()
    emb, found = _lookup(texts, saved, np.load(directory / "query_embeddings.npy"), "запросов")
    if not found.all() and encode_missing is not None:
        missing = [t for t, f in zip(texts, found) if not f]
        print(f"  докодирую {len(missing):,} запросов моделью из артефакта")
        emb[~found] = encode_missing(missing).astype(np.float16).astype(np.float32)
    return quantize(emb)


# ─────────────────────────── отладочная модель ───────────────────────────

def build_debug_encoder(texts: list, device: str = "cpu", vocab_size: int = 2000, dim: int = 32) -> BiEncoder:
    """Крошечная модель со случайными весами и токенизатором, обученным на переданных текстах.
    Нужна только для smoke-теста пайплайна без скачивания весов (SMOKE_TEST=1)."""
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    torch.manual_seed(0)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    model = BertModel(BertConfig(vocab_size=fast.vocab_size, hidden_size=dim, num_hidden_layers=2,
                                 num_attention_heads=2, intermediate_size=2 * dim, max_position_embeddings=512))
    return BiEncoder(model, fast, device)
