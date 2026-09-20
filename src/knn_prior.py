"""
P(микрокатегория | запрос) по похожим запросам train (v5).

Зачем. Априорная вероятность микрокатегории из priors.py опирается на леммы запроса:
для знакомых текстов она точная, а для новых (62% бенчмарка) сваливается к отдельным леммам
и категории поиска. Смысловые соседи дают лучшую оценку: у запроса «замена гидроаккумулятора»
нет знакомых лемм, но есть близкие по смыслу тексты train («ремонт скважины», «насос для
скважины»), и по ним видно, какие микрокатегории выбирали.

Как считается (для одного запроса):
  1. k ближайших текстов train по векторам дообученного энкодера - только среди текстов,
     которые есть в строках статистик этой выборки (иначе утечка эталона);
  2. у каждого соседа - распределение выбранных микрокатегорий;
  3. среднее распределений с весами  max(sim − sim_лучшего + margin, 0) · n / (n + shrink):
     в счёт идут только соседи, почти такие же близкие, как лучший, а текстам с одной-двумя
     строками доверие меньше.

Воспроизводимость. Вектора квантуются в целые числа с шагом 2^-10: сумма произведений тогда
по модулю < 2^24 и во float32 считается точно при любом порядке сложения, то есть поиск соседей
не зависит от BLAS и числа потоков. Ничьи разрешаются по номеру текста.

Вектора всех текстов train - дополнение к артефакту 03_embeddings (train_queries.*):
их один раз кодирует ноутбук 05 моделью из артефакта, дальше они только читаются.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .encoder import QUANT_SCALE, query_texts
from .repro import file_md5

KNN_SCALE = 2 ** 10
TEXTS_FILE = "train_queries.parquet"
EMB_FILE = "train_query_embeddings.npy"
MANIFEST_FILE = "train_queries.json"


# ─────────────────────────── тексты запросов train ───────────────────────────

def train_text_codes(train: pd.DataFrame):
    """Уникальные тексты запросов train в формате энкодера (запрос + фильтры), отсортированные,
    и номер текста для каждой строки train."""
    codes, texts = pd.factorize(pd.Series(query_texts(train), dtype=object), sort=True)
    return list(texts), codes.astype(np.int32)


def train_query_embeddings(directory, texts: list, encode, meta: dict = None) -> np.ndarray:
    """
    Вектора текстов train (float16) из дополнения к артефакту. Если файлов нет или каких-то
    текстов в них не хватает, недостающие кодируются функцией encode(texts) -> np.ndarray
    и дополнение перезаписывается. Повторный запуск берёт вектора из файлов - ответ не зависит
    от того, на каком железе работала нейросеть.
    """
    directory = Path(directory)
    saved_texts, saved = [], np.zeros((0, 0), np.float16)
    if (directory / TEXTS_FILE).is_file() and (directory / EMB_FILE).is_file():
        saved_texts = pd.read_parquet(directory / TEXTS_FILE)["text"].tolist()
        saved = np.load(directory / EMB_FILE)
    position = {t: i for i, t in enumerate(saved_texts)}
    missing = [t for t in texts if t not in position]
    if missing:
        print(f"кодирую {len(missing):,} текстов запросов train (есть в артефакте: {len(texts) - len(missing):,})")
        new = encode(missing).astype(np.float16)
        all_texts = saved_texts + missing
        all_emb = new if not len(saved) else np.concatenate([saved, new])
        order = np.argsort(np.array(all_texts, dtype=object), kind="stable")
        all_texts, saved = [all_texts[i] for i in order], all_emb[order]
        np.save(directory / EMB_FILE, saved)
        pd.DataFrame({"text": all_texts}).to_parquet(directory / TEXTS_FILE, index=False)
        manifest = dict(meta or {}, n_texts=len(all_texts), dim=int(saved.shape[1]),
                        md5={f: file_md5(directory / f) for f in (TEXTS_FILE, EMB_FILE)})
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
        position = {t: i for i, t in enumerate(all_texts)}
    return saved[np.array([position[t] for t in texts], dtype=np.int64)]


# ─────────────────────────── точный поиск соседей ───────────────────────────

class TextIndex:
    """Вектора текстов train, квантованные с шагом 2^-10 (int16), и точный поиск ближайших."""

    def __init__(self, emb: np.ndarray):
        self.codes = np.rint(emb.astype(np.float32) * KNN_SCALE).astype(np.int16)
        self.n = len(self.codes)
        self.shift = max(self.n, 1).bit_length()

    def search(self, query_q: np.ndarray, allowed: np.ndarray, k: int, q_chunk: int = 1024,
               t_chunk: int = 16384):
        """
        query_q - вектора запросов, квантованные encoder.quantize (шаг 2^-14);
        allowed - маска текстов, среди которых искать.
        Возвращает (номера соседей int32 [n, k'], близость float32 [n, k']), k' = min(k, allowed.sum()),
        соседи по убыванию близости, при равенстве - меньший номер текста первым.
        """
        ids = np.flatnonzero(allowed)
        k = min(k, len(ids))
        n_q = len(query_q)
        if k == 0:
            return np.zeros((n_q, 0), np.int32), np.zeros((n_q, 0), np.float32)
        qq = np.rint(query_q / (QUANT_SCALE // KNN_SCALE)).astype(np.float32)   # тот же шаг 2^-10
        out_key = np.empty((n_q, k), dtype=np.int64)
        for qs in range(0, n_q, q_chunk):
            q_block = qq[qs:qs + q_chunk]
            best = None
            for ts in range(0, len(ids), t_chunk):
                t_ids = ids[ts:ts + t_chunk]
                score = q_block @ self.codes[t_ids].astype(np.float32).T      # целые числа, точно
                key = score.astype(np.int64) << self.shift
                key += (self.n - 1 - t_ids)[None, :]                         # тай-брейк: меньший номер выше
                if key.shape[1] > k:
                    key = np.take_along_axis(key, np.argpartition(-key, k - 1, axis=1)[:, :k], axis=1)
                best = key if best is None else np.concatenate([best, key], axis=1)
                if best.shape[1] > k:
                    best = np.take_along_axis(best, np.argpartition(-best, k - 1, axis=1)[:, :k], axis=1)
            out_key[qs:qs + q_chunk] = -np.sort(-best, axis=1)
        score = out_key >> self.shift
        idx = (self.n - 1 - (out_key - (score << self.shift))).astype(np.int32)
        return idx, (score / float(KNN_SCALE) ** 2).astype(np.float32)


# ─────────────────────────── распределение микрокатегорий по соседям ───────────────────────────

class KnnPrior:
    def __init__(self, margin: float, shrink: float):
        self.margin, self.shrink = margin, shrink

    def fit(self, text_code: np.ndarray, micro_idx: np.ndarray, n_texts: int, n_micro: int) -> "KnnPrior":
        """Строки статистик: номер текста и код микрокатегории выбранного объявления."""
        counts = sp.coo_matrix((np.ones(len(text_code)), (text_code, micro_idx)), shape=(n_texts, n_micro)).tocsr()
        n = np.asarray(counts.sum(axis=1)).ravel()
        self.available = n > 0
        self.dist = (sp.diags(np.divide(1.0, n, out=np.zeros_like(n), where=n > 0)) @ counts).tocsr()
        self.trust = n / (n + self.shrink)
        self.n_micro = n_micro
        return self

    def transform(self, nbr_idx: np.ndarray, nbr_sim: np.ndarray):
        """(P(микрокатегория) по соседям [n, n_micro] float32, близость лучшего соседа [n] float32)."""
        n_q, k = nbr_idx.shape
        if k == 0:
            return np.zeros((n_q, self.n_micro), np.float32), np.zeros(n_q, np.float32)
        sim = nbr_sim.astype(np.float64)
        w = np.maximum(sim - sim[:, :1] + self.margin, 0.0) * self.trust[nbr_idx]
        W = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n_q), k), nbr_idx.ravel())), shape=(n_q, len(self.trust)))
        prior = (W @ self.dist).toarray()
        total = w.sum(axis=1, keepdims=True)
        prior = np.divide(prior, total, out=np.zeros_like(prior), where=total > 0)
        return prior.astype(np.float32), nbr_sim[:, 0].astype(np.float32)
