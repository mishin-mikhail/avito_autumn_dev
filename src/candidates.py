"""
Пул кандидатов и признаки пар «запрос — объявление».

Весь корпус скорится плотно (батчами по cfg.batch_size запросов), затем
объединяются несколько списков — у каждого своя «специализация»:
  * src_text       топ по тексту (BM25 полей + покрытие лемм запроса);
  * src_text_loc   топ по тексту с приоритетом близких объявлений;
  * src_prior_loc  близкие объявления из самых вероятных микрокатегорий;
  * src_memo       объявления, выбранные по такому же запросу в train
                   (только если объявления корпуса встречаются в train).
«Близость» = max(совпадение локации, P(перехода локаций), exp(-расстояние / geo_near_km)).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bm25 import BM25Field, CoverageField, row_max_normalize
from .filters import normalize_params, parse_filter_pairs, pool_match_share

BASE_FEATURES = [
    "title", "params", "desc", "cov",           # текст
    "filt", "filt_exact",                       # фильтры поиска
    "logp",                                     # log P(микрокатегория | запрос)
    "loc_same", "loc_p", "loc_logp", "log_dist",  # локация
    "rating_ok", "rating", "log_reviews",       # качество объявления
]
ITEM_STAT_FEATURES = ["log_pop", "log_memo"]
SOURCES = ["src_text", "src_text_loc", "src_prior_loc", "src_memo"]


def feature_names(use_item_stats: bool) -> list:
    return BASE_FEATURES + (ITEM_STAT_FEATURES if use_item_stats else [])


# ─────────────────────────── словари id ───────────────────────────

class Vocab:
    """Строковые id → коды 0..n-1; неизвестное значение → код n (`unknown`)."""

    def __init__(self, values):
        self.index = {v: i for i, v in enumerate(sorted(dict.fromkeys(values)))}
        self.unknown = len(self.index)

    def encode(self, values) -> np.ndarray:
        get, unk = self.index.get, self.unknown
        return np.fromiter((get(v, unk) for v in values), dtype=np.int64, count=len(values))

    def __len__(self):
        return len(self.index)

    @property
    def size_with_unknown(self) -> int:
        return len(self.index) + 1


@dataclass
class Vocabs:
    loc: Vocab
    micro: Vocab


def build_vocabs(query_frames: list, item_frames: list) -> Vocabs:
    """Общие словари по всем источникам, чтобы коды совпадали на валидации и бенчмарке."""
    locs = [f["search_location_id"] for f in query_frames] + [f["item_location_id"] for f in item_frames]
    micros = [f["item_microcat_id"] for f in item_frames]
    return Vocabs(loc=Vocab(pd.concat(locs).tolist()), micro=Vocab(pd.concat(micros).tolist()))


# ─────────────────────────── корпус и запросы ───────────────────────────

@dataclass
class Corpus:
    item_ids: np.ndarray
    rank: np.ndarray          # ранг item_id в лексикографическом порядке (тай-брейк)
    loc: np.ndarray
    micro: np.ndarray
    lat32: np.ndarray         # координаты (float32 — для расчёта расстояний батчами)
    lon32: np.ndarray
    rating_raw: np.ndarray    # с NaN — для проверки фильтра по рейтингу
    rating: np.ndarray
    log_reviews: np.ndarray
    log_pop: np.ndarray
    params_norm: list         # параметры с сохранённым регистром — для filt_exact
    fields: dict              # BM25 по полям
    cov: CoverageField        # покрытие лемм запроса заголовком и параметрами

    @property
    def n(self) -> int:
        return len(self.item_ids)


def build_corpus(items: pd.DataFrame, lemmas: dict, cfg, vocabs: Vocabs) -> Corpus:
    """lemmas — лемматизированные поля {'title','params','desc'} в порядке строк items."""
    ids = items["item_id"].to_numpy(dtype=object)
    assert len(pd.unique(ids)) == len(ids), "item_id в корпусе должны быть уникальны"
    rank = np.empty(len(ids), dtype=np.int64)
    rank[np.argsort(ids.astype(str), kind="stable")] = np.arange(len(ids))

    fields = {name: BM25Field(cfg.bm25_k1, cfg.bm25_b).fit(docs) for name, docs in lemmas.items()}
    cov = CoverageField().fit([f"{t} {p}" for t, p in zip(lemmas["title"], lemmas["params"])])
    rating_raw = items["item_rating"].to_numpy(np.float32)
    reviews = np.nan_to_num(items["item_rating_reviews_count"].to_numpy(np.float32), nan=0.0)
    return Corpus(
        item_ids=ids, rank=rank,
        loc=vocabs.loc.encode(items["item_location_id"].tolist()),
        micro=vocabs.micro.encode(items["item_microcat_id"].tolist()),
        lat32=items["item_latitude"].to_numpy(np.float32),
        lon32=items["item_longitude"].to_numpy(np.float32),
        rating_raw=rating_raw,
        rating=np.nan_to_num(rating_raw, nan=0.0),
        log_reviews=np.log1p(np.maximum(reviews, 0)).astype(np.float32),
        log_pop=np.zeros(len(ids), np.float32),
        params_norm=[normalize_params(t) for t in items["item_infm_params_text"]],
        fields=fields, cov=cov,
    )


@dataclass
class QuerySet:
    lemmas: list          # леммы текста запроса через пробел
    lemma_key: list       # отсортированные уникальные леммы («мешок лемм»)
    filt_lemmas: list     # леммы фильтров
    filter_pairs: list    # разобранные пары фильтров для точного сопоставления
    loc: np.ndarray
    loc_unknown: int      # код «неизвестной локации» (с ним совпадение не засчитывается)
    rating_thr: np.ndarray
    prior: np.ndarray     # запросы × микрокатегории

    @property
    def n(self) -> int:
        return len(self.lemmas)


def build_queries(q: pd.DataFrame, lem, vocabs: Vocabs, prior_model) -> QuerySet:
    lemma_key = lem.key_many(q["search_query"])
    return QuerySet(
        lemmas=lem.join_many(q["search_query"]),
        lemma_key=lemma_key,
        filt_lemmas=lem.join_many(q["search_infm_params_text"]),
        filter_pairs=[parse_filter_pairs(t) for t in q["search_infm_params_text"]],
        loc=vocabs.loc.encode(q["search_location_id"].tolist()),
        loc_unknown=vocabs.loc.unknown,
        rating_thr=q["rating_thr"].to_numpy(np.float32),
        prior=prior_model.transform(lemma_key, q["search_category"].tolist()),
    )


# ─────────────────────────── отбор топ-K ───────────────────────────

def topk_rows(score: np.ndarray, k: int, tie: np.ndarray, decimals: int) -> np.ndarray:
    """
    Индексы топ-k по строкам без случайных ничьих.
    Ключ = округлённый скор (старшие биты) + тай-брейк по item_id (младшие биты):
    все ключи различны, поэтому результат однозначен на любой машине.
    """
    k = min(k, score.shape[1])
    shift = int(len(tie)).bit_length()
    key = score.astype(np.float64)
    key *= 10 ** decimals
    np.rint(key, out=key)
    key = key.astype(np.int64)
    key <<= shift
    key += tie[None, :]
    np.negative(key, out=key)                     # argpartition ищет минимумы
    idx = np.argpartition(key, k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(key, idx, axis=1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


# ─────────────────────────── пул ───────────────────────────

def _memo_counts(uniq: np.ndarray, m_idx: np.ndarray, m_cnt: np.ndarray) -> np.ndarray:
    """Счётчики «памяти» для отсортированного массива кандидатов uniq."""
    out = np.zeros(len(uniq), np.float32)
    if len(m_idx):
        order = np.argsort(m_idx)
        pos = np.minimum(np.searchsorted(m_idx[order], uniq), len(m_idx) - 1)
        hit = m_idx[order][pos] == uniq
        out[hit] = m_cnt[order][pos[hit]]
    return out


def generate_pool(corpus: Corpus, qs: QuerySet, geo, cfg, item_stats=None, verbose: bool = True) -> pd.DataFrame:
    """Строки пула: q (номер запроса), item (индекс в корпусе), признаки и флаги источников."""
    tie = (corpus.n - 1 - corpus.rank).astype(np.int64)   # меньший item_id → больший тай-брейк
    dec, f = cfg.score_decimals, corpus.fields
    Q = {name: f[name].query_matrix(qs.lemmas) for name in ("title", "params", "desc")}
    Q_filt = f["params"].query_matrix(qs.filt_lemmas)
    Q_cov = corpus.cov.query_matrix(qs.lemmas)
    qlen = np.array([max(len(k.split()), 1) for k in qs.lemma_key], dtype=np.float32)
    no_memo = (np.zeros(0, np.int64), np.zeros(0, np.float32))

    parts = []
    for start in range(0, qs.n, cfg.batch_size):
        stop = min(start + cfg.batch_size, qs.n)
        sl = slice(start, stop)
        q_loc = qs.loc[sl]

        # --- плотные скоры: запросы батча × весь корпус ---
        title = row_max_normalize(f["title"].score(Q["title"][sl]))
        params = row_max_normalize(f["params"].score(Q["params"][sl]))
        desc = row_max_normalize(f["desc"].score(Q["desc"][sl]))
        filt = row_max_normalize(f["params"].score(Q_filt[sl]))
        cov = corpus.cov.score(Q_cov[sl]) / qlen[sl, None]
        prior = qs.prior[sl][:, corpus.micro]
        same = ((corpus.loc[None, :] == q_loc[:, None]) & (q_loc[:, None] != qs.loc_unknown)).astype(np.float32)
        loc_p, dist, known = geo.batch(q_loc, corpus)
        proximity = np.exp(dist / np.float32(-cfg.geo_near_km))
        proximity[~known] = 0.0
        np.maximum(proximity, same, out=proximity)
        np.maximum(proximity, loc_p, out=proximity)

        # --- списки кандидатов ---
        text0 = title + 0.5 * params + 0.3 * desc + cov   # фиксированная формула только для отбора
        lists = {"src_text": topk_rows(text0, cfg.pool_k_text, tie, dec)}
        lists["src_prior_loc"] = topk_rows(prior + proximity + 1e-3 * text0, cfg.pool_k_prior_loc, tie, dec)
        proximity *= 3.0
        proximity += text0
        lists["src_text_loc"] = topk_rows(proximity, cfg.pool_k_text_loc, tie, dec)
        del proximity, text0

        rows, cols, memo, flags = [], [], [], {k: [] for k in SOURCES}
        for r in range(stop - start):
            m_idx, m_cnt = item_stats.memo_for(qs.lemma_key[start + r]) if item_stats else no_memo
            m_idx, m_cnt = m_idx[: cfg.pool_k_memo], m_cnt[: cfg.pool_k_memo]
            per_source = [lists[k][r] for k in SOURCES[:3]] + [m_idx]
            uniq = np.unique(np.concatenate(per_source))       # отсортировано → детерминировано
            rows.append(np.full(len(uniq), r, dtype=np.int64))
            cols.append(uniq)
            memo.append(_memo_counts(uniq, m_idx, m_cnt))
            for k, arr in zip(SOURCES, per_source):
                flags[k].append(np.isin(uniq, arr))

        # --- признаки для строк пула ---
        rr, cc = np.concatenate(rows), np.concatenate(cols)
        thr = qs.rating_thr[start + rr]
        part = pd.DataFrame({
            "q": (start + rr).astype(np.int32),
            "item": cc.astype(np.int32),
            "title": title[rr, cc], "params": params[rr, cc], "desc": desc[rr, cc], "cov": cov[rr, cc],
            "filt": filt[rr, cc],
            "logp": np.log(prior[rr, cc] + 1e-6).astype(np.float32),
            "loc_same": same[rr, cc],
            "loc_p": loc_p[rr, cc],
            "loc_logp": np.log(loc_p[rr, cc] + 1e-4).astype(np.float32),
            "log_dist": np.log1p(dist[rr, cc]).astype(np.float32),
            "rating_ok": (np.isnan(thr) | (corpus.rating_raw[cc] >= thr)).astype(np.float32),
            "rating": corpus.rating[cc],
            "log_reviews": corpus.log_reviews[cc],
            "log_pop": corpus.log_pop[cc],
            "log_memo": np.log1p(np.concatenate(memo)).astype(np.float32),
        })
        for k in SOURCES:
            part[k] = np.concatenate(flags[k])
        parts.append(part)
        # освобождаем плотные массивы до следующего батча, иначе пик памяти удваивается
        del title, params, desc, filt, cov, prior, same, loc_p, dist, lists
        if verbose and (start // cfg.batch_size) % 5 == 0:
            print(f"  пул: {stop}/{qs.n} запросов")

    pool = pd.concat(parts, ignore_index=True)
    pool["filt_exact"] = pool_match_share(pool["q"].to_numpy(), pool["item"].to_numpy(),
                                          qs.filter_pairs, corpus.params_norm)
    return pool
