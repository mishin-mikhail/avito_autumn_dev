"""
Генерация пула кандидатов и признаков для каждой пары «запрос — объявление».

Для каждого запроса весь корпус скорится плотно (батчами), затем объединяются
несколько списков, каждый отвечает за свой тип «попадания»:
  * text       — топ по тексту (BM25 по полям + покрытие лемм запроса);
  * text_loc   — то же, но объявления своей локации идут первыми;
  * prior_loc  — объявления своей локации из самых вероятных микрокатегорий;
  * memo       — объявления, выбранные по такому же запросу в train.
Пул (~1000 кандидатов на запрос) затем переранжируется линейной формулой
(tuning.py), а на следующих этапах — обученной моделью.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bm25 import BM25Field, CoverageField, row_max_normalize
from .data import item_texts

FEATURES = ["title", "params", "desc", "filt", "cov", "logp", "loc", "loc_deliv",
            "rating_ok", "rating", "log_reviews", "log_pop", "log_memo"]
SOURCES = ["src_text", "src_text_loc", "src_prior_loc", "src_memo"]


class Vocab:
    """Отображение строковых id в целые коды. Неизвестное значение → missing."""

    def __init__(self, values):
        self.index = {v: i for i, v in enumerate(sorted(dict.fromkeys(values)))}

    def encode(self, values, missing: int = -1) -> np.ndarray:
        return np.fromiter((self.index.get(v, missing) for v in values), dtype=np.int64, count=len(values))

    def __len__(self):
        return len(self.index)


@dataclass
class Corpus:
    item_ids: np.ndarray
    rank: np.ndarray        # ранг item_id в лексикографическом порядке (тай-брейк)
    loc: np.ndarray
    micro: np.ndarray
    rating_raw: np.ndarray  # с NaN — для проверки фильтра
    rating: np.ndarray
    log_reviews: np.ndarray
    log_pop: np.ndarray
    fields: dict
    cov: CoverageField

    @property
    def n(self) -> int:
        return len(self.item_ids)


def build_corpus(items: pd.DataFrame, lem, cfg, loc_vocab: Vocab, micro_vocab: Vocab) -> Corpus:
    ids = items["item_id"].to_numpy(dtype=object)
    assert len(pd.unique(ids)) == len(ids), "item_id в корпусе должны быть уникальны"
    rank = np.empty(len(ids), dtype=np.int64)
    rank[np.argsort(ids.astype(str), kind="stable")] = np.arange(len(ids))

    texts = item_texts(items, cfg.desc_max_chars)
    lem_texts = {k: lem.join_many(v) for k, v in texts.items()}
    fields = {k: BM25Field(cfg.bm25_k1, cfg.bm25_b).fit(v) for k, v in lem_texts.items()}
    cov = CoverageField().fit([f"{t} {p}" for t, p in zip(lem_texts["title"], lem_texts["params"])])

    rating_raw = items["item_rating"].to_numpy(np.float32)
    reviews = np.nan_to_num(items["item_rating_reviews_count"].to_numpy(np.float32), nan=0.0)
    return Corpus(
        item_ids=ids, rank=rank,
        loc=loc_vocab.encode(items["item_location_id"].tolist(), missing=-2),
        micro=micro_vocab.encode(items["item_microcat_id"].tolist(), missing=0),
        rating_raw=rating_raw,
        rating=np.nan_to_num(rating_raw, nan=0.0),
        log_reviews=np.log1p(np.maximum(reviews, 0)).astype(np.float32),
        log_pop=np.zeros(len(ids), np.float32),
        fields=fields, cov=cov,
    )


@dataclass
class QuerySet:
    lemmas: list        # леммы текста запроса (строка через пробел)
    lemma_key: list     # отсортированные уникальные леммы
    filt_lemmas: list   # леммы фильтров
    loc: np.ndarray
    deliv: np.ndarray
    rating_thr: np.ndarray
    prior: np.ndarray   # запросы × микрокатегории

    @property
    def n(self) -> int:
        return len(self.lemmas)


def build_queries(q: pd.DataFrame, lem, loc_vocab: Vocab, prior_model) -> QuerySet:
    lemma_key = lem.key_many(q["search_query"])
    return QuerySet(
        lemmas=lem.join_many(q["search_query"]),
        lemma_key=lemma_key,
        filt_lemmas=lem.join_many(q["search_infm_params_text"]),
        loc=loc_vocab.encode(q["search_location_id"].tolist(), missing=-1),
        deliv=q["search_is_delivery_search"].to_numpy(np.float32),
        rating_thr=q["rating_thr"].to_numpy(np.float32),
        prior=prior_model.transform(lemma_key, q["search_category"].tolist()),
    )


def topk_rows(score: np.ndarray, k: int, tie: np.ndarray, decimals: int) -> np.ndarray:
    """
    Индексы топ-k по строкам без случайных ничьих.
    Ключ = округлённый скор (старшие биты) + тай-брейк по item_id (младшие биты),
    поэтому все ключи различны и результат однозначен на любой машине.
    """
    k = min(k, score.shape[1])
    shift = int(len(tie)).bit_length()
    s = np.rint(score.astype(np.float64) * 10 ** decimals).astype(np.int64)
    key = (s << shift) + tie[None, :]
    idx = np.argpartition(-key, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(key, idx, axis=1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def generate_pool(corpus: Corpus, qs: QuerySet, cfg, item_stats=None, verbose: bool = True) -> pd.DataFrame:
    tie = (corpus.n - 1 - corpus.rank).astype(np.int64)   # меньший item_id → больший тай-брейк
    f = corpus.fields
    Qt, Qp, Qd = (f[k].query_matrix(qs.lemmas) for k in ("title", "params", "desc"))
    Qf = f["params"].query_matrix(qs.filt_lemmas)
    Qc = corpus.cov.query_matrix(qs.lemmas)
    qlen = np.array([max(len(k.split()), 1) for k in qs.lemma_key], dtype=np.float32)
    dec = cfg.score_decimals

    parts = []
    for s in range(0, qs.n, cfg.batch_size):
        e = min(s + cfg.batch_size, qs.n)
        sl = slice(s, e)
        title = row_max_normalize(f["title"].score(Qt[sl]))
        params = row_max_normalize(f["params"].score(Qp[sl]))
        desc = row_max_normalize(f["desc"].score(Qd[sl]))
        filt = row_max_normalize(f["params"].score(Qf[sl]))
        cov = corpus.cov.score(Qc[sl]) / qlen[sl, None]
        prior = qs.prior[sl][:, corpus.micro]
        loc = (corpus.loc[None, :] == qs.loc[sl, None]).astype(np.float32)

        # «дефолтная» текстовая формула только для отбора кандидатов в пул
        text0 = title + 0.5 * params + 0.3 * desc + cov
        lists = {
            "src_text": topk_rows(text0, cfg.pool_k_text, tie, dec),
            "src_text_loc": topk_rows(text0 + 3.0 * loc, cfg.pool_k_text_loc, tie, dec),
            "src_prior_loc": topk_rows(prior + loc + 1e-3 * text0, cfg.pool_k_prior_loc, tie, dec),
        }

        rows, cols, memo_cnt, flags = [], [], [], {k: [] for k in SOURCES}
        for r in range(e - s):
            if item_stats is not None:
                m_idx, m_cnt = item_stats.memo_for(qs.lemma_key[s + r])
                m_idx, m_cnt = m_idx[: cfg.pool_k_memo], m_cnt[: cfg.pool_k_memo]
            else:
                m_idx, m_cnt = np.zeros(0, np.int64), np.zeros(0, np.float32)
            per_list = [lists[k][r] for k in SOURCES[:3]] + [m_idx]
            uniq = np.unique(np.concatenate(per_list))          # отсортировано → детерминировано
            rows.append(np.full(len(uniq), r, dtype=np.int64))
            cols.append(uniq)
            for k, arr in zip(SOURCES, per_list):
                flags[k].append(np.isin(uniq, arr))
            mc = np.zeros(len(uniq), np.float32)
            if len(m_idx):
                o = np.argsort(m_idx)
                pos = np.searchsorted(m_idx[o], uniq)
                pos = np.minimum(pos, len(m_idx) - 1)
                hit = m_idx[o][pos] == uniq
                mc[hit] = m_cnt[o][pos[hit]]
            memo_cnt.append(mc)

        rr, cc = np.concatenate(rows), np.concatenate(cols)
        thr = qs.rating_thr[s + rr]
        part = pd.DataFrame({
            "q": (s + rr).astype(np.int32),
            "item": cc.astype(np.int32),
            "title": title[rr, cc], "params": params[rr, cc], "desc": desc[rr, cc],
            "filt": filt[rr, cc], "cov": cov[rr, cc],
            "logp": np.log(prior[rr, cc] + 1e-6).astype(np.float32),
            "loc": loc[rr, cc],
            "loc_deliv": loc[rr, cc] * qs.deliv[s + rr],
            "rating_ok": (np.isnan(thr) | (corpus.rating_raw[cc] >= thr)).astype(np.float32),
            "rating": corpus.rating[cc],
            "log_reviews": corpus.log_reviews[cc],
            "log_pop": corpus.log_pop[cc],
            "log_memo": np.log1p(np.concatenate(memo_cnt)).astype(np.float32),
        })
        for k in SOURCES:
            part[k] = np.concatenate(flags[k])
        parts.append(part)
        if verbose and (s // cfg.batch_size) % 5 == 0:
            print(f"  pool: {e}/{qs.n} запросов, строк в батче {len(part)}")
    return pd.concat(parts, ignore_index=True)


def attach_labels(pool: pd.DataFrame, corpus: Corpus, truth: list) -> pd.DataFrame:
    """label = 1, если объявление из пула входит в эталон запроса."""
    idx_of = {v: i for i, v in enumerate(corpus.item_ids)}
    pairs = pd.DataFrame(
        [(qi, idx_of[t]) for qi, rel in enumerate(truth) for t in rel if t in idx_of],
        columns=["q", "item"],
    ).astype(np.int32)
    pairs["label"] = np.int8(1)
    pool = pool.merge(pairs, on=["q", "item"], how="left", sort=False)
    pool["label"] = pool["label"].fillna(0).astype(np.int8)
    return pool


def source_recall(pool: pd.DataFrame, n_rel: np.ndarray) -> pd.DataFrame:
    """Какую долю эталона ловит каждый список и весь пул (recall при своём K)."""
    out = {}
    pos = pool[pool["label"] == 1]
    for k in SOURCES + ["pool"]:
        hits = pos if k == "pool" else pos[pos[k]]
        per_q = np.bincount(hits["q"], minlength=len(n_rel)) / np.maximum(n_rel, 1)
        out[k] = float(per_q[n_rel > 0].mean())
    sizes = pool.groupby("q").size()
    return pd.DataFrame({"recall": out}).assign(
        avg_pool_size=float(sizes.mean()))
