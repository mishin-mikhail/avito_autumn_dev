"""
Выбор топ-50 из пула: линейная формула над признаками и подбор её весов.

Скор считается поэлементными операциями в фиксированном порядке (без BLAS):
сложение и умножение IEEE-754 дают одинаковый результат на любой машине,
а матричное умножение в BLAS может менять порядок суммирования.
"""
import numpy as np
import pandas as pd


def weights_vector(names: list, weights) -> np.ndarray:
    """dict/пары (признак, вес) → вектор в порядке names; отсутствующие признаки = 0."""
    w = dict(weights)
    return np.array([w.get(n, 0.0) for n in names], dtype=np.float64)


def linear_score(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = np.zeros(F.shape[0], dtype=np.float64)
    for j in range(F.shape[1]):
        if w[j] != 0.0:
            s += w[j] * F[:, j]
    return s


def rank_order(q: np.ndarray, rank: np.ndarray, score: np.ndarray, decimals: int) -> np.ndarray:
    """Порядок строк: запрос ↑, округлённый скор ↓, item_id ↑.
    Все три ключа упаковываются в одно int64 (быстрее lexsort), если хватает битов."""
    s = np.rint(score * 10 ** decimals).astype(np.int64)
    s_min, s_max = int(s.min()), int(s.max())
    q_bits = max(int(q.max()), 1).bit_length()
    s_bits = max(s_max - s_min, 1).bit_length()
    r_bits = max(int(rank.max()), 1).bit_length()
    if q_bits + s_bits + r_bits <= 62:
        key = (q.astype(np.int64) << (s_bits + r_bits)) | ((s_max - s) << r_bits) | rank.astype(np.int64)
        return np.argsort(key, kind="stable")
    return np.lexsort((rank, -s, q))


def topk_mask(q, rank, score, k, decimals):
    """Отсортированный порядок строк и маска «строка попала в топ-k своего запроса»."""
    order = rank_order(q, rank, score, decimals)
    qs = q[order]
    first = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    group_start = np.repeat(first, np.diff(np.r_[first, len(qs)]))
    return order, (np.arange(len(qs)) - group_start) < k


def pool_recall(q, rank, label, n_rel, score, k, decimals) -> float:
    """Recall@k по пулу. Запросы с n_rel = 0 не учитываются (так задаются подвыборки)."""
    order, sel = topk_mask(q, rank, score, k, decimals)
    hits = np.bincount(q[order][sel], weights=label[order][sel], minlength=len(n_rel))
    ok = n_rel > 0
    return float((hits[ok] / n_rel[ok]).mean())


class PoolData:
    """Пул в виде массивов, готовых для многократной оценки весов."""

    def __init__(self, pool: pd.DataFrame, corpus, features: list, n_rel: np.ndarray):
        self.features = features
        self.F = pool[features].to_numpy(np.float64)
        self.q = pool["q"].to_numpy(np.int64)
        self.rank = corpus.rank[pool["item"].to_numpy()]
        self.label = pool["label"].to_numpy(np.float64)
        self.n_rel = n_rel

    def recall(self, w, k, decimals, query_mask=None) -> float:
        """query_mask - булев массив по запросам: считать только эти запросы."""
        if query_mask is None:
            return pool_recall(self.q, self.rank, self.label, self.n_rel,
                               linear_score(self.F, w), k, decimals)
        rows = query_mask[self.q]
        return pool_recall(self.q[rows], self.rank[rows], self.label[rows], self.n_rel * query_mask,
                           linear_score(self.F[rows], w), k, decimals)


def coordinate_ascent(data: PoolData, w0, grid, passes, k, decimals, query_mask=None, verbose=True):
    """Жадный покоординатный перебор по фиксированной сетке. Принимается только строгое улучшение,
    порядок признаков и значений фиксирован → результат детерминирован."""
    w = np.asarray(w0, dtype=np.float64).copy()
    best = data.recall(w, k, decimals, query_mask)
    if verbose:
        print(f"  старт: recall@{k} = {best:.5f}")
    for p in range(passes):
        for j in range(len(data.features)):
            for v in grid:
                if v == w[j]:
                    continue
                trial = w.copy()
                trial[j] = v
                r = data.recall(trial, k, decimals, query_mask)
                if r > best + 1e-9:
                    best, w = r, trial
        if verbose:
            nz = ", ".join(f"{n}={x:g}" for n, x in zip(data.features, w) if x != 0)
            print(f"  проход {p + 1}: recall@{k} = {best:.5f} | {nz}")
    return w, best


def half_split_cv(data: PoolData, w0, grid, passes, k, decimals) -> dict:
    """Честная оценка подбора весов: учим на чётных запросах - меряем на нечётных, и наоборот.
    Запросы идут в md5-порядке, поэтому половины случайны и сбалансированы."""
    odd = np.arange(len(data.n_rel)) % 2 == 1
    out = {}
    for name, train_mask in (("чётные→нечётные", ~odd), ("нечётные→чётные", odd)):
        w, _ = coordinate_ascent(data, w0, grid, passes, k, decimals, train_mask, verbose=False)
        out[name] = data.recall(w, k, decimals, ~train_mask)
    return out


def predict_from_scores(pool: pd.DataFrame, corpus, score: np.ndarray, k: int, decimals: int,
                        n_queries: int) -> list:
    """Топ-k item_id для каждого запроса по готовому скору строк пула.
    Если в пуле меньше k кандидатов - добор популярными объявлениями."""
    q = pool["q"].to_numpy(np.int64)
    items = pool["item"].to_numpy(np.int64)
    order, sel = topk_mask(q, corpus.rank[items], score, k, decimals)
    q_top, i_top = q[order][sel], items[order][sel]
    bounds = np.searchsorted(q_top, np.arange(n_queries + 1))
    fallback = np.lexsort((corpus.rank, -corpus.log_reviews, -corpus.log_pop))

    predictions = []
    for qi in range(n_queries):
        top = i_top[bounds[qi]:bounds[qi + 1]].tolist()
        if len(top) < k:
            seen = dict.fromkeys(top)
            top += [int(i) for i in fallback[: k + len(top)] if i not in seen][: k - len(top)]
        predictions.append([corpus.item_ids[i] for i in top])
    return predictions


def predict_from_pool(pool: pd.DataFrame, corpus, features, w, k, decimals, n_queries) -> list:
    """Топ-k item_id по линейной формуле."""
    score = linear_score(pool[features].to_numpy(np.float64), np.asarray(w, np.float64))
    return predict_from_scores(pool, corpus, score, k, decimals, n_queries)
