"""
Линейная формула над признаками пула и подбор весов координатным спуском.

Скор считается поэлементными операциями в фиксированном порядке (без BLAS):
сложение и умножение IEEE-754 дают одинаковый результат на любой машине,
а матричное умножение в BLAS может менять порядок суммирования.
"""
import numpy as np
import pandas as pd


def linear_score(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = np.zeros(F.shape[0], dtype=np.float64)
    for j in range(F.shape[1]):
        if w[j] != 0.0:
            s += w[j] * F[:, j]
    return s


def topk_mask(q: np.ndarray, rank: np.ndarray, score: np.ndarray, k: int, decimals: int):
    """Порядок строк пула (запрос ↑, скор ↓, item_id ↑) и маска «попал в топ-k»."""
    s_int = np.rint(score * 10 ** decimals).astype(np.int64)
    order = np.lexsort((rank, -s_int, q))
    q_sorted = q[order]
    starts = np.searchsorted(q_sorted, q_sorted, side="left")
    pos = np.arange(len(order)) - starts
    return order, pos < k


def pool_recall(q, rank, label, n_rel, score, k, decimals) -> float:
    order, sel = topk_mask(q, rank, score, k, decimals)
    hits = np.bincount(q[order][sel], weights=label[order][sel], minlength=len(n_rel))
    ok = n_rel > 0
    return float((hits[ok] / n_rel[ok]).mean())


def coordinate_ascent(F, names, q, rank, label, n_rel, w0, grid, passes, k, decimals, verbose=True):
    """Жадный покоординатный перебор по фиксированной сетке. Строгое улучшение → детерминизм."""
    F = F.astype(np.float64)
    w = np.asarray(w0, dtype=np.float64).copy()
    best = pool_recall(q, rank, label, n_rel, linear_score(F, w), k, decimals)
    if verbose:
        print(f"  старт: recall@{k} = {best:.5f}")
    for p in range(passes):
        for j, name in enumerate(names):
            for v in grid:
                if v == w[j]:
                    continue
                w2 = w.copy()
                w2[j] = v
                r = pool_recall(q, rank, label, n_rel, linear_score(F, w2), k, decimals)
                if r > best + 1e-9:
                    best, w = r, w2
        if verbose:
            print(f"  проход {p + 1}: recall@{k} = {best:.5f} | "
                  + ", ".join(f"{n}={x:g}" for n, x in zip(names, w) if x != 0))
    return w, best


def predict_from_pool(pool: pd.DataFrame, corpus, feat_names, w, k, decimals, n_queries,
                      fallback_order: np.ndarray) -> list:
    """Топ-k item_id для каждого запроса; если в пуле меньше k — добираем из fallback."""
    q = pool["q"].to_numpy(np.int64)
    items = pool["item"].to_numpy(np.int64)
    score = linear_score(pool[feat_names].to_numpy(np.float64), np.asarray(w, np.float64))
    order, sel = topk_mask(q, corpus.rank[items], score, k, decimals)
    q_sel, i_sel = q[order][sel], items[order][sel]
    bounds = np.searchsorted(q_sel, np.arange(n_queries + 1))
    preds = []
    for qi in range(n_queries):
        top = i_sel[bounds[qi]:bounds[qi + 1]].tolist()
        if len(top) < k:
            have = dict.fromkeys(top)
            for it in fallback_order:
                if len(top) >= k:
                    break
                if it not in have:
                    top.append(int(it))
                    have[it] = None
        preds.append([corpus.item_ids[i] for i in top])
    return preds
