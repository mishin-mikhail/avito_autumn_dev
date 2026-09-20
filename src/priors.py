"""
Статистики из train.

1. P(микрокатегория | запрос) со сглаживанием по цепочке
   «текст+категория поиска» → «текст» → «леммы по отдельности» → «категория поиска» → «всё».
   Для коротких запросов вроде «автоподбор» это почти однозначное указание на
   нужную подкатегорию, а цепочка сглаживания даёт осмысленный ответ и для
   запросов, которых в train не было.
2. Статистики по item_id: общая популярность и «память» (сколько раз объявление
   выбирали по такому же мешку лемм). Полезны, только если объявления корпуса
   встречаются в train.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

_SEP = "\x1f"


def _count_matrix(keys: pd.Series, cols: np.ndarray, weights: np.ndarray, n_cols: int):
    """Ключи (строки) × колонки → разреженная матрица сумм; ключи в отсортированном порядке."""
    uniq = np.array(sorted(dict.fromkeys(keys)), dtype=object)
    index = {k: i for i, k in enumerate(uniq)}
    rows = np.fromiter((index[k] for k in keys), dtype=np.int64, count=len(keys))
    m = sp.coo_matrix((weights, (rows, cols)), shape=(len(uniq), n_cols)).tocsr()
    return index, m


def _lookup(index: dict, m: sp.csr_matrix, keys) -> np.ndarray:
    """Строки матрицы по ключам; неизвестный ключ → нулевая строка."""
    zero = sp.csr_matrix((1, m.shape[1]), dtype=m.dtype)
    mm = sp.vstack([m, zero]).tocsr()
    rows = [index.get(k, m.shape[0]) for k in keys]
    return mm[rows].toarray()


class MicrocatPrior:
    def __init__(self, alpha: float = 5.0, beta: float = 20.0):
        self.alpha, self.beta = alpha, beta

    def fit(self, lemma_key: pd.Series, category: pd.Series, micro_idx: np.ndarray,
            n_micro: int) -> "MicrocatPrior":
        df = pd.DataFrame({"lk": lemma_key.to_numpy(dtype=object),
                           "cat": category.to_numpy(dtype=object),
                           "m": micro_idx})
        agg = df.groupby(["lk", "cat", "m"], sort=True).size().reset_index(name="c")
        c = agg["c"].to_numpy(np.float64)
        m = agg["m"].to_numpy(np.int64)
        self.n_micro = n_micro

        self.glob = np.bincount(m, weights=c, minlength=n_micro) + 1.0
        self.glob /= self.glob.sum()
        self.cat_index, self.C_cat = _count_matrix(agg["cat"], m, c, n_micro)
        self.text_index, self.C_text = _count_matrix(agg["lk"], m, c, n_micro)
        self.tc_index, self.C_tc = _count_matrix(agg["lk"] + _SEP + agg["cat"], m, c, n_micro)

        # леммы по отдельности: (строки agg × леммы)ᵀ @ (строки agg × микрокатегории)
        self.tok_vec = CountVectorizer(analyzer=str.split, binary=True, dtype=np.float64)
        R = self.tok_vec.fit_transform(agg["lk"].tolist())
        Mh = sp.csr_matrix((c, (np.arange(len(agg)), m)), shape=(len(agg), n_micro))
        self.C_tok = (R.T @ Mh).tocsr()
        return self

    def text_counts(self, lemma_key: list) -> np.ndarray:
        """Сколько строк статистик приходится на каждый «мешок лемм» (0 - текст новый)."""
        totals = np.asarray(self.C_text.sum(axis=1)).ravel()
        return np.array([totals[self.text_index[k]] if k in self.text_index else 0.0 for k in lemma_key])

    def transform(self, lemma_key: list, category: list) -> np.ndarray:
        """Плотная матрица P(m | запрос), запросы × микрокатегории (float32)."""
        a, b = self.alpha, self.beta
        c_cat = _lookup(self.cat_index, self.C_cat, category)
        p_cat = (c_cat + self.glob[None, :]) / (c_cat.sum(1, keepdims=True) + 1.0)

        Q = self.tok_vec.transform(lemma_key)
        c_tok = (Q @ self.C_tok).toarray()
        p_tok = (c_tok + b * p_cat) / (c_tok.sum(1, keepdims=True) + b)

        c_text = _lookup(self.text_index, self.C_text, lemma_key)
        p_text = (c_text + a * p_tok) / (c_text.sum(1, keepdims=True) + a)

        keys_tc = [f"{k}{_SEP}{c}" for k, c in zip(lemma_key, category)]
        c_tc = _lookup(self.tc_index, self.C_tc, keys_tc)
        p = (c_tc + a * p_text) / (c_tc.sum(1, keepdims=True) + a)
        return p.astype(np.float32)


class ItemStats:
    """Популярность объявлений и «память» (мешок лемм запроса → объявления)."""

    def fit(self, lemma_key: pd.Series, item_idx: np.ndarray, n_items: int) -> "ItemStats":
        ok = item_idx >= 0                      # объявления вне корпуса пропускаем
        self.pop = np.bincount(item_idx[ok], minlength=n_items).astype(np.float32)
        df = pd.DataFrame({"lk": lemma_key.to_numpy(dtype=object)[ok], "i": item_idx[ok]})
        agg = df.groupby(["lk", "i"], sort=True).size().reset_index(name="c")
        # внутри ключа: по убыванию счётчика, затем по индексу (детерминированно)
        agg = agg.sort_values(["lk", "c", "i"], ascending=[True, False, True], kind="stable")
        self._items = agg["i"].to_numpy(np.int64)
        self._counts = agg["c"].to_numpy(np.float32)
        keys = agg["lk"].to_numpy(dtype=object)
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.zeros(0, int)
        ends = np.r_[starts[1:], len(keys)]
        self._span = {keys[s]: (int(s), int(e)) for s, e in zip(starts, ends)}
        return self

    def memo_for(self, key: str):
        """Объявления, выбранные по этому мешку лемм: (индексы, счётчики), по убыванию счётчика."""
        s, e = self._span.get(key, (0, 0))
        return self._items[s:e], self._counts[s:e]
