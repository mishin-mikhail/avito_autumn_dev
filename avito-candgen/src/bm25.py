"""
BM25 на разреженных матрицах (scipy).

Своя реализация вместо rank_bm25: прозрачная, быстрая и полностью
детерминированная (CountVectorizer сортирует словарь, умножение разреженных
матриц в scipy однопоточное).
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


def _vectorizer(binary: bool = False) -> CountVectorizer:
    # на вход подаются уже лемматизированные строки, поэтому просто split
    return CountVectorizer(analyzer=str.split, binary=binary, dtype=np.float32)


class BM25Field:
    """BM25-индекс одного текстового поля."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list) -> "BM25Field":
        self.vec = _vectorizer()
        tf = self.vec.fit_transform(docs).tocsr()
        n_docs, n_terms = tf.shape
        df = np.bincount(tf.indices, minlength=n_terms)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(dl.mean()) if dl.mean() > 0 else 1.0
        norm = (self.k1 * (1.0 - self.b + self.b * dl / avgdl)).astype(np.float32)
        row_norm = np.repeat(norm, np.diff(tf.indptr))
        tf.data = tf.data * (self.k1 + 1.0) / (tf.data + row_norm) * idf[tf.indices]
        self.W_t = tf.T.tocsr()          # термины × документы
        self.n_docs = n_docs
        return self

    def query_matrix(self, queries: list) -> sp.csr_matrix:
        q = self.vec.transform(queries).tocsr()
        q.data[:] = 1.0                  # каждый термин запроса учитывается один раз
        return q

    def score(self, Q: sp.csr_matrix) -> np.ndarray:
        """Плотная матрица скоров (запросы × документы)."""
        return (Q @ self.W_t).toarray()


class CoverageField:
    """Сколько уникальных лемм запроса встретилось в документе (для доли покрытия)."""

    def fit(self, docs: list) -> "CoverageField":
        self.vec = _vectorizer(binary=True)
        self.B_t = self.vec.fit_transform(docs).T.tocsr()
        return self

    def query_matrix(self, queries: list) -> sp.csr_matrix:
        q = self.vec.transform(queries).tocsr()
        q.data[:] = 1.0
        return q

    def score(self, Q: sp.csr_matrix) -> np.ndarray:
        return (Q @ self.B_t).toarray()


def row_max_normalize(x: np.ndarray) -> np.ndarray:
    """Делит строку на её максимум: скоры разных запросов становятся сопоставимы."""
    m = x.max(axis=1, keepdims=True)
    return x / np.where(m > 0, m, 1.0)
