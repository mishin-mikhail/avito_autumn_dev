"""
Символьные n-граммы: сходство, устойчивое к опечаткам и оборванным словам
(«расрочку» ~ «рассрочку», «натяжные потолки в» ~ «натяжные потолки»).

TF-IDF по n-граммам 3-5 символов внутри слов (char_wb), нормировка L2 →
скалярное произведение = косинусное сходство.
Для воспроизводимости не используется max_features: отбор по частоте в sklearn
сортирует нестабильно, и порядок при равных частотах зависит от сборки numpy.
Словарь ограничивается только min_df (детерминированная маска).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import normalize_query


class CharIndex:
    def __init__(self, ngram_range=(3, 5), min_df: int = 3):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(ngram_range), min_df=min_df,
                                   sublinear_tf=True, dtype=np.float32, lowercase=False)

    def fit(self, docs: list) -> "CharIndex":
        X = self.vec.fit_transform([normalize_query(d) for d in docs])
        self.X_t = X.T.tocsr()                    # n-граммы × документы
        return self

    def query_matrix(self, queries: list):
        return self.vec.transform([normalize_query(q) for q in queries]).tocsr()

    def score(self, Q) -> np.ndarray:
        """Косинусное сходство (запросы × документы), float32."""
        return (Q @ self.X_t).toarray()
