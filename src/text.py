"""
Нормализация и лемматизация русских текстов.

Используем pymorphy3 (open-source, MIT), чтобы «телевизоров» и «телевизор»
совпадали. Слова лемматизируются один раз и кешируются: словарь уникальных
слов на порядки меньше общего числа токенов.
"""
import re

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-zа-я0-9]+")
_CYR_RE = re.compile(r"[а-я]")

# Небольшой собственный список стоп-слов (без скачивания nltk, чтобы не ходить в сеть)
STOPWORDS = frozenset("""
а без более бы был была были было быть в вам вас весь во вот все всего всех вы
где да даже для до его ее если есть еще же за и из или им их к как ко когда кто
ли либо мы на над не нет ни но о об однако он она они оно от по под при про с
со так также такой там те то того тоже только том ты у уже чем что чтобы эта
эти это этот я который свой наш ваш мой
""".split())


def normalize_text(s) -> str:
    """Нижний регистр и ё→е. Не-строки (None/NaN) превращаются в пустую строку."""
    if not isinstance(s, str):
        return ""
    return s.lower().replace("ё", "е")


def normalize_query(s) -> str:
    """Каноническая форма запроса: только буквы/цифры, одиночные пробелы."""
    return _NON_ALNUM_RE.sub(" ", normalize_text(s)).strip()


def tokenize(s) -> list:
    return _TOKEN_RE.findall(normalize_text(s))


class Lemmatizer:
    """Лемматизатор с кешем. Результат детерминирован (берём первый разбор pymorphy3)."""

    def __init__(self):
        import pymorphy3
        self._morph = pymorphy3.MorphAnalyzer()
        self._cache: dict = {}

    def lemma(self, word: str) -> str:
        r = self._cache.get(word)
        if r is None:
            if _CYR_RE.search(word) and not word.isdigit():
                r = self._morph.parse(word)[0].normal_form.replace("ё", "е")
            else:
                r = word  # латиница и числа остаются как есть
            self._cache[word] = r
        return r

    def lemmas(self, text) -> list:
        """Список лемм без стоп-слов и однобуквенных «мусорных» токенов (цифры оставляем)."""
        out = []
        for w in tokenize(text):
            if w in STOPWORDS or (len(w) == 1 and not w.isdigit()):
                continue
            lw = self.lemma(w)
            if lw not in STOPWORDS:
                out.append(lw)
        return out

    def join_many(self, texts) -> list:
        """Тексты → строки лемм через пробел (формат для CountVectorizer(analyzer=str.split))."""
        return [" ".join(self.lemmas(t)) for t in texts]

    def key_many(self, texts) -> list:
        """Тексты → «мешок лемм»: уникальные леммы в отсортированном порядке.
        Так «скупка телевизоров» и «телевизор скупка» получают один ключ."""
        return [" ".join(sorted(dict.fromkeys(self.lemmas(t)))) for t in texts]

    @property
    def cache_size(self) -> int:
        return len(self._cache)
