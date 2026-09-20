"""
Точное сопоставление фильтров поиска с параметрами объявления.

Фильтр поиска - это склеенные пары «ключ значение»:
    «Тип услуги Телевизоры Вид услуги Ремонт и обслуживание техники»
В параметрах объявления те же пары идут вперемешку с другими полями:
    «Вид услуги Ремонт и обслуживание техники Тип услуги Телевизоры Место оказания услуг ...»

Разбираем фильтр на пары по известным ключам и проверяем, что каждая пара
дословно встречается в параметрах объявления и после неё начинается следующий
ключ (слово с заглавной буквы) или строка заканчивается. Регистр важен:
«Тип услуги Автосервис» и «Тип услуги автосервиса ...» - разные пары.
"""
import re

import numpy as np

# «Вид услуги», «Тип услуги» и уточнённые ключи вида «Тип услуги автосервиса»;
# значение всегда начинается с заглавной буквы или цифры
_KEY_RE = re.compile(r"(Вид услуги|Тип услуги(?: [а-я]+)?)(?= [A-ZА-Я0-9]|$)")
# флаги без значения - только границы, сами по себе не проверяются
_FLAG_RE = re.compile(r"Онлайн-запись|Рейтинг пользователя")
_SPACE_RE = re.compile(r"\s+")
_PATTERN_CACHE: dict = {}


def normalize_params(text) -> str:
    """ё→е (с сохранением регистра) и одиночные пробелы."""
    if not isinstance(text, str):
        return ""
    return _SPACE_RE.sub(" ", text.replace("ё", "е").replace("Ё", "Е")).strip()


def parse_filter_pairs(text) -> tuple:
    """«Тип услуги X Вид услуги Y» → ('Тип услуги X', 'Вид услуги Y'). Пустые значения пропускаются."""
    t = normalize_params(text)
    keys = list(_KEY_RE.finditer(t))
    if not keys:
        return ()
    bounds = sorted([m.start() for m in keys] + [m.start() for m in _FLAG_RE.finditer(t)] + [len(t)])
    pairs = []
    for m in keys:
        end = next(b for b in bounds if b > m.start())
        value = t[m.end():end].strip()
        if value:
            pairs.append(f"{m.group(1)} {value}")
    return tuple(dict.fromkeys(pairs))


def _pattern(pair: str):
    p = _PATTERN_CACHE.get(pair)
    if p is None:
        p = re.compile(re.escape(pair) + r"(?= [A-ZА-Я0-9]|$)")
        _PATTERN_CACHE[pair] = p
    return p


def match_share(pairs: tuple, params_norm: str) -> float:
    """Доля пар фильтра, найденных в параметрах объявления."""
    return sum(_pattern(p).search(params_norm) is not None for p in pairs) / len(pairs)


def pool_match_share(q: np.ndarray, item: np.ndarray, pairs_per_query: list, params_norm: list) -> np.ndarray:
    """Признак filt_exact для всех строк пула (0 для запросов без разобранных фильтров)."""
    out = np.zeros(len(q), dtype=np.float32)
    for row in range(len(q)):
        pairs = pairs_per_query[q[row]]
        if pairs:
            out[row] = match_share(pairs, params_norm[item[row]])
    return out
