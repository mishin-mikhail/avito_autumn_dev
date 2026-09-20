"""
Загрузка и приведение данных к единому виду.

Ключевые решения:
  * все идентификаторы (item_id, локации, категории) - строки; числовые id
    приводятся к строке без «.0», чтобы search_location_id и item_location_id
    сравнивались корректно при любых dtype в parquet;
  * «запрос» = группа строк train с одинаковыми признаками запроса
    (нормализованный текст + локация + доставка + фильтры + категория).
"""
import re

import numpy as np
import pandas as pd

from .text import normalize_query, normalize_text

QUERY_COLS = ["search_query", "search_location_id", "search_is_delivery_search",
              "search_infm_params_text", "search_category"]
ITEM_TEXT_COLS = ["item_title_raw", "item_description_raw", "item_infm_params_text"]
ITEM_ID_COLS = ["item_category_id", "item_microcat_id", "item_location_id"]
ITEM_NUM_COLS = ["item_price", "item_rating", "item_rating_reviews_count",
                 "item_latitude", "item_longitude", "item_is_phone_hidden",
                 "item_is_message_forbidden"]
ITEM_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_SEP = "\x1f"  # разделитель полей в составных ключах (не встречается в текстах)


def norm_id_series(s: pd.Series) -> pd.Series:
    """Идентификатор → str. 123.0 → '123', пропуск → '<NA>'."""
    if pd.api.types.is_float_dtype(s):
        v = s.dropna()
        if len(v) == 0 or bool((v == np.floor(v)).all()):
            s = s.astype("Int64")
    out = s.astype("string").str.strip().fillna("<NA>")
    return pd.Series(out.to_numpy(dtype=object), index=s.index, name=s.name)


def _norm_text_series(s: pd.Series) -> pd.Series:
    return pd.Series(s.to_numpy(dtype=object), index=s.index, name=s.name).where(s.notna(), "")


def prepare_query_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит search_* колонки к единому виду и добавляет norm_text, filters_norm, query_key."""
    df["search_query"] = _norm_text_series(df["search_query"])
    df["search_infm_params_text"] = _norm_text_series(df["search_infm_params_text"])
    df["search_location_id"] = norm_id_series(df["search_location_id"])
    df["search_category"] = norm_id_series(df["search_category"])
    df["search_is_delivery_search"] = (
        pd.to_numeric(df["search_is_delivery_search"], errors="coerce").fillna(0).astype(np.int8)
    )
    df["norm_text"] = [normalize_query(x) for x in df["search_query"]]
    df["filters_norm"] = [normalize_text(x).strip() for x in df["search_infm_params_text"]]
    df["query_key"] = (
        df["norm_text"] + _SEP + df["search_location_id"] + _SEP
        + df["search_is_delivery_search"].astype(str) + _SEP
        + df["filters_norm"] + _SEP + df["search_category"]
    )
    df["rating_thr"] = parse_rating_filter(df["filters_norm"])
    return df


def prepare_item_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ITEM_TEXT_COLS:
        if c in df.columns:
            df[c] = _norm_text_series(df[c])
    for c in ITEM_ID_COLS:
        if c in df.columns:
            df[c] = norm_id_series(df[c])
    for c in ITEM_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    if "item_id" in df.columns:
        df["item_id"] = norm_id_series(df["item_id"])
    return df


_RATING_RE = re.compile(r"рейтинг[^0-9]{0,40}?(\d(?:[.,]\d)?)")


def parse_rating_filter(filters_norm: pd.Series) -> np.ndarray:
    """«Рейтинг пользователя 4 звезды и выше» → 4.0; нет фильтра → NaN."""
    out = np.full(len(filters_norm), np.nan, dtype=np.float32)
    for i, t in enumerate(filters_norm):
        if t and "рейтинг" in t:
            m = _RATING_RE.search(t)
            if m:
                out[i] = float(m.group(1).replace(",", "."))
    return out


def parquet_columns(path) -> list:
    import pyarrow.parquet as pq
    return list(pq.read_schema(path).names)


def load_train(data_dir, with_description: bool = False) -> pd.DataFrame:
    """train без описаний (они тяжёлые и нужны только для валидационных позитивов)."""
    path = data_dir / "train.parquet"
    cols = parquet_columns(path)
    if "item_id" not in cols:
        raise ValueError("В train.parquet нет item_id - нужна другая схема валидации, см. README.")
    use = [c for c in cols if with_description or c != "item_description_raw"]
    df = pd.read_parquet(path, columns=use)
    return prepare_item_columns(prepare_query_columns(df))


def load_train_items_text(data_dir, item_ids) -> pd.DataFrame:
    """Полные признаки (с описанием) только для заданных item_id - экономит память."""
    path = data_dir / "train.parquet"
    item_cols = [c for c in parquet_columns(path) if c.startswith("item_")]
    ids = sorted(dict.fromkeys(item_ids))
    df = pd.read_parquet(path, columns=item_cols, filters=[("item_id", "in", ids)])
    df = prepare_item_columns(df)
    # у одного объявления в разных строках train признаки могут отличаться (объявление
    # редактировали); берём первую строку в порядке файла - это детерминированно
    return df.drop_duplicates("item_id", keep="first").reset_index(drop=True)


def load_benchmark(data_dir):
    queries = prepare_query_columns(pd.read_parquet(data_dir / "benchmark_queries.parquet"))
    queries["query_id"] = queries["query_id"].astype(str)
    items = prepare_item_columns(pd.read_parquet(data_dir / "benchmark_items.parquet"))
    return queries, items
