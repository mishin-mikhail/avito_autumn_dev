"""
Локальная валидация: псевдо-бенчмарк из train.

«Запрос» = группа строк train с одинаковым query_key. Валидация повторяет
состав бенчмарка по трём осям, от которых сильно зависит качество:
  * текст запроса знакомый (встречается в train) или новый;
  * есть ли у запроса фильтры;
  * тип локации поиска: «обычная» (бывает у объявлений) или «только поисковая»
    (региональный id, у объявлений не встречается).
Доли ячеек берутся из бенчмарка, отбор внутри ячейки — по md5 (детерминированно).

Как получаются «новые» и «знакомые» запросы:
  * доля text_holdout_frac текстов убирается из train целиком → кандидаты в новые;
  * знакомые — группы, чей текст остаётся в train в других группах.
"""
import numpy as np
import pandas as pd

from .repro import md5_hex

SEEN, UNSEEN = "знакомый", "новый"


def add_query_segments(df: pd.DataFrame, item_locations: pd.Index) -> pd.DataFrame:
    """seg_filter и seg_loc (seg_text зависит от разбиения и ставится отдельно)."""
    df["seg_filter"] = np.where(df["filters_norm"] != "", "фильтр есть", "фильтра нет")
    df["seg_loc"] = np.where(df["search_location_id"].isin(item_locations),
                             "локация обычная", "локация только поисковая")
    df["stratum"] = df["seg_filter"] + " | " + df["seg_loc"]
    return df


def mark_seen(df: pd.DataFrame, known_texts) -> pd.DataFrame:
    df["seg_text"] = np.where(df["norm_text"].isin(pd.Index(known_texts)), SEEN, UNSEEN)
    return df


def _allocate(shares: pd.Series, n: int) -> pd.Series:
    """Разбивает n по долям методом наибольших остатков (сумма ровно n, без случайности)."""
    raw = shares.sort_index() / shares.sum() * n
    base = np.floor(raw).astype(int)
    extra = (raw - base).sort_values(ascending=False, kind="stable").index[: n - int(base.sum())]
    base.loc[extra] += 1
    return base


def build_validation_split(train: pd.DataFrame, bench_q: pd.DataFrame, n_val: int,
                           text_holdout_frac: float, salt: str):
    """
    Возвращает:
      val_keys   — query_key валидационных запросов (в md5-порядке);
      val_seen   — словарь query_key → «знакомый»/«новый»;
      fold_mask  — булев массив по строкам train: True = строка идёт в train-фолд;
      report     — таблица «цель vs факт» по ячейкам стратификации.
    """
    # 1. тексты, целиком убранные из train
    texts = pd.unique(train["norm_text"].to_numpy(dtype=object))
    t_unit = np.array([int(md5_hex(f"{salt}|t|{t}")[:12], 16) / float(1 << 48) for t in texts])
    holdout_texts = pd.Index(texts[t_unit < text_holdout_frac])
    row_holdout = train["norm_text"].isin(holdout_texts).to_numpy()

    # 2. группы в md5-порядке
    groups = train[["query_key", "norm_text", "stratum"]].drop_duplicates("query_key").copy()
    groups["h"] = [md5_hex(f"{salt}|g|{k}") for k in groups["query_key"]]
    groups = groups.sort_values(["h", "query_key"], kind="stable").reset_index(drop=True)
    in_holdout = groups["norm_text"].isin(holdout_texts)

    unseen_pool = groups[in_holdout]
    # у знакомого запроса текст должен остаться хотя бы в одной группе train-фолда
    seen_pool = groups[~in_holdout]
    n_groups = seen_pool["norm_text"].map(seen_pool["norm_text"].value_counts()).to_numpy()
    rank_in_text = seen_pool.groupby("norm_text", sort=False).cumcount().to_numpy()
    seen_pool = seen_pool[rank_in_text < n_groups - 1]

    # 3. отбор по ячейкам в пропорциях бенчмарка
    target = _allocate(bench_q.groupby(["seg_text", "stratum"]).size(), n_val)
    picked, rows = [], []
    for (seg_text, stratum), n in target.items():
        pool = unseen_pool if seg_text == UNSEEN else seen_pool
        chosen = pool.loc[pool["stratum"] == stratum, ["query_key", "h"]].head(n)
        picked.append(chosen.assign(seg_text=seg_text))
        rows.append((seg_text, stratum, int(n), len(chosen)))
    picked = pd.concat(picked).sort_values(["h", "query_key"], kind="stable")

    report = pd.DataFrame(rows, columns=["текст", "страта", "цель", "факт"])
    if (report["факт"] < report["цель"]).any():
        print("[warn] в некоторых ячейках не хватило групп — см. таблицу")

    seen_keys = pd.Index(picked.loc[picked["seg_text"] == SEEN, "query_key"])
    fold_mask = ~(row_holdout | train["query_key"].isin(seen_keys).to_numpy())
    val_seen = dict(zip(picked["query_key"], picked["seg_text"]))
    return picked["query_key"].tolist(), val_seen, fold_mask, report


def build_val_queries(train: pd.DataFrame, val_keys: list, val_seen: dict):
    """Таблица валидационных запросов (одна строка на ключ) и эталонные объявления."""
    rows = train[train["query_key"].isin(pd.Index(val_keys))]
    queries = (rows.drop_duplicates("query_key", keep="first")
               .set_index("query_key").loc[val_keys].reset_index())
    queries.insert(0, "query_id", [f"val_{i:05d}" for i in range(len(queries))])
    queries["seg_text"] = queries["query_key"].map(val_seen)
    truth_map = (rows.groupby("query_key", sort=True)["item_id"]
                 .agg(lambda s: sorted(dict.fromkeys(s))).to_dict())
    return queries, [truth_map[k] for k in val_keys]


def recall_at_k(predictions, truth, k: int = 50) -> float:
    """Recall@K ровно как в условии: среднее по запросам |топ-K ∩ rel| / |rel|."""
    return float(np.mean(per_query_recall(predictions, truth, k)))


def per_query_recall(predictions, truth, k: int = 50) -> np.ndarray:
    return np.array([sum(r in frozenset(p[:k]) for r in rel) / len(rel)
                     for p, rel in zip(predictions, truth)])
