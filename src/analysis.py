"""Разбор качества: полнота по сегментам и примеры промахов."""
import numpy as np
import pandas as pd

from .candidates import SOURCES


def attach_labels(pool: pd.DataFrame, corpus, truth: list) -> pd.DataFrame:
    """label = 1, если объявление из пула входит в эталон запроса."""
    idx_of = {v: i for i, v in enumerate(corpus.item_ids)}
    pairs = pd.DataFrame([(qi, idx_of[t]) for qi, rel in enumerate(truth) for t in rel if t in idx_of],
                         columns=["q", "item"]).astype(np.int32)
    pairs["label"] = np.int8(1)
    pool = pool.merge(pairs, on=["q", "item"], how="left", sort=False)
    pool["label"] = pool["label"].fillna(0).astype(np.int8)
    return pool


def pool_hit_rate(pool: pd.DataFrame, n_rel: np.ndarray) -> np.ndarray:
    """Доля эталона, попавшая в пул, — по каждому запросу."""
    hits = np.bincount(pool.loc[pool["label"] == 1, "q"], minlength=len(n_rel))
    return hits / np.maximum(n_rel, 1)


def source_recall(pool: pd.DataFrame, n_rel: np.ndarray) -> pd.DataFrame:
    """Какую долю эталона ловит каждый список (при своём K) и весь пул."""
    pos = pool[pool["label"] == 1]
    out = {}
    for name in SOURCES + ["pool"]:
        hits = pos if name == "pool" else pos[pos[name]]
        out[name] = float((np.bincount(hits["q"], minlength=len(n_rel)) / n_rel).mean())
    return pd.DataFrame({"recall": out}).assign(
        avg_candidates=float(pool.groupby("q").size().mean()))


def segment_table(queries: pd.DataFrame, recall50: np.ndarray, pool_hit: np.ndarray,
                  seg_cols=("seg_text", "seg_filter", "seg_loc")) -> pd.DataFrame:
    """Recall@50 и полнота пула по каждому сегменту."""
    df = queries[list(seg_cols)].assign(pool=pool_hit, recall50=recall50)
    parts = []
    for col in seg_cols:
        t = df.groupby(col).agg(n=("recall50", "size"), share=("recall50", "size"),
                                pool=("pool", "mean"), recall50=("recall50", "mean"))
        t["share"] = t["share"] / len(df)
        parts.append(t.rename_axis("сегмент").assign(ось=col).reset_index())
    return pd.concat(parts).set_index(["ось", "сегмент"]).round(4)


def error_examples(predictions, truth, queries: pd.DataFrame, items: pd.DataFrame,
                   pool_hit: np.ndarray, n: int = 20) -> pd.DataFrame:
    """Запросы без единого попадания в топ-50: что искали и что на самом деле выбрали."""
    row_of = {v: i for i, v in enumerate(items["item_id"])}
    rows = []
    for qi, (pred, rel) in enumerate(zip(predictions, truth)):
        if frozenset(pred) & frozenset(rel):
            continue
        gold = items.iloc[row_of[rel[0]]]
        rows.append({
            "запрос": queries.at[qi, "search_query"],
            "фильтры": queries.at[qi, "search_infm_params_text"],
            "та же локация": gold["item_location_id"] == queries.at[qi, "search_location_id"],
            "в пуле": pool_hit[qi] > 0,
            "заголовок эталона": gold["item_title_raw"],
            "параметры эталона": gold["item_infm_params_text"][:120],
        })
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)
