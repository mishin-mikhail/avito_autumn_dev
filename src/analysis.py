"""Разбор качества: полнота по сегментам и примеры промахов."""
import numpy as np
import pandas as pd

from .candidates import ALL_SOURCES


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
    for name in [s for s in ALL_SOURCES if s in pool.columns] + ["pool"]:
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


V2_SOURCES = ["src_text", "src_text_loc", "src_prior_loc", "src_memo"]   # пул как в v2


def v2_rows(pool: pd.DataFrame) -> pd.DataFrame:
    """Строки пула, которые были бы и в пуле v2 (без списка по символьному сходству)."""
    return pool[pool[V2_SOURCES].any(axis=1)]


def difficulty_table(pools: dict, k: int = 50) -> pd.DataFrame:
    """
    Насколько «плотная» конкуренция у запросов разных наборов (медианы по запросам):
      * объявлений корпуса в той же локации;
      * объявлений «рядом» (близость > 0.5);
      * максимальный BM25 заголовка (насколько много явных текстовых совпадений);
      * скор формулы v2 у k-го кандидата (порог попадания в топ-k).
    pools: имя → пул v3 с колонкой stage1.
    """
    rows = {}
    for name, pool in pools.items():
        per_q = pool.groupby("q").agg(same_loc=("q_n_same_loc", "first"), near=("q_n_near", "first"),
                                      title_max=("q_title_max", "first"))
        kth = (pool[pool["stage1_rank"] == k - 1].set_index("q")["stage1"]
               .reindex(per_q.index))
        rows[name] = {
            "запросов": len(per_q),
            "объявлений в той же локации": float(np.expm1(per_q["same_loc"]).median()),
            "объявлений рядом": float(per_q["near"].median()),
            "max BM25 заголовка": float(per_q["title_max"].median()),
            f"скор v2 у {k}-го": float(kth.median()),
            "доля «региональных»": float(pool.groupby("q")["q_region"].first().mean()),
        }
    return pd.DataFrame(rows).T.round(3)
