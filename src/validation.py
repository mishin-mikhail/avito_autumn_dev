"""
Локальная валидация: псевдо-бенчмарк из train.

Идея. Бенчмарк — это ~2.5 тыс. «запросов» (групп признаков запроса), у каждого
1–2+ выбранных объявления. Часть текстов запросов бенчмарка встречается в train
(«знакомые»), часть — нет («новые»). Для знакомых работают статистики из train,
для новых — нет, поэтому валидация должна повторять ту же пропорцию, иначе
оценка будет смещённой.

Схема (всё детерминировано через md5):
  1. доля text_holdout_frac текстов убирается из train целиком → кандидаты в «новые»;
  2. из остальных групп берутся «знакомые»: их текст продолжает встречаться в train
     в других группах (как у знакомых запросов бенчмарка);
  3. число новых и знакомых подбирается под долю знакомых в бенчмарке.
"""
import numpy as np
import pandas as pd

from .repro import md5_hex


def seen_share(query_texts, train_texts) -> float:
    """Доля запросов, чей нормализованный текст встречается в train."""
    known = pd.Index(pd.unique(np.asarray(train_texts, dtype=object)))
    return float(pd.Index(query_texts).isin(known).mean())


def build_validation_split(train: pd.DataFrame, bench_seen_share: float, n_val: int,
                           text_holdout_frac: float, salt: str):
    """
    Возвращает:
      val_keys   — список query_key валидационных запросов (в md5-порядке);
      fold_mask  — булев массив по строкам train: True = строка идёт в «train-фолд»;
      info       — словарь со статистикой разбиения.
    """
    # --- 1. тексты, целиком убранные из train ---
    texts = pd.unique(train["norm_text"].to_numpy(dtype=object))
    t_hash = np.array([int(md5_hex(f"{salt}|t|{t}")[:12], 16) / float(1 << 48) for t in texts])
    ho_texts = pd.Index(texts[t_hash < text_holdout_frac])
    row_ho = train["norm_text"].isin(ho_texts).to_numpy()

    # --- таблица групп в md5-порядке ---
    groups = train[["query_key", "norm_text"]].drop_duplicates("query_key").copy()
    groups["h"] = [md5_hex(f"{salt}|g|{k}") for k in groups["query_key"]]
    groups = groups.sort_values(["h", "query_key"], kind="stable").reset_index(drop=True)
    g_ho = groups["norm_text"].isin(ho_texts)

    n_unseen = int(round(n_val * (1.0 - bench_seen_share)))
    n_seen = n_val - n_unseen

    # --- 2. «новые» запросы: группы убранных текстов ---
    unseen = groups.loc[g_ho, "query_key"].head(n_unseen).tolist()

    # --- 3. «знакомые»: у текста должна остаться хотя бы одна группа в train-фолде ---
    cand = groups.loc[~g_ho].copy()
    n_groups_per_text = cand["norm_text"].map(cand["norm_text"].value_counts())
    rank_in_text = cand.groupby("norm_text", sort=False).cumcount()
    allowed = cand[rank_in_text.to_numpy() < n_groups_per_text.to_numpy() - 1]
    seen = allowed["query_key"].head(n_seen).tolist()

    if len(unseen) < n_unseen or len(seen) < n_seen:
        print(f"[warn] не хватило групп: новых {len(unseen)}/{n_unseen}, знакомых {len(seen)}/{n_seen}")

    val_keys = unseen + seen
    row_val_seen = train["query_key"].isin(pd.Index(seen)).to_numpy()
    fold_mask = ~(row_ho | row_val_seen)
    info = {
        "n_val": len(val_keys), "n_unseen": len(unseen), "n_seen": len(seen),
        "bench_seen_share": bench_seen_share,
        "n_holdout_texts": len(ho_texts),
        "train_rows_total": int(len(train)), "train_rows_fold": int(fold_mask.sum()),
    }
    return val_keys, fold_mask, info


def build_val_queries(train: pd.DataFrame, val_keys: list):
    """Таблица валидационных запросов (одна строка на ключ) и эталонные объявления."""
    sub = train[train["query_key"].isin(pd.Index(val_keys))]
    q = sub.drop_duplicates("query_key", keep="first").set_index("query_key").loc[val_keys].reset_index()
    q.insert(0, "query_id", [f"val_{i:05d}" for i in range(len(q))])
    truth_map = (sub.groupby("query_key", sort=True)["item_id"]
                 .agg(lambda s: sorted(dict.fromkeys(s))).to_dict())
    truth = [truth_map[k] for k in val_keys]
    return q, truth


def recall_at_k(predictions, truth, k: int = 50) -> float:
    """Recall@K ровно как в условии: среднее по запросам |топ-K ∩ rel| / |rel|."""
    vals = []
    for pred, rel in zip(predictions, truth):
        if len(rel) == 0:
            continue
        top = frozenset(pred[:k])
        vals.append(sum(r in top for r in rel) / len(rel))
    return float(np.mean(vals)) if vals else 0.0


def recall_by_segment(predictions, truth, segment, k: int = 50) -> pd.DataFrame:
    """Recall@K в разрезе сегментов (например, знакомые/новые, доставка/нет)."""
    rows = []
    for pred, rel, seg in zip(predictions, truth, segment):
        top = frozenset(pred[:k])
        rows.append((seg, sum(r in top for r in rel) / len(rel)))
    df = pd.DataFrame(rows, columns=["segment", "recall"])
    return df.groupby("segment").agg(n=("recall", "size"), recall=("recall", "mean"))
