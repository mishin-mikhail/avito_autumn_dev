"""
Выборки запросов из train для v3: валидация и фолды ранкера.

Общий принцип тот же, что в validation.py (v2), с тремя отличиями:
  * «новые» запросы берутся по одному на текст — в бенчмарке новые тексты
    почти всегда редкие и уникальные, а выбор по группам перекашивал выборку
    в сторону популярных текстов;
  * можно ограничить кандидатов (eligible) — например, запросами, чьи выбранные
    объявления лежат в корпусе бенчмарка (схема «в корпусе»);
  * разбиение строится поверх base_mask — строк, уже доступных для статистик
    (так фолды ранкера не пересекаются с валидацией).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .repro import md5_hex, md5_unit
from .validation import SEEN, UNSEEN, _allocate, build_val_queries


@dataclass
class QuerySample:
    name: str
    queries: pd.DataFrame     # одна строка на запрос (+ seg_text)
    truth: list               # эталонные item_id
    stats_mask: np.ndarray    # строки train, по которым можно считать статистики для этих запросов
    report: pd.DataFrame      # цель vs факт по ячейкам

    @property
    def keys(self) -> list:
        return self.queries["query_key"].tolist()

    @property
    def n_rel(self) -> np.ndarray:
        return np.array([len(t) for t in self.truth], dtype=np.float64)


def group_table(train: pd.DataFrame, corpus_ids) -> pd.DataFrame:
    """Одна строка на группу-запрос: текст, страта и флаг «все выбранные объявления в корпусе»."""
    in_corpus = train["item_id"].isin(pd.Index(corpus_ids)).groupby(train["query_key"], sort=False).all()
    groups = train[["query_key", "norm_text", "stratum"]].drop_duplicates("query_key")
    return groups.assign(in_corpus=groups["query_key"].map(in_corpus).to_numpy()).reset_index(drop=True)


def sample_queries(name: str, train: pd.DataFrame, groups: pd.DataFrame, target: pd.Series,
                   base_mask: np.ndarray, eligible: pd.Index, holdout_frac: float, salt: str) -> QuerySample:
    """
    target    — число запросов в каждой ячейке (seg_text, stratum);
    base_mask — строки train, доступные этой выборке;
    eligible  — ключи групп, из которых можно выбирать запросы.
    """
    base = train.loc[base_mask, ["query_key", "norm_text"]]
    base_texts = pd.unique(base["norm_text"].to_numpy(dtype=object))
    holdout = pd.Index([t for t in base_texts if md5_unit(f"{salt}|t|{t}") < holdout_frac])

    g = groups[groups["query_key"].isin(pd.Index(base["query_key"].unique()))].copy()
    g["h"] = [md5_hex(f"{salt}|g|{k}") for k in g["query_key"]]
    g = g.sort_values(["h", "query_key"], kind="stable").reset_index(drop=True)
    g_holdout = g["norm_text"].isin(holdout).to_numpy()
    g_eligible = g["query_key"].isin(eligible).to_numpy()

    # новые: по одному запросу на убранный текст
    unseen_pool = g[g_holdout & g_eligible].drop_duplicates("norm_text", keep="first")
    # знакомые: у текста остаётся хотя бы одна группа среди доступных строк
    rest = g[~g_holdout]
    n_groups = rest["norm_text"].map(rest["norm_text"].value_counts())
    rest = rest.assign(rank_in_text=rest.groupby("norm_text", sort=False).cumcount(), n_groups=n_groups)
    seen_pool = rest[rest["query_key"].isin(eligible) & (rest["rank_in_text"] < rest["n_groups"] - 1)]

    picked, rows = [], []
    for (seg_text, stratum), n in target.items():
        pool = unseen_pool if seg_text == UNSEEN else seen_pool
        chosen = pool.loc[pool["stratum"] == stratum, ["query_key", "h"]].head(int(n))
        picked.append(chosen.assign(seg_text=seg_text))
        rows.append((seg_text, stratum, int(n), len(chosen)))
    picked = pd.concat(picked).sort_values(["h", "query_key"], kind="stable")
    report = pd.DataFrame(rows, columns=["текст", "страта", "цель", "факт"])
    if (report["факт"] < report["цель"]).any():
        print(f"[warn] {name}: в некоторых ячейках не хватило запросов — "
              f"{report['факт'].sum()} из {report['цель'].sum()}")

    seen_keys = pd.Index(picked.loc[picked["seg_text"] == SEEN, "query_key"])
    removed = train["norm_text"].isin(holdout).to_numpy() | train["query_key"].isin(seen_keys).to_numpy()
    stats_mask = base_mask & ~removed

    keys = picked["query_key"].tolist()
    queries, truth = build_val_queries(train, keys, dict(zip(keys, picked["seg_text"])))
    return QuerySample(name, queries, truth, stats_mask, report)


def bench_targets(bench_q: pd.DataFrame, n: int) -> pd.Series:
    """Число запросов в каждой ячейке (знакомый/новый × страта) в пропорциях бенчмарка."""
    return _allocate(bench_q.groupby(["seg_text", "stratum"]).size(), n)
