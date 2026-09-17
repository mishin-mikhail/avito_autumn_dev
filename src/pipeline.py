"""
Сквозной пайплайн одного «этапа» (валидация или бенчмарк):
  корпус → статистики из train → запросы → пул кандидатов с признаками.

Валидация и бенчмарк проходят через одну и ту же функцию build_stage —
отличаются только корпус, запросы и строки train, по которым считаются статистики.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import attach_labels
from .candidates import Corpus, QuerySet, Vocabs, build_corpus, build_queries, feature_names, generate_pool
from .geo import LocationModel
from .priors import ItemStats, MicrocatPrior
from .utils import timer

ITEM_TEXT_FIELDS = {"title": "item_title_raw", "params": "item_infm_params_text", "desc": "item_description_raw"}


def add_lemma_keys(df: pd.DataFrame, lem) -> pd.DataFrame:
    """«Мешок лемм» запроса. Лемматизируются только уникальные тексты."""
    uniq = pd.unique(df["search_query"].to_numpy(dtype=object))
    df["lemma_key"] = df["search_query"].map(dict(zip(uniq, lem.key_many(uniq))))
    return df


class ItemLemmaCache:
    """Лемматизированные поля объявлений по item_id: корпус валидации почти совпадает
    с корпусом бенчмарка, и второй раз его лемматизировать не нужно."""

    def __init__(self, lem, desc_max_chars: int):
        self.lem, self.desc_max_chars = lem, desc_max_chars
        self._store: dict = {}

    def get(self, items: pd.DataFrame) -> dict:
        ids = items["item_id"].tolist()
        todo = [i for i, item_id in enumerate(ids) if item_id not in self._store]
        if todo:
            sub = items.iloc[todo]
            desc = [d[: self.desc_max_chars] for d in sub["item_description_raw"]]
            lemmas = zip(self.lem.join_many(sub["item_title_raw"]),
                         self.lem.join_many(sub["item_infm_params_text"]),
                         self.lem.join_many(desc))
            for item_id, triple in zip(sub["item_id"], lemmas):
                self._store[item_id] = triple
        rows = [self._store[i] for i in ids]
        return {name: [r[j] for r in rows] for j, name in enumerate(ITEM_TEXT_FIELDS)}


@dataclass
class Stage:
    name: str
    items: pd.DataFrame
    queries: pd.DataFrame
    corpus: Corpus
    query_set: QuerySet
    pool: pd.DataFrame
    features: list


def build_stage(name: str, items: pd.DataFrame, queries: pd.DataFrame, stats_rows: pd.DataFrame, *,
                lem, cache: ItemLemmaCache, vocabs: Vocabs, cfg, use_item_stats: bool,
                truth: list = None) -> Stage:
    """
    items       — корпус, среди которого ищем;
    queries     — запросы этапа;
    stats_rows  — строки train, по которым считаются статистики (для валидации — только train-фолд!);
    truth       — эталон (для валидации): добавляет в пул колонку label.
    """
    with timer(f"{name}: индекс корпуса"):
        corpus = build_corpus(items, cache.get(items), cfg, vocabs)

    with timer(f"{name}: статистики train"):
        prior = MicrocatPrior(cfg.prior_alpha, cfg.prior_beta).fit(
            stats_rows["lemma_key"], stats_rows["search_category"],
            vocabs.micro.encode(stats_rows["item_microcat_id"].tolist()), vocabs.micro.size_with_unknown)
        geo = LocationModel(cfg.geo_alpha).fit(stats_rows, items, vocabs.loc)
        item_stats = None
        if use_item_stats:
            row_of = {v: i for i, v in enumerate(corpus.item_ids)}
            item_idx = np.fromiter((row_of.get(v, -1) for v in stats_rows["item_id"]),
                                   np.int64, len(stats_rows))
            item_stats = ItemStats().fit(stats_rows["lemma_key"], item_idx, corpus.n)
            corpus.log_pop = np.log1p(item_stats.pop).astype(np.float32)

    with timer(f"{name}: пул кандидатов"):
        query_set = build_queries(queries, lem, vocabs, prior)
        pool = generate_pool(corpus, query_set, geo, cfg, item_stats)
        if truth is not None:
            pool = attach_labels(pool, corpus, truth)

    print(f"{name}: запросов {query_set.n:,}, объявлений {corpus.n:,}, строк пула {len(pool):,} "
          f"(~{len(pool) / query_set.n:.0f} на запрос)")
    return Stage(name, items, queries, corpus, query_set, pool, feature_names(use_item_stats))
