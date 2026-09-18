"""
Сквозной пайплайн: корпус → статистики из train → запросы → пул кандидатов с признаками.

* build_stage   — всё сразу для одного набора запросов (ноутбук v2);
* build_index   — только индекс корпуса (v3: один корпус на валидацию и все фолды ранкера);
* build_pool    — статистики + пул для набора запросов поверх готового индекса.
Результат build_stage = build_index + build_pool без расширения, то есть в точности v2.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import attach_labels
from .candidates import Corpus, QuerySet, Vocabs, build_corpus, build_queries, feature_names, generate_pool
from .geo import LocationModel
from .params import place_text, service_text
from .priors import ItemStats, MicrocatPrior
from .utils import timer

# колонки train, нужные для статистик (чтобы не копировать всю таблицу)
STATS_COLS = ["lemma_key", "search_category", "search_location_id", "item_id", "item_microcat_id",
              "item_location_id", "item_latitude", "item_longitude"]
V2_LEMMA_FIELDS = ("title", "params", "desc")
EXT_LEMMA_FIELDS = V2_LEMMA_FIELDS + ("service", "place")


def add_lemma_keys(df: pd.DataFrame, lem) -> pd.DataFrame:
    """«Мешок лемм» запроса. Лемматизируются только уникальные тексты."""
    uniq = pd.unique(df["search_query"].to_numpy(dtype=object))
    df["lemma_key"] = df["search_query"].map(dict(zip(uniq, lem.key_many(uniq))))
    return df


class ItemLemmaCache:
    """Лемматизированные поля объявлений по item_id. Корпуса валидации и бенчмарка почти
    совпадают, поэтому каждое объявление лемматизируется один раз."""

    def __init__(self, lem, desc_max_chars: int):
        self.lem, self.desc_max_chars = lem, desc_max_chars
        self._store: dict = {}          # поле → {item_id: строка лемм}

    def _raw(self, field: str, items: pd.DataFrame) -> list:
        if field == "title":
            return items["item_title_raw"].tolist()
        if field == "params":
            return items["item_infm_params_text"].tolist()
        if field == "desc":
            return [d[: self.desc_max_chars] for d in items["item_description_raw"]]
        if field == "service":
            return [service_text(p) for p in items["item_infm_params_text"]]
        if field == "place":
            return [place_text(p) for p in items["item_infm_params_text"]]
        raise KeyError(field)

    def get(self, items: pd.DataFrame, fields=V2_LEMMA_FIELDS) -> dict:
        ids = items["item_id"].tolist()
        out = {}
        for field in fields:
            store = self._store.setdefault(field, {})
            todo = [i for i, item_id in enumerate(ids) if item_id not in store]
            if todo:
                sub = items.iloc[todo]
                store.update(zip(sub["item_id"], self.lem.join_many(self._raw(field, sub))))
            out[field] = [store[i] for i in ids]
        return out


@dataclass
class Stage:
    name: str
    items: pd.DataFrame
    queries: pd.DataFrame
    corpus: Corpus
    query_set: QuerySet
    pool: pd.DataFrame
    features: list


def build_index(name: str, items: pd.DataFrame, *, cache: ItemLemmaCache, vocabs: Vocabs, cfg,
                ext_cfg=None) -> Corpus:
    fields = V2_LEMMA_FIELDS if ext_cfg is None else EXT_LEMMA_FIELDS
    with timer(f"{name}: индекс корпуса"):
        return build_corpus(items, cache.get(items, fields), cfg, vocabs, ext_cfg)


def build_pool(name: str, corpus: Corpus, items: pd.DataFrame, queries: pd.DataFrame,
               stats_rows: pd.DataFrame, *, lem, vocabs: Vocabs, cfg, use_item_stats: bool,
               truth: list = None, ext_cfg=None, verbose: bool = True):
    """
    stats_rows — строки train, по которым считаются статистики (без строк самих запросов!);
    truth      — эталон: добавляет в пул колонку label.
    Возвращает (QuerySet, пул).
    """
    with timer(f"{name}: статистики и пул"):
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

        query_set = build_queries(queries, lem, vocabs, prior, ext=ext_cfg is not None)
        pool = generate_pool(corpus, query_set, geo, cfg, item_stats, verbose=verbose, ext_cfg=ext_cfg)
        if truth is not None:
            pool = attach_labels(pool, corpus, truth)
    print(f"{name}: запросов {query_set.n:,}, строк пула {len(pool):,} (~{len(pool) / query_set.n:.0f} на запрос)")
    return query_set, pool


def build_stage(name: str, items: pd.DataFrame, queries: pd.DataFrame, stats_rows: pd.DataFrame, *,
                lem, cache: ItemLemmaCache, vocabs: Vocabs, cfg, use_item_stats: bool,
                truth: list = None) -> Stage:
    """Пайплайн v2 целиком: индекс корпуса + пул для одного набора запросов."""
    corpus = build_index(name, items, cache=cache, vocabs=vocabs, cfg=cfg)
    query_set, pool = build_pool(name, corpus, items, queries, stats_rows, lem=lem, vocabs=vocabs, cfg=cfg,
                                 use_item_stats=use_item_stats, truth=truth)
    return Stage(name, items, queries, corpus, query_set, pool, feature_names(use_item_stats))
