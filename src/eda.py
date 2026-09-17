"""
EDA: ответы на вопросы, от которых зависят решения.

Каждая функция печатает короткую сводку и возвращает словарь с ключевыми
числами — часть из них дальше используется в пайплайне автоматически.
"""
import numpy as np
import pandas as pd

from .filters import match_share, normalize_params, parse_filter_pairs
from .geo import planar_km
from .utils import section


def query_groups(train: pd.DataFrame) -> dict:
    section("Сколько объявлений выбирают по одному запросу")
    sizes = train.groupby("query_key")["item_id"].nunique()
    print(f"групп-запросов: {len(sizes):,} | текстов: {train['norm_text'].nunique():,} | "
          f"объявлений: {train['item_id'].nunique():,}")
    print(f"объявлений на запрос: среднее {sizes.mean():.2f}, медиана {sizes.median():.0f}, "
          f"p90 {sizes.quantile(.9):.0f}, p99 {sizes.quantile(.99):.0f}, max {sizes.max()}")
    dist = sizes.clip(upper=5).value_counts(normalize=True).sort_index().round(3)
    print("доли групп (5 = «5 и больше»):", dist.to_dict())
    return {"mean_items_per_query": float(sizes.mean())}


def overlap(train: pd.DataFrame, bench_q: pd.DataFrame, bench_items: pd.DataFrame,
            min_item_overlap: float) -> dict:
    section("Пересечение бенчмарка с train")
    train_ids = pd.Index(train["item_id"].unique())
    item_share = float(bench_items["item_id"].isin(train_ids).mean())
    text_share = float(bench_q["norm_text"].isin(pd.Index(train["norm_text"].unique())).mean())
    key_share = float(bench_q["query_key"].isin(pd.Index(train["query_key"].unique())).mean())
    print(f"объявлений корпуса, встречающихся в train: {item_share:.3f}")
    print(f"запросов, чей текст встречается в train:  {text_share:.3f}")
    print(f"запросов, чей полный ключ есть в train:   {key_share:.3f}")
    use = item_share >= min_item_overlap
    print(f"→ статистики по item_id {'ВКЛЮЧЕНЫ' if use else 'выключены'} (порог {min_item_overlap})")
    return {"item_overlap": item_share, "text_seen": text_share, "key_seen": key_share,
            "use_item_stats": use}


def locations(train: pd.DataFrame, bench_q: pd.DataFrame, bench_items: pd.DataFrame) -> dict:
    section("Локации")
    item_locs = pd.Index(pd.concat([train["item_location_id"], bench_items["item_location_id"]]).unique())
    same = train["search_location_id"].to_numpy() == train["item_location_id"].to_numpy()
    mism = train[~same]
    region_like = ~mism["search_location_id"].isin(item_locs)
    print(f"пар, где локация объявления = локации поиска: {same.mean():.3f}")
    print(f"из несовпадающих: локация поиска никогда не бывает у объявлений: {region_like.mean():.3f}")
    print(f"запросов бенчмарка с такой («только поисковой») локацией: "
          f"{(~bench_q['search_location_id'].isin(item_locs)).mean():.3f}")

    centers = train.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
    m = mism.join(centers, on="search_location_id", rsuffix="_c")
    km = pd.Series(planar_km(m["item_latitude_c"], m["item_longitude_c"],
                             m["item_latitude"], m["item_longitude"]))
    print("расстояние до объявления при несовпадении, км: "
          + ", ".join(f"p{int(p * 100)}={km.quantile(p):.0f}" for p in (.5, .75, .9, .95)))

    trans = mism.groupby(["search_location_id", "item_location_id"]).size().sort_values(ascending=False)
    print(f"переходов «поиск → объявление»: {len(trans):,}; "
          f"несовпадающих пар в переходах, встреченных ≥5 раз: {trans[trans >= 5].sum() / len(mism):.3f}")
    print("самые частые переходы:\n" + trans.head(5).to_string())
    return {"item_locations": item_locs, "same_location_share": float(same.mean())}


def filters(train: pd.DataFrame, bench_q: pd.DataFrame, sample_size: int, seed: int) -> dict:
    section("Фильтры поиска")
    print(f"запросов с фильтрами: train {(train['filters_norm'] != '').mean():.3f} | "
          f"бенчмарк {(bench_q['filters_norm'] != '').mean():.3f}")
    print(f"поиск с доставкой: train {train['search_is_delivery_search'].mean():.5f} | "
          f"бенчмарк {bench_q['search_is_delivery_search'].mean():.5f}")

    has_thr = ~np.isnan(train["rating_thr"].to_numpy())
    r, t = train["item_rating"].to_numpy()[has_thr], train["rating_thr"].to_numpy()[has_thr]
    print(f"фильтр по рейтингу: {has_thr.mean():.4f} пар; из них рейтинг ≥ порога {np.mean(r >= t):.3f}")

    # соблюдается ли фильтр «дословно»: детерминированная подвыборка пар с фильтрами
    with_f = np.flatnonzero((train["filters_norm"] != "").to_numpy())
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(with_f, size=min(sample_size, len(with_f)), replace=False))
    sub = train.iloc[idx]
    pairs = [parse_filter_pairs(x) for x in sub["search_infm_params_text"]]
    shares = [match_share(p, normalize_params(t)) for p, t in zip(pairs, sub["item_infm_params_text"])
              if p]
    print(f"пар с разобранными фильтрами: {np.mean([bool(p) for p in pairs]):.3f}; "
          f"из них все пары найдены в параметрах объявления: {np.mean(np.array(shares) == 1):.3f}, "
          f"средняя доля найденных: {np.mean(shares):.3f}")
    top = train["search_infm_params_text"].replace("", "<без фильтра>").value_counts().head(8)
    print("частые фильтры:\n" + top.to_string())
    return {"filter_exact_compliance": float(np.mean(np.array(shares) == 1)) if shares else float("nan")}


def categories(train: pd.DataFrame, bench_items: pd.DataFrame) -> dict:
    section("Категории")
    print(f"категорий поиска: {train['search_category'].nunique()} "
          f"(самая частая — {train['search_category'].value_counts(normalize=True).iloc[0]:.4f} строк)")
    print(f"микрокатегорий в train: {train['item_microcat_id'].nunique()}; объявлений корпуса "
          f"в незнакомых train микрокатегориях: "
          f"{(~bench_items['item_microcat_id'].isin(pd.Index(train['item_microcat_id'].unique()))).mean():.3f}")
    print(f"объявлений корпуса в самой частой категории: "
          f"{bench_items['item_category_id'].value_counts(normalize=True).iloc[0]:.3f}")
    return {}
