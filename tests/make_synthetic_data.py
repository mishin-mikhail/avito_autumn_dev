"""
Синтетические данные в схеме соревнования — для smoke-теста пайплайна
без настоящих файлов:  python tests/make_synthetic_data.py --out data_synth
Затем:  DATA_DIR=data_synth jupyter nbconvert --execute notebooks/01_eda_baseline.ipynb
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SERVICES = {
    "автоподбор": ["автоподбор", "осмотр автомобиля", "проверка авто", "диагностика перед покупкой"],
    "скупка": ["скупка телевизоров", "выкуп техники", "скупка ноутбуков", "продать телевизор"],
    "баня": ["баня на дровах", "аренда бани", "сауна", "русская баня с парной"],
    "домофон": ["монтаж видеодомофонов", "установка домофона", "ремонт домофона"],
    "колл-центр": ["обзвон по базе", "холодные звонки", "оператор колл центра"],
    "ремонт": ["ремонт квартир", "отделка под ключ", "укладка плитки", "штукатурка стен"],
    "красота": ["маникюр", "наращивание ресниц", "стрижка мужская", "макияж"],
    "грузоперевозки": ["грузоперевозки", "переезд квартиры", "грузчики", "газель с грузчиками"],
}
FILTERS = ["", "", "", "Рейтинг пользователя 4 звезды и выше", "Вид услуги Красота, здоровье",
           "Вид услуги Ремонт и отделка"]


def hexid(rng, n):
    return [f"{x:016x}" for x in rng.integers(0, 2**63, size=n, dtype=np.int64)]


def main(out: Path, n_items=6000, n_train=30000, n_bench=300, seed=0):
    rng = np.random.default_rng(seed)
    cats = list(SERVICES)
    micro_of = {c: [f"{i * 10 + j}" for j in range(2)] for i, c in enumerate(cats)}
    locs = [621540, 637640, 653240, 641780]

    # --- объявления ---
    cat_i = rng.integers(0, len(cats), n_items)
    items = pd.DataFrame({
        "item_id": hexid(rng, n_items),
        "item_title_raw": [f"{rng.choice(SERVICES[cats[c]]).capitalize()} недорого" for c in cat_i],
        "item_description_raw": [f"Опытный мастер. {' '.join(rng.choice(SERVICES[cats[c]], 3))}. Звоните!"
                                 for c in cat_i],
        "item_infm_params_text": [f"Вид услуги {cats[c]}" for c in cat_i],
        "item_category_id": [100 + c for c in cat_i],
        "item_microcat_id": [int(rng.choice(micro_of[cats[c]])) for c in cat_i],
        "item_price": rng.integers(500, 50000, n_items).astype(float),
        "item_rating": np.where(rng.random(n_items) < 0.3, np.nan, rng.uniform(3, 5, n_items)),
        "item_rating_reviews_count": rng.integers(0, 200, n_items).astype(float),
        "item_location_id": rng.choice(locs, n_items),
        "item_latitude": rng.uniform(55, 56, n_items),
        "item_longitude": rng.uniform(37, 38, n_items),
        "item_is_phone_hidden": rng.integers(0, 2, n_items),
        "item_is_message_forbidden": rng.integers(0, 2, n_items),
    })

    def make_pairs(n):
        rows = []
        for _ in range(n):
            c = int(rng.integers(0, len(cats)))
            q = rng.choice(SERVICES[cats[c]])
            loc = int(rng.choice(locs))
            deliv = int(rng.random() < 0.1)
            pool = items[(items["item_category_id"] == 100 + c)
                         & ((items["item_location_id"] == loc) if not deliv else True)]
            it = pool.iloc[int(rng.integers(0, len(pool)))]
            rows.append({"search_query": q, "search_location_id": loc,
                         "search_is_delivery_search": deliv,
                         "search_infm_params_text": rng.choice(FILTERS),
                         "search_category": f"cat_{c % 3}", **it.to_dict()})
        return pd.DataFrame(rows)

    train = make_pairs(n_train)
    bench_pairs = make_pairs(n_bench * 2).drop_duplicates(
        ["search_query", "search_location_id", "search_is_delivery_search",
         "search_infm_params_text", "search_category"]).head(n_bench)
    qcols = ["search_query", "search_location_id", "search_is_delivery_search",
             "search_infm_params_text", "search_category"]
    bench_q = bench_pairs[qcols].copy()
    bench_q.insert(0, "query_id", [f"{i:016d}"[-16:].replace("0", "Q", 1) for i in range(len(bench_q))])
    # корпус бенчмарка: часть объявлений train + все позитивы бенчмарка
    sub = items.sample(frac=0.7, random_state=seed)
    bench_items = pd.concat([sub, items[items["item_id"].isin(bench_pairs["item_id"])]]
                            ).drop_duplicates("item_id")

    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    bench_q.to_parquet(out / "benchmark_queries.parquet", index=False)
    bench_items.to_parquet(out / "benchmark_items.parquet", index=False)
    print(f"saved to {out}: train {train.shape}, bench_q {bench_q.shape}, bench_items {bench_items.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data_synth"))
    main(ap.parse_args().out)
