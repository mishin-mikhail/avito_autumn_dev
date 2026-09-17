"""
Синтетические данные в схеме соревнования — smoke-тест пайплайна без настоящих файлов.

    python tests/make_synthetic_data.py --out data_synth
    DATA_DIR=data_synth jupyter nbconvert --to notebook --execute notebooks/01_eda_baseline.ipynb

Воспроизводит ключевые особенности реальных данных: параметры объявлений в виде
«Вид услуги … Место оказания услуг … Тип услуги …», фильтры в том же формате,
«региональные» локации поиска, которых нет у объявлений, и координаты.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# вид услуги → {тип услуги → формулировки}
SERVICES = {
    "Автосервис, аренда": {"Автосервис": ["автоподбор", "осмотр автомобиля", "русификация авто"],
                           "Аренда авто": ["аренда автодома", "авто в рассрочку"]},
    "Ремонт и обслуживание техники": {"Телевизоры": ["ремонт телевизоров", "скупка телевизоров"],
                                      "Компьютеры": ["ремонт ноутбуков", "выкуп техники"]},
    "Красота, здоровье": {"Маникюр, педикюр": ["маникюр", "наращивание ногтей"],
                          "Услуги парикмахера": ["стрижка мужская", "окрашивание волос на дому"]},
    "Ремонт и отделка": {"Плиточные работы": ["укладка плитки", "ремонт ванной"],
                         "Сантехника": ["замена смесителя", "очистка жироуловителя"]},
    "Праздники, мероприятия": {"Аренда площадок": ["аренда лофта", "баня на дровах"]},
}
CITIES = {  # код локации: (город, широта, долгота)
    637640: ("Москва", 55.75, 37.62), 653240: ("Санкт-Петербург", 59.94, 30.31),
    640860: ("Нижний Новгород", 56.30, 43.94), 652000: ("Казань", 55.79, 49.12),
    633540: ("Подольск", 55.43, 37.54),
}
REGIONS = {107620: [637640, 633540], 107621: [653240]}   # «региональные» id поиска


def hexid(rng, n):
    return [f"{x:016x}" for x in rng.integers(0, 2**62, size=n, dtype=np.int64)]


def main(out: Path, n_items=6000, n_train=30000, n_bench=300, seed=0):
    rng = np.random.default_rng(seed)
    combos = [(vid, tip, phr) for vid, types in SERVICES.items() for tip, phr in types.items()]
    city_codes = list(CITIES)

    # ── объявления ──
    ci = rng.integers(0, len(combos), n_items)
    loc = rng.choice(city_codes, n_items)
    rows = []
    for c, l in zip(ci, loc):
        vid, tip, phrases = combos[c]
        city, lat, lon = CITIES[l]
        rows.append({
            "item_title_raw": f"{rng.choice(phrases).capitalize()} {rng.choice(['недорого', 'с гарантией', ''])}".strip(),
            "item_description_raw": f"Опытный мастер. {'. '.join(rng.choice(phrases, 3))}. Работаем в городе {city}.",
            "item_infm_params_text": f"Вид услуги {vid} Место оказания услуг {city}, улица Ленина, {rng.integers(1, 99)} "
                                     f"Тип услуги {tip} Тип стоимости за услугу Опыт работы Больше 5 лет",
            "item_category_id": 114, "item_microcat_id": 1000 + c,
            "item_price": float(rng.integers(500, 50000)),
            "item_rating": np.nan if rng.random() < 0.1 else rng.uniform(3, 5),
            "item_rating_reviews_count": np.nan if rng.random() < 0.05 else float(rng.integers(0, 300)),
            "item_location_id": int(l),
            "item_latitude": lat + rng.normal(0, 0.1), "item_longitude": lon + rng.normal(0, 0.1),
            "item_is_phone_hidden": float(rng.integers(0, 2)), "item_is_message_forbidden": 0.0,
        })
    items = pd.DataFrame(rows)
    items.insert(0, "item_id", hexid(rng, n_items))

    # ── пары «запрос → выбранное объявление» ──
    def make_pairs(n):
        out = []
        for _ in range(n):
            c = int(rng.integers(0, len(combos)))
            vid, tip, phrases = combos[c]
            search_loc = int(rng.choice(city_codes + list(REGIONS), p=[.15] * 5 + [.15, .10]))
            item_locs = REGIONS.get(search_loc, [search_loc])
            if rng.random() < 0.15:                      # иногда выбирают соседний город
                item_locs = city_codes
            cand = items[(items["item_microcat_id"] == 1000 + c) & items["item_location_id"].isin(item_locs)]
            if cand.empty:
                continue
            filt = rng.choice(["", "", f"Вид услуги {vid}", f"Тип услуги {tip} Вид услуги {vid}",
                               "Рейтинг пользователя 4 звезды и выше", "Вид услуги"])
            if "Рейтинг" in filt:
                cand = cand[cand["item_rating"] >= 4] if (cand["item_rating"] >= 4).any() else cand
            it = cand.iloc[int(rng.integers(0, len(cand)))]
            out.append({"search_query": str(rng.choice(phrases)), "search_location_id": search_loc,
                        "search_is_delivery_search": 0, "search_infm_params_text": str(filt),
                        "search_category": 114, **it.to_dict()})
        return pd.DataFrame(out)

    train = make_pairs(n_train)
    qcols = ["search_query", "search_location_id", "search_is_delivery_search",
             "search_infm_params_text", "search_category"]
    bench_pairs = make_pairs(n_bench * 2).drop_duplicates(qcols).head(n_bench)
    bench_q = bench_pairs[qcols].reset_index(drop=True)
    bench_q.insert(0, "query_id", [f"Q{i:015d}" for i in range(len(bench_q))])
    # корпус бенчмарка: случайная часть объявлений + все позитивы бенчмарка
    corpus = pd.concat([items.sample(frac=0.7, random_state=seed),
                        items[items["item_id"].isin(bench_pairs["item_id"])]]).drop_duplicates("item_id")

    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    bench_q.to_parquet(out / "benchmark_queries.parquet", index=False)
    corpus.to_parquet(out / "benchmark_items.parquet", index=False)
    print(f"{out}: train {train.shape}, запросы {bench_q.shape}, корпус {corpus.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data_synth"))
    main(ap.parse_args().out)
