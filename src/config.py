"""
Единый конфиг решения.

Все гиперпараметры живут здесь, а не разбросаны по коду: так проще
воспроизводить эксперименты и описывать их в README.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    # --- воспроизводимость ---
    seed: int = 42
    n_threads: int = 4            # фиксируем число потоков BLAS
    score_decimals: int = 6       # скоры округляются перед сортировкой (гасим шум float)

    # --- локальная валидация ---
    n_val_queries: int = 2500     # размер псевдо-бенчмарка (в бенчмарке 2452);
                                  # состав стратифицирован под бенчмарк: знакомый/новый текст × фильтр × тип локации
    text_holdout_frac: float = 0.05  # доля текстов запросов, целиком убранных из train
    val_salt: str = "val-v1"      # соль для md5-разбиения; другая соль = другое разбиение

    # --- тексты ---
    desc_max_chars: int = 2000    # описание обрезаем: хвосты длинных описаний шумят

    # --- BM25 ---
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    batch_size: int = 64          # запросов за раз при плотном скоринге всего корпуса;
                                  # влияет только на память (~1 ГБ на батч), не на результат

    # --- локация ---
    geo_alpha: float = 1.0        # сглаживание P(локация объявления | локация поиска) к совпадению
    geo_near_km: float = 30.0     # масштаб «близости» для отбора кандидатов в пул

    # --- пул кандидатов (объединение нескольких списков) ---
    pool_k_text: int = 400        # топ по тексту
    pool_k_text_loc: int = 400    # топ по тексту с приоритетом близких объявлений
    pool_k_prior_loc: int = 400   # близкие объявления из вероятных микрокатегорий
    pool_k_memo: int = 200        # объявления, выбранные по этому же запросу в train
    top_k: int = 50

    # --- априорные вероятности микрокатегорий (сглаживание) ---
    prior_alpha: float = 5.0
    prior_beta: float = 20.0

    # статистики по item_id (популярность, «память» запрос→объявление) включаем,
    # только если объявления корпуса действительно встречаются в train
    item_stats_min_overlap: float = 0.5

    # --- подбор весов линейной формулы ---
    tune_grid: tuple = (-1.0, -0.25, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    tune_passes: int = 2
    # стартовая точка — веса, подобранные в v1 (Recall@50 = 0.842 на валидации v1)
    init_weights: tuple = (("title", 0.25), ("params", -0.25), ("desc", 0.5), ("filt", 0.5),
                           ("cov", 0.5), ("logp", 0.1), ("loc_same", 2.0))

    version: str = "v2"

    def as_dict(self) -> dict:
        return asdict(self)


CFG = Config()


@dataclass(frozen=True)
class RankerConfig:
    """Параметры v3: трудная валидация, расширенные признаки, LightGBM-ранкер.
    Общие параметры пайплайна (BM25, пул, локации) берутся из Config."""
    version: str = "v3"

    # --- опорные значения отправки v2 (тег v2) ---
    v2_lb: float = 0.8313
    v2_answer_md5: str = "2de61da58afd87fc244e225b7cccde58"
    v2_weights: tuple = (("title", 2.0), ("params", 0.1), ("desc", 2.0), ("cov", 1.0), ("filt", 2.0),
                         ("filt_exact", 0.25), ("logp", 0.1), ("loc_same", 2.0), ("loc_p", 2.0),
                         ("loc_logp", 0.25))

    # --- выборка запросов ---
    # "auto": из двух схем валидации берётся та, где веса v2 ближе к результату на лидерборде
    val_scheme: str = "auto"          # "auto" | "injected" | "in_corpus"
    val_salt: str = "val-v3"
    n_val_queries: int = 2500
    text_holdout_frac: float = 0.2    # у «новых» запросов берётся один запрос на текст
    n_folds: int = 4                  # фолды для ранкера: n_folds-1 на обучение, последний — ранняя остановка
    fold_queries: int = 4000
    fold_holdout_frac: float = 0.3    # у фолдов выше: «новых» текстов среди запросов «в корпусе» немного

    # --- расширенный пул ---
    pool_k_char: int = 300
    char_ngram: tuple = (3, 5)
    char_min_df: int = 3

    # --- обучающая выборка ранкера (на запрос) ---
    sample_hard: int = 150            # лучшие по формуле v2
    sample_char: int = 50             # лучшие по символьному сходству среди остальных
    sample_random: int = 100          # случайные из оставшихся

    # --- LightGBM ---
    objectives: tuple = ("lambdarank", "binary")
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.8
    lambda_l2: float = 1.0
    max_rounds: int = 1500
    early_stopping: int = 100

    def as_dict(self) -> dict:
        return asdict(self)


RANKER_CFG = RankerConfig()
