"""
Единый конфиг решения.

Все гиперпараметры живут здесь, а не разбросаны по коду: так проще
воспроизводить эксперименты и описывать их в README.
"""
from dataclasses import asdict, dataclass, replace


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
    # стартовая точка - веса, подобранные в v1 (Recall@50 = 0.842 на валидации v1)
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
    n_folds: int = 4                  # фолды для ранкера: n_folds-1 на обучение, последний - ранняя остановка
    fold_queries: int = 4000
    fold_holdout_frac: float = 0.3    # у фолдов выше: «новых» текстов среди запросов «в корпусе» немного
    # «новые» запросы: False - в порядке хеша группы (v3; перекос в сторону популярных текстов),
    # True - все отложенные тексты равновероятны, как редкие новые тексты бенчмарка (v4)
    uniform_unseen_texts: bool = False

    # --- расширенный пул ---
    pool_k_char: int = 300
    pool_k_dense: int = 300           # список по эмбеддингам (v4; без эмбеддингов не используется)
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

    # --- v5 (значения по умолчанию выключают всё новое: пул и признаки v3/v4 не меняются) ---
    # 1. «размытые» локации поиска: регион или город, откуда чаще выбирают объявления в других городах
    geo_v5: bool = False                # признаки loc_pn, loc_cover, dist_rel, q_self_share, q_diffuse
    region_cover_mass: float = 0.95     # «ядро» локации поиска - города, куда уходит 95% переходов
    region_radius_quantile: float = 0.8 # радиус локации поиска - этот квантиль расстояний выбранных объявлений
    region_radius_max_km: float = 300.0
    region_self_share: float = 0.5      # размытая = в свою же локацию уходит меньше этой доли переходов
    pool_k_region: int = 0              # длина каждого из трёх списков для размытых локаций (0 - выкл.)
    region_lists: tuple = ("text", "prior", "dense")   # какие из трёх списков строить (v6: только dense)
    # 2. P(микрокатегория) по похожим запросам train (соседи по эмбеддингам)
    pool_k_knn: int = 0                 # список «близкие объявления вероятных микрокатегорий по соседям»
    knn_neighbors: int = 30             # сколько ближайших текстов train рассматривать
    knn_margin: float = 0.05            # вес соседа = max(sim − sim_лучшего + margin, 0)
    knn_shrink: float = 2.0             # доверие к тексту с n строками: n / (n + shrink)
    # 3. поиск по эмбеддингам без учёта локации
    pool_k_dense_pure: int = 0
    # v6: итоговый скор - среднее мест в запросе по ранкерам всех целевых функций (если на фолде лучше)
    ensemble: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


RANKER_CFG = RankerConfig()

# v5: что меняется относительно v4 (ноутбук 05_ranker_v5.ipynb)
RANKER_V5 = replace(
    RANKER_CFG, version="v5",
    uniform_unseen_texts=True,          # как в v4
    geo_v5=True, pool_k_region=200,     # размытые локации
    pool_k_knn=300,                     # микрокатегории по соседним запросам
    pool_k_dense=400, pool_k_dense_pure=150,   # расширенный поиск по эмбеддингам (в v4: 300 и 0)
)

# v6: по разбору v5 - из списков для размытых локаций оставлен только список по эмбеддингам
# (текстовый и «по микрокатегориям» ничего не добавили к полноте), он и список src_dense_loc длиннее;
# итог - среднее двух ранкеров. Эмбеддинги - второй раунд дообучения (EMB_CFG_V6).
RANKER_V6 = replace(
    RANKER_V5, version="v6",
    region_lists=("dense",), pool_k_region=300,
    pool_k_dense=500,
    ensemble=True,
)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Параметры двухбашенного энкодера (v4): дообучение и кодирование корпуса."""
    # кандидаты: multilingual-e5-base (базовый) и deepvk/USER-base (тот же e5, дообучен на русском).
    # Оба требуют префиксов "query: " и "passage: ".
    candidates: tuple = ("intfloat/multilingual-e5-base", "deepvk/USER-base")

    max_len_query: int = 48
    max_len_item: int = 160
    desc_chars: int = 300          # сколько символов описания попадает в текст объявления

    # дообучение: InfoNCE с негативами из батча + трудными негативами из корпуса
    train_pairs: int = 400_000     # сколько пар train взять (0 - все)
    batch_size: int = 0            # 0 - подобрать под память GPU (больше батч - больше негативов)
    max_batch_size: int = 512      # верхняя граница автоподбора
    hard_negatives: int = 2        # трудных негативов на запрос
    hard_neg_skip: int = 5         # первые ранги не берём: там часто объявления, которые тоже подходят
    hard_neg_depth: int = 50       # негатив выбирается случайно из рангов [skip, depth)
    epochs: int = 1
    lr: float = 2e-5
    warmup_frac: float = 0.05
    temperature: float = 0.02
    max_grad_norm: float = 1.0
    encode_batch: int = 512
    # "auto": bf16 на GPU, который его умеет (A100 и новее), иначе fp16; ещё "bf16" | "fp16" | "off"
    amp_dtype: str = "auto"
    # градиентные чекпоинты: память ценой ~30% скорости - зато помещается батч в 2-3 раза больше
    grad_checkpointing: bool = True

    # оценка качества поиска по векторам (Recall@100) до и после дообучения
    zero_shot_queries: int = 2000

    # --- версия артефакта ---
    version: str = "v4"
    artifact_name: str = "embeddings"      # папка артефакта внутри WORK_DIR
    init_model: str = ""                   # дообучать уже дообученную модель (путь внутри WORK_DIR); "" - с нуля
    # не брать в трудные негативы объявления той же микрокатегории, что и позитив:
    # это чаще всего тоже подходящие объявления («Скупка телевизоров» для «скупка б/у техники»)
    neg_exclude_same_micro: bool = False
    freeze_word_embeddings: bool = False   # не обучать матрицу эмбеддингов слов (экономия памяти GPU)

    def as_dict(self) -> dict:
        return asdict(self)


EMB_CFG = EmbeddingConfig()

# v6: второй раунд дообучения - от модели v4, с очищенными трудными негативами
EMB_CFG_V6 = replace(
    EMB_CFG, version="v6", artifact_name="embeddings_v6", init_model="embeddings/model",
    neg_exclude_same_micro=True, hard_neg_skip=3, hard_neg_depth=100,
    train_pairs=0,          # все свободные пары train (~264 тыс.)
    lr=1e-5,                # модель уже дообучена - шаг меньше
)

# large (для смеси v8): multilingual-e5-large (560 млн параметров, размерность 1024), один раунд с очищенными
# негативами. 150 тыс. пар: большая модель учится примерно в 2,5 раза медленнее base (~2 ч на A100 20 ГБ)
EMB_CFG_LARGE = replace(
    EMB_CFG, version="large", artifact_name="embeddings_large",
    candidates=("intfloat/multilingual-e5-large",),
    neg_exclude_same_micro=True, hard_neg_skip=3, hard_neg_depth=100,
    train_pairs=150_000, lr=1e-5,
    freeze_word_embeddings=True,     # иначе на 20 ГБ помещается батч ~64 - мало негативов
)
