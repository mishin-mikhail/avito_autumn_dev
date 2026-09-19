"""
Обученный ранкер (LightGBM) поверх пула кандидатов.

Схема:
  1. stage1 — линейная формула v2 (веса из отправки v2): её скор и ранг — признаки ранкера
     и основа для отбора «трудных» негативов;
  2. обучающая выборка — фолды запросов из train; для каждого запроса берутся все позитивы
     и часть негативов (трудные, символьно похожие, случайные), чтобы не учиться на ~20 млн строк;
  3. ранняя остановка — по Recall@50 на полном пуле отдельного фолда;
  4. предсказание — по всему пулу, топ-50 по скору ранкера.

Детерминизм LightGBM: deterministic=True, force_row_wise=True, фиксированные сиды и число
потоков, без бэггинга строк. Скоры округляются перед сортировкой, как и в линейной формуле.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidates import BASE_FEATURES, DENSE_FEATURES, EXT_FEATURES, V5_FEATURES
from .ranking import linear_score, rank_order, pool_recall, weights_vector

STAGE1_FEATURES = ["stage1", "stage1_rank"]
RANKER_FEATURES = BASE_FEATURES + EXT_FEATURES + STAGE1_FEATURES


def ranker_features(use_dense: bool = False, v5: bool = False) -> list:
    """Признаки ранкера; с эмбеддингами (v4) добавляются dense и rank_dense_loc, в v5 — V5_FEATURES."""
    return RANKER_FEATURES + (DENSE_FEATURES if use_dense else []) + (V5_FEATURES if v5 else [])


def positions_in_query(q: np.ndarray, rank: np.ndarray, score: np.ndarray, decimals: int) -> np.ndarray:
    """Место каждой строки внутри своего запроса при сортировке по score (0 — лучшая)."""
    order = rank_order(q, rank, score, decimals)
    qs = q[order]
    first = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    pos_sorted = np.arange(len(qs)) - np.repeat(first, np.diff(np.r_[first, len(qs)]))
    out = np.empty(len(q), dtype=np.int64)
    out[order] = pos_sorted
    return out


def add_stage1(pool: pd.DataFrame, corpus, v2_weights, decimals: int) -> pd.DataFrame:
    w = weights_vector(BASE_FEATURES, v2_weights)
    score = linear_score(pool[BASE_FEATURES].to_numpy(np.float64), w)
    pool["stage1"] = score.astype(np.float32)
    pool["stage1_rank"] = positions_in_query(pool["q"].to_numpy(np.int64), corpus.rank[pool["item"].to_numpy()],
                                             score, decimals).astype(np.float32)
    return pool


def sample_training_rows(pool: pd.DataFrame, rcfg, seed: int) -> pd.DataFrame:
    """Все позитивы + трудные, символьно похожие и случайные негативы каждого запроса."""
    label = pool["label"].to_numpy()
    hard = pool["stage1_rank"].to_numpy() < rcfg.sample_hard
    char = ~hard & (pool["rank_char_loc"].to_numpy() < rcfg.sample_char)
    rest = (label == 0) & ~hard & ~char

    rng = np.random.default_rng(seed)
    key = rng.random(len(pool))
    q = pool["q"].to_numpy()
    idx = np.flatnonzero(rest)
    order = idx[np.lexsort((key[idx], q[idx]))]
    q_sorted = q[order]
    first = np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1]])
    pos = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
    random_pick = np.zeros(len(pool), dtype=bool)
    random_pick[order[pos < rcfg.sample_random]] = True

    keep = (label == 1) | hard | char | random_pick
    return pool.loc[keep].reset_index(drop=True)


def _params(objective: str, rcfg, seed: int, n_threads: int) -> dict:
    params = dict(
        objective=objective, metric="None", learning_rate=rcfg.learning_rate,
        num_leaves=rcfg.num_leaves, min_data_in_leaf=rcfg.min_data_in_leaf,
        feature_fraction=rcfg.feature_fraction, lambda_l2=rcfg.lambda_l2,
        seed=seed, feature_fraction_seed=seed, data_random_seed=seed, bagging_seed=seed,
        deterministic=True, force_row_wise=True, num_threads=n_threads, verbosity=-1,
    )
    if objective == "lambdarank":
        params["lambdarank_truncation_level"] = 100
    return params


def _group_sizes(gid: np.ndarray) -> np.ndarray:
    """Размеры подряд идущих групп (строки должны быть отсортированы по gid)."""
    first = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    return np.diff(np.r_[first, len(gid)])


def train_ranker(train_rows: pd.DataFrame, valid_pool: pd.DataFrame, valid_n_rel: np.ndarray, valid_rank: np.ndarray,
                 objective: str, rcfg, seed: int, n_threads: int, k: int, decimals: int, log_every: int = 100,
                 features: list = None):
    """
    train_rows — выборка строк с колонками gid (номер запроса, строки сгруппированы), label и признаками;
    valid_pool — полный пул фолда для ранней остановки (с label), valid_rank — ранги item_id его строк.
    Возвращает (модель, лучшая итерация, лучший Recall@k на фолде).
    """
    features = features or RANKER_FEATURES
    params = _params(objective, rcfg, seed, n_threads)   # те же параметры и для датасета, и для обучения
    gid = train_rows["gid"].to_numpy()
    label = train_rows["label"].to_numpy()
    # запросы без позитивов ничему не учат
    keep = np.flatnonzero(pd.Series(label).groupby(gid).transform("max").to_numpy() > 0)
    # Датасеты собираются сразу (construct): LightGBM переводит признаки в свои гистограммы,
    # и исходные float32-массивы освобождаются до начала обучения — так ниже пик памяти.
    dtrain = lgb.Dataset(train_rows[features].to_numpy(np.float32)[keep], label=label[keep],
                         group=_group_sizes(gid[keep]), feature_name=features,
                         params=params, free_raw_data=True).construct()
    vq = valid_pool["q"].to_numpy(np.int64)
    v_label = valid_pool["label"].to_numpy(np.float64)
    dvalid = lgb.Dataset(valid_pool[features].to_numpy(np.float32), label=valid_pool["label"].to_numpy(),
                         group=_group_sizes(vq), reference=dtrain, params=params, free_raw_data=True).construct()

    def recall_metric(preds, _data):
        return f"recall@{k}", pool_recall(vq, valid_rank, v_label, valid_n_rel,
                                          np.asarray(preds, dtype=np.float64), k, decimals), True

    booster = lgb.train(
        params, dtrain, num_boost_round=rcfg.max_rounds,
        valid_sets=[dvalid], valid_names=["fold"], feval=recall_metric,
        callbacks=[lgb.early_stopping(rcfg.early_stopping, first_metric_only=True, verbose=False),
                   lgb.log_evaluation(log_every)],
    )
    return booster, booster.best_iteration, booster.best_score["fold"][f"recall@{k}"]


def ranker_score(booster, pool: pd.DataFrame, n_threads: int, features: list = None) -> np.ndarray:
    return booster.predict(pool[features or RANKER_FEATURES].to_numpy(np.float32),
                           num_iteration=booster.best_iteration, num_threads=n_threads).astype(np.float64)


def importance_table(booster) -> pd.DataFrame:
    gain = booster.feature_importance("gain")
    return (pd.DataFrame({"признак": booster.feature_name(), "gain": gain})
            .assign(доля=lambda d: d["gain"] / d["gain"].sum())
            .sort_values(["gain", "признак"], ascending=[False, True], kind="stable")
            .reset_index(drop=True))
