"""
Пул кандидатов и признаки пар «запрос — объявление».

Весь корпус скорится плотно (батчами по cfg.batch_size запросов), затем
объединяются несколько списков — у каждого своя «специализация»:
  * src_text       топ по тексту (BM25 полей + покрытие лемм запроса);
  * src_text_loc   топ по тексту с приоритетом близких объявлений;
  * src_prior_loc  близкие объявления из самых вероятных микрокатегорий;
  * src_memo       объявления, выбранные по такому же запросу в train
                   (только если объявления корпуса встречаются в train).
«Близость» = max(совпадение локации, P(перехода локаций), exp(-расстояние / geo_near_km)).

Расширение v3 (включается, если корпус построен с ext_cfg): ещё один список
  * src_char_loc   топ по символьному сходству с приоритетом близких объявлений,
и дополнительные признаки для ранкера (EXT_FEATURES). Без расширения пул и признаки
в точности совпадают с v2.

Расширение v4 (если в корпус переданы вектора объявлений): список
  * src_dense_loc  топ по близости эмбеддингов с приоритетом близких объявлений,
и признаки dense, rank_dense_loc (DENSE_FEATURES).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bm25 import BM25Field, CoverageField, row_max_normalize
from .charsim import CharIndex
from .encoder import exact_dot
from .filters import normalize_params, parse_filter_pairs, pool_match_share
from .params import service_text

BASE_FEATURES = [
    "title", "params", "desc", "cov",           # текст
    "filt", "filt_exact",                       # фильтры поиска
    "logp",                                     # log P(микрокатегория | запрос)
    "loc_same", "loc_p", "loc_logp", "log_dist",  # локация
    "rating_ok", "rating", "log_reviews",       # качество объявления
]
ITEM_STAT_FEATURES = ["log_pop", "log_memo"]
SOURCES = ["src_text", "src_text_loc", "src_prior_loc", "src_memo"]
ALL_SOURCES = SOURCES + ["src_char_loc", "src_dense_loc"]      # src_char_loc — v3, src_dense_loc — v4
V2_FIELDS = ("title", "params", "desc")

# признаки расширения v3: пара «запрос — объявление», запрос, объявление
EXT_PAIR_FEATURES = ["service", "cov_place", "char", "text0", "prior_rank", "dist_km",
                     "rank_text", "rank_text_loc", "rank_prior_loc", "rank_char_loc"]
EXT_QUERY_FEATURES = ["q_len", "q_n_pairs", "q_region", "q_text_cnt", "q_prior_max", "q_prior_ent",
                      "q_title_max", "q_n_same_loc", "q_n_near", "q_pool_size"]
EXT_ITEM_FEATURES = ["i_log_price", "i_title_len", "i_desc_len", "i_phone_hidden", "i_msg_forbidden"]
EXT_FEATURES = EXT_PAIR_FEATURES + EXT_QUERY_FEATURES + EXT_ITEM_FEATURES
DENSE_FEATURES = ["dense", "rank_dense_loc"]      # v4, только если есть вектора


def feature_names(use_item_stats: bool) -> list:
    return BASE_FEATURES + (ITEM_STAT_FEATURES if use_item_stats else [])


# ─────────────────────────── словари id ───────────────────────────

class Vocab:
    """Строковые id → коды 0..n-1; неизвестное значение → код n (`unknown`)."""

    def __init__(self, values):
        self.index = {v: i for i, v in enumerate(sorted(dict.fromkeys(values)))}
        self.unknown = len(self.index)

    def encode(self, values) -> np.ndarray:
        get, unk = self.index.get, self.unknown
        return np.fromiter((get(v, unk) for v in values), dtype=np.int64, count=len(values))

    def __len__(self):
        return len(self.index)

    @property
    def size_with_unknown(self) -> int:
        return len(self.index) + 1


@dataclass
class Vocabs:
    loc: Vocab
    micro: Vocab


def build_vocabs(query_frames: list, item_frames: list) -> Vocabs:
    """Общие словари по всем источникам, чтобы коды совпадали на валидации и бенчмарке."""
    locs = [f["search_location_id"] for f in query_frames] + [f["item_location_id"] for f in item_frames]
    micros = [f["item_microcat_id"] for f in item_frames]
    return Vocabs(loc=Vocab(pd.concat(locs).tolist()), micro=Vocab(pd.concat(micros).tolist()))


# ─────────────────────────── корпус и запросы ───────────────────────────

@dataclass
class ExtIndex:
    """Индексы и свойства объявлений для расширения v3."""
    service: BM25Field          # BM25 по значениям «Вид/Тип услуги»
    place: CoverageField        # покрытие лемм запроса адресом («город в запросе»)
    char: CharIndex             # символьные n-граммы по заголовку и услуге
    log_price: np.ndarray
    title_len: np.ndarray
    desc_len: np.ndarray
    phone_hidden: np.ndarray
    msg_forbidden: np.ndarray
    loc_count: np.ndarray       # число объявлений корпуса в каждой локации (по кодам)
    emb: np.ndarray = None      # вектора объявлений (v4), квантованные (encoder.quantize)


def _build_ext(items: pd.DataFrame, lemmas: dict, cfg, ext_cfg, n_loc_codes: int, loc: np.ndarray,
               item_emb: np.ndarray = None) -> ExtIndex:
    def col(name):
        return np.nan_to_num(items[name].to_numpy(np.float32), nan=0.0)

    char_docs = [f"{t} {service_text(p)}" for t, p in zip(items["item_title_raw"], items["item_infm_params_text"])]
    return ExtIndex(
        service=BM25Field(cfg.bm25_k1, cfg.bm25_b).fit(lemmas["service"]),
        place=CoverageField().fit(lemmas["place"]),
        char=CharIndex(ext_cfg.char_ngram, ext_cfg.char_min_df).fit(char_docs),
        log_price=np.log1p(np.maximum(col("item_price"), 0)).astype(np.float32),
        title_len=items["item_title_raw"].str.len().to_numpy(np.float32),
        desc_len=np.log1p(items["item_description_raw"].str.len().to_numpy(np.float32)),
        phone_hidden=col("item_is_phone_hidden"),
        msg_forbidden=col("item_is_message_forbidden"),
        loc_count=np.bincount(loc, minlength=n_loc_codes).astype(np.float32),
        emb=item_emb,
    )


@dataclass
class Corpus:
    item_ids: np.ndarray
    rank: np.ndarray          # ранг item_id в лексикографическом порядке (тай-брейк)
    loc: np.ndarray
    micro: np.ndarray
    lat32: np.ndarray         # координаты (float32 — для расчёта расстояний батчами)
    lon32: np.ndarray
    rating_raw: np.ndarray    # с NaN — для проверки фильтра по рейтингу
    rating: np.ndarray
    log_reviews: np.ndarray
    log_pop: np.ndarray
    params_norm: list         # параметры с сохранённым регистром — для filt_exact
    fields: dict              # BM25 по полям
    cov: CoverageField        # покрытие лемм запроса заголовком и параметрами
    ext: ExtIndex = None      # расширение v3 (None — пул как в v2)

    @property
    def n(self) -> int:
        return len(self.item_ids)


def build_corpus(items: pd.DataFrame, lemmas: dict, cfg, vocabs: Vocabs, ext_cfg=None,
                 item_emb: np.ndarray = None) -> Corpus:
    """lemmas — лемматизированные поля в порядке строк items: title, params, desc
    (+ service, place, если ext_cfg задан — тогда строится расширение v3)."""
    ids = items["item_id"].to_numpy(dtype=object)
    assert len(pd.unique(ids)) == len(ids), "item_id в корпусе должны быть уникальны"
    rank = np.empty(len(ids), dtype=np.int64)
    rank[np.argsort(ids.astype(str), kind="stable")] = np.arange(len(ids))

    fields = {name: BM25Field(cfg.bm25_k1, cfg.bm25_b).fit(lemmas[name]) for name in V2_FIELDS}
    cov = CoverageField().fit([f"{t} {p}" for t, p in zip(lemmas["title"], lemmas["params"])])
    rating_raw = items["item_rating"].to_numpy(np.float32)
    reviews = np.nan_to_num(items["item_rating_reviews_count"].to_numpy(np.float32), nan=0.0)
    loc = vocabs.loc.encode(items["item_location_id"].tolist())
    ext = (None if ext_cfg is None else
           _build_ext(items, lemmas, cfg, ext_cfg, vocabs.loc.size_with_unknown, loc, item_emb))
    return Corpus(
        item_ids=ids, rank=rank,
        loc=loc,
        micro=vocabs.micro.encode(items["item_microcat_id"].tolist()),
        lat32=items["item_latitude"].to_numpy(np.float32),
        lon32=items["item_longitude"].to_numpy(np.float32),
        rating_raw=rating_raw,
        rating=np.nan_to_num(rating_raw, nan=0.0),
        log_reviews=np.log1p(np.maximum(reviews, 0)).astype(np.float32),
        log_pop=np.zeros(len(ids), np.float32),
        params_norm=[normalize_params(t) for t in items["item_infm_params_text"]],
        fields=fields, cov=cov, ext=ext,
    )


@dataclass
class QuerySet:
    lemmas: list          # леммы текста запроса через пробел
    lemma_key: list       # отсортированные уникальные леммы («мешок лемм»)
    filt_lemmas: list     # леммы фильтров
    filter_pairs: list    # разобранные пары фильтров для точного сопоставления
    loc: np.ndarray
    loc_unknown: int      # код «неизвестной локации» (с ним совпадение не засчитывается)
    rating_thr: np.ndarray
    prior: np.ndarray     # запросы × микрокатегории
    raw_text: list = None           # исходные тексты (для символьного сходства, v3)
    text_count: np.ndarray = None   # сколько раз «мешок лемм» встречался в статистиках (v3)
    emb: np.ndarray = None          # вектора запросов (v4), квантованные (encoder.quantize)

    @property
    def n(self) -> int:
        return len(self.lemmas)


def build_queries(q: pd.DataFrame, lem, vocabs: Vocabs, prior_model, ext: bool = False,
                  query_emb: np.ndarray = None) -> QuerySet:
    lemma_key = lem.key_many(q["search_query"])
    return QuerySet(
        raw_text=q["search_query"].tolist() if ext else None,
        emb=query_emb,
        text_count=prior_model.text_counts(lemma_key) if ext else None,
        lemmas=lem.join_many(q["search_query"]),
        lemma_key=lemma_key,
        filt_lemmas=lem.join_many(q["search_infm_params_text"]),
        filter_pairs=[parse_filter_pairs(t) for t in q["search_infm_params_text"]],
        loc=vocabs.loc.encode(q["search_location_id"].tolist()),
        loc_unknown=vocabs.loc.unknown,
        rating_thr=q["rating_thr"].to_numpy(np.float32),
        prior=prior_model.transform(lemma_key, q["search_category"].tolist()),
    )


# ─────────────────────────── отбор топ-K ───────────────────────────

def topk_rows(score: np.ndarray, k: int, tie: np.ndarray, decimals: int) -> np.ndarray:
    """
    Индексы топ-k по строкам без случайных ничьих.
    Ключ = округлённый скор (старшие биты) + тай-брейк по item_id (младшие биты):
    все ключи различны, поэтому результат однозначен на любой машине.
    """
    k = min(k, score.shape[1])
    shift = int(len(tie)).bit_length()
    key = score.astype(np.float64)
    key *= 10 ** decimals
    np.rint(key, out=key)
    key = key.astype(np.int64)
    key <<= shift
    key += tie[None, :]
    np.negative(key, out=key)                     # argpartition ищет минимумы
    idx = np.argpartition(key, k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(key, idx, axis=1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


# ─────────────────────────── пул ───────────────────────────

def _positions(uniq: np.ndarray, arr: np.ndarray, missing: int) -> np.ndarray:
    """Позиция каждого элемента uniq в списке arr (missing, если его там нет)."""
    out = np.full(len(uniq), missing, dtype=np.int32)
    if len(arr):
        order = np.argsort(arr, kind="stable")
        pos = np.minimum(np.searchsorted(arr[order], uniq), len(arr) - 1)
        hit = arr[order][pos] == uniq
        out[hit] = order[pos[hit]]
    return out


def _query_context(corpus: Corpus, qs: QuerySet) -> dict:
    """Признаки запроса, постоянные для всех его кандидатов (для ранкера)."""
    p = qs.prior.astype(np.float64)
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    loc_count = corpus.ext.loc_count[qs.loc]
    return {
        "q_len": np.array([len(k.split()) for k in qs.lemma_key], dtype=np.float32),
        "q_n_pairs": np.array([len(x) for x in qs.filter_pairs], dtype=np.float32),
        "q_region": (loc_count == 0).astype(np.float32),
        "q_text_cnt": np.log1p(qs.text_count).astype(np.float32),
        "q_prior_max": p.max(axis=1).astype(np.float32),
        "q_prior_ent": entropy.astype(np.float32),
        "q_n_same_loc": np.log1p(loc_count).astype(np.float32),
    }


def _memo_counts(uniq: np.ndarray, m_idx: np.ndarray, m_cnt: np.ndarray) -> np.ndarray:
    """Счётчики «памяти» для отсортированного массива кандидатов uniq."""
    out = np.zeros(len(uniq), np.float32)
    if len(m_idx):
        order = np.argsort(m_idx)
        pos = np.minimum(np.searchsorted(m_idx[order], uniq), len(m_idx) - 1)
        hit = m_idx[order][pos] == uniq
        out[hit] = m_cnt[order][pos[hit]]
    return out


def generate_pool(corpus: Corpus, qs: QuerySet, geo, cfg, item_stats=None, verbose: bool = True,
                  ext_cfg=None) -> pd.DataFrame:
    """Строки пула: q (номер запроса), item (индекс в корпусе), признаки и флаги источников.
    Расширение v3 включается, если корпус построен с ext_cfg и ext_cfg передан сюда."""
    ext = ext_cfg is not None and corpus.ext is not None
    dense = ext and corpus.ext.emb is not None and qs.emb is not None
    tie = (corpus.n - 1 - corpus.rank).astype(np.int64)   # меньший item_id → больший тай-брейк
    dec, f = cfg.score_decimals, corpus.fields
    Q = {name: f[name].query_matrix(qs.lemmas) for name in V2_FIELDS}
    Q_filt = f["params"].query_matrix(qs.filt_lemmas)
    Q_cov = corpus.cov.query_matrix(qs.lemmas)
    qlen = np.array([max(len(k.split()), 1) for k in qs.lemma_key], dtype=np.float32)
    no_memo = (np.zeros(0, np.int64), np.zeros(0, np.float32))
    list_names = ["src_text", "src_text_loc", "src_prior_loc"]
    if ext:
        E = corpus.ext
        list_names.append("src_char_loc")
        if dense:
            list_names.append("src_dense_loc")
        QE = {"service": E.service.query_matrix(qs.lemmas), "place": E.place.query_matrix(qs.lemmas),
              "char": E.char.query_matrix(qs.raw_text)}
        context = _query_context(corpus, qs)
    source_names = list_names + ["src_memo"]

    parts = []
    for start in range(0, qs.n, cfg.batch_size):
        stop = min(start + cfg.batch_size, qs.n)
        sl = slice(start, stop)
        q_loc = qs.loc[sl]

        # --- плотные скоры: запросы батча × весь корпус ---
        title_raw = f["title"].score(Q["title"][sl])
        title_max = title_raw.max(axis=1)
        title = row_max_normalize(title_raw)
        del title_raw
        params = row_max_normalize(f["params"].score(Q["params"][sl]))
        desc = row_max_normalize(f["desc"].score(Q["desc"][sl]))
        filt = row_max_normalize(f["params"].score(Q_filt[sl]))
        cov = corpus.cov.score(Q_cov[sl]) / qlen[sl, None]
        prior = qs.prior[sl][:, corpus.micro]
        same = ((corpus.loc[None, :] == q_loc[:, None]) & (q_loc[:, None] != qs.loc_unknown)).astype(np.float32)
        loc_p, dist, known = geo.batch(q_loc, corpus)
        proximity = np.exp(dist / np.float32(-cfg.geo_near_km))
        proximity[~known] = 0.0
        np.maximum(proximity, same, out=proximity)
        np.maximum(proximity, loc_p, out=proximity)

        # --- списки кандидатов ---
        text0 = title + 0.5 * params + 0.3 * desc + cov   # фиксированная формула только для отбора
        lists = {"src_text": topk_rows(text0, cfg.pool_k_text, tie, dec)}
        lists["src_prior_loc"] = topk_rows(prior + proximity + 1e-3 * text0, cfg.pool_k_prior_loc, tie, dec)
        if ext:
            service = row_max_normalize(E.service.score(QE["service"][sl]))
            cov_place = E.place.score(QE["place"][sl]) / qlen[sl, None]
            char = E.char.score(QE["char"][sl])
            n_near = (proximity > 0.5).sum(axis=1).astype(np.float32)
            lists["src_char_loc"] = topk_rows(char + 3.0 * proximity, ext_cfg.pool_k_char, tie, dec)
            if dense:
                # точная (не зависящая от BLAS) близость; float32 — как у остальных плотных скоров
                dense_sim = exact_dot(qs.emb[sl], E.emb).astype(np.float32)
                lists["src_dense_loc"] = topk_rows(dense_sim + 3.0 * proximity, ext_cfg.pool_k_dense, tie, dec)
            micro_rank = np.empty(qs.prior[sl].shape, dtype=np.int32)
            np.put_along_axis(micro_rank, np.argsort(-qs.prior[sl], axis=1, kind="stable"),
                              np.arange(qs.prior.shape[1], dtype=np.int32)[None, :], axis=1)
        proximity *= 3.0
        proximity += text0
        lists["src_text_loc"] = topk_rows(proximity, cfg.pool_k_text_loc, tie, dec)
        del proximity

        rows, cols, memo = [], [], []
        flags = {k: [] for k in source_names}
        ranks = {k: [] for k in list_names}
        for r in range(stop - start):
            m_idx, m_cnt = item_stats.memo_for(qs.lemma_key[start + r]) if item_stats else no_memo
            m_idx, m_cnt = m_idx[: cfg.pool_k_memo], m_cnt[: cfg.pool_k_memo]
            per_source = [lists[k][r] for k in list_names] + [m_idx]
            uniq = np.unique(np.concatenate(per_source))       # отсортировано → детерминировано
            rows.append(np.full(len(uniq), r, dtype=np.int64))
            cols.append(uniq)
            memo.append(_memo_counts(uniq, m_idx, m_cnt))
            for k, arr in zip(source_names, per_source):
                flags[k].append(np.isin(uniq, arr))
            if ext:
                for k in list_names:
                    ranks[k].append(_positions(uniq, lists[k][r], missing=len(lists[k][r])))

        # --- признаки для строк пула ---
        rr, cc = np.concatenate(rows), np.concatenate(cols)
        thr = qs.rating_thr[start + rr]
        part = pd.DataFrame({
            "q": (start + rr).astype(np.int32),
            "item": cc.astype(np.int32),
            "title": title[rr, cc], "params": params[rr, cc], "desc": desc[rr, cc], "cov": cov[rr, cc],
            "filt": filt[rr, cc],
            "logp": np.log(prior[rr, cc] + 1e-6).astype(np.float32),
            "loc_same": same[rr, cc],
            "loc_p": loc_p[rr, cc],
            "loc_logp": np.log(loc_p[rr, cc] + 1e-4).astype(np.float32),
            "log_dist": np.log1p(dist[rr, cc]).astype(np.float32),
            "rating_ok": (np.isnan(thr) | (corpus.rating_raw[cc] >= thr)).astype(np.float32),
            "rating": corpus.rating[cc],
            "log_reviews": corpus.log_reviews[cc],
            "log_pop": corpus.log_pop[cc],
            "log_memo": np.log1p(np.concatenate(memo)).astype(np.float32),
        })
        for k in source_names:
            part[k] = np.concatenate(flags[k])
        if ext:
            qi = start + rr
            part = part.assign(
                service=service[rr, cc], cov_place=cov_place[rr, cc], char=char[rr, cc],
                text0=text0[rr, cc],
                prior_rank=micro_rank[rr, corpus.micro[cc]].astype(np.float32),
                dist_km=dist[rr, cc],
                **{"rank_" + k[4:]: np.concatenate(ranks[k]).astype(np.float32)
                   for k in list_names if k != "src_dense_loc"},
                **{name: values[qi] for name, values in context.items()},
                q_title_max=title_max[rr], q_n_near=n_near[rr],
                i_log_price=E.log_price[cc], i_title_len=E.title_len[cc], i_desc_len=E.desc_len[cc],
                i_phone_hidden=E.phone_hidden[cc], i_msg_forbidden=E.msg_forbidden[cc],
            )
            if dense:
                part["dense"] = dense_sim[rr, cc]
                part["rank_dense_loc"] = np.concatenate(ranks["src_dense_loc"]).astype(np.float32)
                del dense_sim
            del service, cov_place, char, micro_rank, n_near
        parts.append(part)
        # освобождаем плотные массивы до следующего батча, иначе пик памяти удваивается
        del title, params, desc, filt, cov, prior, same, loc_p, dist, lists, text0
        if verbose and (start // cfg.batch_size) % 5 == 0:
            print(f"  пул: {stop}/{qs.n} запросов")

    pool = pd.concat(parts, ignore_index=True)
    pool["filt_exact"] = pool_match_share(pool["q"].to_numpy(), pool["item"].to_numpy(),
                                          qs.filter_pairs, corpus.params_norm)
    if ext:
        sizes = np.bincount(pool["q"].to_numpy(), minlength=qs.n).astype(np.float32)
        pool["q_pool_size"] = sizes[pool["q"].to_numpy()]
    return pool
