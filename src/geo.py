"""
Локационные признаки.

EDA показал, что локация - самый сильный сигнал, но не жёсткий фильтр:
  * в 17% пар train локация объявления ≠ локации поиска;
  * в 64% таких пар локация поиска никогда не бывает локацией объявления -
    это «региональные» id (например, 107620 почти всегда ведёт в 637640);
  * остальные случаи - в основном соседние населённые пункты (медиана 16 км).

Поэтому вместо бинарного совпадения используются:
  1. P(локация объявления | локация поиска) по train со сглаживанием к совпадению;
  2. расстояние от центра локации поиска до координат объявления.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

KM_PER_DEGREE = 111.195


def planar_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Расстояние в км в приближении плоской Земли (точно на десятках-сотнях км).
    Только умножения, сложения и sqrt (точно округляемые по IEEE) плюс один cos на точку
    - результат практически не зависит от машины."""
    lat1 = np.asarray(lat1, dtype=np.float64)
    cos_lat = np.cos(np.radians(lat1))
    dy = np.asarray(lat2, dtype=np.float64) - lat1
    dx = (np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64)) * cos_lat
    return KM_PER_DEGREE * np.sqrt(dx * dx + dy * dy)


class LocationModel:
    def __init__(self, alpha: float = 1.0, max_km: float = 20000.0):
        self.alpha = alpha      # вес «совпадения» как априорного перехода
        self.max_km = max_km    # потолок расстояния (и значение для объявлений без координат)

    def fit(self, rows: pd.DataFrame, items: pd.DataFrame, loc_vocab) -> "LocationModel":
        """rows - пары train (поиск→объявление), items - корпус (для центров «новых» локаций)."""
        n_known = len(loc_vocab)
        n = n_known + 1                                   # последний код - «неизвестная локация»
        s = loc_vocab.encode(rows["search_location_id"].tolist())
        i = loc_vocab.encode(rows["item_location_id"].tolist())
        ok = (s < n_known) & (i < n_known)

        # --- матрица переходов: (счётчики + alpha на диагонали) / сумма строки ---
        counts = sp.coo_matrix((np.ones(ok.sum()), (s[ok], i[ok])), shape=(n, n)).tocsr()
        diag = sp.diags(np.r_[np.full(n_known, self.alpha), 0.0])
        counts = (counts + diag).tocsr()
        row_sum = np.asarray(counts.sum(axis=1)).ravel()
        inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
        self.P = (sp.diags(inv) @ counts).tocsr().astype(np.float32)

        # --- центры: медиана координат объявлений; train-пары приоритетнее корпуса ---
        self.lat = np.full(n, np.nan)
        self.lon = np.full(n, np.nan)
        item_codes = loc_vocab.encode(items["item_location_id"].tolist())
        for codes, lat, lon in (
            (item_codes, items["item_latitude"], items["item_longitude"]),
            (s, rows["item_latitude"], rows["item_longitude"]),
        ):
            df = pd.DataFrame({"code": codes, "lat": lat.to_numpy(np.float64),
                               "lon": lon.to_numpy(np.float64)}).dropna()
            med = df[df["code"] < n_known].groupby("code", sort=True)[["lat", "lon"]].median()
            self.lat[med.index.to_numpy()] = med["lat"].to_numpy()
            self.lon[med.index.to_numpy()] = med["lon"].to_numpy()
        return self

    # ─────────── v5: «размытые» локации поиска ───────────
    # Регион (например, «Московская область») или город, из которого часто выбирают объявления
    # в соседних городах. Для них P(перехода) размазана по многим городам и каждое значение мало,
    # а расстояние до медианного центра с масштабом 30 км не отличает свой областной город от чужого.
    # Поэтому для каждой локации поиска считаются:
    #   * P_norm - P(перехода), делённая на максимум в строке: главный город = 1, остальные - доля от него;
    #   * cover  - 1 − (доля переходов в более популярные города): 1 у главного, ~0 в хвосте;
    #              «ядро» локации - города с cover > 1 − region_cover_mass;
    #   * radius - квантиль расстояний выбранных объявлений до центра (для города ~ geo_near_km,
    #              для области - десятки и сотни км);
    #   * self_share - доля переходов в саму себя (без сглаживания); NaN, если переходов нет.

    def fit_spread(self, rows: pd.DataFrame, loc_vocab, rcfg, near_km: float) -> "LocationModel":
        """Вызывается после fit на тех же строках. rcfg - RankerConfig v5, near_km - минимальный радиус."""
        n_known = len(loc_vocab)
        n = self.P.shape[0]
        P = self.P.tocoo()
        df = pd.DataFrame({"r": P.row, "c": P.col, "p": P.data.astype(np.float64)})
        df = df.sort_values(["r", "p", "c"], ascending=[True, False, True], kind="stable")
        row_max = df.groupby("r", sort=False)["p"].transform("max").to_numpy()
        before = df.groupby("r", sort=False)["p"].cumsum().to_numpy() - df["p"].to_numpy()
        shape = (n, n)
        self.P_norm = sp.csr_matrix((df["p"].to_numpy() / row_max, (df["r"], df["c"])), shape=shape, dtype=np.float32)
        self.cover = sp.csr_matrix((np.maximum(1.0 - before, 1e-6), (df["r"], df["c"])), shape=shape,
                                   dtype=np.float32)
        self.cover_min = np.float32(1.0 - rcfg.region_cover_mass)

        s = loc_vocab.encode(rows["search_location_id"].tolist())
        i = loc_vocab.encode(rows["item_location_id"].tolist())
        ok = (s < n_known) & (i < n_known)
        total = np.bincount(s[ok], minlength=n).astype(np.float64)
        same = np.bincount(s[ok & (s == i)], minlength=n).astype(np.float64)
        self.self_share = np.divide(same, total, out=np.full(n, np.nan), where=total > 0).astype(np.float32)

        # радиус: квантиль расстояний «центр локации поиска → выбранное объявление»
        lat, lon = rows["item_latitude"].to_numpy(np.float64), rows["item_longitude"].to_numpy(np.float64)
        has = (s < n_known) & ~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(self.lat[s])
        dist = planar_km(self.lat[s[has]], self.lon[s[has]], lat[has], lon[has])
        q = pd.Series(dist).groupby(s[has], sort=True).quantile(rcfg.region_radius_quantile)
        self.radius = np.full(n, near_km, dtype=np.float32)
        self.radius[q.index.to_numpy()] = np.clip(q.to_numpy(), near_km, rcfg.region_radius_max_km)
        return self

    def batch_spread(self, q_loc: np.ndarray, corpus):
        """P_norm и cover для батча запросов × корпус, радиус и доля «в себя» для каждого запроса."""
        return (self.P_norm[q_loc].toarray()[:, corpus.loc], self.cover[q_loc].toarray()[:, corpus.loc],
                self.radius[q_loc], self.self_share[q_loc])

    def batch(self, q_loc: np.ndarray, corpus):
        """Для батча запросов: P(перехода), расстояние (км) и маска «центр локации известен».
        Расстояние считается во float32 с операциями на месте - это самые большие массивы батча."""
        loc_p = self.P[q_loc].toarray()[:, corpus.loc]
        lat_q, lon_q = self.lat[q_loc], self.lon[q_loc]
        known = ~np.isnan(lat_q)
        lat0 = np.where(known, lat_q, 0.0).astype(np.float32)[:, None]
        lon0 = np.where(known, lon_q, 0.0).astype(np.float32)[:, None]
        cos0 = np.cos(np.radians(np.where(known, lat_q, 0.0))).astype(np.float32)[:, None]

        dx = corpus.lon32[None, :] - lon0
        dx *= cos0
        np.multiply(dx, dx, out=dx)
        dy = corpus.lat32[None, :] - lat0
        np.multiply(dy, dy, out=dy)
        dx += dy
        del dy
        np.sqrt(dx, out=dx)
        dx *= np.float32(KM_PER_DEGREE)
        np.nan_to_num(dx, copy=False, nan=self.max_km)       # объявления без координат - «далеко»
        np.minimum(dx, np.float32(self.max_km), out=dx)
        dx[~known] = 0.0          # константа внутри запроса ни на что не влияет
        return loc_p, dx, known
