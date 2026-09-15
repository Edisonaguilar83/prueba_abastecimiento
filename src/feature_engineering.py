import numpy as np
import pandas as pd

GROUPS = ["id_tienda", "id_producto"]


def add_features(df):
    out = df.sort_values(GROUPS + ["fecha"]).copy()
    g = out.groupby(GROUPS, sort=False)["unidades_vendidas"]
    for lag in [1, 7, 14, 21, 28]:
        out[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    out["rolling_mean_7"] = shifted.groupby([out[c] for c in GROUPS], sort=False).transform(lambda s: s.rolling(7, min_periods=7).mean())
    out["rolling_mean_14"] = shifted.groupby([out[c] for c in GROUPS], sort=False).transform(lambda s: s.rolling(14, min_periods=14).mean())
    out["rolling_mean_28"] = shifted.groupby([out[c] for c in GROUPS], sort=False).transform(lambda s: s.rolling(28, min_periods=28).mean())
    out["rolling_std_7"] = shifted.groupby([out[c] for c in GROUPS], sort=False).transform(lambda s: s.rolling(7, min_periods=7).std())
    out["rolling_std_14"] = shifted.groupby([out[c] for c in GROUPS], sort=False).transform(lambda s: s.rolling(14, min_periods=14).std())
    out["mean_last_7"] = out["rolling_mean_7"]
    out["mean_prev_7"] = out.groupby(GROUPS, sort=False)["mean_last_7"].shift(7)
    out["trend_7"] = out["mean_last_7"] - out["mean_prev_7"]
    out["day_of_week"] = out["fecha"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([4, 5, 6]).astype(int)
    out["week"] = out["fecha"].dt.isocalendar().week.astype(int)
    return out


def add_weekly_target(df):
    out = df.sort_values(GROUPS + ["fecha"]).copy()
    def target_series(s):
        rev = s.shift(-1).iloc[::-1]
        return rev.rolling(7, min_periods=7).sum().iloc[::-1]
    out["target_week"] = out.groupby(GROUPS, sort=False)["unidades_vendidas"].transform(target_series)
    return out


def build_weekly_dataset(enriched):
    out = add_weekly_target(add_features(enriched))
    return out.dropna(subset=["target_week", "lag_28", "rolling_mean_28"]).reset_index(drop=True)
