import numpy as np
import pandas as pd
from scipy.stats import norm


def residual_profile(residuals, by=None):
    if by is None:
        arr = np.asarray(residuals, dtype=float)
        return {"bias": float(np.mean(arr)), "sigma": float(np.std(arr, ddof=1))}
    g = residuals.groupby(by)["residual"] if hasattr(residuals, "groupby") else None
    if g is None:
        raise ValueError("Con by se espera un DataFrame con columna residual")
    out = g.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "bias", "std": "sigma"})
    return out


def build_uncertainty_profile(oof_df, by="id_producto", prediction_col="pred_champion"):
    x = oof_df.copy()
    x["residual"] = x["actual_week"] - x[prediction_col]
    profile = residual_profile(x, by=by)
    profile["sigma"] = profile["sigma"].fillna(profile["sigma"].median())
    profile["bias"] = profile["bias"].fillna(0.0)
    return profile


def normal_interval(mu, sigma, level=0.80):
    z = norm.ppf((1 + level) / 2)
    return np.maximum(mu - z * sigma, 0), mu + z * sigma


def add_uncertainty(forecast_df, profile, key="id_producto"):
    out = forecast_df.merge(profile[[key, "bias", "sigma"]], on=key, how="left", validate="many_to_one")
    out["bias"] = out["bias"].fillna(0.0)
    fallback = float(profile["sigma"].median()) if len(profile) else 0.0
    out["sigma_error"] = out["sigma"].fillna(fallback)
    out["forecast_ajustado"] = np.maximum(out["forecast_next_week"] + out["bias"], 0.0)
    out["pi80_lower"], out["pi80_upper"] = normal_interval(out["forecast_ajustado"], out["sigma_error"], level=0.80)
    return out
