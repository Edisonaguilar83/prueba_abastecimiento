"""Forecasting reproducible para Caso A.

Incluye:
- Seasonal Naive como baseline.
- Strategy A: XGBoost directo al total de los siguientes 7 días.
- Strategy B: siete modelos XGBoost directos, uno por horizonte diario, agregados a semana.
- Backtesting rolling-origin y entrenamiento final para el forecast futuro.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import regression_metrics

CATEGORICAL = ["id_tienda", "id_producto", "ciudad"]
EXCLUDE = {"fecha", "unidades_vendidas", "target_week"}
DEFAULT_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8
)


def make_preprocessor(X):
    cat = [c for c in CATEGORICAL if c in X.columns]
    num = [c for c in X.columns if c not in cat and c not in EXCLUDE and pd.api.types.is_numeric_dtype(X[c])]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num),
    ])


def fit_xgb(X_train, y_train, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = XGBRegressor(
        **p, objective="reg:squarederror", random_state=123, n_jobs=4
    )
    prep = make_preprocessor(X_train)
    Xt = prep.fit_transform(X_train)
    model.fit(Xt, y_train, verbose=False)
    return prep, model


def predict(prep, model, X):
    return np.maximum(model.predict(prep.transform(X)), 0.0)


def feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE]


def weekly_actuals(df, origin):
    origin = pd.Timestamp(origin)
    future = df[(df["fecha"] > origin) & (df["fecha"] <= origin + pd.Timedelta(days=7))]
    return future.groupby(["id_tienda", "id_producto"], as_index=False)["unidades_vendidas"].sum().rename(columns={"unidades_vendidas": "actual_week"})


def seasonal_naive_forecast(df, origin):
    origin = pd.Timestamp(origin)
    hist = df[(df["fecha"] > origin - pd.Timedelta(days=7)) & (df["fecha"] <= origin)]
    out = hist.groupby(["id_tienda", "id_producto"], as_index=False)["unidades_vendidas"].sum()
    return out.rename(columns={"unidades_vendidas": "prediction"})


def strategy_a_forecast(train_df, forecast_rows):
    train = train_df.dropna(subset=["target_week"]).copy()
    cols = feature_columns(train)
    prep, model = fit_xgb(train[cols], train["target_week"])
    return predict(prep, model, forecast_rows[cols])


def _daily_target(df, horizon):
    g = df.sort_values(["id_tienda", "id_producto", "fecha"]).groupby(["id_tienda", "id_producto"], sort=False)["unidades_vendidas"]
    return g.shift(-horizon)


def strategy_b_forecast(train_df, forecast_rows, origin):
    """Direct multi-horizon: un modelo por h=1..7, todos desde el mismo origen."""
    df = train_df.sort_values(["id_tienda", "id_producto", "fecha"]).copy()
    cols = feature_columns(df)
    preds = []
    for h in range(1, 8):
        y = _daily_target(df, h)
        eligible = df.loc[y.notna() & (df["fecha"] + pd.Timedelta(days=h) <= pd.Timestamp(origin))].copy()
        y_h = y.loc[eligible.index]
        prep, model = fit_xgb(eligible[cols], y_h)
        preds.append(predict(prep, model, forecast_rows[cols]))
    return np.sum(preds, axis=0)


def backtest_forecasting(daily_df, origins):
    """Rolling-origin backtest. Cada fold solo entrena con targets conocidos al origen."""
    df = daily_df.sort_values(["id_tienda", "id_producto", "fecha"]).copy()
    feature_cols = feature_columns(df)
    rows = []
    for origin in pd.to_datetime(origins):
        actual = weekly_actuals(df, origin)
        forecast_rows = df[df["fecha"] == origin].copy()
        if len(forecast_rows) != df[["id_tienda", "id_producto"]].drop_duplicates().shape[0]:
            raise ValueError(f"Origen {origin.date()} no tiene una fila por SKU-tienda")

        train_a = df[(df["fecha"] + pd.Timedelta(days=7) <= origin)].copy()
        pred_a = strategy_a_forecast(train_a, forecast_rows)
        pred_naive = seasonal_naive_forecast(df, origin).rename(columns={"prediction": "pred_naive"})
        pred_b = strategy_b_forecast(df[df["fecha"] <= origin], forecast_rows, origin)

        base = forecast_rows[["id_tienda", "id_producto"]].copy()
        base["origin"] = origin
        base["pred_naive"] = pred_naive.set_index(["id_tienda", "id_producto"]).reindex(pd.MultiIndex.from_frame(base[["id_tienda", "id_producto"]]))["pred_naive"].to_numpy()
        base["pred_A_week"] = pred_a
        base["pred_B_week"] = pred_b
        base["pred_champion"] = pred_b
        base = base.merge(actual, on=["id_tienda", "id_producto"], how="left", validate="one_to_one")
        rows.append(base)
    bt = pd.concat(rows, ignore_index=True)
    metrics = []
    for name, col in [("Seasonal Naive", "pred_naive"), ("Strategy A — weekly direct", "pred_A_week"), ("Strategy B — 7 daily direct", "pred_B_week")]:
        m = regression_metrics(bt["actual_week"], bt[col])
        metrics.append({"estrategia": name, **m})
    return bt, pd.DataFrame(metrics)


def fit_final_forecasts(daily_df, origin):
    """Entrena con toda la información cuyo target ya era conocido al origen y predice origin+1..origin+7."""
    origin = pd.Timestamp(origin)
    df = daily_df.sort_values(["id_tienda", "id_producto", "fecha"]).copy()
    forecast_rows = df[df["fecha"] == origin].copy()
    if forecast_rows.empty:
        raise ValueError(f"No existe fila de origen {origin.date()}")
    train_a = df[(df["fecha"] + pd.Timedelta(days=7) <= origin)].copy()
    pred_a = strategy_a_forecast(train_a, forecast_rows)
    pred_b = strategy_b_forecast(df[df["fecha"] <= origin], forecast_rows, origin + pd.Timedelta(days=7))
    naive = seasonal_naive_forecast(df, origin).rename(columns={"prediction": "pred_naive"})
    key = forecast_rows[["id_tienda", "id_producto"]].copy()
    key = key.merge(naive, on=["id_tienda", "id_producto"], how="left", validate="one_to_one")
    key["pred_A_week"] = pred_a
    key["forecast_next_week"] = pred_b
    key["champion"] = "Strategy B — 7 daily direct"
    key["pred_B_week"] = pred_b
    key["pred_blend_week"] = 0.5 * pred_a + 0.5 * pred_b
    key["fecha"] = origin
    return key


def run_forecast(processed_path, output_path):
    df = pd.read_csv(processed_path, parse_dates=["fecha"])
    origin = df["fecha"].max()
    result = fit_final_forecasts(df, origin)
    catalog = df[["id_producto", "nombre"]].drop_duplicates("id_producto") if "nombre" in df.columns else None
    if catalog is not None:
        result = result.merge(catalog, on="id_producto", how="left", validate="many_to_one")
    result = result[["fecha", "id_tienda", "id_producto", "nombre", "pred_naive", "pred_A_week", "pred_B_week", "pred_blend_week", "forecast_next_week", "champion"]]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    daily = root / "data" / "processed" / "daily" / "dataset_daily_features.csv"
    output = root / "outputs" / "forecast_next_week.csv"
    if not daily.exists():
        raise FileNotFoundError("Ejecuta primero: python src/data_preparation.py")
    result = run_forecast(daily, output)
    print(f"Pronósticos generados: {len(result):,} series SKU-tienda")
    print(f"Origen: {result['fecha'].iloc[0].date()} | Horizonte: 7 días")
    print(f"Archivo: {output}")
