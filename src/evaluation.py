import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return np.abs(y_true - y_pred).sum() / denom if denom else np.nan


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "WAPE": float(wape(y_true, y_pred)),
    }


def inventory_backtest(oof, cost_underage, cost_overage):
    """Contrafactual por unidad: compara target q con demanda semanal observada.
    No reconstruye inventario histórico; por ello q se interpreta como disponibilidad
    total de la semana en cada SKU-tienda.
    """
    x = oof.copy()
    results = []
    for policy, qcol in [("forecast", "q_forecast"), ("economic", "q_economic"), ("p80", "q_p80")]:
        q = x[qcol].to_numpy(float)
        d = x["actual_week"].to_numpy(float)
        shortage = np.maximum(d - q, 0)
        overstock = np.maximum(q - d, 0)
        cost = cost_underage * shortage + cost_overage * overstock
        results.append(pd.DataFrame({
            "politica": policy,
            "costo": cost,
            "faltante": shortage,
            "excedente": overstock,
            "fill_units": np.minimum(d, q),
            "demanda": d,
        }))
    return pd.concat(results, ignore_index=True)
