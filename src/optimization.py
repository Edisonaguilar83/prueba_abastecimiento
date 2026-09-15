import math
import numpy as np
from scipy.stats import norm


def economic_order(forecast, sigma, stock_actual, margin_unit, unit_cost, storage_week):
    underage = max(float(margin_unit), 0.0)
    overage = max(float(unit_cost) + float(storage_week), 1e-12)
    cr = underage / (underage + overage) if underage > 0 else 0.0
    z = norm.ppf(cr) if 0 < cr < 1 else 0.0
    target = max(float(forecast) + z * max(float(sigma), 0.0), 0.0)
    order = max(0, math.ceil(target - float(stock_actual)))
    return {"critical_ratio": cr, "z_economico": z, "stock_objetivo": target, "pedido_optimo": order}


def expected_cost_normal(mu, sigma, q, underage, overage):
    mu, sigma, q = float(mu), float(sigma), float(q)
    if sigma <= 0:
        shortage = max(mu - q, 0)
        overstock = max(q - mu, 0)
    else:
        a = (q - mu) / sigma
        phi = norm.pdf(a)
        Phi = norm.cdf(a)
        shortage = sigma * phi + (mu - q) * (1 - Phi)
        overstock = (q - mu) * Phi + sigma * phi
    return underage * shortage + overage * overstock


def policy_targets(forecast, sigma, margin_unit, unit_cost, storage_week):
    x = economic_order(forecast, sigma, 0, margin_unit, unit_cost, storage_week)
    q80 = max(float(forecast) + norm.ppf(0.80) * max(float(sigma), 0.0), 0.0)
    return x["critical_ratio"], x["z_economico"], x["stock_objetivo"], q80
