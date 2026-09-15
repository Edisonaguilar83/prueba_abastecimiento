"""Pipeline end-to-end reproducible: forecasting, incertidumbre y optimización."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.forecasting import backtest_forecasting, fit_final_forecasts
from src.uncertainty import build_uncertainty_profile, add_uncertainty
from src.optimization import economic_order, expected_cost_normal


def run_end_to_end(root):
    root = Path(root)
    daily_path = root / "data" / "processed" / "daily" / "dataset_daily_features.csv"
    raw = root / "data" / "raw"
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    daily = pd.read_csv(daily_path, parse_dates=["fecha"])
    inv = pd.read_csv(raw / "inventario_actual.csv")
    cat = pd.read_csv(raw / "catalogo_productos.csv")
    origin = daily["fecha"].max()

    # Rolling-origin folds: seis semanas completas, terminando antes del origen final.
    origins = pd.date_range(origin - pd.Timedelta(days=42), origin - pd.Timedelta(days=7), freq="7D")
    oof, metrics = backtest_forecasting(daily, origins)
    metrics.to_csv(outputs / "forecast_metrics.csv", index=False)
    oof.to_csv(outputs / "forecast_backtest_oof.csv", index=False)

    final = fit_final_forecasts(daily, origin)
    final.to_csv(outputs / "forecast_next_week.csv", index=False)

    profile = build_uncertainty_profile(oof, by="id_producto", prediction_col="pred_champion")
    profile.to_csv(outputs / "uncertainty_profile.csv", index=False)
    unc = add_uncertainty(final, profile, key="id_producto")

    rec = unc.merge(inv, on=["id_tienda", "id_producto"], how="left", validate="one_to_one")
    rec = rec.merge(cat, on="id_producto", how="left", validate="many_to_one")
    rows = []
    for r in rec.itertuples(index=False):
        margin = r.precio_venta - r.costo_unitario
        econ = economic_order(r.forecast_ajustado, r.sigma_error, r.stock_actual, margin, r.costo_unitario, r.costo_almacenamiento_semanal)
        qf = max(r.forecast_ajustado, 0.0)
        q80 = max(r.forecast_ajustado + 0.8416212336 * max(r.sigma_error, 0), 0.0)
        cr = econ["critical_ratio"]
        risk = "Conservadora" if cr < 0.60 else ("Balanceada" if cr <= 0.65 else "Agresiva")
        base_order = max(0, int(np.ceil(qf - r.stock_actual)))
        rows.append({
            "fecha": r.fecha, "id_tienda": r.id_tienda, "id_producto": r.id_producto, "nombre": r.nombre,
            "stock_actual": r.stock_actual, "forecast_next_week": r.forecast_next_week, "forecast_ajustado": r.forecast_ajustado,
            "bias_error": r.bias, "sigma_error": r.sigma_error, "pi80_lower": r.pi80_lower, "pi80_upper": r.pi80_upper,
            "critical_ratio": cr, "politica_riesgo": risk, "costo_stockout": margin,
            "costo_overstock": r.costo_unitario + r.costo_almacenamiento_semanal, "demanda_objetivo_economica": econ["stock_objetivo"],
            "stock_objetivo": econ["stock_objetivo"], "pedido_base": base_order, "pedido_optimo": econ["pedido_optimo"],
            "q_p80": q80,
            "costo_esperado_base": expected_cost_normal(r.forecast_ajustado, r.sigma_error, qf, margin, r.costo_unitario+r.costo_almacenamiento_semanal),
            "costo_esperado_optimo": expected_cost_normal(r.forecast_ajustado, r.sigma_error, econ["stock_objetivo"], margin, r.costo_unitario+r.costo_almacenamiento_semanal),
        })
    rec = pd.DataFrame(rows)
    rec["ahorro_costo_esperado"] = rec["costo_esperado_base"] - rec["costo_esperado_optimo"]
    rec.to_csv(outputs / "resultado_final_supply_optimization.csv", index=False)

    # Backtest de política: usa exclusivamente forecasts OOF de Strategy A y su incertidumbre OOF.
    bt = oof.merge(daily[["id_tienda", "id_producto", "fecha", "unidades_vendidas"]], left_on=["id_tienda", "id_producto", "origin"], right_on=["id_tienda", "id_producto", "fecha"], how="left")
    bt = bt.merge(cat, on="id_producto", how="left", validate="many_to_one")
    prof = profile.copy()
    bt = bt.merge(prof, on="id_producto", how="left", validate="many_to_one")
    bt["margen_unitario"] = bt["precio_venta"] - bt["costo_unitario"]
    bt["forecast_ajustado"] = np.maximum(bt.pred_champion + bt.bias, 0)
    bt["q_forecast"] = bt.forecast_ajustado
    bt["q_economic"] = [economic_order(f,s,0,m,c,st)["stock_objetivo"] for f,s,m,c,st in zip(bt.forecast_ajustado,bt.sigma,bt.margen_unitario,bt.costo_unitario,bt.costo_almacenamiento_semanal)]
    bt["q_p80"] = np.maximum(bt.forecast_ajustado + 0.8416212336 * bt.sigma.fillna(bt.sigma.median()), 0)
    bt["forecast_shortage"] = np.maximum(bt.actual_week-bt.q_forecast,0)
    bt["forecast_overstock"] = np.maximum(bt.q_forecast-bt.actual_week,0)
    bt["economic_shortage"] = np.maximum(bt.actual_week-bt.q_economic,0)
    bt["economic_overstock"] = np.maximum(bt.q_economic-bt.actual_week,0)
    bt["p80_shortage"] = np.maximum(bt.actual_week-bt.q_p80,0)
    bt["p80_overstock"] = np.maximum(bt.q_p80-bt.actual_week,0)
    cu=bt.margen_unitario; co=bt.costo_unitario+bt.costo_almacenamiento_semanal
    for pol in ["forecast","economic","p80"]:
        bt[f"{pol}_cost"] = cu * bt[f"{pol}_shortage"] + co * bt[f"{pol}_overstock"]
        bt[f"{pol}_fill_rate"] = np.where(bt.actual_week>0, np.minimum(bt.actual_week, bt[f"q_{pol}"])/bt.actual_week, 1.0)
    bt.to_csv(outputs / "backtest_politica_abastecimiento_detalle.csv", index=False)

    summary=[]
    for pol in ["forecast","economic","p80"]:
        cost=bt[f"{pol}_cost"].sum(); shortage=bt[f"{pol}_shortage"].sum(); over=bt[f"{pol}_overstock"].sum(); demand=bt.actual_week.sum()
        summary.append({"politica":pol,"costo_total":round(cost),"costo_promedio":cost/len(bt),"faltante_unidades":round(shortage),"excedente_unidades":round(over),"fill_rate":1-shortage/demand,"pedido_promedio":bt[f"q_{pol}"].mean(),"ahorro_vs_forecast":round(bt.forecast_cost.sum()-cost)})
    pd.DataFrame(summary).to_csv(outputs / "backtest_politica_abastecimiento_resumen.csv", index=False)
    byprod=[]
    for pid,g in bt.groupby(["id_producto","nombre"]):
        cf=g.forecast_cost.sum(); ce=g.economic_cost.sum()
        byprod.append({"id_producto":pid[0],"nombre":pid[1],"costo_forecast":round(cf),"costo_economic":round(ce),"faltante_economic":round(g.economic_shortage.sum()),"excedente_economic":round(g.economic_overstock.sum()),"fill_rate_economic":1-g.economic_shortage.sum()/g.actual_week.sum(),"critical_ratio":(g.margen_unitario.iloc[0]/(g.margen_unitario.iloc[0]+g.costo_unitario.iloc[0]+g.costo_almacenamiento_semanal.iloc[0])),"ahorro_vs_forecast":round(cf-ce)})
    pd.DataFrame(byprod).sort_values("ahorro_vs_forecast",ascending=False).to_csv(outputs / "backtest_politica_abastecimiento_producto.csv", index=False)

    # Recomendaciones A/B: mantener el output solicitado, pero ahora sale del pipeline.
    orders = final.merge(inv,on=["id_tienda","id_producto"],validate="one_to_one").merge(cat,on="id_producto",validate="many_to_one")
    orders["margen_unitario"] = orders["precio_venta"] - orders["costo_unitario"]
    orders["order_A"] = np.maximum(np.ceil(orders.forecast_next_week-orders.stock_actual),0).astype(int)
    orders["order_B"] = np.maximum(np.ceil(orders.pred_B_week-orders.stock_actual),0).astype(int)
    orders["order_blend"] = np.maximum(np.ceil(orders.pred_blend_week-orders.stock_actual),0).astype(int)
    orders[["fecha","id_tienda","id_producto","nombre","stock_actual","forecast_next_week","pred_B_week","pred_blend_week","order_A","order_B","order_blend","margen_unitario","costo_almacenamiento_semanal"]].rename(columns={"forecast_next_week":"pred_A_week"}).to_csv(outputs / "recomendaciones_strategy_A_B.csv",index=False)
    return metrics, rec, bt
