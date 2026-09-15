"""MVP de visualización — Tostao, Case A: Supply Optimization.

Muestra las salidas de `src.pipeline` (vía `main.py`): qué pedir, cuánto y
por qué (explicación determinista por SKU-tienda), el desempeño de
forecasting (rolling-origin) y el backtest económico de la política de
abastecimiento frente al enfoque puramente basado en el pronóstico.

Ejecutar desde la raíz de este proyecto:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.data_preparation import build_processed_datasets  # noqa: E402
from src.pipeline import run_end_to_end  # noqa: E402

st.set_page_config(page_title="Tostao — Optimización de Abastecimiento", layout="wide")

OUTPUTS = RAIZ / "outputs"

ARCHIVOS = {
    "metrics": OUTPUTS / "forecast_metrics.csv",
    "oof": OUTPUTS / "forecast_backtest_oof.csv",
    "next_week": OUTPUTS / "forecast_next_week.csv",
    "uncertainty": OUTPUTS / "uncertainty_profile.csv",
    "resultado": OUTPUTS / "resultado_final_supply_optimization.csv",
    "bt_detalle": OUTPUTS / "backtest_politica_abastecimiento_detalle.csv",
    "bt_resumen": OUTPUTS / "backtest_politica_abastecimiento_resumen.csv",
    "bt_producto": OUTPUTS / "backtest_politica_abastecimiento_producto.csv",
    "recomendaciones_ab": OUTPUTS / "recomendaciones_strategy_A_B.csv",
}


def _archivos_existen() -> bool:
    return all(ruta.exists() for ruta in ARCHIVOS.values())


def _ejecutar_pipeline() -> None:
    build_processed_datasets(RAIZ / "data" / "raw", RAIZ / "data" / "processed")
    run_end_to_end(RAIZ)


@st.cache_data(show_spinner=False)
def _cargar_salidas(_marca_cache: float) -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.read_csv(ARCHIVOS["metrics"]),
        "oof": pd.read_csv(ARCHIVOS["oof"], parse_dates=["origin"]),
        "next_week": pd.read_csv(ARCHIVOS["next_week"], parse_dates=["fecha"]),
        "resultado": pd.read_csv(ARCHIVOS["resultado"], parse_dates=["fecha"]),
        "bt_resumen": pd.read_csv(ARCHIVOS["bt_resumen"]),
        "bt_producto": pd.read_csv(ARCHIVOS["bt_producto"]),
    }


def _explicacion(fila: pd.Series) -> str:
    return (
        f"Para **{fila['nombre']}** en **{fila['id_tienda']}**: el pronóstico ajustado para la "
        f"próxima semana es de {fila['forecast_ajustado']:.1f} unidades (sesgo histórico "
        f"{fila['bias_error']:+.2f}, desviación estándar del error {fila['sigma_error']:.1f}; "
        f"intervalo 80% aproximado [{fila['pi80_lower']:.0f}, {fila['pi80_upper']:.0f}]). "
        f"Con un margen unitario de ${fila['costo_stockout']:,.0f} frente a un costo de exceso de "
        f"${fila['costo_overstock']:,.0f}, el ratio crítico newsvendor es {fila['critical_ratio']:.0%}, "
        f"lo que corresponde a una política **{fila['politica_riesgo']}**. "
        f"El stock objetivo es {fila['stock_objetivo']:.1f} unidades; con {fila['stock_actual']:.0f} "
        f"unidades actuales en tienda, se recomienda pedir **{int(fila['pedido_optimo'])} unidades**. "
        f"Esto implica un ahorro esperado de ${fila['ahorro_costo_esperado']:,.0f} frente a pedir "
        f"únicamente el pronóstico base ({int(fila['pedido_base'])} unidades)."
    )


st.title("Tostao — Optimización de Abastecimiento (Case A)")
st.caption(
    "Sistema que decide, para cada tienda y producto, cuántas unidades pedir la próxima semana: "
    "forecasting + incertidumbre out-of-fold + política newsvendor, traducido en un pedido y su explicación."
)

if not _archivos_existen():
    st.warning("Todavía no hay resultados generados en outputs/.")
    if st.button("Generar resultados (corre el pipeline completo)", type="primary"):
        with st.spinner("Preparando datos, corriendo el backtest de forecasting y la optimización económica..."):
            _ejecutar_pipeline()
        st.cache_data.clear()
        st.rerun()
    st.stop()

marca_cache = max(ruta.stat().st_mtime for ruta in ARCHIVOS.values())
datos = _cargar_salidas(marca_cache)
metrics, oof, next_week = datos["metrics"], datos["oof"], datos["next_week"]
resultado, bt_resumen, bt_producto = datos["resultado"], datos["bt_resumen"], datos["bt_producto"]

if st.button("Volver a generar (reentrena y recorre el backtest)"):
    with st.spinner("Preparando datos, corriendo el backtest de forecasting y la optimización económica..."):
        _ejecutar_pipeline()
    st.cache_data.clear()
    st.rerun()

# --- KPIs ---------------------------------------------------------------
champion = next_week["champion"].iloc[0] if "champion" in next_week.columns else metrics.loc[metrics["WAPE"].idxmin(), "estrategia"]
wape_champion = metrics.loc[metrics["estrategia"] == champion, "WAPE"]
wape_champion = float(wape_champion.iloc[0]) if len(wape_champion) else float(metrics["WAPE"].min())
fila_economic = bt_resumen.loc[bt_resumen["politica"] == "economic"].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("SKUs con pedido óptimo > 0", int((resultado["pedido_optimo"] > 0).sum()))
col2.metric("Unidades totales a pedir", int(resultado["pedido_optimo"].sum()))
col3.metric(f"WAPE del champion ({champion})", f"{wape_champion:.1%}")
col4.metric("Fill rate política económica", f"{fila_economic['fill_rate']:.1%}")

st.divider()

# --- Tabla filtrable de recomendaciones ---------------------------------
st.subheader("Recomendaciones de pedido")

col_izq, col_der = st.columns(2)
tiendas_sel = col_izq.multiselect("Tienda", sorted(resultado["id_tienda"].unique()))
productos_sel = col_der.multiselect("Producto", sorted(resultado["id_producto"].unique()))

tabla = resultado.copy()
if tiendas_sel:
    tabla = tabla[tabla["id_tienda"].isin(tiendas_sel)]
if productos_sel:
    tabla = tabla[tabla["id_producto"].isin(productos_sel)]
tabla = tabla.sort_values("pedido_optimo", ascending=False)

st.dataframe(
    tabla[
        [
            "id_tienda",
            "id_producto",
            "nombre",
            "forecast_ajustado",
            "critical_ratio",
            "politica_riesgo",
            "stock_actual",
            "stock_objetivo",
            "pedido_base",
            "pedido_optimo",
            "ahorro_costo_esperado",
        ]
    ].style.format(
        {
            "forecast_ajustado": "{:.1f}",
            "critical_ratio": "{:.0%}",
            "stock_actual": "{:.0f}",
            "stock_objetivo": "{:.1f}",
            "ahorro_costo_esperado": "${:,.0f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

st.caption(f"{len(tabla)} de {len(resultado)} series (tienda x producto)")

if len(tabla) > 0:
    opciones = list(tabla.index)
    fila_sel = st.selectbox(
        "Ver la explicación de una recomendación",
        opciones,
        format_func=lambda i: f"{tabla.loc[i, 'id_tienda']} / {tabla.loc[i, 'id_producto']} ({tabla.loc[i, 'nombre']})",
    )
    st.info(_explicacion(tabla.loc[fila_sel]))

st.divider()

# --- Forecasting: validación temporal -------------------------------------
st.subheader("Desempeño del forecasting (rolling-origin)")

col_a, col_b = st.columns(2)

fig_metrics = px.bar(
    metrics.melt(id_vars="estrategia", value_vars=["RMSE", "MAE", "WAPE"], var_name="métrica", value_name="valor"),
    x="estrategia",
    y="valor",
    color="métrica",
    barmode="group",
    title="Comparación de estrategias de forecasting",
)
col_a.plotly_chart(fig_metrics, width="stretch")

oof_wape = (
    oof.assign(error_abs=(oof["actual_week"] - oof["pred_champion"]).abs())
    .groupby("origin")
    .apply(lambda g: g["error_abs"].sum() / g["actual_week"].sum(), include_groups=False)
    .reset_index(name="wape")
)
fig_wape = px.line(oof_wape, x="origin", y="wape", markers=True, title=f"WAPE semanal del champion ({champion})")
fig_wape.update_yaxes(tickformat=".0%", title="WAPE")
fig_wape.update_xaxes(title="Semana (origen)")
col_b.plotly_chart(fig_wape, width="stretch")

st.divider()

# --- Backtest económico de la política -------------------------------------
st.subheader("Backtest de política: ¿vale la pena la incertidumbre + newsvendor?")
st.caption(
    "Comparación contrafactual entre pedir solo el pronóstico, aplicar la política económica (newsvendor) "
    "o cubrir el percentil 80 de la demanda, usando el forecast out-of-fold del champion."
)

col_c, col_d = st.columns(2)
fig_costo = px.bar(bt_resumen, x="politica", y="costo_total", title="Costo contrafactual total (pesos)")
col_c.plotly_chart(fig_costo, width="stretch")

fig_fill = px.bar(bt_resumen, x="politica", y="fill_rate", title="Fill rate por política")
fig_fill.update_yaxes(tickformat=".0%")
col_d.plotly_chart(fig_fill, width="stretch")

col_e, col_f = st.columns(2)
fig_faltante = px.bar(bt_resumen, x="politica", y="faltante_unidades", title="Unidades faltantes")
col_e.plotly_chart(fig_faltante, width="stretch")

fig_excedente = px.bar(bt_resumen, x="politica", y="excedente_unidades", title="Unidades excedentes")
col_f.plotly_chart(fig_excedente, width="stretch")

st.subheader("Ahorro de la política económica por producto")
st.dataframe(
    bt_producto.style.format(
        {
            "costo_forecast": "${:,.0f}",
            "costo_economic": "${:,.0f}",
            "fill_rate_economic": "{:.0%}",
            "critical_ratio": "{:.0%}",
            "ahorro_vs_forecast": "${:,.0f}",
        }
    ),
    width="stretch",
    hide_index=True,
)
