# Supuestos y límites

1. La demanda de los próximos 7 días se modela directamente como suma semanal.
2. No se utiliza `ground_truth_trends.csv` como predictor; solo sirve para diagnóstico.
3. Las variables móviles usan `shift(1)` para evitar leakage.
4. La incertidumbre se aproxima inicialmente mediante residuos históricos y distribución normal.
5. El costo de quiebre se aproxima al margen unitario perdido.
6. El costo de exceso se aproxima como costo unitario + almacenamiento semanal. No hay datos de merma, salvamento ni lead time.
7. El backtest de abastecimiento es contrafactual porque solo existe un snapshot de inventario.
8. El ahorro reportado es estimado por backtesting, no ahorro histórico realizado.
