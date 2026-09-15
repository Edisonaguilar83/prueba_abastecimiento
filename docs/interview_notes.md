# Guion de entrevista — Case A

## ¿Por qué forecast semanal directo?
El negocio solicita la demanda de la siguiente semana. Un target semanal evita acumular siete errores diarios y está alineado con la decisión de compra.

## ¿Por qué validación temporal?
Una validación aleatoria mezcla pasado y futuro y puede producir leakage. Los folds respetan el orden temporal.

## ¿Por qué RMSE, MAE y WAPE?
RMSE penaliza errores grandes; MAE es interpretable en unidades; WAPE permite comunicar el error relativo al volumen vendido.

## ¿Cómo pasa ML a una decisión de negocio?
El forecast no es la decisión final. Se agrega incertidumbre y se pondera el costo de faltante frente al costo de exceso mediante un critical ratio económico.

## ¿Qué mejoraría?
Lead time, promociones, festivos, merma, capacidad, múltiplos de pedido, conformal prediction y monitoreo de drift.
