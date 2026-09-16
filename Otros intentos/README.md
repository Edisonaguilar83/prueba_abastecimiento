Intento de optimización de abastecimiento

Descripción

Este archivo corresponde a un intento inicial de optimización de abastecimiento desarrollado en Excel a partir de información histórica de ventas.

El objetivo del ejercicio fue utilizar el comportamiento histórico de las ventas para analizar la demanda por tienda, producto y periodo, y construir una primera aproximación que pudiera servir como apoyo para decisiones de abastecimiento.

Estructura del archivo

El libro contiene las siguientes hojas:

Modelado

Dimensión reportada por Excel: 25 filas × 182 columnas.

ventas_historicas

Dimensión reportada por Excel: 14,561 filas × 18 columnas.

Campos encontrados en la primera fila: fecha, id_tienda, id_producto, unidades_vendidas, Dia, Semana.

1. Histórico de ventas — ventas_historicas

Esta hoja constituye la base del análisis. Los campos identificados permiten relacionar la venta con una fecha, una tienda y un producto.

Campo

Descripción

fecha

Fecha de la observación de venta.

id_tienda

Identificador de la tienda.

id_producto

Identificador del producto.

unidades_vendidas

Cantidad de unidades vendidas; es la principal variable utilizada para representar la demanda observada.

Dia

Día de la semana derivado de la fecha.

Semana

Agrupación temporal utilizada para analizar el comportamiento semanal.

Cobertura encontrada

Registros de ventas: 14,560.

Tiendas identificadas: 20.

Productos identificados: 8.

Periodo observado: 2024-01-01 00:00:00 a 2024-03-31 00:00:00.

Unidades vendidas acumuladas: 191,861.

Promedio de unidades por registro: 13.18.

2. Modelado — Modelado

Esta hoja concentra el resultado del modelado realizado en Excel.

Se observa una organización de las ventas para analizar las unidades vendidas por tienda y producto, con identificadores de tiendas del tipo STORE_XX y productos del tipo PROD_XXX.

La finalidad aparente es transformar el histórico detallado en una estructura más sencilla para comparar el comportamiento de la demanda entre tiendas y productos.

Enfoque del intento

El ejercicio puede entenderse como una primera aproximación basada en los siguientes pasos:

Tomar el histórico de ventas como representación de la demanda observada.

Incorporar variables temporales como día y semana.

Analizar las unidades vendidas por tienda y producto.

Consolidar la información en una hoja de modelado.

Utilizar los resultados históricos como insumo para plantear decisiones de abastecimiento.

Por qué se considera un intento fallido de optimización

El archivo permite analizar y consolidar la demanda, pero no constituye todavía un modelo completo de optimización de abastecimiento.

La principal limitación es que el ejercicio se apoya fundamentalmente en las ventas históricas y no se observa una formulación explícita de una función objetivo y restricciones de inventario que permitan determinar matemáticamente cuánto abastecer.

No se identifican de forma explícita, dentro del modelo, variables como:

Inventario disponible o inventario inicial.

Stock de seguridad.

Punto de reorden.

Lead time de proveedores.

Capacidad de almacenamiento.

Costos de compra.

Costos de mantener inventario.

Costos asociados al quiebre de stock.

Cantidades mínimas o máximas de pedido.

Restricciones de presupuesto.

Nivel de servicio objetivo.

Pedidos o abastecimientos históricos.

Por lo tanto, el resultado se acerca más a un análisis exploratorio de demanda y agregación de ventas que a un optimizador de abastecimiento operativo.

Aprendizaje del intento

Este primer ejercicio permitió identificar que disponer únicamente del histórico de ventas no es suficiente para construir una solución de abastecimiento integral.

Para pasar de un análisis descriptivo a una solución de optimización sería necesario complementar el histórico con información operativa y definir claramente:

Demanda → Pronóstico → Inventario → Restricciones → Optimización → Cantidad recomendada de abastecimiento

Propuesta para una siguiente versión

Una evolución natural del proyecto podría incorporar:

Pronóstico de demanda: estimar las ventas futuras por tienda y producto.

Inventario: incluir inventario actual y movimientos de entrada/salida.

Lead time: considerar el tiempo de reposición de cada producto/proveedor.

Stock de seguridad: proteger el nivel de servicio ante variaciones de la demanda.

Políticas de inventario: definir puntos de reorden y cantidades objetivo.

Restricciones: incorporar capacidad, presupuesto, mínimos de compra y otras restricciones reales.

Optimización: formular una función objetivo que permita determinar cantidades de abastecimiento bajo las restricciones definidas.

Validación: comparar la solución propuesta contra el comportamiento histórico y medir indicadores como nivel de servicio, inventario promedio y quiebres.

Conclusión

Este archivo representa un primer intento exploratorio de optimización de abastecimiento. Su principal aporte fue estructurar el histórico de ventas y analizar la demanda por tienda, producto y periodo.

El intento no llegó a una optimización integral porque faltaban variables operativas y una formulación matemática que conectara la demanda con inventarios, costos, restricciones y cantidades óptimas de abastecimiento.

Aun así, el ejercicio sirve como punto de partida para identificar los datos necesarios y diseñar una solución de abastecimiento más completa.
