## Proyecto de Predicción de Precios de Carros

Este proyecto tiene como objetivo realizar la predicción de precios de carros utilizando tres algoritmos de aprendizaje automático: Regresion Lineal, K-Nearest Neighbors y Random Forest.

# Descripción del Proyecto

El proyecto se centra en analizar un conjunto de datos que contiene información sobre carros, como el año, kilometraje, mediana de precios por modelo, mediana de precios por año y mediana de precios por kilometraje. Con base en estos datos, se busca construir modelos de aprendizaje automático capaces de predecir el precio de un carro dado.

# Algoritmos Utilizados

* Regresion Lineal: Este algoritmo busca establecer una relación lineal entre las variables independientes (año, kilometraje, etc.) y la variable dependiente (precio). Utiliza la técnica de mínimos cuadrados para encontrar la mejor línea de ajuste.

* K-Nearest Neighbors (KNN): En este algoritmo, se busca predecir el precio de un carro basándose en los precios de los carros vecinos más cercanos en términos de caraterísticas. Se utiliza una medida de distancia para determinar qué carros son los más cercanos.

* Random Forest: Este algoritmo combina múltiples árboles de decisión para generar predicciones. Cada árbol se entrena con una muestra aleatoria del conjunto de datos y las predicciones finales se obtienen mediante la votación de todos los árboles.

## Pasos del Proyecto

* Carga de datos: Se cargan los datos del conjunto de datos que contiene la información de los carros.
* Preprocesamiento de datos: Se realizan operaciones de limpieza, como eliminar filas con los valores faltantes y duplicados, y seleccionar las columnas relevantes para el análisis.
* División de datos: Se dividen los datos en conjuntos de entrenamiento y de prueba, lo que nos permite evaluar el rendimiento de los modelos de manera independiente.
* Entrenamiento de modelos: Se entrenan los modelos de Regresión Lineal, KNN e Random Forest utilizando el conjunto de entrenamiento.
* Evaluación de Modelos: Se evalúa el rendimineto de cada modelo utilizando métricas como el Error Cuadrático Medio (RMSE) y el Error Absoluto Medio (MAE).
* Predicción de precios: Se utilizan los modelos entrenados para realizar predicciones de precios en el conjunto de prueba.
* Visualización de resultados: Se generan gráficos de dispersión para comparar las predicciones con los valores reales y analizar el redimiento de cada modelo.

# Resultados y Conclusiones

Se analizan los resultados obtenidos en términos de métricas de cada rendimiento y gráficos de dispersión. Se discuten las fortalezas y debilidades de cada algoritmo y se concluye cuál es el más adecuado para la tarea de predicción de precios de carros.

