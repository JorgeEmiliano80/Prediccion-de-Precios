import warnings

# Gráficos
import matplotlib.pyplot as plt

# Operaciones matemáticas y en matrices
import numpy as np
import pandas as pd
from math import sqrt

# Algoritmos
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Metricas
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Surppress warnings
warnings.filterwarnings("ignore")

# Cargar el dataset y realizar limpieza de datos
dataframe = pd.read_csv("dataset_filtered.csv", header=0)
dataframe = dataframe.sample(frac=0.1, random_state=42) ###
dataframe = dataframe.dropna().drop_duplicates().drop(dataframe.columns[0], axis=1)
print(dataframe.columns)

y_dataframe = dataframe[['Price']]
X_dataframe = dataframe.loc[:, dataframe.columns != 'Price']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_dataframe, y_dataframe, test_size=0.2, random_state=42)

# Generar los gráficos de dispersion
models = [
    ('Random Forest', RandomForestRegressor()),
    ('Linear Regression', LinearRegression()),
    ('K Nearest Neighbors', KNeighborsRegressor())
]

for name, model in models:
    print('Entrenando con ', name)
    mdl = model.fit(X_train, y_train)
    mdl_predict = mdl.predict(X_test)

    plt.scatter(mdl_predict, y_test, color="gray")

    plt.title('Gráfico de dispersión' + name)
    plt.xlabel('Predicción')
    plt.ylabel('Observando')
    plt.grid(False)
    plt.savefig(name.replace(' ', '') + '.png')
    plt.show()

    rmse = sqrt(mean_squared_error(y_test, mdl_predict))
    mae = mean_absolute_error(y_test, mdl_predict)
    print("RMSE (Raíz del Erro Cuadrático Medio): ", rmse)
    print("MAE (Error Absoluto Medio): ", mae)
    print()
                

# Sintonización de Hiperparámetros (ejemplo utilizando Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_model_predict = best_model.predict(X_test)
best_model_rmse = sqrt(mean_squared_error(y_test, best_model_predict))
best_model_mae = mean_absolute_error(y_test,best_model_predict)

print("Mejor modelo (Random Forest) despues de sintonización de hiperparámetros: ")
print("RMSE (Raíz del Error Cuadrático Medio): ", best_model_rmse)
print("MAE (Error Absoluto Medio): ", best_model_mae)
print()

# Validación cruzada
cv_scores = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=5)
cv_rmse = np.sqrt(-cv_scores)
cv_mae = -cv_scores

print("Resultados de validación cruzada: ")
print("Media del RMSE: ", np.mean(cv_rmse))
print("Media del MAE: ", np.mean(cv_mae))
print()

######################################################################
algs = [RandomForestRegressor(), LinearRegression(), KNeighborsRegressor()]
media_rmses = []
media_maes = []

for n in range(10):
    train, test = train_test_split(dataframe, test_size=0.20, random_state=n+1)

    X_train = train.loc[:, dataframe.columns != 'Price']
    y_train = train['Price']
    X_test = test.loc[:, dataframe.columns != 'Price']
    y_test = test['Price']

    for model in algs:
        print("Entrenando con ", model)
        mdl = model.fit(X_train, y_train)
        mdl_predict = mdl.predict(X_test)

        rmse = sqrt(mean_absolute_error(y_test, mdl_predict))
        mae = mean_absolute_error(y_test, mdl_predict)

        print("RMSE (Raíz del Error Cuadrático Medio): ", rmse)
        print("MAE (Error Absoluto Medio): ", mae)
        print()
        media_rmses.append(rmse)
        media_maes.append(mae)

for count, model_name in enumerate(['Random Forest', 'Linear Regression', 'KNearest Neighbors']):
    vec_rmse = media_rmses[count::3]
    vec_mae = media_maes[count::3]

    print("Media del RMSE del ",  model_name + ":", + np.mean(vec_rmse))
    print("Media del MAE del ", model_name + ":", + np.mean(vec_mae))
    print()
    
