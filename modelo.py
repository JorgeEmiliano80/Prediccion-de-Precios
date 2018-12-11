#
#
#

#
import warnings

# Graficos
import matplotlib.pyplot as plt

# Operações matemáticas e em matriz
import numpy as np
import pandas as pd
from math import sqrt
# Algoritmos
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Métricas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Suppress warnings
warnings.filterwarnings("ignore")


#
dataframe = pd.read_csv("dataset_filtered.csv", header=0)
dataframe = dataframe.dropna()
dataframe = dataframe.drop_duplicates()
dataframe = dataframe.drop(dataframe.columns[0], axis=1)


print(dataframe.columns)

# y_dataframe = dataframe[['Price']]
# X_dataframe = dataframe.loc[:, dataframe.columns != 'Price']



################################# Gerar os gráficos de dispersão ##################################
# Comentar este bloco após gerar

# model = RandomForestRegressor()
# model = LinearRegression()
# model = KNeighborsRegressor()
#
# print('treinando...')
# mdl = model.fit(X_train, y_train)
# mdl_predict = mdl.predict(X_test)
#
# plt.scatter(mdl_predict, y_test, color="gray")
#
# plt.title('Gráfico de dispersão Random Forest')
# plt.title('Gráfico de dispersão Linear Regression')
# plt.title('Gráfico de dispersão K Nearest Neighbors')
#
# plt.xlabel('predição')
# plt.ylabel('observado')
# plt.grid(False)
# plt.show()

# plt.savefig('RandomForest.png')
# plt.savefig('LinearRegression.png')
# plt.savefig('KNearestNeighbors.png')

########################################################################################

algs = [RandomForestRegressor(),
        LinearRegression(),
        KNeighborsRegressor()]

media_rmses = []
media_maes = []

for n in range(10):
    # , random_state=i+1
    train, test = train_test_split(dataframe, test_size=0.20, random_state=n+1)

    X_train = train.loc[:, dataframe.columns != 'Price']
    y_train = train['Price']
    X_test = test.loc[:, dataframe.columns != 'Price']
    y_test = test['Price']

    for model in algs:
        alg = model
        print("Treinando com {}".format(model))

        mdl = alg.fit(X_train, y_train)
        mdl_predict = mdl.predict(X_test)

        rmse = sqrt(mean_squared_error(y_test, mdl_predict))
        mae = mean_absolute_error(y_test, mdl_predict)

        print("RMSE (Raíz do Erro Quadrático Médio): ", rmse)
        print("MAE (Erro Absoluto Médio): ", mae)
        media_rmses.append(rmse)
        media_maes.append(mae)


for count in range(0, 3):
    name = ''
    vec_rmse = []
    vec_mae = []

    if count == 0:
        name = 'Random Forest'
    elif count == 1:
        name = 'Linear Regression'
    else:
        name = 'K Nearest Neighbors'

    for i in range(count, len(media_rmses), 3):
        vec_rmse.append(media_rmses[i])
        vec_mae.append(media_maes[i])

    print("Média do RMSE do " + name + ": " + str(np.mean(vec_rmse)))
    print("Média do MAE do " + name + ": " + str(np.mean(vec_mae)))



