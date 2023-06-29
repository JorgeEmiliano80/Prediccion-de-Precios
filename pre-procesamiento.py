#
import warnings
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Paso 1: Leer el archivo .csv
dataframe = pd.read_csv('/Users/jorgeemiliano/Desktop/predicao-precos/true_car_listings.csv')

# Paso 2: Retirar la variable Vin;
#         Retirar las variables City y State;
#         Pasar las columnas Model y Make para mayúscula caso tenga algun ruido (typo) en los datos;
#         Remover los duplicados;
dataframe_filtered = dataframe.drop(['Vin'], axis=1)
dataframe_filtered = dataframe_filtered.drop(columns=['City', 'State'], axis=1)
dataframe_filtered['Model'] = dataframe_filtered['model'].str.upper()
dataframe_filtered['Make'] = dataframe_filtered['Make'].str.upper()
dataframe_filtered = dataframe_filtered.drop_duplicates()

# Paso 3: Hallar la mediana y la media para:
#         - Make Model
#         - Make Model Year
#         - Make Model Mileage

# - Make Model
mediana_precio_modelo = dataframe_filtered.groupby(["Make", "Model"]) ["Price"]\
                                         .median().reset_index(name='mediana_precio_modelo')

media_precio_modelo = dataframe_filtered.groupby(["Make", "Model"])["Price"]\
                                        .mean().reset_index(name='media_precio_modelo')

# - Make Model Year
mediana_precio_ano = dataframe_filtered.groupby(["Make", "Model", "Year"])["Price"]\
                                        .median().reset_index(name='mediana_precio_ano')

media_precio_ano = dataframe_filtered.groupby(["Make", "Model", "Year"])["Price"]\
                                        .mean().reset_index(name='media_precio_ano')

# - Make Model Mileage
mediana_precio_mileage = dataframe_filtered.groupby(["Make", "Model", "Mileage"])["Price"]\
                                            .median().reset_index(name='mediana_precio_mileage')

media_precio_mileage = dataframe_filtered.groupby(["Make", "Model", "Mileage"])["Price"]\
                                        .mean().reset_index(name='media_precio_mileage')


# Paso 4: Merge del Paso 3 con nuestro dataframe_filtered
dataframe_filtered = pd.merge(dataframe_filtered, mediana_precio_modelo, left_on=['Make', 'Model'],
                              right_on=['Make', 'Model'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered,media_precio_modelo, left_on=['Make', 'Model'],
                              right_on=['Make', 'Model'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered, mediana_precio_ano, left_on=['Make', 'Model', 'Year'],
                              right_on=['Make', 'Model', 'Year'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered, media_precio_ano, left_on=['Make', 'Model', 'Year'],
                              right_on=['Make', 'Model', 'Year'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered, mediana_precio_mileage, left_on=['Make', 'Model', 'Mileage'],
                              right_on=['Make', 'Model', 'Mileage'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered, media_precio_mileage, left_on=['Make', 'Model', 'Mileage'],
                              right_on=['Make', 'Model', 'Mileage'], how='left')

# Paso 5: Remover Make y Model
dataframe_filtered = dataframe_filtered.drop(columns=['Make', 'Model'])

# Paso 6: Pasar el dataframe para .csv para ser usado en el modelo
print(dataframe_filtered.shape[0])
print(dataframe_filtered.columns)
dataframe_filtered.to_csv("Dataset_filtered.csv")