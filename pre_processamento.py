#
import warnings
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")


# Passo 1: Ler o arquivo .csv
dataframe = pd.read_csv('true_car_listings.csv')


# Passo 2: Retirar a variável Vin;
#          Retirar as variáveis City e State;
#          Passar as colunas Model e Make para maiúsculo caso tenha algum ruído (typo) nos dados;
#          Remover as duplicatas;
dataframe_filtered = dataframe.drop(['Vin'], axis=1)
dataframe_filtered = dataframe_filtered.drop(columns=['City', 'State'], axis=1)
dataframe_filtered['Model'] = dataframe_filtered['Model'].str.upper()
dataframe_filtered['Make'] = dataframe_filtered['Make'].str.upper()
dataframe_filtered = dataframe_filtered.drop_duplicates()


# Passo 3: Achar a mediana e a média para:
#           - Make Model
#           - Make Model Year
#           - Make Model Mileage

# - Make Model
mediana_preco_modelo = dataframe_filtered.groupby(["Make", "Model"])["Price"]\
                                        .median().reset_index(name='mediana_preco_modelo')

media_preco_modelo = dataframe_filtered.groupby(["Make", "Model"])["Price"]\
                                        .mean().reset_index(name='media_preco_modelo')

# - Make Model Year
mediana_preco_ano = dataframe_filtered.groupby(["Make", "Model", "Year"])["Price"]\
                                        .median().reset_index(name='mediana_preco_ano')

media_preco_ano = dataframe_filtered.groupby(["Make", "Model", "Year"])["Price"]\
                                        .mean().reset_index(name='media_preco_ano')

# - Make Model Mileage
mediana_preco_mileage = dataframe_filtered.groupby(["Make", "Model", "Mileage"])["Price"]\
                                        .median().reset_index(name='mediana_preco_mileage')

media_preco_mileage = dataframe_filtered.groupby(["Make", "Model", "Mileage"])["Price"]\
                                        .mean().reset_index(name='media_preco_mileage')


# Passo 4: Merge do Passo 3 com o nosso dataframe_filtered
dataframe_filtered = pd.merge(dataframe_filtered, mediana_preco_modelo, left_on=['Make', 'Model'],
                              right_on=['Make', 'Model'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered, media_preco_modelo, left_on=['Make', 'Model'],
                              right_on=['Make', 'Model'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered, mediana_preco_ano, left_on=['Make', 'Model', 'Year'],
                              right_on=['Make', 'Model', 'Year'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered, media_preco_ano, left_on=['Make', 'Model', 'Year'],
                              right_on=['Make', 'Model', 'Year'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered, mediana_preco_mileage, left_on=['Make', 'Model', 'Mileage'],
                              right_on=['Make', 'Model', 'Mileage'], how='left')

dataframe_filtered = pd.merge(dataframe_filtered, media_preco_mileage, left_on=['Make', 'Model', 'Mileage'],
                              right_on=['Make', 'Model', 'Mileage'], how='left')


# Passo 5: Remover Make e Model
dataframe_filtered = dataframe_filtered.drop(columns=['Make', 'Model'])


# Passo 6: Passar o dataframe para .csv para ser usado no modelo
print(dataframe_filtered.shape[0])
print(dataframe_filtered.columns)
dataframe_filtered.to_csv("dataset_filtered.csv")





