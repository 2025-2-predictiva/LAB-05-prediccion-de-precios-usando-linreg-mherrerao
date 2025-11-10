#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os 
import pandas as pd
import gzip
import json
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


# Cargar la data
train_data = pd.read_csv("files/input/train_data.csv.zip", index_col=False)
test_data = pd.read_csv("files/input/test_data.csv.zip", index_col=False)


# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.

train_data["Age"] = 2021 - train_data["Year"]
test_data["Age"] = 2021 - test_data["Year"]

# - Elimine las columnas 'Year' y 'Car_Name'.

train_data.drop(columns=['Year', 'Car_Name'], inplace=True)
test_data.drop(columns=['Year', 'Car_Name'], inplace=True)


# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

x_train = train_data.drop(columns="Present_Price")
y_train = train_data["Present_Price"]

x_test = test_data.drop(columns="Present_Price")
y_test = test_data["Present_Price"]


# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.

def make_pipeline():

    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numerical_features = ['Selling_Price', 'Driven_kms', 'Owner', 'Age']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('numerica',MinMaxScaler(), numerical_features),
        ],
    )

    #pipeline
    pipeline=Pipeline(
        [
            ("preprocessor",preprocessor),
            ('feature_selection',SelectKBest(f_regression)),
            ('classifier', LinearRegression())
        ]
    )

    return pipeline


# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.

def make_grid_search(pipeline, x_train, y_train):

    param_grid = {
        'feature_selection__k':[11],
        # 'classifier__fit_intercept':[True,False], # Controla si el modelo calcula el intercepto (β₀) o no.
        # 'classifier__positive':[True,False] # Si lo pone en True, el modelo solo puede tener coeficientes positivos (es decir, todos los β deben ser mayores o iguales a 0).
    }

    model=GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        )

    model.fit(x_train, y_train)

    return model


# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

def save_estimator(estimator):
    models_path = "files/models"
    os.makedirs(models_path, exist_ok=True)
    model_file = os.path.join(models_path, "model.pkl.gz")

    with gzip.open(model_file, "wb") as file:
        pickle.dump(estimator, file)



# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}

def calculate_metrics(model, x_train, y_train, x_test, y_test):
    os.makedirs("files/output", exist_ok=True)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Métricas
    metrics = [
        {
        'type': 'metrics',
        'dataset': 'train',
        'r2': r2_score (y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'mad': median_absolute_error(y_train, y_train_pred),
        },
        {
        'type': 'metrics',
        'dataset': 'test',
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mad': median_absolute_error(y_test, y_test_pred),
        
        }
    ]

    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            json.dump(metric, f)
            f.write("\n")


def main():
        pipeline = make_pipeline()
        model = make_grid_search(pipeline, x_train, y_train)
        save_estimator(model)
        calculate_metrics(model, x_train, y_train, x_test, y_test)

    
if __name__ == "__main__":
    main()