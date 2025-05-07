import pandas as pd
import os
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters,ComprehensiveFCParameters,EfficientFCParameters
from multiprocessing import freeze_support
import lime
import lime.lime_tabular
import shap
from tsfresh.feature_selection import select_features
from sklearn.cluster import KMeans
from tsfresh.feature_selection.relevance import calculate_relevance_table
from sklearn.feature_selection import VarianceThreshold
from tsfresh.utilities.dataframe_functions import impute
from sklearn.feature_selection import SelectFromModel
from tsfresh.feature_extraction.data import Timeseries
from sklearn.feature_selection import mutual_info_regression


magnitud_data_path = '../TFG/TFG_ALBERTO_MODELADO/magnitudmed.csv'
terremotos_data_path = '../TFG/TFG_ALBERTO_MODELADO/terremotosacum.csv'

magnitud_data = pd.read_csv(magnitud_data_path)
terremotos_data = pd.read_csv(terremotos_data_path)

magnitud_data['fecha'] = pd.to_datetime(magnitud_data['fecha'])
terremotos_data['fecha'] = pd.to_datetime(terremotos_data['fecha'])

def plot_feature_importance(feature_names, feature_importance_values, top_n=10):
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_values
    }).sort_values(by='importance', ascending=False)

    feature_importance_df = feature_importance_df.head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_importance_df['importance'], y=feature_importance_df['feature'])
    
    plt.title("Importancia de Características en Decision Tree")
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.grid(True)
    
    plt.show()


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    naive_pred = y_train[:-1]
    mae_naive = np.mean(np.abs(y_train[1:] - naive_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / mae_naive

def plot_predictions_vs_true(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Valor Real', color='blue')
    plt.plot(y_pred, label='Predicción', color='orange', linestyle='dashed')
    plt.title(f'Predicciones vs Valores Reales - {model_name}')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.title(f'Distribución de los Residuos - {model_name}')
    plt.xlabel('Residuos')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

def extract_tsfresh_features(data):
    data = data.reset_index()
    for col in data.columns:
        if col not in ["fecha", "index"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    settings = MinimalFCParameters()

    #relevant_functions = ['mean', 'variance', 'standard_deviation'] 
    #settings = {key: value for key, value in settings.items() if key in relevant_functions}

    extracted_features = extract_features(
        data,
        column_id="index", 
        column_sort="fecha",
        default_fc_parameters=settings,
    )
    

    return extracted_features

def extract_similar_tsfresh_features2(extracted_features, feature_names):
    data = data.reset_index()
    for col in data.columns:
        if col not in ["fecha", "index"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    
    settings = EfficientFCParameters()

    # Funciones estadísticas básicas que podrían ser similares a las características dadas
    relevant_functions = ['mean', 'variance', 'standard_deviation']
    settings = {key: value for key, value in settings.items() if key in relevant_functions}

    # Extraemos las características
    extracted_features = extract_features(
        data,
        column_id="index",
        column_sort="fecha",
        default_fc_parameters=settings,
    )
    
    # Filtramos las características extraídas para mantener solo aquellas que son similares
    similar_features = extracted_features[
        [feat for feat in extracted_features.columns if any(feat.startswith(f) for f in feature_names)]
    ]
    
    # Extraer nuevas características similares a las seleccionadas
    extracted_features_additional = extract_features(
        data,
        column_id="index",
        column_sort="fecha",
        default_fc_parameters=settings,
    )

    # Filtrar características adicionales para mantener solo aquellas similares a las iniciales
    additional_similar_features = extracted_features_additional[
        [feat for feat in extracted_features_additional.columns if any(feat.startswith(f) for f in feature_names)]
    ]

    # Combinar las características originales con las adicionales similares
    combined_features = pd.concat([similar_features, additional_similar_features], axis=1)

    return combined_features


def optimize_hiperparams_forest(data):
    data = data.sort_values(by='fecha')
    values = data.drop(columns=['fecha'])
    W = 7  # Número de pasos en el pasado
    T = 1  # Número de pasos en el futuro

    X, y = [], []
    for i in range(len(values) - W - T + 1):
        past_window = values.iloc[i:i + W].values.flatten()
        future_targets = values.iloc[i + W:i + W + T].values.flatten()
        X.append(past_window)
        y.append(future_targets)

    X = np.array(X)
    y = np.array(y)

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 30, 100)
        max_depth = trial.suggest_int("max_depth", 2, 7)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 5)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=1
        )
        
        mse_list = []  
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_list.append(mse)

        return np.mean(mse_list)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    print(f"Mejores parámetros para Random Forest: {best_params}")
    return best_params

def optimize_forest(data, output_filename, best_params, threshold):
    data = extract_tsfresh_features(data) 
    values = data.values 
    W = 7  # Número de pasos en el pasado
    T = 1  # Número de pasos en el futuro

    X, y = [], []
    for i in range(len(values) - W - T + 1):
        past_window = values[i:i + W].flatten()
        future_targets = values[i + W:i + W + T].flatten()
        X.append(past_window)
        y.append(future_targets)

    X = np.array(X)
    y = np.array(y)

    tscv = TimeSeriesSplit(n_splits=5)

    best_model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        bootstrap=best_params["bootstrap"],
        random_state=1
    )

    maes, mses, rmses, mases, wapes = [], [], [], [], []
    all_y_test, all_y_pred = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mase = mean_absolute_scaled_error(y_test, y_pred, y_train)
        wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test))

        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
        mases.append(mase)
        wapes.append(wape)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
    

    print(f"Errores para {output_filename}:")
    print(f"Promedio MAE: {np.mean(maes):.4f}")
    print(f"Promedio MSE: {np.mean(mses):.4f}")
    print(f"Promedio RMSE: {np.mean(rmses):.4f}")
    print(f"Promedio MASE: {np.mean(mases):.4f}")
    print(f"Promedio WAPE: {np.mean(wapes):.4f}")


    # Guardamos predicciones en CSV
    output = pd.DataFrame(all_y_pred, columns=[f"salida_{i}" for i in range(all_y_pred.shape[1])])
    output.to_csv(output_filename, index=False)
    print(f"Archivo '{output_filename}' creado con éxito.")


def optimize_hiperparams_tree(data):
    data = data.sort_values(by='fecha')
    values = data.drop(columns=['fecha'])

    W = 7  # Número de pasos en el pasado
    T = 1  # Número de pasos en el futuro

    X, y = [], []
    for i in range(len(values) - W - T + 1):
        past_window = values.iloc[i:i + W].values.flatten()
        future_targets = values.iloc[i + W:i + W + T].values.flatten()
        X.append(past_window)
        y.append(future_targets)

    X = np.array(X)
    y = np.array(y)

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 2, 8)
        min_samples_split = trial.suggest_int("min_samples_split", 10, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 14)

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=1
        )
        mse_list = []  
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_list.append(mse)

        return np.mean(mse_list)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    print(f"Mejores parámetros para Decision Tree: {best_params}")
    return best_params


def optimize_tree(data,output_filename, best_params):
    data = extract_tsfresh_features(data)
    values = data.values 
    W = 7  # Número de pasos en el pasado
    T = 1  # Número de pasos en el futuro

    X, y = [], []
    for i in range(len(values) - W - T + 1):
        past_window = values[i:i + W].flatten()
        future_targets = values[i + W:i + W + T].flatten()
        X.append(past_window)
        y.append(future_targets)

    X = np.array(X)
    y = np.array(y)

    tscv = TimeSeriesSplit(n_splits=5)

    best_model = DecisionTreeRegressor(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=1
    )
    
    maes, mses, rmses, mases, wapes = [], [], [], [], []
    all_y_test, all_y_pred = [], []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Entrenamos el modelo
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Calculamos las métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mase = mean_absolute_scaled_error(y_test, y_pred, y_train)
        wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test))

        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
        mases.append(mase)
        wapes.append(wape)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)

    # Imprimimos los errores promedio
    print(f"Errores para {output_filename}:")
    print(f"Promedio MAE: {np.mean(maes):.4f}")
    print(f"Promedio MSE: {np.mean(mses):.4f}")
    print(f"Promedio RMSE: {np.mean(rmses):.4f}")
    print(f"Promedio MASE: {np.mean(mases):.4f}")
    print(f"Promedio WAPE: {np.mean(wapes):.4f}")

    output = pd.DataFrame(all_y_pred, columns=[f"salida_{i}" for i in range(all_y_pred.shape[1])])
    output.to_csv(output_filename, index=False)
    print(f"Archivo '{output_filename}' creado con éxito.")



hiperparams_file = "../TFG_ALBERTO_MODELADO/Modelos/hiperparametros.txt"


def cargar_hiperparametros():
    if not os.path.exists(hiperparams_file):
        return {}

    hiperparametros = {}
    with open(hiperparams_file, "r") as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            hiperparametros[key] = eval(value)  
    return hiperparametros

def guardar_hiperparametros(hiperparametros):
    with open(hiperparams_file, "w") as f:
        for key, value in hiperparametros.items():
            f.write(f"{key}:{value}\n")

hiperparametros = cargar_hiperparametros()


if "hiper_tree_mag" not in hiperparametros:
    hiperparametros["hiper_tree_mag"] = optimize_hiperparams_tree(magnitud_data)
if "hiper_forest_mag" not in hiperparametros:
    hiperparametros["hiper_forest_mag"] = optimize_hiperparams_forest(magnitud_data)
if "hiper_tree_num" not in hiperparametros:
    hiperparametros["hiper_tree_num"] = optimize_hiperparams_tree(terremotos_data)
if "hiper_forest_num" not in hiperparametros:
    hiperparametros["hiper_forest_num"] = optimize_hiperparams_forest(terremotos_data)

guardar_hiperparametros(hiperparametros)

if __name__ == '__main__':
    freeze_support()  # Esto es útil si planeas crear un ejecutable
    #optimize_tree(magnitud_data, 'opt_predicciones_magnitud_tree.csv', hiperparametros["hiper_tree_mag"])
    #optimize_tree(terremotos_data, 'opt_predicciones_terremotos_tree.csv', hiperparametros["hiper_tree_num"])
    optimize_forest(magnitud_data, '../TFG/TFG_ALBERTO_MODELADO/Iteración3/magnitudmed_pred.csv', hiperparametros["hiper_forest_mag"], threshold=0.01)
    optimize_forest(terremotos_data, '../TFG/TFG_ALBERTO_MODELADO/Iteración3/terremotosacum_pred.csv', hiperparametros["hiper_forest_num"], threshold=0.01)

