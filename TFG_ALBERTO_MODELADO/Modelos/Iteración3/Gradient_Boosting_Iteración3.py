import optuna
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor,XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

magnitud_data_path = '../TFG_ALBERTO/Modelos/magnitudmed.csv'
terremotos_data_path = '../TFG_ALBERTO/Modelos/terremotosacum.csv'

magnitud_data = pd.read_csv(magnitud_data_path)
terremotos_data = pd.read_csv(terremotos_data_path)

magnitud_data['fecha'] = pd.to_datetime(magnitud_data['fecha'])
terremotos_data['fecha'] = pd.to_datetime(terremotos_data['fecha'])


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


def optimize_hiperparams(data):
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
        
        n_estimators = trial.suggest_int('n_estimators', 30,100)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.05, log=True)
        max_depth = trial.suggest_int('max_depth',1, 5)
        subsample = trial.suggest_float('subsample', 0.5, 1)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 1)
        min_child_weight = trial.suggest_int('min_child_weight', 8, 10)
        gamma = trial.suggest_float('gamma', 0.17, 0.18)

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            tree_method="hist",
            random_state=42
        )

        mse_list = []  
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train, 
                eval_set=[(X_test, y_test)],
                verbose=False)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_list.append(mse)

        return np.mean(mse_list)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    print(f"Mejores parámetros para Decision Tree: {best_params}")
    return best_params

def optimize_XGBRegressor(data,output_filename, best_params):
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

    best_model = XGBRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        min_child_weight=best_params["min_child_weight"],
        gamma=best_params["gamma"],
        random_state=42,
        tree_method="hist"
    )

    maes, mses, rmses, mases = [], [], [], []
    all_y_test, all_y_pred = [], []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_model.fit(X_train, y_train, 
                eval_set=[(X_test, y_test)],
                #early_stopping_rounds=5, 
                verbose=True)
        
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mase = mean_absolute_scaled_error(y_test, y_pred, y_train)

        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
        mases.append(mase)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
    
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)

    print(f"Errores para {output_filename}:")
    print(f"Promedio MAE: {np.mean(maes):.4f}")
    print(f"Promedio MSE: {np.mean(mses):.4f}")
    print(f"Promedio RMSE: {np.mean(rmses):.4f}")
    print(f"Promedio MASE: {np.mean(mases):.4f}")

    plot_predictions_vs_true(all_y_test, all_y_pred, "XGBRegressor (Optimized)")
    plot_residuals(all_y_test, all_y_pred, "XGBRegressor (Optimized)")

    output = pd.DataFrame(all_y_pred, columns=[f"salida_{i}" for i in range(all_y_pred.shape[1])])
    output.to_csv(output_filename, index=False)
    print(f"Archivo '{output_filename}' creado con éxito.")

#optimize_XGBRegressor(magnitud_data, 'opt_predicciones_magnitud_XGBRegressor.csv')
#optimize_XGBRegressor(terremotos_data, 'opt_predicciones_terremotos_XGBRegressor.csv')

hiperparams_file = "../TFG_ALBERTO/Modelos/hiperparametros_3.txt"


def cargar_hiperparametros():
    if not os.path.exists(hiperparams_file):
        return {}

    hiperparametros = {}
    with open(hiperparams_file, "r") as f:
        for line in f:
            key, value = line.strip().split(":", 1)
            hiperparametros[key] = eval(value)  # Convierte de string a diccionario
    return hiperparametros

# Función para guardar hiperparámetros en el archivo
def guardar_hiperparametros(hiperparametros):
    with open(hiperparams_file, "w") as f:
        for key, value in hiperparametros.items():
            f.write(f"{key}:{value}\n")

# Cargar hiperparámetros previos
hiperparametros = cargar_hiperparametros()

# Si no existen, calcular y guardar
if "hiper_XGBRegressor_mag" not in hiperparametros:
    hiperparametros["hiper_XGBRegressor_mag"] = optimize_hiperparams(magnitud_data)
if "hiper_XGBRegressor_num" not in hiperparametros:
    hiperparametros["hiper_XGBRegressor_num"] = optimize_hiperparams(terremotos_data)

# Guardar los hiperparámetros calculados
guardar_hiperparametros(hiperparametros)

#optimize_XGBRegressor(magnitud_data, 'optun_predicciones_magnitud_XGBRegressor.csv', hiperparametros["hiper_XGBRegressor_mag"])
optimize_XGBRegressor(terremotos_data, 'optun_predicciones_terremotos_XGBRegressor.csv', hiperparametros["hiper_XGBRegressor_num"])
