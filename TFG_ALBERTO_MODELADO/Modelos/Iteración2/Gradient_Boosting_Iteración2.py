import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
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


def prepare_data(data, W, T):
    X, y = [], []
    for i in range(len(data) - W - T + 1):
        past_window = data.iloc[i:i + W].values.flatten()
        future_targets = data.iloc[i + W:i + W + T].values.flatten()
        X.append(past_window)
        y.append(future_targets)
    return np.array(X), np.array(y)

def process_Gradient_Boosting(data, output_filename, n_estimators=50, learning_rate=0.05):
    data = data.sort_values(by='fecha')
    values = data.drop(columns=['fecha'])
    W = 7  # Number of steps in the past
    T = 1  # Number of steps in the future

    X, y = prepare_data(values, W, T)

    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42,n_jobs=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train, 
            eval_set=[(X_test, y_test)],
            #early_stopping_rounds=5, 
            verbose=True)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mase = mean_absolute_scaled_error(y_test, y_pred, y_train)

    print(f"Errores para {output_filename}:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MASE: {mase:.4f}")

    plot_predictions_vs_true(y_test, y_pred, "XGBRegressor")
    plot_residuals(y_test, y_pred , "XGBRegressor")

    output = pd.DataFrame(y_pred, columns=[f"output_{i}" for i in range(y_pred.shape[1])])
    output.to_csv(output_filename, index=False)
    print(f"File '{output_filename}' created successfully.")

def process_Gradient_Boosting_split(data, output_filename, n_splits=5, n_estimators=100, learning_rate=0.02):
    data = data.sort_values(by='fecha')
    values = data.drop(columns=['fecha'])
    W = 7  # Ventana del pasado
    T = 1  # Horizonte de predicción

    X, y = prepare_data(values, W, T)

    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, tree_method='hist',random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    maes, mses, rmses, mases = [], [], [], []
    all_y_test, all_y_pred = [], []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train, 
            eval_set=[(X_test, y_test)],
            #early_stopping_rounds=5, 
            verbose=True)
        
        y_pred = model.predict(X_test)

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

    plot_predictions_vs_true(all_y_test, all_y_pred, "XGBRegressor")
    plot_residuals(all_y_test, all_y_pred, "XGBRegressor")

    output = pd.DataFrame(all_y_pred, columns=[f"output_{i}" for i in range(all_y_pred.shape[1])])
    output.to_csv(output_filename, index=False)
    print(f"Archivo '{output_filename}' creado correctamente.")

# Ejecutar el procesamiento
#process_Gradient_Boosting_split(magnitud_data, "predicciones_magnitud_Gradient_Boosting.csv")
process_Gradient_Boosting_split(terremotos_data, "predicciones_terremotos_Gradient_Boosting.csv")
