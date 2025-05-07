import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar los datos
magnitud_data_path = '../TFG_ALBERTO/Modelos/magnitudmed.csv'
magnitud_data = pd.read_csv(magnitud_data_path)

terremotos_data_path = '../TFG_ALBERTO/Modelos/terremotosacum.csv'
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

def p_forest(data, output_filename):
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

    pipeline = Pipeline(steps=[('model', RandomForestRegressor(n_estimators=50, random_state=1))])

    tscv = TimeSeriesSplit(n_splits=5)  
    
    all_y_test, all_y_pred = [], []
    maes, mses, rmses, mases = [], [], [], []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

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

    #plot_predictions_vs_true(all_y_test, all_y_pred, "RandomForestRegressor")
    #plot_residuals(all_y_test, all_y_pred, "RandomForestRegressor")
    
    output = pd.DataFrame(all_y_pred, columns=[f"salida_{i}" for i in range(all_y_pred.shape[1])])
    output.to_csv(output_filename, index=False)
    print(f"Archivo '{output_filename}' creado con éxito.")

# Función para procesar con DecisionTree
def p_tree(data, output_filename):
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

    pipeline = Pipeline(steps=[('model', DecisionTreeRegressor(random_state=1))])


    tscv = TimeSeriesSplit(n_splits=5)  

    all_y_test, all_y_pred = [], []
    maes, mses, rmses, mases = [], [], [], []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

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

    #plot_predictions_vs_true(all_y_test, all_y_pred, "DecisionTreeRegressor")
    #plot_residuals(all_y_test, all_y_pred, "DecisionTreeRegressor")

    output = pd.DataFrame(all_y_pred, columns=[f"salida_{i}" for i in range(all_y_pred.shape[1])])
    output.to_csv(output_filename, index=False)
    print(f"Archivo '{output_filename}' creado con éxito.")

#p_tree(magnitud_data, 'predicciones_magnitud_tree.csv')
#p_tree(terremotos_data, 'predicciones_terremotos_tree.csv')

p_forest(magnitud_data, '../TFG_ALBERTO/Modelos/Iteración2/predicciones_magnitud_forest.csv')
p_forest(terremotos_data, '../TFG_ALBERTO/Modelos/Iteración2/predicciones_terremotos_forest.csv')
