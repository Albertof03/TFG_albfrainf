import optuna
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters,EfficientFCParameters
from multiprocessing import freeze_support
from sklearn.feature_selection import VarianceThreshold
from tsfresh.utilities.distribution import MultiprocessingDistributor
import multiprocessing
from tsfresh.utilities.dataframe_functions import roll_time_series
from sklearn.cluster import KMeans
import lime
import lime.lime_tabular
import shap

magnitud_data_path = '../TFG/TFG_ALBERTO_MODELADO/magnitudmed.csv'
terremotos_data_path = '../TFG/TFG_ALBERTO_MODELADO/terremotosacum.csv'

magnitud_data = pd.read_csv(magnitud_data_path)
terremotos_data = pd.read_csv(terremotos_data_path)

magnitud_data['fecha'] = pd.to_datetime(magnitud_data['fecha'])
terremotos_data['fecha'] = pd.to_datetime(terremotos_data['fecha'])

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Scrollbar, Canvas, Frame
import numpy as np
import math

def scatter_scrollable(y_true, y_pred, column_names=None):
    num_columnas = y_true.shape[1]
    n_cols = 3
    n_rows = math.ceil(num_columnas / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows))  # Tamaño ajustado

    axes = axes.flatten()

    for i in range(num_columnas):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=15, color='green', edgecolors='k', linewidths=0.3)
        ax.plot([y_true[:, i].min(), y_true[:, i].max()],
                [y_true[:, i].min(), y_true[:, i].max()],
                'r--', linewidth=1)
        col_name = column_names[i] if column_names is not None else f"Columna {i}"
        ax.set_title(col_name, fontsize=8)
        ax.set_xlabel("Real", fontsize=7)
        ax.set_ylabel("Predicho", fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, linewidth=0.3)

    for j in range(num_columnas, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(pad=1.0)

    # Tkinter scrollable UI
    root = tk.Tk()
    root.title("Comparación Reales vs Predichos")

    canvas = Canvas(root)
    frame = Frame(canvas)
    scroll_y = Scrollbar(root, orient="vertical", command=canvas.yview)

    canvas.configure(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", on_configure)

    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack()

    root.mainloop()

def plot_feature_importance(model, feature_names):
    """
    Genera un gráfico de barras con la importancia de las características en el modelo XGBoost.
    """
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance_df['importance'], y=feature_importance_df['feature'])
    plt.title("Importancia de Características en XGBoost")
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
"""
def remove_constant_features(X):
    print(X)
    # Inicializar el selector de varianza con un umbral de 0 para eliminar solo características constantes
    selector = VarianceThreshold(threshold=0)
    selector.fit(X)
    
    # Obtener los índices de las características que no son constantes
    non_constant_features = selector.get_support(indices=True)
    
    # Identificar y mostrar las características constantes
    constant_features = [column for column in X.columns if column not in X.columns[non_constant_features]]
    if constant_features:
        print(f"Características constantes eliminadas: {constant_features}")
    else:
        print("No se encontraron características constantes.")
    
    # Retornar el DataFrame sin las características constantes
    data=X.iloc[:, non_constant_features]
    print(data)
    return data
"""
"""
def add_temporal_features(df):
    df = df.copy()
    df["dayofweek"] = df["fecha"].dt.dayofweek
    df["day"] = df["fecha"].dt.day
    df["month"] = df["fecha"].dt.month
    df["year"] = df["fecha"].dt.year
    df["is_weekend"] = df["dayofweek"] >= 5
    df["dayofyear"] = df["fecha"].dt.dayofyear
    return df

def add_fourier_features(df, period, order, colname="fecha"):
    t = pd.to_datetime(df[colname])
    t = (t - t.min()).dt.days.values
    for i in range(1, order + 1):
        df[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        df[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)
    return df
"""
def extract_tsfresh_features(data):
    target_columns = [col for col in data.columns]
    data["fecha"] = pd.to_datetime(data["fecha"])
    data = data.sort_values(by="fecha")
    data["id"] = 0
    df_rolled = roll_time_series(
        data, column_id="id", column_sort="fecha",
         max_timeshift=7, min_timeshift=7, rolling_direction=1
    )
    print(df_rolled)
    settings = EfficientFCParameters()
    relevant_functions = ['mean','median','linear_trend','standard_deviation','minimum', 'variance', 'maximum', 'sum'] 
    settings = {key: value for key, value in settings.items() if key in relevant_functions}
    num_workers = multiprocessing.cpu_count()  # Obtener núcleos disponibles
    distributor = MultiprocessingDistributor(n_workers=num_workers, disable_progressbar=False)
    df_features = extract_features(df_rolled, column_id="id", column_sort="fecha",default_fc_parameters=settings, n_jobs=-1, distributor=distributor)
    #df_features=remove_constant_features(df_features)
    df_targets = data[target_columns].shift(-7)  # Mover 7 días hacia arrib
    df_targets = df_targets.iloc[:len(df_features)]  
    return df_features, df_targets

def optimize_hiperparams(data):
    X,y = extract_tsfresh_features(data) 
    X = pd.DataFrame(X)
    X=X.reset_index(drop=True)
    y=y.drop(columns=['fecha'])

    X, y = X.to_numpy(), y.to_numpy()

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
    X,y = extract_tsfresh_features(data) 
    X = pd.DataFrame(X)  # Restaurar DataFrame si se convirtió en array
    X=X.reset_index(drop=True)
    fechas_prediccion = y["fecha"].reset_index(drop=True)
    y=y.drop(columns=['fecha'])
    original_colums=y.columns.tolist()
    feature_names = X.columns.tolist()

    X, y = X.to_numpy(), y.to_numpy()
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = XGBRegressor(**best_params, random_state=42, tree_method="hist")
    
    maes, mses, rmses, mases, wapes = [], [], [], [], []
    all_y_test, all_y_pred, all_fechas_test = [], [], []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        fechas_test = fechas_prediccion.iloc[test_idx].reset_index(drop=True)
        
        best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
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
        all_fechas_test.extend(fechas_test)
    
    # try:
    #     scatter_scrollable(
    #         np.array(all_y_test).reshape(-1, len(original_colums)), 
    #         np.array(all_y_pred).reshape(-1, len(original_colums)), 
    #         column_names=original_colums
    #     )
    # except Exception as e:
    #     print(f"Error al mostrar la interfaz interactiva de gráficos: {e}")
        
    last_data = data.copy()
    for _ in range(7):
        X_input, _ = extract_tsfresh_features(last_data)
        next_prediction = best_model.predict(X_input.iloc[[-1]])[0]
        new_row = last_data.iloc[-1].copy()
        new_row.update({col: next_prediction[idx-1] for idx, col in enumerate(new_row.index) if col != "fecha"})
        new_row["fecha"] = last_data["fecha"].max() + pd.Timedelta(days=1)
        print(new_row)
        all_fechas_test.append(new_row["fecha"])
        last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)
        all_y_pred.append(new_row.drop("fecha").to_numpy())
        print(all_y_pred)
    
    print(f"Errores para {output_filename}:")
    print(f"Promedio MAE: {np.mean(maes):.4f}")
    print(f"Promedio MSE: {np.mean(mses):.4f}")
    print(f"Promedio RMSE: {np.mean(rmses):.4f}")
    print(f"Promedio MASE: {np.mean(mases):.4f}")
    print(f"Promedio WAPE: {np.mean(wapes):.4f}")


    output_pred = pd.DataFrame(all_y_pred, columns=original_colums)
    output_pred["fecha"] = all_fechas_test 
    output_pred = output_pred[["fecha"] + original_colums]
    print(output_pred)
    output_pred.to_csv(output_filename, index=False)
    print(f"Archivo de predicciones guardado en: {output_filename}")

    num_samples = min(30, X_test.shape[0])  
    kmeans = KMeans(n_clusters=num_samples, random_state=1, n_init=10)
    cluster_labels = kmeans.fit_predict(X_test)

    selected_indices = []
    for cluster in range(num_samples):
        cluster_points = np.where(cluster_labels == cluster)[0]
        centroid = kmeans.cluster_centers_[cluster]
        closest_index = cluster_points[np.argmin(np.linalg.norm(X_test[cluster_points] - centroid, axis=1))]
        selected_indices.append(closest_index)
    
    X_test_sample = X_test[selected_indices, :]
    
    explainer_shap = shap.TreeExplainer(best_model, approximate=True)
    shap_values = explainer_shap.shap_values(X_test_sample)
    shap_values = shap_values[:, :, 0]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, plot_size=(10, 6), show=False)
    plt.title("Importancia de características SHAP", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    #PROBAR
    # Mostrar SHAP dependence plot para la característica más importante
    most_important_feature = important_shap_features[0]

    # Selecciona automáticamente una variable de interacción si no se especifica
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        most_important_feature, 
        shap_values, 
        X_test_sample, 
        feature_names=feature_names, 
        interaction_index=None,  # Puedes cambiarlo a un nombre concreto si deseas controlar la interacción
        show=False
    )
    plt.title(f"Dependencia de SHAP para '{most_important_feature}'", fontsize=14)
    plt.tight_layout()
    plt.show()

    shap_importances = np.mean(np.abs(shap_values), axis=0)
    top_n = 10
    sorted_idx = np.argsort(shap_importances)[-top_n:][::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = shap_importances[sorted_idx]

    plt.figure(figsize=(10, 5))
    plt.barh(sorted_features[::-1], sorted_importances[::-1], color='crimson')
    plt.xlabel("Valor SHAP (impacto promedio)", fontsize=12)
    plt.title(f"Top {top_n} características más importantes (SHAP)", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    important_shap_features = [feature_names[i] for i in np.argsort(shap_importances)[::-1]]


    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train, mode='regression', 
        feature_names=feature_names, 
        discretize_continuous=True
    )
    
    lime_importances = []
    for idx in selected_indices:
        exp = explainer_lime.explain_instance(X_test[idx], best_model.predict, num_features=10)
        lime_importances.append(dict(exp.as_list()))

    mean_lime_importance = {feature: np.mean([imp.get(feature, 0) for imp in lime_importances]) for feature in lime_importances[0]}
    print("Importancias medias de LIME:")
    print(mean_lime_importance)

    important_lime_features = [feature for feature, importance in sorted(mean_lime_importance.items(), key=lambda item: item[1], reverse=True)]
    
    final_important_features = list(set(important_shap_features + important_lime_features))
    print("Características más importantes de SHAP y LIME:")
    print(final_important_features)
    
#optimize_XGBRegressor(magnitud_data, 'opt_predicciones_magnitud_XGBRegressor.csv')
#optimize_XGBRegressor(terremotos_data, 'opt_predicciones_terremotos_XGBRegressor.csv')

hiperparams_file = "../TFG_ALBERTO_MODELADO/Modelos/hiperparametros.txt"


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

if __name__ == '__main__':
    freeze_support()  # Esto es útil si planeas crear un ejecutable
    optimize_XGBRegressor(magnitud_data, '../TFG/TFG_ALBERTO_MODELADO/Iteración4/magnitudmed_pred.csv', hiperparametros["hiper_XGBRegressor_mag"])
    optimize_XGBRegressor(terremotos_data, '../TFG/TFG_ALBERTO_MODELADO/Iteración4/terremotosacum_pred.csv', hiperparametros["hiper_XGBRegressor_num"])
