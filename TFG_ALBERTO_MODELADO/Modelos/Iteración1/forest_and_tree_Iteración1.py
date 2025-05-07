import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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




magnitud_data_path = '../TFG/TFG_ALBERTO_MODELADO/magnitudmed.csv'
magnitud_data = pd.read_csv(magnitud_data_path)

terremotos_data_path = '../TFG/TFG_ALBERTO_MODELADO/terremotosacum.csv'
terremotos_data = pd.read_csv(terremotos_data_path)

magnitud_data['fecha'] = pd.to_datetime(magnitud_data['fecha'])
terremotos_data['fecha'] = pd.to_datetime(terremotos_data['fecha'])


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    naive_pred = y_train[:-1]
    mae_naive = np.mean(np.abs(y_train[1:] - naive_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / mae_naive

def plot_predictions_vs_true(y_true, y_pred, model_name):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    plt.plot(y_true, label='Valor Real', color='blue', linewidth=1.2, alpha=0.7)
    plt.plot(y_pred, label='Predicción', color='orange', linestyle='dashed', linewidth=1.2, alpha=0.7)
    
    plt.title(f'Predicciones vs Valores Reales - {model_name}', fontsize=14)
    plt.xlabel('Índice', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    
    plt.legend(loc='upper right', fontsize=10, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
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
    W = 7  # Número de pasos en el pasado
    T = 1  # Número de pasos en el futuro

    X, y = [], []
    print()
    for i in range(len(data) - W - T + 1):
        past_window = data.iloc[i:i + W].values
        future_targets = data.iloc[i + W:i + W + T].values
        X.append(past_window)
        y.append(future_targets)

    X = np.array(X)  # Shape (num_samples, W, num_features)
    y = np.array(y)  # Shape (num_samples, T, num_features)

    # Transformamos X en 2D (num_samples, W * num_features)
    X = X.reshape(X.shape[0], -1)

    # Si la variable y tiene más de una dimensión, la reducimos
    y = y.reshape(y.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[('model', RandomForestRegressor(n_estimators=100,random_state=1))])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)


    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mase = mean_absolute_scaled_error(y_test, y_pred, y_train)

    print(f"Errores para {output_filename}:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MASE: {mase:.4f}")

    #plot_predictions_vs_true(y_test, y_pred, "Random Forest")
    #plot_residuals(y_test, y_pred, "Random Forest")
    scatter_scrollable(y_test, y_pred)

    output = pd.DataFrame(y_pred, columns=[f"salida_{i}" for i in range(y_pred.shape[1])])
    output.to_csv(output_filename, index=False)
    print(f"Archivo '{output_filename}' creado con éxito.")

def p_tree(data, output_filename):
    W = 7  # Número de pasos en el pasado
    T = 1  # Número de pasos en el futuro

    X, y = [], []
    print()
    for i in range(len(data) - W - T + 1):
        past_window = data.iloc[i:i + W].values
        future_targets = data.iloc[i + W:i + W + T].values
        X.append(past_window)
        y.append(future_targets)

    X = np.array(X)  # Shape (num_samples, W, num_features)
    y = np.array(y)  # Shape (num_samples, T, num_features)

    # Transformamos X en 2D (num_samples, W * num_features)
    X = X.reshape(X.shape[0], -1)

    # Si la variable y tiene más de una dimensión, la reducimos
    y = y.reshape(y.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[('model', DecisionTreeRegressor(random_state=1))])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    #MAE, MSE, RMSE y MASE
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mase = mean_absolute_scaled_error(y_test, y_pred, y_train)


    
    print(f"Errores para {output_filename}:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MASE: {mase:.4f}")

    #plot_predictions_vs_true(y_test, y_pred, "Decision Tree")
    #plot_residuals(y_test, y_pred, "Decision Tree")
    #scatter_scrollable(y_test, y_pred)

    output = pd.DataFrame(y_pred, columns=[f"salida_{i}" for i in range(y_pred.shape[1])])
    output.to_csv(output_filename, index=False)
    print(f"Archivo '{output_filename}' creado con éxito.")

p_tree(magnitud_data.drop(columns=['fecha']), '../TFG/TFG_ALBERTO_MODELADO/Iteración1/magnitudmed_pred.csv')
p_tree(terremotos_data.drop(columns=['fecha']), '../TFG/TFG_ALBERTO_MODELADO/Iteración1/terremotosacum_pred.csv')
#p_forest(magnitud_data.drop(columns=['fecha']), '../TFG_ALBERTO/Modelos/Iteración1/predicciones_magnitud_forest.csv')
#p_forest(terremotos_data.drop(columns=['fecha']), 'predicciones_terremotos_forest.csv')
