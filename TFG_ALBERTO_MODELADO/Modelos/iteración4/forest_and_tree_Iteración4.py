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
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from tsfresh.utilities.dataframe_functions import roll_time_series
from datetime import timedelta
from sklearn.feature_selection import VarianceThreshold
from tsfresh.utilities.distribution import MultiprocessingDistributor
import multiprocessing



os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"


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
"""
def select_featuress(X, y, method='mutual_info', percentile=10, n_splits=5):
    selected_features_indices = []
    kf = TimeSeriesSplit(n_splits=n_splits)

    for train_index, _ in kf.split(X):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]

        selected_indices_fold = []

        cont=0
        for col in y_train.columns: 
            cont=cont+1
            print(cont)
            if method == 'f_regression':
                selector = SelectPercentile(score_func=f_regression, percentile=percentile)
            elif method == 'mutual_info':
                selector = SelectPercentile(score_func=mutual_info_regression, percentile=percentile)
            else:
                raise ValueError("Invalid method. Use 'f_regression' or 'mutual_info'")

            selector.fit(X_train, y_train[col])  # Seleccionamos características para cada variable
            selected_indices_fold.append(selector.get_support(indices=True))

        # Unimos los índices seleccionados de todas las variables objetivo en esta iteración
        selected_features_indices.append(np.unique(np.concatenate(selected_indices_fold)))

        print(selected_features_indices)

    # Convertimos la lista de índices seleccionados a una matriz y calculamos la frecuencia de cada índice
    # Convertir la lista de arrays en una lista plana y asegurarse de que sean enteros
    all_selected_indices = np.concatenate([np.array(lst, dtype=int) for lst in selected_features_indices])
    print(all_selected_indices)
    selected_indices_counts = np.bincount(all_selected_indices)
    print(selected_indices_counts)


    # Seleccionamos características que fueron seleccionadas al menos en la mitad de los pliegues
    selected_indices = np.where(selected_indices_counts >= (n_splits // 2))[0]
    print(selected_indices)

    return selected_indices

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


def optimize_hiperparams_forest(data):
    X,y = extract_tsfresh_features(data) 
    X = pd.DataFrame(X)
    X=X.reset_index(drop=True)
    y=y.drop(columns=['fecha'])

    X, y = X.to_numpy(), y.to_numpy()

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
    X,y = extract_tsfresh_features(data) 
    X = pd.DataFrame(X)  # Restaurar DataFrame si se convirtió en array
    X=X.reset_index(drop=True)
    fechas_prediccion = y["fecha"].reset_index(drop=True)
    y=y.drop(columns=['fecha'])
    original_colums=y.columns.tolist()
    feature_names = X.columns.tolist()

    X, y = X.to_numpy(), y.to_numpy()
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
    all_y_test, all_y_pred, all_fechas_test = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        fechas_test = fechas_prediccion.iloc[test_idx].reset_index(drop=True)

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
        all_fechas_test.extend(fechas_test)
    
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

def optimize_hiperparams_tree(data):
    X,y = extract_tsfresh_features(data) 
    X = pd.DataFrame(X)
    X=X.reset_index(drop=True)
    y=y.drop(columns=['fecha'])
    X, y = X.to_numpy(), y.to_numpy()
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
    X,y = extract_tsfresh_features(data) 
    X = pd.DataFrame(X)  # Restaurar DataFrame si se convirtió en array
    X=X.reset_index(drop=True)
    fechas_prediccion = y["fecha"].reset_index(drop=True)
    y=y.drop(columns=['fecha'])
    original_colums=y.columns.tolist()
    feature_names = X.columns.tolist()

    X, y = X.to_numpy(), y.to_numpy()
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = DecisionTreeRegressor(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=1
    )
    
    maes, mses, rmses, mases, wapes = [], [], [], [], []
    all_y_test, all_y_pred, all_fechas_test = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        fechas_test = fechas_prediccion.iloc[test_idx].reset_index(drop=True)

        # Entrenamos el modelo
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
        all_fechas_test.extend(fechas_test)

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
    #optimize_tree(magnitud_data, '../TFG/TFG_ALBERTO_MODELADO/Iteración4/magnitudmed_pred.csv', hiperparametros["hiper_tree_mag"])
    #optimize_tree(terremotos_data, '../TFG/TFG_ALBERTO_MODELADO/Iteración4/terremotosacum_pred.csv', hiperparametros["hiper_tree_num"])
    #optimize_forest(magnitud_data, '../TFG/TFG_ALBERTO_MODELADO/Iteración4/magnitudmed_pred.csv', hiperparametros["hiper_forest_mag"], threshold=0.01)
    #optimize_forest(terremotos_data, '../TFG/TFG_ALBERTO_MODELADO/Iteración4/terremotosacum_pred.csv', hiperparametros["hiper_forest_num"], threshold=0.01)

