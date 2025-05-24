import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from multiprocessing import freeze_support
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


def plot_mean_predictions(y_true, y_pred):
    avg_true = np.mean(y_true, axis=1)
    avg_pred = np.mean(y_pred, axis=1)

    plt.figure(figsize=(14, 5))
    plt.plot(avg_true, label='Promedio Real', color='blue', linewidth=1)
    plt.plot(avg_pred, label='Promedio Predicho', color='orange', linestyle='--', linewidth=1)

    plt.title('Evolución del valor promedio de todas las variables objetivo')
    plt.xlabel('Índice temporal')
    plt.ylabel('Valor promedio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    naive_pred = y_train[:-1]
    mae_naive = np.mean(np.abs(y_train[1:] - naive_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / mae_naive

def extract_tsfresh_features(data):
    target_columns = [col for col in data.columns]
    data["fecha"] = pd.to_datetime(data["fecha"])
    data = data.sort_values(by="fecha")
    data["id"] = 0
    df_rolled = roll_time_series(
        data, column_id="id", column_sort="fecha",
         max_timeshift=7, min_timeshift=7, rolling_direction=1
    )
    settings = EfficientFCParameters()
    relevant_functions = ['mean','median','linear_trend','standard_deviation','minimum', 'variance', 'maximum', 'sum'] 
    settings = {key: value for key, value in settings.items() if key in relevant_functions}
    num_workers = multiprocessing.cpu_count()
    distributor = MultiprocessingDistributor(n_workers=num_workers, disable_progressbar=False)
    df_features = extract_features(df_rolled, column_id="id", column_sort="fecha",default_fc_parameters=settings, n_jobs=-1, distributor=distributor)
    df_targets = data[target_columns].shift(-7)
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
    X = pd.DataFrame(X)
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

    plot_mean_predictions(all_y_test, all_y_pred)

    last_data = data.copy()
    for _ in range(7):
        X_input, _ = extract_tsfresh_features(last_data)
        next_prediction = best_model.predict(X_input.iloc[[-1]])[0]
        new_row = last_data.iloc[-1].copy()
        new_row.update({col: next_prediction[idx-1] for idx, col in enumerate(new_row.index) if col != "fecha"})
        new_row["fecha"] = last_data["fecha"].max() + pd.Timedelta(days=1)
        all_fechas_test.append(new_row["fecha"])
        last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)
        all_y_pred.append(new_row.drop("fecha").to_numpy())
    
    print(f"Errores para {output_filename}:")
    print(f"Promedio MAE: {np.mean(maes):.4f}")
    print(f"Promedio MSE: {np.mean(mses):.4f}")
    print(f"Promedio RMSE: {np.mean(rmses):.4f}")
    print(f"Promedio MASE: {np.mean(mases):.4f}")
    print(f"Promedio WAPE: {np.mean(wapes):.4f}")

    output_pred = pd.DataFrame(all_y_pred, columns=original_colums)
    output_pred["fecha"] = all_fechas_test 
    output_pred = output_pred[["fecha"] + original_colums]
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
    
    shap_importances = np.mean(np.abs(shap_values), axis=0)
    top_n = 10
    sorted_idx = np.argsort(shap_importances)[-top_n:][::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = shap_importances[sorted_idx]
    important_shap_features = [feature_names[i] for i in np.argsort(shap_importances)[::-1]]

    plt.figure(figsize=(10, 5))
    plt.barh(sorted_features[::-1], sorted_importances[::-1], color='crimson')
    plt.xlabel("Valor SHAP (impacto promedio)", fontsize=12)
    plt.title(f"Top {top_n} características más importantes (SHAP)", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

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

if __name__ == '__main__':
    freeze_support()
    hiperparametros_magnitud=optimize_hiperparams(magnitud_data)
    optimize_XGBRegressor(magnitud_data, '../TFG/TFG_ALBERTO_MODELADO/magnitudmed_pred.csv',  hiperparametros_magnitud)
    hiperparametros_terremotos=optimize_hiperparams(terremotos_data)
    optimize_XGBRegressor(terremotos_data, '../TFG/TFG_ALBERTO_MODELADO/terremotosacum_pred.csv', hiperparametros_terremotos)
