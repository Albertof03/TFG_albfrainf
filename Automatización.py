import subprocess
import time
import psycopg2
import csv
import os
import pandas as pd
from psycopg2.extras import execute_values

# --- CONFIGURACIÓN ---
DB_CONFIG = {
    "dbname": "MYtfg",
    "user": "postgres",
    "password": "r1u2i3z4",
    "host": "localhost",
    "port": "5432"
}

RUTA_SALIDA_CSVS = "../TFG/TFG_ALBERTO_MODELADO"
RUTA_SALIDA_CSVS2 = "../TFG/TFG_ALBERTO_WEB/backend/datos"
SCRIPT_MODELO = "../TFG/TFG_ALBERTO_MODELADO/Modelos/Iteración3/forest_and_tree_Iteración3.py"
SCRIPT_NODE = "../TFG/TFG_ALBERTO_DATOS/index.js"

CSV_ESPERADOS = [
    "magnitudmed_pred.csv",
    "terremotosacum_pred.csv"
]

VISTAS_A_REFRESCAR = [
    "terremotos_con_clusters_def",
    "terremotos_con_clusters_def_text",
    "terremotos_por_cluster_dia_def",
    "vista_magnitud_media_acumulativa",
    "vista_terremotos_acumulativos"
]

# --- FUNCIONES ---

def ejecutar_script(cmd, ruta):
    try:
        resultado = subprocess.run(cmd + [ruta], capture_output=True, text=True, check=True)
        print(f"[✓] Script ejecutado ({ruta}):\n{resultado.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[X] Error ejecutando {ruta}:\n{e.stderr}")

def refrescar_vistas(vistas):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        for vista in vistas:
            print(f"[→] Refrescando vista: {vista}")
            cursor.execute(f"REFRESH MATERIALIZED VIEW {vista};")
            conn.commit()
            print(f"[✓] Vista {vista} actualizada. Esperando 30s...")
            time.sleep(30)
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[X] Error al refrescar vistas:\n{e}")

def exportar_vista_a_csv(nombre_vista, ruta_salida):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {nombre_vista}")
            columnas = [desc[0] for desc in cursor.description]
            with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
                escritor_csv = csv.writer(f)
                escritor_csv.writerow(columnas)
                escritor_csv.writerows(cursor.fetchall())
            print(f"[✓] Exportación completada: {ruta_salida}")
    except Exception as e:
        print(f"[X] Error exportando {nombre_vista}:\n{e}")
    finally:
        if conn:
            conn.close()

def esperar_csvs(ruta_directorio, archivos_esperados):
    print(f"[...] Esperando sin límite a que se generen los archivos: {archivos_esperados}")
    while True:
        archivos_actuales = os.listdir(ruta_directorio)
        if all(nombre in archivos_actuales for nombre in archivos_esperados):
            print(f"[✓] Todos los archivos generados: {archivos_esperados}")
            return True
        time.sleep(5)

def actualizar_vistas_materializadas_desde_csvs(ruta_csvs, archivos_objetivo):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        for archivo in archivos_objetivo:
            ruta_completa = os.path.join(ruta_csvs, archivo)
            if not os.path.exists(ruta_completa):
                print(f"[X] No se encontró el archivo esperado: {archivo}")
                continue

            nombre_vista = os.path.splitext(archivo)[0]
            df = pd.read_csv(ruta_completa)

            if df.empty:
                print(f"[!] El archivo {archivo} está vacío. Se omite.")
                continue

            print(f"[→] Cargando {archivo} como vista materializada '{nombre_vista}'...")
            cursor.execute(f"DROP MATERIALIZED VIEW IF EXISTS {nombre_vista}")

            columnas = ', '.join([f'"{col}"' for col in df.columns])
            values_placeholder = ', '.join(["%s"] * len(df.columns))
            all_values = df.values.tolist()
            values_sql = ', '.join(
                cursor.mogrify(f"({values_placeholder})", row).decode("utf-8") for row in all_values
            )

            create_view_sql = f"""
                CREATE MATERIALIZED VIEW {nombre_vista} ({columnas}) AS
                SELECT * FROM (VALUES {values_sql}) AS t({columnas});
            """
            cursor.execute(create_view_sql)
            conn.commit()
            print(f"[✓] Vista materializada '{nombre_vista}' actualizada.")

            try:
                os.remove(ruta_completa)
                print(f"[✓] Archivo eliminado: {ruta_completa}")
            except Exception as e:
                print(f"[!] No se pudo eliminar {ruta_completa}: {e}")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[X] Error actualizando vistas materializadas:\n{e}")

# --- FLUJO COMPLETO ---

# Ejecutar script JS y refrescar vistas
ejecutar_script(["node"], SCRIPT_NODE)
#refrescar_vistas(VISTAS_A_REFRESCAR)


# Exportar vistas de entrada al modelo
exportar_vista_a_csv("public.vista_magnitud_media_acumulativa", f"{RUTA_SALIDA_CSVS}/magnitudmed.csv")
exportar_vista_a_csv("public.vista_terremotos_acumulativos", f"{RUTA_SALIDA_CSVS}/terremotosacum.csv")

# Ejecutar modelo
ejecutar_script(["python"], SCRIPT_MODELO)

# Esperar y cargar predicciones
if esperar_csvs(RUTA_SALIDA_CSVS, CSV_ESPERADOS):
    actualizar_vistas_materializadas_desde_csvs(RUTA_SALIDA_CSVS, CSV_ESPERADOS)
else:
    print("[X] No se generaron todos los archivos esperados. Proceso detenido.")

exportar_vista_a_csv("public.terremotosacum_pred", f"{RUTA_SALIDA_CSVS2}/predicciones_terremotos.csv")
exportar_vista_a_csv("public.magnitudmed_pred", f"{RUTA_SALIDA_CSVS2}/predicciones_magnitud.csv")
exportar_vista_a_csv("public.terremotos_con_clusters_def_text", f"{RUTA_SALIDA_CSVS2}/bounding_boxs.csv")

print("[✓] FLUJO TERMINADO")
