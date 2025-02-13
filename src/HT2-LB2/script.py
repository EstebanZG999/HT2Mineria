import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# Definir la ruta al archivo
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")  # Carpeta donde se guarda el archivo
data_path = os.path.join(data_dir, "movies.csv")  # Archivo de entrada
clean_data_path = os.path.join(data_dir, "movies_clean.csv")  # Archivo de salida

data_dir = os.path.normpath(data_dir)
data_path = os.path.normpath(data_path)
clean_data_path = os.path.normpath(clean_data_path)

print(f"📂 Ruta al archivo de entrada: {data_path}")
print(f"📂 Ruta al archivo de salida: {clean_data_path}")

# Verificar si el archivo existe
if not os.path.isfile(data_path):
    print("\n❌ El archivo movies.csv no se encuentra en la ruta especificada.")
else:
    print("\n✅ El archivo movies.csv ha sido encontrado correctamente.")

    # Cargar el dataset
    df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Convertir 'releaseDate' a tipo fecha
    df["releaseDate"] = pd.to_datetime(df["releaseDate"], errors="coerce")

    # Mostrar información general
    print("\n🔍 Información general del dataset:")
    print(df.info())

    # Revisar datos faltantes
    print("\n⚠️  Datos faltantes en el dataset:")
    print(df.isnull().sum())

    # Descripción estadística de las variables numéricas
    print("\n📊 Estadísticas de las variables numéricas:")
    print(df.describe().applymap(lambda x: f"{x:,.2f}"))

    # Crear la carpeta 'data/' si no existe
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📂 Carpeta creada: {data_dir}")

    # Guardar el dataset limpio
    df.to_csv(clean_data_path, index=False)
    print(f"\n✅ Datos guardados en: {clean_data_path}")

    ### Clasificación Automática de Variables ###
    # Diccionario para clasificar las variables
    classification = {}

    for column in df.columns:
        dtype = df[column].dtype  # Obtener el tipo de dato de la columna
        
        if dtype == "object":
            classification[column] = "Cualitativa Nominal"
        elif dtype == "int64":
            classification[column] = "Cuantitativa Discreta"
        elif dtype == "float64":
            classification[column] = "Cuantitativa Continua"
        elif "datetime" in str(dtype):
            classification[column] = "Cualitativa Nominal"
    
    # Correcciones manuales para ciertas variables mal detectadas
    continuous_vars = ["budget", "revenue", "runtime", "popularity", "voteAvg", "actorsPopularity"]
    discrete_vars = ["castWomenAmount", "castMenAmount"]

    for var in continuous_vars:
        if var in classification:
            classification[var] = "Cuantitativa Continua"

    for var in discrete_vars:
        if var in classification:
            classification[var] = "Cuantitativa Discreta"
    
    # Convertir la clasificación a un DataFrame
    classification_df = pd.DataFrame(list(classification.items()), columns=["Variable", "Tipo"])

    # Mostrar la clasificación
    print("\n📌 Clasificación de las Variables:")
    print(classification_df)

    # Convertir variables numéricas que puedan estar en texto
    for var in continuous_vars:
        df[var] = pd.to_numeric(df[var], errors='coerce') 

    ### Análisis de Distribución Normal ###
    print("\n📊 Generando gráficos de distribución...")

    for var in continuous_vars:
        plt.figure(figsize=(8, 4))

        try:
            # Filtrar valores extremos usando el método IQR (Rango Intercuartílico)
            Q1 = np.percentile(df[var].dropna(), 25)
            Q3 = np.percentile(df[var].dropna(), 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Aplicar filtro solo a `actorsPopularity` para mejorar rendimiento
            if var == "actorsPopularity":
                filtered_data = df[(df[var] >= lower_bound) & (df[var] <= upper_bound)][var]

                # Agregar un límite en el eje X para evitar que el gráfico se deforme
                max_x = min(upper_bound, filtered_data.max())  

                sns.histplot(filtered_data, kde=True, bins=20)
                plt.xlim(left=filtered_data.min(), right=max_x)
                plt.title(f"Distribución de {var} (Filtrada)")
            else:
                filtered_data = df[var]
                sns.histplot(filtered_data, kde=True, bins=30)
                plt.title(f"Distribución de {var}")

            plt.xlabel(var)
            plt.ylabel("Frecuencia")
            plt.show()

        except Exception as e:
            print(f"⚠️ No se pudo graficar {var} debido a un error: {e}")
            
    print("\nProceso terminado...")

    ### Pruebas de Normalidad ###
    print("\n📊 Pruebas de Normalidad:")
    normality_results = []

    for var in continuous_vars:
        data = df[var].dropna()  # Eliminar valores nulos
        shapiro_test = stats.shapiro(data) if len(data) < 5000 else (None, None)
        ks_test = stats.kstest(data, 'norm')

        normality_results.append({
            "Variable": var,
            "Shapiro-Wilk p-valor": f"{shapiro_test[1]:.6f}" if shapiro_test[1] is not None else "N/A",
            "Kolmogorov-Smirnov p-valor": ks_test.pvalue
        })

    normality_df = pd.DataFrame(normality_results)
    print(normality_df)

    ### Tablas de Frecuencia de Variables Cualitativas ###
    print("\n📊 Tablas de Frecuencias de Variables Cualitativas:")
    qualitative_vars = ["genres", "productionCompany", "productionCountry", "originalLanguage"]

    for var in qualitative_vars:
        print(f"\n🔹 {var}:")
        print(df[var].value_counts().head(10))
        # Para poder observar todos los datos
        # print(df[var].value_counts().to_frame().rename(columns={var: "count"}))