import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurar el estilo de Seaborn
sns.set_theme(style="whitegrid", context="talk")

def cargar_datos(archivo_csv):
    """Cargue de archivo CSV"""
    try:
        df = pd.read_csv(archivo_csv)
        print(f"Forma del DataFrame (filas, columnas): {df.shape}")
        print("\nPRIMERAS 5 FILAS:")
        print(df.head())
        print("\nINFORMACIÓN GENERAL (tipos de datos):")
        df.info()
        print("\nCANTIDAD DE NULOS POR COLUMNA:")
        print(df.isnull().sum())
        print(f"\nCANTIDAD DE FILAS DUPLICADAS: {df.duplicated().sum()}")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{archivo_csv}' no existe en la ruta especificada.")
        return None

def limpiar_datos(df):
    df_limpio = df.copy()
    filas_antes = len(df_limpio)
    
    #Eliminar filas duplicadas
    df_limpio = df_limpio.drop_duplicates()
    
    #Manejo de valores faltantes en variables numéricas ('consumo_kwh_mes' y 'area_m2')
    for col in ['consumo_kwh_mes', 'area_m2']:
        if col in df_limpio.columns and df_limpio[col].isnull().sum() > 0:
            mediana_val = df_limpio[col].median()
            df_limpio[col] = df_limpio[col].fillna(mediana_val)
            print(f"Valores nulos en '{col}' imputados con la mediana: {mediana_val}")

    #Limpieza de texto o conversión a numérico por si hay celdas mal formateadas
    for col in ['consumo_kwh_mes', 'area_m2']:
        if df_limpio[col].dtype == 'object':
            df_limpio[col] = df_limpio[col].astype(str).str.replace(',', '.').str.extract(r'([0-9]+\.?[0-9]*)')[0]
            df_limpio[col] = pd.to_numeric(df_limpio[col])

    filas_despues = len(df_limpio)
    print(f"\nFilas antes de la limpieza: {filas_antes}")
    print(f"Filas después de la limpieza: {filas_despues}")
    
    print("\nDistribución de clases objetivo ('consumo_ineficiente') final:")
    print(df_limpio['consumo_ineficiente'].value_counts())

    #Guardar en data/grupo_13_limpio.csv
    if not os.path.exists('data'):
        os.makedirs('data')
    ruta_salida = 'data/grupo_13_limpio.csv'
    df_limpio.to_csv(ruta_salida, index=False)
    print(f"\nDataset limpio guardado exitosamente en: {ruta_salida}")
    
    return df_limpio

def visualizar_datos(df):
    """Tarea 4: Genera y guarda los gráficos con Seaborn."""
    
    if not os.path.exists('data'):
        os.makedirs('data')

    # Gráfico 1: Dispersión de las dos variables numéricas coloreada por la columna objetivo
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, 
        x='consumo_kwh_mes', 
        y='area_m2', 
        hue='consumo_ineficiente', 
        palette='Set1', 
        s=80, 
        alpha=0.8
    )
    plt.title('Dispersión: Consumo kWh/Mes vs Área (m²)')
    plt.xlabel('Consumo kWh/Mes (x1)')
    plt.ylabel('Área m² (x2)')
    plt.legend(title='Consumo Ineficiente', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/grafico_dispersion.png', dpi=150)
    plt.close()

    # Gráfico 2: Boxplot que compare una variable numérica entre las clases 0 y 1
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df, 
        x='consumo_ineficiente', 
        y='consumo_kwh_mes', 
        palette='Set2'
    )
    plt.title('Comparación de Consumo kWh/Mes según Ineficiencia')
    plt.xlabel('Consumo Ineficiente (0 = Normal, 1 = Ineficiente)')
    plt.ylabel('Consumo kWh/Mes')
    plt.tight_layout()
    plt.savefig('data/grafico_boxplot.png', dpi=150)
    plt.close()
    
    print("Gráficos 'grafico_dispersion.png' y 'grafico_boxplot.png' generados y guardados en la carpeta data/.")

# ==========================================
# FUNCIONES AUXILIARES PARA EL PERCEPTRÓN (NumPy)
# ==========================================
def sigmoide(z):
    """Función de activación sigmoide (evitando desbordamientos numéricos)."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def entrenar_perceptron(X, y, epochs=300, lr=0.1):
    print("\n" + "+" + "="*60 + "+")
    print(" TAREA 5: PERCEPTRÓN DESDE CERO CON NUMPY")
    print("+" + "="*60 + "+")
    
    n_muestras, n_caracteristicas = X.shape
    
    # Inicialización de pesos y sesgo (b)
    np.random.seed(42)
    w = np.random.randn(n_caracteristicas)
    b = 0.0
    
    print(f"Pesos iniciales (w1, w2): {w}")
    print(f"Sesgo inicial (b): {b}")
    
    historial_perdida = []
    
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoide(z)
        
        # Pérdida logarítmica (Binary Cross-Entropy)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        perdida = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        historial_perdida.append(perdida)
        
        # Gradientes
        dw = (1 / n_muestras) * np.dot(X.T, (y_pred - y))
        db = (1 / n_muestras) * np.sum(y_pred - y)
        
        # Actualización
        w -= lr * dw
        b -= lr * db

    # Métricas finales
    z_final = np.dot(X, w) + b
    y_pred_final = sigmoide(z_final)
    predicciones_clase = (y_pred_final >= 0.5).astype(int)
    exactitud = np.mean(predicciones_clase == y)
    
    print(f"Pesos finales (w1, w2): {w}")
    print(f"Sesgo final (b): {b}")
    print(f"Pérdida inicial: {historial_perdida[0]:.4f}")
    print(f"Pérdida final: {historial_perdida[-1]:.4f}")
    print(f"Exactitud final (Accuracy): {exactitud * 100:.2f}%")
    
    # Guardar Curva de Aprendizaje
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), historial_perdida, color='b', lw=2)
    plt.title('Curva de Aprendizaje (Pérdida por Época)')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida Logarítmica')
    plt.tight_layout()
    plt.savefig('data/curva_aprendizaje.png', dpi=150)
    plt.close()
    
    return w, b

def generar_frontera_decision(X_original, y, w, b, media_x, desv_x):
    """Genera el gráfico de la frontera de decisión en las unidades originales."""
    plt.figure(figsize=(9, 6))
    
    # Puntos originales
    plt.scatter(X_original[:, 0], X_original[:, 1], c=y, cmap='Set1', edgecolor='k', s=80)
    
    x1_min, x1_max = X_original[:, 0].min() - 1, X_original[:, 0].max() + 1
    x1_vals = np.linspace(x1_min, x1_max, 100)
    
    w1, w2 = w[0], w[1]
    m0, m1 = media_x[0], media_x[1]
    d0, d1 = desv_x[0], desv_x[1]
    
    # Despejar x2 (area_m2) en función de x1 (consumo_kwh_mes)
    x2_vals = m1 - (d1 / w2) * (b + (w1 / d0) * (x1_vals - m0))
    
    plt.plot(x1_vals, x2_vals, color='green', linewidth=2.5, label='Frontera de Decisión')
    plt.title('Frontera de Decisión del Perceptrón (Unidades Originales)')
    plt.xlabel('Consumo kWh/Mes (x1)')
    plt.ylabel('Área m² (x2)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/frontera_decision.png', dpi=150)
    plt.close()
    print("Gráficos 'curva_aprendizaje.png' y 'frontera_decision.png' generados con éxito.")

def main():
    archivo_csv = 'grupo_13.csv'
    
    df = cargar_datos(archivo_csv)
    if df is None:
        return

    df_limpio = limpiar_datos(df)

    visualizar_datos(df_limpio)

    X_raw = df_limpio[['consumo_kwh_mes', 'area_m2']].values
    y = df_limpio['consumo_ineficiente'].values

    media_x = np.mean(X_raw, axis=0)
    desv_x = np.std(X_raw, axis=0)
    desv_x[desv_x == 0] = 1.0  # Evitar división por cero
    
    X_estandarizada = (X_raw - media_x) / desv_x

    # Entrenar modelo
    w, b = entrenar_perceptron(X_estandarizada, y, epochs=300, lr=0.1)

    # Generar gráfico de frontera de decisión
    generar_frontera_decision(X_raw, y, w, b, media_x, desv_x)

if __name__ == '__main__':
    main()