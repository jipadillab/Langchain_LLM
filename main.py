import streamlit as st
import pandas as pd
import numpy as np
import time

# --- Configuración de la página ---
st.set_page_config(
    page_title="Visualizador de Datos Aleatorios",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Visualizador de Datos Aleatorios con Streamlit")
st.write("Esta aplicación genera y muestra datos aleatorios en tiempo real.")

# --- Sidebar para controles ---
st.sidebar.header("Opciones de Visualización")
num_filas = st.sidebar.slider("Número de filas", min_value=10, max_value=1000, value=100, step=10)
intervalo_actualizacion = st.sidebar.slider("Intervalo de actualización (segundos)", min_value=1, max_value=10, value=3, step=1)
columnas_mostrar = st.sidebar.multiselect(
    "Columnas a mostrar",
    options=["Columna A", "Columna B", "Columna C", "Columna D"],
    default=["Columna A", "Columna B"]
)

# --- Generación de datos aleatorios ---
@st.cache_data(ttl=intervalo_actualizacion) # Cachea los datos por el intervalo de actualización
def generar_datos(num_filas):
    data = {
        'Columna A': np.random.rand(num_filas) * 100,
        'Columna B': np.random.randint(0, 1000, num_filas),
        'Columna C': np.random.randn(num_filas) * 50 + 100,
        'Columna D': np.random.choice(['Categoría X', 'Categoría Y', 'Categoría Z'], num_filas)
    }
    df = pd.DataFrame(data)
    return df

# --- Contenedor para los datos y gráficos ---
placeholder = st.empty()

while True:
    df = generar_datos(num_filas)
    df_mostrar = df[columnas_mostrar] # Mostrar solo las columnas seleccionadas

    with placeholder.container():
        st.subheader(f"Datos Aleatorios (Última actualización: {pd.Timestamp.now().strftime('%H:%M:%S')})")
        st.dataframe(df_mostrar)

        st.subheader("Gráfico de Líneas (Columna A vs Columna B)")
        if "Columna A" in columnas_mostrar and "Columna B" in columnas_mostrar:
            st.line_chart(df_mostrar[["Columna A", "Columna B"]])
        else:
            st.info("Selecciona 'Columna A' y 'Columna B' para ver el gráfico de líneas.")

        st.subheader("Histograma (Columna C)")
        if "Columna C" in columnas_mostrar:
            fig, ax = plt.subplots()
            ax.hist(df_mostrar["Columna C"], bins=20)
            st.pyplot(fig)
        else:
            st.info("Selecciona 'Columna C' para ver el histograma.")

    time.sleep(intervalo_actualizacion)