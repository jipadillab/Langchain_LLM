import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os

# --- Configuración de la página ---
st.set_page_config(
    page_title=" 🌱 Asistente Agrícola con IA (Hugging Face)",
    page_icon="👨‍🌾",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🌱 Asistente Agrícola con IA (Hugging Face)")
st.write("¡Hola! Soy tu asistente agrícola impulsado por IA, ahora desde la nube con Hugging Face. "
         "Pregúntame sobre cultivos, plagas, suelos, fertilizantes y más.")

# --- Configuración de Hugging Face ---
# **ADVERTENCIA DE SEGURIDAD:**
# Nunca incluyas tu token directamente en el código fuente en un repositorio público.
# Para despliegues reales, usa os.environ.get("HUGGINGFACEHUB_API_TOKEN")
# o los secretos de Streamlit (st.secrets["HUGGINGFACEHUB_API_TOKEN"]).
# Aquí lo incluimos directamente para fines de prueba y demostración.
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Modelos recomendados y GRATUITOS de Hugging Face para LangChain:
# Puedes buscar más en Hugging Face Hub que sean compatibles con text-generation.
# Modelos buenos y relativamente pequeños para empezar (pueden ser lentos en la versión gratuita):
# - google/flan-t5-xxl (grande, pero un buen ejemplo de LLM)
# - google/flan-t5-base (más pequeño y rápido)
# - gpt2 (muy básico, solo para probar la conexión)
# - microsoft/DialoGPT-medium (diseñado para chat, pero no para instrucciones complejas)

# Recomendación: Empieza con un modelo como 'google/flan-t5-base' o 'HuggingFaceH4/zephyr-7b-beta'
# (este último puede ser más lento si no tienes acceso a inferencia acelerada)
# Para este ejemplo, usaremos 'google/flan-t5-base' por su disponibilidad general y tamaño manejable.
repo_id = "google/flan-t5-base"

try:
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_length": 500} # Ajusta la temperatura y la longitud máxima
    )
except Exception as e:
    st.error(f"Error al conectar con Hugging Face Hub. Asegúrate de que tu token es válido "
             f"y el modelo '{repo_id}' existe y es accesible. Error: {e}")
    st.stop()

# Plantilla de prompt para el asistente agrícola
template = """
Eres un asistente de inteligencia artificial experto en agricultura. Tu objetivo es proporcionar información precisa, útil y práctica sobre una amplia gama de temas agrícolas. Responde a las preguntas de los usuarios de manera clara, concisa y fácil de entender.

Temas que puedes cubrir:
- Cultivos (siembra, cuidado, cosecha, enfermedades, etc.)
- Plagas y enfermedades (identificación, prevención, tratamiento)
- Suelos (tipos, análisis, mejora, nutrientes)
- Fertilización (tipos de fertilizantes, cuándo y cómo aplicarlos)
- Riego (métodos, necesidades de agua por cultivo)
- Ganadería (conceptos básicos, salud animal, alimentación, etc. - si el contexto lo permite)
- Maquinaria agrícola (usos básicos, mantenimiento - si el contexto lo permite)

Si la pregunta no está relacionada con la agricultura, por favor, indícalo amablemente y redirige al usuario a temas agrícolas.

Pregunta del usuario: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# --- Interfaz de usuario de Streamlit ---

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes históricos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
user_question = st.chat_input("Haz tu pregunta sobre agricultura aquí...")

if user_question:
    # Añadir pregunta del usuario al historial
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                # Generar respuesta usando LangChain
                response = llm_chain.run(question=user_question)
                st.markdown(response)
                # Añadir respuesta del asistente al historial
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Ocurrió un error al generar la respuesta. Por favor, inténtalo de nuevo. Error: {e}")

# --- Información Adicional en Sidebar ---
st.sidebar.header("Acerca de este Asistente")
st.sidebar.info(
    "Este asistente utiliza **Hugging Face Hub** para acceder a modelos de lenguaje grandes (LLMs) "
    "y **LangChain** para la orquestación. Es una herramienta experimental y la calidad de las respuestas "
    "depende del modelo LLM subyacente y la calidad del prompt."
)
st.sidebar.write("Desarrollado con ❤️ para la comunidad agrícola.")

st.sidebar.header("Consideraciones de Hugging Face")
st.sidebar.markdown(
    """
    1.  **Token de API:** Asegúrate de que tu token de Hugging Face (`HF_TOKEN` o `HUGGINGFACEHUB_API_TOKEN`) esté configurado correctamente.
    2.  **Modelos Gratuitos:** La inferencia gratuita en Hugging Face puede ser lenta o tener límites de uso. Considera modelos más pequeños para mejor rendimiento.
    3.  **Tipo de Modelo:** El modelo (`repo_id`) debe ser apto para "text generation" o "conversational" para funcionar bien como asistente. `google/flan-t5-base` es un buen punto de partida.
    """
)
