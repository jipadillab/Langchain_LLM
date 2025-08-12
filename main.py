import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os # Necesario para interactuar con variables de entorno

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
# **MANEJO SEGURO DEL TOKEN:**
# Streamlit Community Cloud permite usar st.secrets para variables de entorno.
# No pegues tu token directamente aquí. Lo configuraremos en el panel de Streamlit Cloud.
# Esto intentará leer el token de los secretos de Streamlit o de una variable de entorno local.
try:
    hf_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except KeyError:
    hf_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not hf_api_token:
    st.error("Error: Token de Hugging Face API no configurado. "
             "Por favor, añade 'HUGGINGFACEHUB_API_TOKEN' a tus secretos de Streamlit "
             "o como una variable de entorno.")
    st.stop()

# Configura la variable de entorno para LangChain
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token

# Modelo de Hugging Face. 'google/flan-t5-base' es una buena opción gratuita y de propósito general.
# Considera que los modelos grandes como Llama 2 o Mistral pueden ser muy lentos o no funcionar
# en la inferencia gratuita de Hugging Face.
repo_id = "google/flan-t5-base" # Puedes cambiarlo por "google/flan-t5-large" si quieres algo más potente (puede ser más lento)

try:
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_length": 500} # Ajusta la creatividad y la longitud de la respuesta
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
    1.  **Token de API:** Asegúrate de que tu token de Hugging Face (`HUGGINGFACEHUB_API_TOKEN`) esté configurado correctamente en los secretos de Streamlit.
    2.  **Modelos Gratuitos:** La inferencia gratuita en Hugging Face puede ser lenta o tener límites de uso. Considera modelos más pequeños para un mejor rendimiento.
    3.  **Tipo de Modelo:** El modelo (`repo_id`) debe ser apto para "text generation" o "conversational" para funcionar bien como asistente. `google/flan-t5-base` es un buen punto de partida.
    """
)
