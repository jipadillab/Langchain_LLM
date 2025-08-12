import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# --- Configuración de la página ---
st.set_page_config(
    page_title=" 🌱 Asistente Agrícola con IA",
    page_icon="👨‍🌾",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🌱 Asistente Agrícola con IA")
st.write("¡Hola! Soy tu asistente agrícola impulsado por IA. Pregúntame sobre cultivos, plagas, suelos, fertilizantes y más. "
         "Estoy aquí para ayudarte a optimizar tus prácticas agrícolas.")

# --- Configuración de Ollama y LangChain ---
# Asegúrate de que Ollama esté corriendo y de que hayas descargado un modelo como 'llama2' o 'mistral'.
# Puedes cambiar el modelo aquí si tienes otro descargado.
try:
    llm = Ollama(model="llama2") # O "mistral" si prefieres ese modelo
except Exception as e:
    st.error(f"Error al conectar con Ollama. Asegúrate de que Ollama esté ejecutándose y el modelo 'llama2' "
             f"(o el que hayas especificado) esté descargado. Error: {e}")
    st.stop() # Detiene la ejecución si no se puede conectar con Ollama

# Plantilla de prompt para el asistente agrícola
# Es crucial una buena ingeniería de prompts para obtener respuestas relevantes.
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
    "Este asistente utiliza **Ollama** para ejecutar modelos de lenguaje grandes de forma local y **LangChain** "
    "para la orquestación. Es una herramienta experimental y la calidad de las respuestas depende del modelo LLM "
    "subyacente y la calidad del prompt."
)
st.sidebar.write("Desarrollado con ❤️ para la comunidad agrícola.")

st.sidebar.header("Pasos para usar Ollama")
st.sidebar.markdown(
    """
    1.  **Descarga e instala Ollama:** Visita [ollama.com](https://ollama.com/)
    2.  **Descarga un modelo:** Abre tu terminal y ejecuta `ollama run llama2` o `ollama run mistral`. Esto descargará el modelo y lo iniciará. Asegúrate de que el nombre del modelo coincida con el `model` configurado en `main.py`.
    3.  **Ejecuta Streamlit:** Navega al directorio de tu proyecto en la terminal y ejecuta `streamlit run main.py`.
    """
)
