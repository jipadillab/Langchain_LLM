import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

# Título de la app
st.set_page_config(page_title="Agente Agrícola", page_icon="🌱")
st.title("🌱 Agente Agrícola Asistente")
st.markdown("Asistente de IA para consultas sobre agricultura, cultivos y buenas prácticas.")

# Cargar token de Hugging Face desde Secrets (en Streamlit Cloud)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Falta el token de Hugging Face. Configure `HF_TOKEN` en los secretos de Streamlit.")
    st.stop()

# Inicializar modelo Hugging Face
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# Memoria conversacional
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Entrada del usuario
user_input = st.text_input("💬 Escribe tu pregunta agrícola aquí:")

# Procesar respuesta
if st.button("Responder"):
    if user_input.strip() == "":
        st.warning("Por favor escribe una pregunta.")
    else:
        with st.spinner("Pensando..."):
            respuesta = conversation.predict(input=user_input)
            st.markdown(f"**Respuesta:** {respuesta}")

# Historial
if st.checkbox("Mostrar historial de conversación"):
    st.write(memory. Buffer)
