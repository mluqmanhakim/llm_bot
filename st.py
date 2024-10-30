import streamlit as st

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@st.cache_resource
def initialize_models():
    llm = Ollama(model="tinyllama:latest", request_timeout=120.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    return embed_model, llm
    
@st.cache_resource()
def load_index():
    with st.spinner(text="Loading the index"):
        storage_context = StorageContext.from_defaults(persist_dir="bot_index")
        index = load_index_from_storage(storage_context)
        return index


if __name__ == '__main__':
    st.title("Review Bot")
    embed_model, llm = initialize_models()
    index = load_index()

    if "chat_engine" not in st.session_state.keys():
        # st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        st.session_state.query_engine = index.as_query_engine()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # response = st.session_state.chat_engine.chat(prompt)
        response = st.session_state.query_engine.query(prompt)
        
        with st.chat_message("assistant"):
            response = st.write(response.response)

        st.session_state.messages.append({"role": "assistant", "content": response})