import streamlit as st

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever


@st.cache_resource
def initialize_models():
    llm = Ollama(model="tinyllama:latest", request_timeout=120.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    return embed_model, llm


@st.cache_resource()
def load_index(context_path):
    with st.spinner(text="Loading the index"):
        storage_context = StorageContext.from_defaults(persist_dir=context_path)
        index = load_index_from_storage(storage_context)
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=3,
        )
        return index, retriever


def prepare_prompt(query, retriever):
    query_context = retriever.retrieve(query)

    context_str = ""
    for i in range(len(query_context)):
        context_str += query_context[0].get_content() + ". \n"

    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )
    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str=context_str, query_str=query)
    return prompt


if __name__ == "__main__":
    st.title("Review Bot")
    context_path = "bot_index"
    embed_model, llm = initialize_models(context_path)
    index, retriever = load_index()

    if "chat_engine" not in st.session_state.keys():
        st.session_state.llm = llm

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        prompt_mod = prepare_prompt(prompt, retriever)
        response = st.session_state.llm.complete(prompt_mod)

        with st.chat_message("assistant"):
            response = st.write(response.text)

        st.session_state.messages.append({"role": "assistant", "content": response})
