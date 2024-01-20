import openai
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import commonFunction as commonFunc

import os, base64, requests, re
from io import BytesIO
from tempfile import NamedTemporaryFile

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def generate_chat_with_ai(user_prompt):
    openai_llm = ChatOpenAI(
        openai_api_key=st.session_state.openai_api_key,
        model_name=st.session_state.llm_chatmodel,
        temperature=st.session_state.temperature2,
        streaming=True,
        callbacks=[StreamHandler(st.empty())]
    )
    
    # Add the user input to the messages
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    
    try:
        response = openai_llm(st.session_state.messages)
        generated_text = response.content
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    if generated_text is not None:
        # Add the generated output to the messages
        st.session_state.messages.append(response)

    return generated_text

def enable_user_input():
    st.session_state.prompt_exists = True

def get_vector_store(uploaded_file):
    if uploaded_file is None:
        return None

    file_bytes = BytesIO(uploaded_file.read())
    
    # Create a temporary file within the "files/" directory
    with NamedTemporaryFile(dir="files/", delete=False) as file:
        filepath = file.name
        file.write(file_bytes.read())
    
    # Determine the loader based on the file extension.
    if uploaded_file.name.lower().endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif uploaded_file.name.lower().endswith(".txt"):
        loader = TextLoader(filepath)
    elif uploaded_file.name.lower().endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif uploaded_file.name.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
    elif uploaded_file.name.lower().endswith(".html"):
        loader = UnstructuredHTMLLoader(filepath)
    elif uploaded_file.name.lower().endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(filepath)
    else:
        st.error("Please load a file in pdf or txt", icon="🚨")
        if os.path.exists(filepath):
            os.remove(filepath)
        return None

    # Load the document using the selected loader.
    document = loader.load()

    try:
        with st.spinner("Vector store in preparation..."):
            # Split the loaded text into smaller chunks for processing.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                # separators=["\n", "\n\n", "(?<=\. )", "", " "],
            )
            doc = text_splitter.split_documents(document)
            
            # Create a FAISS vector database.
            embeddings = OpenAIEmbeddings(
                openai_api_key=st.session_state.openai_api_key
            )
            
            vector_store = FAISS.from_documents(doc, embeddings)
    except Exception as e:
        vector_store = None
        st.error(f"An error occurred: {e}", icon="🚨")
    finally:
        # Ensure the temporary file is deleted after processing
        if os.path.exists(filepath):
            os.remove(filepath)

    return vector_store

def run_chatdoc():
    # initial system prompts
    general_role = "You are a helpful assistant."
    english_teacher = "You are an English teacher who analyzes texts and corrects any grammatical issues if necessary."
    translator = "You are a translator who translates English into Korean and Korean into English."
    coding_adviser = "You are an expert in coding who provides advice on good coding styles."
    doc_analyzer = "You are an assistant analyzing the document uploaded."
    docQA_analyzer = "You are an assistant analyzing the document uploaded and providing the source"
    roles = (general_role, english_teacher, translator, coding_adviser, doc_analyzer, docQA_analyzer)

    # check ai_role
    if st.session_state.ai_role[1] not in (doc_analyzer):
        st.session_state.ai_role[0] = doc_analyzer
        commonFunc.reset_conversation()
    
    with st.sidebar:
        # GPT llm models
        st.write("")
        st.write("**LLM Models**")  # https://platform.openai.com/docs/models/continuous-model-upgrades
        optLlmModel = st.sidebar.radio(
            label="not display",
            options=(
                "gpt-3.5-turbo",
                "gpt-4"
            ),
            label_visibility="collapsed",
            key="llm_chatmodel",    # <= st.session_state.llm_chatmodel
            # on_change=commonFunc.info_state,   # https://stackoverflow.com/questions/72182849/streamlit-pass-the-widget-input-to-its-own-callback
            # args=("llm_chatmodel",)
            # index=0,
            # horizontal=True,
        )

        # Temperature selection
        st.write("")
        st.write("**Temperature**")
        optTemperature = st.slider(
            label="not display",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature2,
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
            key="temperature2",    # <= st.session_state.temperature2
            # on_change=commonFunc.info_state,   # https://stackoverflow.com/questions/72182849/streamlit-pass-the-widget-input-to-its-own-callback
            # args=("temperature2",)
        )
        st.write("(Higher $\Rightarrow$ More random)")
                
    st.write("")
    st.write("##### Message to AI")
    st.session_state.ai_role[0] = st.selectbox(
        label="AI's role",
        options=roles,
        index=roles.index(st.session_state.ai_role[1]),
        # on_change=commonFunc.reset_conversation,
        label_visibility="collapsed",
    )

    if st.session_state.ai_role[0] != st.session_state.ai_role[1]:
        commonFunc.reset_conversation()

    if st.session_state.ai_role[0] == doc_analyzer:
        st.write("")
        left, right = st.columns([4, 7])
        left.write("##### Document to ask about")
        right.write("If you want a consistent answer, set the Temperature param to 0.")
        uploaded_file = st.file_uploader(
            label="Upload an article",
            type=["txt", "pdf", "docx", "pptx", "csv", "html"],
            accept_multiple_files=False,
            # on_change=commonFunc.reset_conversation(),
            label_visibility="collapsed",
        )
        if st.session_state.vector_store is None:
            # Create the vector store.
            st.session_state.vector_store = get_vector_store(uploaded_file)

            if st.session_state.vector_store is not None:
                st.write(f"Vector store for :blue[[{uploaded_file.name}]] is ready!")
                
    st.write("")
    st.write("##### Conversation with AI")

    # Print conversations
    for human, ai in zip(st.session_state.human_enq, st.session_state.ai_resp):
        with st.chat_message("human"):
            st.write(human)
        with st.chat_message("ai"):
            st.write(ai)

    # Reset the conversation
    st.button(label="Reset the conversation", on_click=commonFunc.reset_conversation)

    # Use your keyboard
    user_input = st.chat_input(
        placeholder="Enter your query",
        on_submit=enable_user_input,
        disabled=not uploaded_file if st.session_state.ai_role[0] == doc_analyzer else False
    )

    if user_input and st.session_state.prompt_exists:
        user_prompt = user_input.strip()

    if st.session_state.prompt_exists:
        with st.chat_message("human"):
            st.write(user_prompt)

        with st.chat_message("ai"):
            generated_text = generate_chat_with_ai(user_prompt)

        st.session_state.prompt_exists = False
        
        if generated_text is not None:
            st.session_state.human_enq.append(user_prompt)
            st.session_state.ai_resp.append(generated_text)
            st.rerun()
           
if __name__ == "__main__":
    if "openai_api_key" not in st.session_state:
        switch_page('Home')

    run_chatdoc()
    st.info(f"llm model : {st.session_state.llm_chatmodel}, temperature : {st.session_state.temperature2}")
