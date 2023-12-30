import streamlit as st
from langchain.schema import SystemMessage, HumanMessage

def initialize_session_state_variables():
    # variables for using OpenAI
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    if "openai" not in st.session_state:
        st.session_state.openai = None

    if "llm_chatmodel" not in st.session_state:
        st.session_state.llm_chatmodel = "gpt-3.5-turbo"
        
    # variables for chatbot
    if "ai_role" not in st.session_state:
        # st.session_state.ai_role = ["You are a helpful assistant.", "You are a helpful assistant."]
        st.session_state.ai_role = 2 * ["You are a helpful assistant."]

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=st.session_state.ai_role[0])
        ]

    if "prompt_exists" not in st.session_state:
        st.session_state.prompt_exists = False

    if "human_enq" not in st.session_state:
        st.session_state.human_enq = []

    if "ai_resp" not in st.session_state:
        st.session_state.ai_resp = []

    if "temperature" not in st.session_state:
        st.session_state.temperature = [0.7, 0.7]

    if "temperature2" not in st.session_state:
        st.session_state.temperature2 = 0.7
        
    # variables for audio and image
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    if "mic_used" not in st.session_state:
        st.session_state.mic_used = False

    if "audio_response" not in st.session_state:
        st.session_state.audio_response = None

    if "audio_response_gtts" not in st.session_state:
        st.session_state.audio_response_gtts = None
        
    if "image_url" not in st.session_state:
        st.session_state.image_url = None

    if "image_description" not in st.session_state:
        st.session_state.image_description = None

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    if "qna" not in st.session_state:
        st.session_state.qna = {"question": "", "answer": ""}

    if "image_source" not in st.session_state:
        st.session_state.image_source = 2 * ["From URL"]

    # variables for RAG
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "sources" not in st.session_state:
        st.session_state.sources = None

    if "memory" not in st.session_state:
        st.session_state.memory = None

def reset_conversation():
    st.session_state.messages = [
        SystemMessage(content=st.session_state.ai_role[0])
    ]
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.prompt_exists = False
    st.session_state.human_enq = []
    st.session_state.ai_resp = []
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.audio_response = None
    st.session_state.audio_response_gtts = None
    st.session_state.vector_store = None
    st.session_state.sources = None
    st.session_state.memory = None

def change_state(variable_name, variable_value):
    st.session_state[variable_name] = variable_value

def info_state(key):
    st.info(st.session_state[key])