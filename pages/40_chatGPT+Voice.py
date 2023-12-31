import openai
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import commonFunction as commonFunc

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler

import os, base64, requests, re
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS

# This is for streaming on Streamlit
# Reference : https://github.com/streamlit/StreamlitLangChain/blob/main/streaming_demo.py
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

def play_audio(audio_response):
    if st.session_state.tts_model == "OpenAI":
        audio_data = audio_response.read()

        # Encode audio data to base64
        b64 = base64.b64encode(audio_data).decode("utf-8")

        # Create a markdown string to embed the audio player with the base64 source
        md = f"""
            <audio controls autoplay style="width: 100%;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
         # Use Streamlit to render the audio player
        st.markdown(md, unsafe_allow_html=True)
    elif st.session_state.tts == "gTTS":
        st.audio(audio_response)

def read_audio(audio_bytes):
    try:
        audio_data = BytesIO(audio_bytes)
        audio_data.name = "recorded_audio.wav"  # dummy name

        transcript = st.session_state.openai.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )
        text = transcript.text
    except Exception as e:
        text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return text

def perform_tts(text):
    try:
        with st.spinner("TTS in progress..."):
            if st.session_state.tts_model == "OpenAI":
                audio_response = st.session_state.openai.audio.speech.create(
                    model="tts-1",
                    voice=st.session_state.tts_voice,
                    input=text,
                )
            elif st.session_state.tts_model == "gTTS":
                tts = gTTS(text=text, lang='en', tld='com', slow=False)
                audio_response = BytesIO()      # convert to file-like object
                tts.write_to_fp(audio_response)

    except Exception as e:
        audio_response = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return audio_response

def run_chatbot():
    # initial system prompts
    general_role = "You are a helpful assistant."
    english_teacher = "You are an English teacher who analyzes texts and corrects any grammatical issues if necessary."
    translator = "You are a translator who translates English into Korean and Korean into English."
    coding_adviser = "You are an expert in coding who provides advice on good coding styles."
    doc_analyzer = "You are an assistant analyzing the document uploaded."
    docQA_analyzer = "You are an assistant analyzing the document uploaded and providing the source"
    roles = (general_role, english_teacher, translator, coding_adviser, doc_analyzer, docQA_analyzer)

    # check ai_role
    if st.session_state.ai_role[1] not in (general_role, english_teacher):
        st.session_state.ai_role[0] = general_role
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
        # Text to Speech selection
        st.write("")
        st.write("**Text to Speech**")
        st.session_state.tts = st.radio(
            label="TTS",
            options=("Disabled", "Enabled", "Auto"),
            horizontal=True,
            index=0,
            label_visibility="visible",
        )
        # TTS model selection
        st.session_state.tts_model = st.radio(
            label="TTS model",
            options=("gTTS", "OpenAI"),
            horizontal=True,
            index=0,
            label_visibility="visible",
        )
        # TTS voice selection
        if st.session_state.tts_model == "OpenAI":
            st.session_state.tts_voice = st.radio(
                label="TTS Voice (for OpenAI)",
                options=("alloy", "echo","fable","onyx","nova","shimmer"),
                horizontal=True,
                index=0,
                label_visibility="visible",
        )
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

    st.write("")
    left, right = st.columns([4, 7])
    left.write("##### Conversation with AI")
    right.write("Click on the mic icon and speak, or type text below.")

    # Print conversations
    for human, ai in zip(st.session_state.human_enq, st.session_state.ai_resp):
        with st.chat_message("human"):
            st.write(human)
        with st.chat_message("ai"):
            st.write(ai)

    # Play TTS
    if st.session_state.audio_response is not None:
        if st.session_state.tts_model == "OpenAI":
            play_audio(st.session_state.audio_response)
            st.session_state.audio_response = None
        elif st.session_state.tts_model == "gTTS":
            st.audio(st.session_state.audio_response)
            # audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{st.session_state.audio_response}">'
            # st.markdown(audio_tag, unsafe_allow_html=True)
            st.session_state.audio_response = None
    
    # Reset the conversation
    st.button(label="Reset the conversation", on_click=commonFunc.reset_conversation)

    # Use your keyboard
    user_input = st.chat_input(
        placeholder="Enter your query",
        on_submit=enable_user_input,
        disabled=False
    )

    # Use your microphone
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes != st.session_state.audio_bytes:
        user_prompt = read_audio(audio_bytes)
        st.session_state.audio_bytes = audio_bytes
        if user_prompt is not None:
            st.session_state.prompt_exists = True
            st.session_state.mic_used = True
    elif user_input and st.session_state.prompt_exists:
        user_prompt = user_input.strip()

    if st.session_state.prompt_exists:
        with st.chat_message("human"):
            st.write(user_prompt)

        with st.chat_message("ai"):
            generated_text = generate_chat_with_ai(user_prompt)

        if generated_text is not None:
            # TTS under two conditions
            cond1 = st.session_state.tts == "Enabled"
            cond2 = st.session_state.tts == "Auto" and st.session_state.mic_used

            if cond1 or cond2:
                st.session_state.audio_response = perform_tts(generated_text)
                    
            st.session_state.mic_used = False
            st.session_state.human_enq.append(user_prompt)
            st.session_state.ai_resp.append(generated_text)
            
        st.session_state.prompt_exists = False

        if generated_text is not None:
           st.rerun()
           
if __name__ == "__main__":
    if "openai_api_key" not in st.session_state:
        switch_page('Home')

    run_chatbot()
    st.info(f"llm model : {st.session_state.llm_chatmodel}, temperature : {st.session_state.temperature2}")