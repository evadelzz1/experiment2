import streamlit as st
import openai
import commonFunction as commonFunc

from dotenv import load_dotenv
import os

def app():

    commonFunc.initialize_session_state_variables()

    st.set_page_config(
        page_title="GPT Apps",
        page_icon="",
    )

    st.title("Welcome to the GPT Apps")

    with st.sidebar:
        # st.info("Select a page above.")
        st.write(
            "<small>**About.**  \n</small>",
            "<small>*blueholelabs*, Dec. 2023  \n</small>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("**OPENAI_API_KEY Selection**")
    choice_api = st.radio(
        label="$\\hspace{0.25em}\\texttt{Choice of API}$",
        options=("Your key", "My key"),
        label_visibility="collapsed",
        horizontal=True,
    )

    authen = False  

    if choice_api == "Your key":
        st.write("**Your API Key**")
        st.session_state.openai_api_key = st.text_input(
            label="$\\hspace{0.25em}\\texttt{Your OpenAI API Key}$",
            type="password",
            placeholder="sk-",
            value="",
            label_visibility="collapsed",
        )
        authen = False if st.session_state.openai_api_key == "" else True
    else:

        if not load_dotenv():
            print("Could not load .env file or it is empty. Please check if it exists and is readable.")
            exit(1)

        st.session_state.user_password = os.getenv("USER_PASSWORD")
        
        # st.session_state.openai_api_key = st.secrets["openai_api_key"]
        # user_password = st.secrets["user_PIN"]

        st.write("**Password**")
        inputPassword = st.text_input(
            label="Enter password", 
            type="password", 
            label_visibility="collapsed"
        )
        if (inputPassword == st.session_state.user_password):
            st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")
            authen = True        

    st.session_state.openai = openai.OpenAI(
        api_key=st.session_state.openai_api_key
    )
    
    if authen == True:
        st.info("Successed Login!")

if __name__ == "__main__":
    app()
