import streamlit as st
import openai
import commonFunction as commonFunc

def app():

    commonFunc.initialize_session_state_variables()

    st.set_page_config(
        page_title="chatGPT App",
        page_icon="👋",
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
    st.write("**API Key Selection**")
    choice_api = st.radio(
        label="$\\hspace{0.25em}\\texttt{Choice of API}$",
        options=("Your key", "My key"),
        label_visibility="collapsed",
        horizontal=True,
    )

    if choice_api == "Your key":
        st.write("**Your API Key**")
        st.session_state.openai_api_key = st.text_input(
            label="$\\hspace{0.25em}\\texttt{Your OpenAI API Key}$",
            type="password",
            placeholder="sk-",
            label_visibility="collapsed",
        )
        authen = False if st.session_state.openai_api_key == "" else True
    else:
        st.session_state.openai_api_key = st.secrets["openai_api_key"]
        stored_pin = st.secrets["user_PIN"]
        st.write("**Password**")
        user_pin = st.text_input(
            label="Enter password", type="password", label_visibility="collapsed"
        )
        authen = user_pin == stored_pin

    st.session_state.openai = openai.OpenAI(
        api_key=st.session_state.openai_api_key
    )

if __name__ == "__main__":
    app()
