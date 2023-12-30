import streamlit as st

st.title("about")

st.write("blueholelabs.inc")

st.write(st.session_state.llm_chatmodel)
st.write(st.session_state.temperature[0])

