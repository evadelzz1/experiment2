import openai
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import commonFunction as commonFunc

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler

import os, base64, requests, re
from io import BytesIO
from PIL import Image, UnidentifiedImageError

def reset_qna_image():
    st.session_state.uploaded_image = None
    st.session_state.qna = {"question": "", "answer": ""}
    
def image_to_base64(image):
    # Convert the image to RGB mode if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the image to a BytesIO object
    buffered_image = BytesIO()
    image.save(buffered_image, format="JPEG")

    # Convert BytesIO to bytes and encode to base64
    img_str = base64.b64encode(buffered_image.getvalue())

    # Convert bytes to string
    base64_image = img_str.decode("utf-8")

    return base64_image

def shorten_image(image, max_pixels=1024):
    if max(image.width, image.height) > max_pixels:
        if image.width > image.height:
            new_width, new_height = 1024, image.height * 1024 // image.width
        else:
            new_width, new_height = image.width * 1024 // image.height, 1024

        image = image.resize((new_width, new_height))

    return image

def is_url(text):
    regex = r"(http|https)://([\w_-]+(?:\.[\w_-]+)+)(:\S*)?"
    p = re.compile(regex)
    match = p.match(text)
    if match:
        return True
    else:
        return False

def openai_create_image(description, model="dall-e-3", size="1024x1024"):
    """
    This function generates image based on user description.

    Args:
        description (string): User description
        model (string): Default set to "dall-e-3"
        size (string): Pixel size of the generated image

    Return:
        URL of the generated image
    """

    try:
        with st.spinner("AI is generating..."):
            response = st.session_state.openai.images.generate(
                model=model,
                prompt=description,
                size=size,
                quality="standard",
                n=1,
            )
        image_url = response.data[0].url
    except Exception as e:
        image_url = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return image_url

def openai_query_image_url(image_url, query, model="gpt-4-vision-preview"):
    """
    This function answers the user's query about the given image from a URL.

    Args:
        image_url (string): URL of the image
        query (string): the user's query
        model (string): default set to "gpt-4-vision-preview"

    Return:
        text as an answer to the user's query.
    """

    try:
        with st.spinner("AI is thinking..."):
            response = st.session_state.openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{query}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{image_url}"},
                            },
                        ],
                    },
                ],
                max_tokens=300,
            )
        generated_text = response.choices[0].message.content
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return generated_text

def openai_query_uploaded_image(image_b64, query, model="gpt-4-vision-preview"):
    """
    This function answers the user's query about the uploaded image.

    Args:
        image_b64 (base64 encoded string): base64 encoded image
        query (string): the user's query
        model (string): default set to "gpt-4-vision-preview"

    Return:
        text as an answer to the user's query.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.session_state.openai_api_key}"
    }

    payload = {
        "model": f"{model}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{query}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    try:
        with st.spinner("AI is thinking..."):
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
        generated_text = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return generated_text

def chat_with_image(model):
    with st.sidebar:
        sources = ("From URL", "Uploaded")
        st.session_state.image_source[0] = st.radio(
            label="Image Source :",
            options=sources,
            index=sources.index(st.session_state.image_source[1]),
            label_visibility="visible",
        )

    st.write("")
    st.write("##### Image to ask about")
    st.write("")

    if st.session_state.image_source[0] == "From URL":
        # Enter a URL
        st.write("###### :blue[Enter the URL of your image]")

        image_url = st.text_input(
            label="URL of the image",
            label_visibility="collapsed",
            value="https://cdn.pixabay.com/photo/2013/02/01/18/14/url-77169_1280.jpg",
            on_change=reset_qna_image
        )
        if image_url:
            if is_url(image_url):
                st.session_state.uploaded_image = image_url
            else:
                st.error("Enter a proper URL", icon="🚨")
    else:
        # Upload an image file
        st.write("###### :blue[Upload your image]")

        image_file = st.file_uploader(
            label="High resolution images will be resized.",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            on_change=reset_qna_image,
        )
        if image_file is not None:
            # Process the uploaded image file
            try:
                image = Image.open(image_file)
                st.session_state.uploaded_image = shorten_image(image, 1024)
            except UnidentifiedImageError as e:
                st.error(f"An error occurred: {e}", icon="🚨")

    # Capture the user's query and provide a response if the image is ready
    if st.session_state.uploaded_image:
        st.image(image=st.session_state.uploaded_image, use_column_width=True)

        # Print query & answer
        if st.session_state.qna["question"] and st.session_state.qna["answer"]:
            with st.chat_message("human"):
                st.write(st.session_state.qna["question"])
            with st.chat_message("ai"):
                st.write(st.session_state.qna["answer"])

        # Use your keyboard
        query = st.chat_input(
            placeholder="Enter your query",
        )
        if query:
            st.session_state.qna["question"] = query
            st.session_state.prompt_exists = True

        if st.session_state.prompt_exists:
            if st.session_state.image_source[0] == "From URL":
                generated_text = openai_query_image_url(
                    image_url=st.session_state.uploaded_image,
                    query=st.session_state.qna["question"],
                    model=model
                )
            else:
                generated_text = openai_query_uploaded_image(
                    image_b64=image_to_base64(st.session_state.uploaded_image),
                    query=st.session_state.qna["question"],
                    model=model
                )

            st.session_state.prompt_exists = False
            if generated_text is not None:
                st.session_state.qna["answer"] = generated_text
                st.rerun()

def create_image_by_prompt(model):
    # Set the image size
    with st.sidebar:
        image_size = st.radio(
            label="Pixel size",
            options=("1024x1024", "1024x1792", "1792x1024"),
            index=0,
            label_visibility="visible",
            # horizontal=True,
            on_change=reset_qna_image,
        )

    st.write("")
    st.write("##### Description for your image")

    if st.session_state.image_url is not None:
        st.info(st.session_state.image_description)
        st.image(image=st.session_state.image_url, use_column_width=True)

    # Get an image description using the keyboard
    text_input = st.chat_input(
        placeholder="Enter a description for your image",
    )
    if text_input:
        st.session_state.image_description = text_input
        st.session_state.prompt_exists = True

    if st.session_state.prompt_exists:
        st.session_state.image_url = openai_create_image(
            st.session_state.image_description, model, image_size
        )
        st.session_state.prompt_exists = False
        if st.session_state.image_url is not None:
            st.rerun()
            

def run_image():
    with st.sidebar:
        st.write("")
        st.write("**Image Models**")
        optImageModel = st.sidebar.radio(
            label="not display",
            options=(
                "dall-e-3",
                "gpt-4-vision-preview"
            ),
            label_visibility="collapsed",
            key="llm_imagemodel",
            index=0,
            # horizontal=True,
        )

    if optImageModel == "dall-e-3":
        create_image_by_prompt(optImageModel)
    else:
        chat_with_image(optImageModel)
           
if __name__ == "__main__":
    if "openai_api_key" not in st.session_state:
        switch_page('Home')

    run_image()