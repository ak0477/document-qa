import streamlit as st
from openai import OpenAI

# create multi page
lab1_page = st.Page("HW1.py", title = "Lab 1", icon = ":material/thumb_up:")
lab2_page = st.Page("lab2.py", title = "Lab 2", icon = ":material/thumb_up:")
lab3_page = st.Page("lab3.py", title = "Lab 3", icon = ":material/thumb_up:")

pg = st.navigation([ lab1_page, lab2_page, lab3_page])
st.set_page_config(page_title= "My Labs", page_icon= ":material/home:", layout = 'wide', initial_sidebar_state = "expanded")
st.title("My Labs")
pg.run()

st.set_page_config
# Show title and description.
st.title("My Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["openai"]['api_key']
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)
