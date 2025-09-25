import streamlit as st
from openai import OpenAI

# create multi page
lab1_page = st.Page("HW1.py", title = "Lab 1", icon = ":material/thumb_up:")
lab2_page = st.Page("lab2.py", title = "Lab 2", icon = ":material/thumb_up:")
lab3_page = st.Page("lab3.py", title = "Lab 3", icon = ":material/thumb_up:")
lab4_page = st.Page("lab4.py", title = "Lab 4", icon = ":material/thumb_up:")
lab5_page = st.Page("lab5.py", title = "Lab 5", icon = ":material/thumb_up:")
pg = st.navigation([ lab1_page, lab2_page, lab3_page, lab4_page, lab5_page])
st.set_page_config(page_title= "My Labs", page_icon= ":material/home:", layout = 'wide', initial_sidebar_state = "expanded")
st.title("My Labs")
pg.run()


# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["openai"]['api_key']
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

