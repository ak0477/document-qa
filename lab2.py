import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF


def read_pdf(uploaded_file) -> str:
    """Extract text from a PDF using PyMuPDF."""
    raw = uploaded_file.read()

    # Reset pointer so Streamlit can re-read file later if needed
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        doc = fitz.open(stream=raw, filetype="pdf")
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text).strip()
    except Exception as e:
        st.error(f"Error reading PDF with PyMuPDF: {e}")
        return ""


# Show title and description.
st.title("Lab 2 - My Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Extension check
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'txt':
            try:
                document = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                document = uploaded_file.read().decode("latin-1", errors="ignore")
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
            if not document:
                st.stop()
        else:
            st.error("Unsupported file type.")
            st.stop()

        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)
