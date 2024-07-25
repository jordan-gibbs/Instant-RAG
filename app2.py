import streamlit as st
import os
import json
import tempfile
from datetime import datetime
from typing_extensions import override
from openai import OpenAI, AssistantEventHandler
import base64
import re
from pypdf import PdfWriter, PdfReader

st.set_page_config(
    page_title="Instant RAG",
    page_icon="✨",
    layout="wide"
)

hide_streamlit_style = """
            <style>
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize the OpenAI client
client = OpenAI()

current_datetime = datetime.now()
date = current_datetime.strftime("%Y-%m-%d")

# Initialize first_message in session state
if "first_message" not in st.session_state:
    st.session_state.first_message = True

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = {}

if "project_description" not in st.session_state:
    st.session_state.project_description = None

if "compressed_file" not in st.session_state:
    st.session_state.compressed_file = None

if "previous_selected_file" not in st.session_state:
    st.session_state.previous_selected_file = None

# Function to normalize file paths
def normalize_path(path):
    return os.path.normpath(path)

def process_uploaded_files(uploaded_files):
    vector_store_dict = {}
    valid_file_streams = []

    for uploaded_file in uploaded_files:
        try:
            file_name = uploaded_file.name
            with open(file_name, "wb") as f:
                f.write(uploaded_file.getvalue())
            valid_file_streams.append(open(file_name, "rb"))
            st.session_state.uploaded_files_info[uploaded_file.name] = file_name
        except Exception as e:
            st.error(f"Unexpected error with file {uploaded_file.name}: {e}")

    if not valid_file_streams:
        st.error("No valid files found in uploaded_files, skipping...")
        return

    # Create the vector store
    vector_store = client.beta.vector_stores.create(name="uploaded_files")

    # Define the chunking strategy
    chunking_strategy = {
        "type": "static",
        "static": {
            "max_chunk_size_tokens": 2048,
            "chunk_overlap_tokens": 1024
        }
    }

    # Upload and poll the file batch
    try:
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=valid_file_streams,
            chunking_strategy=chunking_strategy
        )
    except Exception as e:
        st.error(f"Error uploading files for {vector_store.name}: {e}")
        return
    finally:
        # Close the file streams
        for stream in valid_file_streams:
            stream.close()

    # Add the vector store name and ID to the dictionary
    vector_store_dict[vector_store.name] = vector_store.id

    return vector_store_dict

# Streamlit app interface
st.title("✨Instant RAG")
st.subheader("Create a retrieval augmented generation (RAG) chatbot with your own data - instantly.")

uploaded_files = st.file_uploader("Upload PDF files to chat with", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.session_state.vector_db is None:
    with st.spinner(text="Vectorizing documents..."):
        vector_store_dict = process_uploaded_files(uploaded_files)
        if vector_store_dict:
            st.session_state.vector_db = vector_store_dict['uploaded_files']
        else:
            st.error("Failed to vectorize documents. Please ensure they are OCRd.")

project_description = True
if project_description:
    st.session_state.project_description = project_description

if st.session_state.vector_db and st.session_state.project_description:
    prompt_injection = f""
    prompt_injection_v1 = f""

    # Function to compress images in PDF
    def compress_pdf(input_pdf_path, output_pdf_path):
        writer = PdfWriter(clone_from=input_pdf_path)
        # Remove all images
        writer.remove_images()
        # Apply lossless compression
        for page in writer.pages:
            page.compress_content_streams(level=9)

        with open(output_pdf_path, "wb") as f:
            writer.write(f)

    def read_pdf(file_path):
        with open(file_path, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode('utf-8')
        return encoded_string

    # Main application logic
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)  # Spacer
        selected_file = st.selectbox("Select a document to preview",
                                     options=list(st.session_state.uploaded_files_info.keys()))
        # Check if the selected file has changed
        if selected_file != st.session_state.previous_selected_file:
            st.session_state.previous_selected_file = selected_file
            st.session_state.compressed_file = None  # Reset compressed file

        if selected_file is not None and st.session_state.compressed_file is None:
            file_path = st.session_state.uploaded_files_info[selected_file]

            # Compress the PDF before displaying
            with st.spinner("Loading PDF Preview"):
                compressed_pdf_path = "compressed_example.pdf"
                compress_pdf(file_path, compressed_pdf_path)
                base64_pdf = read_pdf(compressed_pdf_path)
                st.session_state.compressed_file = base64_pdf

        if st.session_state.compressed_file:
            pdf_display = f'''
                   <iframe src="data:application/pdf;base64,{st.session_state.compressed_file}" 
                           width="100%" height="1000" 
                           type="application/pdf"></iframe>
                   '''
            st.markdown(pdf_display, unsafe_allow_html=True)

    with col2:
        # Chat input container at the top
        with st.container():
            st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
            st.subheader("📄Chat with Documents")
            st.markdown("---")
            prompt = st.chat_input(placeholder="Ask anything about your PDF files")
            st.markdown('</div>', unsafe_allow_html=True)

        if prompt:
            # Add user message to chat history
            if st.session_state.first_message:
                st.session_state.messages.append({"role": "user", "content": f"{prompt}{prompt_injection_v1}"})
                st.session_state.first_message = False
            else:
                st.session_state.messages.append({"role": "user", "content": f"{prompt}{prompt_injection}"})

            # Create the assistant with file search capability
            assistant = client.beta.assistants.create(
                instructions=f"You are a document analyst for {st.session_state.project_description}. You only have one dataset available to you, which is a collection of PDFs. You will use this knowledge base to answer questions that the user has about this dataset."
                             f"You will never output anything that isn't directly available in the documents attached to you. The date is {date}.",
                model="gpt-4o",
                tools=[{"type": "file_search"}],
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [f"{st.session_state.vector_db}"]
                    }
                }
            )

            # Create a new thread for each user interaction
            thread = client.beta.threads.create(
                messages=st.session_state.messages,
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [f"{st.session_state.vector_db}"]
                    }
                }
            )


            class EventHandler(AssistantEventHandler):
                @override
                def on_text_created(self, text) -> None:
                    print(f"\nassistant > ", end="", flush=True)

                @override
                def on_tool_call_created(self, tool_call):
                    print(f"\nassistant > {tool_call.type}\n", flush=True)

                @override
                def on_message_done(self, message) -> None:
                    import urllib.parse

                    # Process the assistant's message to handle annotations and citations
                    message_content = message.content[0].text
                    message_text = message_content.value
                    print(f"Original message content: {message_text}")

                    annotations = message_content.annotations

                    # Process each annotation
                    for index, annotation in enumerate(annotations):
                        index += 1  # Adjust for 1-based citation indexing

                        # Handle file citations
                        if file_citation := getattr(annotation, "file_citation", None):
                            cited_file = client.files.retrieve(file_citation.file_id)
                            filename = st.session_state.uploaded_files_info.get(cited_file.filename, cited_file.filename)
                            message_text = re.sub(r"【.*?】", f" [**{filename}**]", message_text)

                    # Escape dollar signs in the message text
                    escaped_content = message_text.replace("$", "\\$")

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": escaped_content})

            # Generate assistant response with event handling
            with st.spinner('Thinking...'):
                with client.beta.threads.runs.stream(
                        thread_id=thread.id,
                        assistant_id=assistant.id,
                        instructions=f"The user has a premium account. The user did not upload the documents, it's a database. The date is {date}. Please always be exact with your output.",
                        event_handler=EventHandler(),
                ) as stream:
                    stream.until_done()

        # Display chat messages from history on app rerun
        for message in reversed(st.session_state.messages):
            cleaned_content = message["content"].replace(prompt_injection, "")
            cleaned_content = cleaned_content.replace(prompt_injection_v1, "")
            with st.chat_message(message["role"]):
                st.markdown(cleaned_content)
