import streamlit as st
import requests
import os

# --- Configuration ---
# Default FastAPI URL, can be overridden by environment variable for flexibility
FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
UPLOAD_ENDPOINT = f"{FASTAPI_BASE_URL}/upload-pdf/"
CHAT_ENDPOINT = f"{FASTAPI_BASE_URL}/chat/"

# --- Helper Functions ---
def upload_pdfs_to_api(uploaded_files_list):
    """
    Sends uploaded PDF files to the FastAPI backend.
    """
    if not uploaded_files_list:
        st.warning("Please upload at least one PDF file.")
        return None
    
    # Prepare files for the 'requests' library.
    # Each file is a tuple: ('files', (filename, file_bytes, content_type))
    files_payload = []
    for uploaded_file in uploaded_files_list:
        files_payload.append(("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")))
    
    try:
        # Make the POST request
        # Increased timeout for potentially large files or slow processing
        response = requests.post(UPLOAD_ENDPOINT, files=files_payload, timeout=300) 
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json() # Return the JSON response from the API
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred during PDF upload: {http_err}")
        try:
            # Try to display more detailed error from server response
            st.error(f"Server response: {response.json().get('detail') if response else 'No response text'}")
        except Exception: # If parsing response JSON fails
            st.error(f"Server raw response: {response.text if response else 'No response text'}")
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Connection error: Could not connect to the backend at {UPLOAD_ENDPOINT}. Ensure the FastAPI server is running.")
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"Timeout error: The request to upload PDFs timed out. {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"An unexpected error occurred during PDF upload: {req_err}")
    return None

def ask_chatbot_api(query: str, top_k: int = 3):
    """
    Sends a query to the FastAPI backend and gets the chatbot's response.
    """
    if not query:
        st.warning("Please enter a question.")
        return None
    
    payload = {"query": query, "top_k_contexts": top_k}
    try:
        # Increased timeout for potentially slow LLM responses
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=180) 
        response.raise_for_status() # Raises an HTTPError for bad responses
        return response.json() # Return the JSON response (should contain "answer")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred while asking chatbot: {http_err}")
        try:
            st.error(f"Server response: {response.json().get('detail') if response else 'No response text'}")
        except Exception:
            st.error(f"Server raw response: {response.text if response else 'No response text'}")
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Connection error: Could not connect to the backend at {CHAT_ENDPOINT}. Ensure the FastAPI server is running.")
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"Timeout error: The request to the chatbot timed out. {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"An unexpected error occurred while asking chatbot: {req_err}")
    return None

# --- Streamlit App Layout ---

# Set page configuration (title, icon, layout)
st.set_page_config(page_title="Construction RAG Chatbot", layout="wide", page_icon="🏗️")

# Main title of the application
st.title("🏗️ Construction Company Document Chatbot")
st.markdown("Upload your construction-related PDF documents and ask questions. The chatbot will answer based on the content of your documents.")

# Initialize chat history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload your PDF documents using the sidebar, then ask me anything about them."}]

# --- Sidebar for PDF Upload and Settings ---
with st.sidebar:
    st.header("📄 Document Management")
    
    # File uploader allows multiple PDFs
    uploaded_pdf_files = st.file_uploader(
        "Upload PDF Files Here", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload one or more PDF documents that the chatbot will use as its knowledge base."
    )
    
    if st.button("Process Uploaded PDFs", key="process_pdfs_button", help="Click to process the uploaded PDFs and add them to the chatbot's knowledge base."):
        if uploaded_pdf_files:
            with st.spinner("Processing PDFs... This might take a few moments depending on file size and number."):
                api_response = upload_pdfs_to_api(uploaded_pdf_files)
                if api_response:
                    st.success(api_response.get("message", "PDFs processed successfully!"))
                    # You can also display more details from api_response.get("details", [])
                    if "details" in api_response:
                        for detail in api_response["details"]:
                            st.info(f"File: {detail['filename']}, Status: {detail['status']}, Chunks: {detail.get('chunks', 'N/A')}")
                # No explicit else here, as upload_pdfs_to_api handles error messages
        else:
            st.warning("No PDF files selected. Please upload files to process.")
            
    st.markdown("---") # Visual separator
    st.subheader("⚙️ Query Settings")
    top_k_contexts_slider = st.slider(
        "Number of relevant document snippets (top_k):", 
        min_value=1, 
        max_value=10, 
        value=3, 
        step=1,
        help="Controls how many relevant text chunks are retrieved from the documents to answer your question. Higher values may provide more context but can also introduce noise."
    )

# --- Main Chat Area ---
st.header("💬 Chat Interface")

# Display existing chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept new user input via chat_input
if user_prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Add user's message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for streaming-like effect
        message_placeholder.markdown("Thinking... 🤔")
        
        api_response = ask_chatbot_api(user_prompt, top_k=top_k_contexts_slider)
        
        if api_response and "answer" in api_response:
            assistant_response_text = api_response["answer"]
            message_placeholder.markdown(assistant_response_text)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response_text})
        else:
            # Fallback response if API call fails or response is malformed
            fallback_response_text = "Sorry, I encountered an issue and couldn't get a response. Please check the backend server or try again."
            message_placeholder.markdown(fallback_response_text)
            st.session_state.messages.append({"role": "assistant", "content": fallback_response_text})

# --- Instructions for Running ---
st.markdown("---")
st.subheader("How to Run This Application:")
st.markdown("""
1.  **Start the FastAPI Backend:**
    Open a terminal and navigate to the project root directory.
    Run the command: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
    Ensure the backend is running before using this Streamlit UI.

2.  **Run this Streamlit Application:**
    Open another terminal (or use the current one if the backend is running in the background).
    Navigate to the project root directory.
    Run the command: `streamlit run streamlit_app.py`

3.  **Set Environment Variables (Optional):**
    If your FastAPI backend is running on a different URL, set the `FASTAPI_URL` environment variable before running Streamlit.
    Example: `FASTAPI_URL=http://your-backend-url:port streamlit run streamlit_app.py`
""")

# --- Footer ---
st.markdown("---")
st.markdown("Developed for the Construction RAG Chatbot System.")
