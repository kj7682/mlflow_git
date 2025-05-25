# PDF Chatbot for Construction Companies

## Overview

This project is a Retrieval Augmented Generation (RAG) chatbot designed for construction companies. It allows users to upload PDF documents (e.g., technical specifications, safety manuals, project plans) and ask questions related to their content. The chatbot leverages these documents as a knowledge base to provide informed answers.

The system extracts text from PDFs, chunks it, generates embeddings, stores them in a vector database, and uses this information alongside a Large Language Model (LLM) to answer user queries. It features a FastAPI backend for processing and chat logic, and a Streamlit UI for user interaction.

## Features

*   **PDF Processing:** Extracts text from uploaded PDF documents.
*   **Text Chunking:** Splits extracted text into manageable chunks for embedding.
*   **Embedding Generation:** Uses sentence-transformer models to convert text chunks into vector embeddings.
*   **Vector Storage:** Stores text embeddings in a FAISS vector database for efficient similarity search.
*   **Contextual Retrieval:** Retrieves relevant text chunks from the vector store based on user queries.
*   **LLM Integration:** Uses a Large Language Model (e.g., GPT-3.5-turbo from OpenAI) to generate answers based on the retrieved context.
*   **FastAPI Backend:** Provides API endpoints for PDF upload and chat functionalities.
*   **Streamlit Web UI:** Offers a user-friendly interface for uploading documents and interacting with the chatbot.
*   **Environment Variable Management:** Securely manages API keys and configurations using `.env` files.
*   **Unit Tested Core Components:** Core PDF processing logic is unit tested using `pytest`.

## Project Structure

```
.
├── app/                       # Core application logic
│   ├── __init__.py
│   ├── chatbot_logic.py       # Handles chat orchestration (retrieval, LLM call)
│   ├── main.py                # FastAPI application, API endpoints
│   ├── pdf_processor.py       # PDF text extraction, chunking, embedding generation
│   └── vector_store.py        # Manages the FAISS vector store
├── data/                      # Stores temporary data like uploaded PDFs
│   └── uploaded_pdfs/         # Temporary storage for PDFs during processing
│   └── .gitkeep
├── vector_db/                 # Stores the FAISS index files
│   └── .gitkeep
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_pdf_processor.py  # Tests for pdf_processor.py
│   └── .gitkeep
├── .env.example               # Example environment file (YOU NEED TO CREATE .env)
├── requirements.txt           # Python dependencies
├── streamlit_app.py           # Streamlit UI application
└── README.md                  # This file
```

*   **`app/`**: Contains the core backend logic.
    *   `pdf_processor.py`: Functions for PDF parsing, text chunking, and embedding generation.
    *   `vector_store.py`: Manages the FAISS vector database.
    *   `chatbot_logic.py`: Orchestrates context retrieval and LLM response generation.
    *   `main.py`: Defines the FastAPI application and its endpoints.
*   **`data/uploaded_pdfs/`**: Temporarily stores PDF files uploaded by the user during processing. This directory is created automatically.
*   **`vector_db/`**: Stores the FAISS index files, which persist the knowledge base. This directory is created automatically.
*   **`tests/`**: Contains unit tests for the application modules.
*   **`streamlit_app.py`**: The Streamlit front-end application.
*   **`requirements.txt`**: Lists all Python dependencies.
*   **`.env.example`**: An example file to guide the creation of a `.env` file for environment variables. **You must create your own `.env` file.**

## Setup Instructions

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Install all required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    You need to configure environment variables, primarily your OpenAI API key.
    *   Create a file named `.env` in the project root directory. You can copy `.env.example` (if provided in the repo, otherwise create it from scratch).
    *   Add your OpenAI API key to the `.env` file:
        ```env
        OPENAI_API_KEY="your_actual_openai_api_key_here"
        ```
    *   (Optional) If you plan to run the FastAPI backend on a URL different from the default `http://localhost:8000` and want the Streamlit app to connect to it, you can set `FASTAPI_URL` in the same `.env` file:
        ```env
        FASTAPI_URL="http://your_backend_host:your_backend_port"
        ```
        The Streamlit app will use this if set, otherwise, it defaults to `http://localhost:8000`.

## Running the Application

The application consists of two main parts: the FastAPI backend and the Streamlit UI. Both need to be running simultaneously for the chatbot to be fully operational.

1.  **Run the FastAPI Backend:**
    Open a terminal, activate your virtual environment, and navigate to the project root. Run the following command:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Enables auto-reloading when code changes (useful for development).
    *   `--host 0.0.0.0`: Makes the server accessible from your local network.
    *   `--port 8000`: Specifies the port (default is 8000).

    You should see output indicating the Uvicorn server is running and that it has loaded your modules (e.g., VectorStoreManager, ChatLogic).

2.  **Run the Streamlit UI:**
    Open a *new* terminal (or use the current one if you ran the backend in the background), activate your virtual environment, and navigate to the project root. Run:
    ```bash
    streamlit run streamlit_app.py
    ```
    This will typically open the Streamlit application in your default web browser. If not, the terminal output will provide the URL (usually `http://localhost:8501`).

## How to Use

1.  **Access the Streamlit UI:** Open your web browser and navigate to the Streamlit application URL (e.g., `http://localhost:8501`).
2.  **Upload PDF Documents:**
    *   In the sidebar of the Streamlit UI, you'll find a "Upload PDF Files Here" section.
    *   Click "Browse files" or drag and drop your PDF documents into the uploader. You can upload multiple files at once.
    *   After selecting your files, click the "Process Uploaded PDFs" button.
    *   Wait for the processing to complete. The UI will show status messages, including details for each processed file.
3.  **Ask Questions:**
    *   Once your documents are processed and added to the knowledge base, you can ask questions using the chat input field at the bottom of the main interface.
    *   Type your question and press Enter.
    *   The chatbot will retrieve relevant information from your uploaded documents and generate an answer using the LLM.
4.  **Adjust Settings (Optional):**
    *   In the sidebar, under "Query Settings," you can use the slider to adjust the "Number of relevant document snippets (top_k)" that the chatbot considers when forming an answer.

## API Endpoints

The FastAPI backend exposes the following main endpoints (useful for programmatic interaction or alternative clients):

*   **`POST /upload-pdf/`**:
    *   **Description:** Uploads one or more PDF files. The text content is extracted, chunked, embedded, and stored in the vector knowledge base.
    *   **Request Body:** `multipart/form-data` with one or more `files` fields, each containing a PDF file.
    *   **Response:** JSON object with a status message and details about processed files (e.g., `{"message": "PDF(s) processed successfully.", "details": [{"filename": "doc1.pdf", "status": "processed_and_added", "chunks": 10}]}`).
*   **`POST /chat/`**:
    *   **Description:** Receives a user query, retrieves relevant context from the knowledge base, and returns an LLM-generated answer.
    *   **Request Body (JSON):**
        ```json
        {
            "query": "Your question here",
            "top_k_contexts": 3 // Optional, default is 3
        }
        ```
    *   **Response:** JSON object `{"answer": "The chatbot's response"}`.
*   **`GET /`**:
    *   **Description:** Root endpoint to check if the API is running.
    *   **Response:** JSON object `{"message": "Construction RAG Chatbot API is running..."}`.

## Testing

Unit tests are implemented using `pytest` for core components.

1.  **Ensure `pytest` is installed** (it's included in `requirements.txt`).
2.  **Navigate to the project root directory** in your terminal (ensure your virtual environment is activated).
3.  **Run tests using the command:**
    ```bash
    pytest
    ```
    Or, for more verbose output:
    ```bash
    pytest -v
    ```
    This will automatically discover and run tests located in the `tests/` directory (e.g., `tests/test_pdf_processor.py`).

---
*Developed for enhanced document interaction in the construction industry.*
