import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables (e.g., for OPENAI_API_KEY)
# This should be called as early as possible, especially before importing modules that might use these variables.
load_dotenv()

from app.vector_store import VectorStoreManager
from app.chatbot_logic import ChatLogic
from app.pdf_processor import extract_text_from_pdf, chunk_text # Assuming these are directly importable

# --- Configuration & Global Initialization ---

# Ensure DATA_DIR for temporary PDF storage exists
DATA_DIR = "data/uploaded_pdfs"
os.makedirs(DATA_DIR, exist_ok=True)

# Define the path for the FAISS index.
# VectorStoreManager expects a full path to the .faiss file, e.g., "vector_db/my_index.faiss"
# It will then use "vector_db" as the folder_path and "my_index" as the index_name.
VECTOR_DB_INDEX_FILE_PATH = "vector_db/main_chatbot_index.faiss" 
# Ensure the parent directory for the FAISS store exists
os.makedirs(os.path.dirname(VECTOR_DB_INDEX_FILE_PATH), exist_ok=True)

# Initialize VectorStoreManager
# The embedding function ('all-MiniLM-L6-v2' by default) is part of VectorStoreManager.
print(f"Initializing VectorStoreManager with index path: {VECTOR_DB_INDEX_FILE_PATH}")
vector_store_manager = VectorStoreManager(index_path=VECTOR_DB_INDEX_FILE_PATH)

# Try to load an existing store. If it doesn't exist, or if loading fails and
# no texts are provided, self.faiss_index in vector_store_manager will be None.
# It will be created/populated upon the first PDF upload via add_texts_to_store.
print("Attempting to load existing vector store...")
vector_store_manager.build_or_load_store() # Initialize by trying to load

# Initialize ChatLogic
# ChatLogic uses OPENAI_API_KEY from .env for its LLM.
print("Initializing ChatLogic...")
chat_logic = ChatLogic(vector_store_manager=vector_store_manager)

app = FastAPI(
    title="Construction RAG Chatbot API",
    description="API for uploading PDF documents and chatting with an AI assistant knowledgeable about their content.",
    version="1.0.0"
)

# --- API Endpoints ---

@app.post("/upload-pdf/", summary="Upload PDF files to the knowledge base")
async def api_upload_pdf(files: List[UploadFile] = File(..., description="List of PDF files to upload.")):
    """
    Uploads one or more PDF files. The text content is extracted, chunked,
    and added to the vector knowledge base.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    processed_files_info = []
    any_new_info_added = False

    for file in files:
        if file.content_type != "application/pdf":
            print(f"Skipping non-PDF file: {file.filename} (Type: {file.content_type})")
            processed_files_info.append({"filename": file.filename, "status": "skipped_invalid_type"})
            continue # Skip to the next file
        
        # Sanitize filename to prevent directory traversal issues, though not strictly necessary here
        # as we define the save path directly.
        safe_filename = os.path.basename(file.filename or "unknown.pdf")
        temp_pdf_path = os.path.join(DATA_DIR, safe_filename)
        
        try:
            print(f"Processing file: {safe_filename}")
            # Save the uploaded file temporarily
            with open(temp_pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"Temporarily saved {safe_filename} to {temp_pdf_path}")

            text_content = extract_text_from_pdf(temp_pdf_path)
            if not text_content:
                print(f"No text content extracted from {safe_filename}.")
                processed_files_info.append({"filename": safe_filename, "status": "skipped_no_text"})
                continue # Skip to the next file if no text

            text_chunks = chunk_text(text_content) # Uses default chunking params

            if text_chunks:
                print(f"Extracted {len(text_chunks)} text chunks from {safe_filename}.")
                # VectorStoreManager's add_texts_to_store handles embedding and saving
                vector_store_manager.add_texts_to_store(text_chunks)
                any_new_info_added = True
                processed_files_info.append({"filename": safe_filename, "status": "processed_and_added", "chunks": len(text_chunks)})
                print(f"Successfully processed and added chunks from {safe_filename} to vector store.")
            else:
                print(f"No text chunks generated from {safe_filename} after extraction.")
                processed_files_info.append({"filename": safe_filename, "status": "skipped_no_chunks"})

        except Exception as e:
            print(f"Error processing file {safe_filename}: {e}")
            # Raising HTTPException will stop processing further files if one fails.
            # Depending on desired behavior, you might want to log and continue.
            # For now, let's inform about the specific file and stop.
            raise HTTPException(status_code=500, detail=f"Error processing file {safe_filename}: {str(e)}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path) # Clean up temp file
                print(f"Cleaned up temporary file: {temp_pdf_path}")
            await file.close() # Close the upload file

    if not any_new_info_added and not processed_files_info:
         return {"message": "No files were provided or processed."}
    elif not any_new_info_added:
        return {"message": "PDF(s) processed, but no new textual information was extracted or added to the knowledge base.", "details": processed_files_info}

    return {"message": "PDF(s) processed successfully.", "details": processed_files_info}


# Pydantic model for chat request
from pydantic import BaseModel
class ChatQuery(BaseModel):
    query: str
    top_k_contexts: Optional[int] = 3 # Default to 3 contexts

@app.post("/chat/", summary="Get an answer from the chatbot")
async def api_chat(query_data: ChatQuery = Body(..., example={"query": "What are the safety regulations for crane operation?"})):
    """
    Receives a user query, retrieves relevant context from the knowledge base,
    and returns an LLM-generated answer.
    """
    user_query = query_data.query
    top_k = query_data.top_k_contexts

    if not user_query:
        raise HTTPException(status_code=400, detail="Query not provided.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k_contexts must be a positive integer.")

    print(f"Received query: '{user_query}' with top_k_contexts: {top_k}")
    
    # Ensure OPENAI_API_KEY is available for ChatLogic
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured on the server.")

    # Ensure vector store is loaded in ChatLogic's VectorStoreManager
    if not chat_logic.vector_store_manager or not chat_logic.vector_store_manager.faiss_index:
        # This might happen if the initial load failed and no PDFs have been uploaded yet.
        # Or if the path was incorrect and build_or_load_store did not result in a usable index.
        return {"answer": "The knowledge base is not yet initialized or is empty. Please upload PDF documents first."}

    answer = chat_logic.get_answer(user_query, top_k_contexts=top_k)
    return {"answer": answer}

# Basic root endpoint for testing
@app.get("/", summary="Root endpoint to check API status")
async def root():
    """
    A simple endpoint to confirm that the API is running.
    """
    return {"message": "Construction RAG Chatbot API is running. Use /upload-pdf/ to add documents and /chat/ to ask questions."}

# To run this application:
# 1. Ensure you have a .env file in the project root with your OPENAI_API_KEY.
#    Example .env content:
#    OPENAI_API_KEY="your_openai_api_key_here"
#
# 2. Install dependencies: pip install -r requirements.txt
#
# 3. Run with Uvicorn:
#    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#
# Example curl commands:
#
# Upload PDF:
# curl -X POST -F "files=@/path/to/your/document1.pdf" -F "files=@/path/to/your/document2.pdf" http://localhost:8000/upload-pdf/
#
# Chat:
# curl -X POST -H "Content-Type: application/json" -d '{"query": "What is the process for concrete curing?"}' http://localhost:8000/chat/
#
# Chat with custom top_k:
# curl -X POST -H "Content-Type: application/json" -d '{"query": "What are safety risks?", "top_k_contexts": 5}' http://localhost:8000/chat/

if __name__ == "__main__":
    # This part is for direct execution (though uvicorn is preferred for development/production)
    import uvicorn
    print("Starting Uvicorn server for app.main from __main__ block...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
