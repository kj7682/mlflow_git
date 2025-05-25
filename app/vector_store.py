import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings # Using SentenceTransformerEmbeddings from Langchain

# Default model for embeddings, consistent with pdf_processor.py
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# Default dimension for 'all-MiniLM-L6-v2'
DEFAULT_EMBEDDING_DIM = 384

class VectorStoreManager:
    """
    Manages a FAISS vector index for storing and searching text embeddings.
    This implementation uses Langchain's FAISS wrapper for easier management
    of text chunks and embeddings.
    """

    def __init__(self,
                 index_path: str = "vector_db/document_index.faiss",
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initializes the VectorStoreManager.

        Args:
            index_path: Path to save/load the FAISS index. The associated .pkl file for
                        docstore and index_to_docstore_id will be in the same directory.
            embedding_model_name: The name of the sentence-transformer model to use for embeddings.
        """
        self.index_path = index_path
        self.index_directory = os.path.dirname(index_path)
        self.index_name = os.path.splitext(os.path.basename(index_path))[0] # e.g., "document_index"

        # Initialize the embedding function
        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        
        # The FAISS index object, will be loaded or created
        self.faiss_index: Optional[FAISS] = None

    def build_or_load_store(self, text_chunks: Optional[List[str]] = None) -> None:
        """
        Loads an existing FAISS index from disk if available, otherwise builds a new one
        from the provided text chunks and saves it.

        Args:
            text_chunks: A list of text chunks to build a new store. If None,
                         tries to load an existing store. If store doesn't exist and
                         text_chunks is None, an error will be raised.
        """
        if os.path.exists(self.index_path):
            print(f"Loading existing FAISS index from {self.index_directory} with index name {self.index_name}")
            try:
                self.faiss_index = FAISS.load_local(
                    folder_path=self.index_directory,
                    embeddings=self.embedding_function,
                    index_name=self.index_name,
                    allow_dangerous_deserialization=True # Required for Langchain FAISS loading
                )
                print("Successfully loaded FAISS index.")
            except Exception as e:
                print(f"Error loading FAISS index: {e}. A new index will be created if text_chunks are provided.")
                self.faiss_index = None # Ensure index is None if loading failed
                if text_chunks:
                    print("Building new store as loading failed.")
                    self._build_new_store(text_chunks)
                else:
                    print("Cannot build new store as no text_chunks provided after loading failed.")
        elif text_chunks:
            print("No existing FAISS index found. Building a new one.")
            self._build_new_store(text_chunks)
        else:
            print("No existing FAISS index found and no text_chunks provided to build a new one.")
            # self.faiss_index remains None

    def _build_new_store(self, text_chunks: List[str]) -> None:
        """
        Helper function to build a new FAISS index from text chunks and save it.

        Args:
            text_chunks: A list of text chunks.
        """
        if not text_chunks:
            print("Cannot build store: No text chunks provided.")
            return
        try:
            print(f"Building new FAISS index with {len(text_chunks)} text chunks.")
            # FAISS.from_texts will generate embeddings using self.embedding_function
            self.faiss_index = FAISS.from_texts(texts=text_chunks, embedding=self.embedding_function)
            self.save_store()
            print(f"Successfully built and saved new FAISS index at {self.index_directory} with name {self.index_name}")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            self.faiss_index = None

    def add_texts_to_store(self, new_text_chunks: List[str]) -> None:
        """
        Adds new text chunks to an existing FAISS index.
        If no index is loaded or built yet, it will build a new one with these chunks.

        Args:
            new_text_chunks: A list of new text chunks to add.
        """
        if not new_text_chunks:
            print("No new text chunks to add.")
            return

        if not self.faiss_index:
            print("No existing index. Building a new store with the provided texts.")
            self._build_new_store(new_text_chunks)
        else:
            print(f"Adding {len(new_text_chunks)} new text chunks to the existing FAISS index.")
            try:
                # Langchain's FAISS `add_texts` handles embedding generation
                self.faiss_index.add_texts(texts=new_text_chunks)
                self.save_store() # Save after adding
                print("Successfully added texts and saved the updated FAISS index.")
            except Exception as e:
                print(f"Error adding texts to FAISS index: {e}")

    def save_store(self) -> None:
        """
        Saves the FAISS index and its associated mapping to disk.
        """
        if self.faiss_index:
            try:
                if not os.path.exists(self.index_directory):
                    os.makedirs(self.index_directory, exist_ok=True)
                self.faiss_index.save_local(folder_path=self.index_directory, index_name=self.index_name)
                print(f"FAISS index saved to {self.index_directory} with index name {self.index_name}")
            except Exception as e:
                print(f"Error saving FAISS index: {e}")
        else:
            print("No FAISS index to save.")

    def search(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches the FAISS index with a query text to find the top k
        most similar text chunks.

        Args:
            query_text: The query text.
            k: The number of similar chunks to retrieve.

        Returns:
            A list of tuples, where each tuple contains (text_chunk, similarity_score).
            Returns an empty list if the index is not initialized or search fails.
        """
        if not self.faiss_index:
            print("FAISS index is not initialized. Cannot perform search.")
            return []
        try:
            # The `similarity_search_with_score` method returns documents and their L2 distance.
            # Smaller distance means more similar.
            results_with_scores = self.faiss_index.similarity_search_with_score(query=query_text, k=k)
            # Format to (text_chunk, score)
            return [(doc.page_content, score) for doc, score in results_with_scores]
        except Exception as e:
            print(f"Error during search in FAISS index: {e}")
            return []

if __name__ == '__main__':
    # Example Usage (Conceptual - requires actual text and embeddings)

    # 0. Instantiate the manager
    # This will use the default 'vector_db/document_index.faiss' and 'all-MiniLM-L6-v2'
    vector_store_manager = VectorStoreManager()

    # 1. Example Text Chunks (In a real scenario, these come from pdf_processor.py)
    sample_chunks = [
        "The quick brown fox jumps over the lazy dog.",
        "Langchain provides tools for building LLM applications.",
        "FAISS is a library for efficient similarity search.",
        "Embeddings are numerical representations of text.",
        "This is a test document for the vector store."
    ]
    
    # Ensure the vector_db directory exists for the example
    if not os.path.exists("vector_db"):
        os.makedirs("vector_db")

    # 2. Build or Load Store
    # If "vector_db/document_index.faiss" and "vector_db/document_index.pkl" exist, it will load.
    # Otherwise, it will build a new store with sample_chunks.
    print("\n--- Attempting to build or load store ---")
    vector_store_manager.build_or_load_store(text_chunks=sample_chunks)

    if vector_store_manager.faiss_index:
        # 3. Search the store
        print("\n--- Searching the store ---")
        query = "What is FAISS?"
        search_results = vector_store_manager.search(query_text=query, k=2)
        if search_results:
            print(f"Search results for '{query}':")
            for i, (chunk, score) in enumerate(search_results):
                print(f"{i+1}. Chunk: \"{chunk}\" (Score: {score:.4f})")
        else:
            print(f"No results found for '{query}'.")

        # 4. Add more documents (optional)
        print("\n--- Adding new documents to the store ---")
        new_chunks = [
            "Another document about vector databases.",
            "Exploring advanced features of FAISS."
        ]
        vector_store_manager.add_texts_to_store(new_text_chunks=new_chunks)
        
        # 5. Search again after adding new documents
        print("\n--- Searching again after adding new documents ---")
        query_2 = "Tell me about vector databases"
        search_results_2 = vector_store_manager.search(query_text=query_2, k=2)
        if search_results_2:
            print(f"Search results for '{query_2}':")
            for i, (chunk, score) in enumerate(search_results_2):
                print(f"{i+1}. Chunk: \"{chunk}\" (Score: {score:.4f})")
        else:
            print(f"No results found for '{query_2}'.")

        # The index is automatically saved when built or when texts are added.
        # You can also call save_store() explicitly if needed, but it's integrated.
        # vector_store_manager.save_store()
    else:
        print("\nFailed to initialize or load the vector store. Example usage cannot proceed.")

    # To clean up the created files for the example:
    # print("\n--- Cleaning up example files ---")
    # example_index_path = "vector_db/document_index.faiss"
    # example_pkl_path = "vector_db/document_index.pkl"
    # if os.path.exists(example_index_path):
    #     os.remove(example_index_path)
    #     print(f"Removed {example_index_path}")
    # if os.path.exists(example_pkl_path):
    #     os.remove(example_pkl_path)
    #     print(f"Removed {example_pkl_path}")
    # if os.path.exists("vector_db") and not os.listdir("vector_db"):
    #      os.rmdir("vector_db")
    #      print("Removed empty vector_db directory")
