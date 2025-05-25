import os
from dotenv import load_dotenv
from typing import List, Tuple
from app.vector_store import VectorStoreManager # Ensure VectorStoreManager is accessible
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, BaseMessage

# Instructions for Environment Variables:
# 1. Create a file named .env in the root directory of this project.
# 2. Add your OpenAI API key to the .env file like this:
#    OPENAI_API_KEY="your_actual_openai_api_key_here"
# 3. Ensure python-dotenv is installed (it's in requirements.txt).
# The ChatLogic class will automatically load this key.

class ChatLogic:
    """
    Handles user queries, retrieves relevant context from the vector store,
    and generates a response using an LLM.
    """

    def __init__(self, 
                 vector_store_manager: VectorStoreManager, 
                 llm_model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7):
        """
        Initializes the ChatLogic.

        Args:
            vector_store_manager: An instance of VectorStoreManager.
            llm_model_name: The name of the OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4").
            temperature: The temperature setting for the LLM (controls randomness).
        """
        load_dotenv() # Load environment variables from .env file
        self.vector_store_manager = vector_store_manager
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables. LLM calls will likely fail.")
            # You could raise an error here or allow it to fail at runtime
            # raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file or environment.")
        
        self.llm = ChatOpenAI(
            model_name=llm_model_name,
            temperature=temperature,
            openai_api_key=api_key # Pass the key explicitly
        )
        print(f"ChatLogic initialized with LLM: {llm_model_name}")

    def retrieve_contexts(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves relevant text chunks from the vector store based on the user query.

        Args:
            query: The user's query string.
            top_k: The number of relevant chunks to retrieve.

        Returns:
            A list of retrieved text chunks (strings).
        """
        if not self.vector_store_manager.faiss_index:
            print("Error: Vector store is not loaded or initialized in VectorStoreManager.")
            return []

        search_results = self.vector_store_manager.search(query_text=query, k=top_k)
        
        # Langchain's FAISS search can return Document objects or tuples (content, score)
        # We need to handle both cases to extract the text content.
        contexts = []
        for result in search_results:
            if hasattr(result, 'page_content'): # If it's a Document object
                contexts.append(result.page_content)
            elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str): # If it's (content, score)
                contexts.append(result[0])
            else:
                print(f"Warning: Unexpected search result format: {result}")
        return contexts

    def _create_prompt_messages(self, query: str, contexts: List[str]) -> List[BaseMessage]:
        """
        Creates a list of messages for the LLM, including system and human messages.
        """
        context_str = "\n---\n".join(contexts)
        
        system_content = (
            "You are a helpful assistant for a construction company. "
            "Answer the following question based *only* on the provided context. "
            "If the context doesn't contain the answer, clearly state that "
            "'I don't have enough information in the provided documents to answer that.' "
            "Do not try to make up an answer if it's not in the context."
        )
        
        human_content = (
            f"Context:\n"
            f"---\n"
            f"{context_str}\n"
            f"---\n\n"
            f"Question: {query}"
        )
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        return messages

    def generate_llm_response(self, query: str, contexts: List[str]) -> str:
        """
        Generates a response from the LLM based on the query and provided contexts.

        Args:
            query: The user's query.
            contexts: A list of context strings retrieved from the vector store.

        Returns:
            The LLM's generated response string.
        """
        if not contexts:
            return "I could not find any relevant information in the documents to form a basis for answering your question."

        messages = self._create_prompt_messages(query, contexts)
        
        try:
            print(f"\nSending prompt to LLM ({self.llm.model_name})...")
            # print("System Prompt:", messages[0].content) # For debugging
            # print("Human Prompt:", messages[1].content) # For debugging
            response = self.llm(messages)
            return response.content
        except Exception as e:
            # Check if the error is due to missing API key specifically
            if "api_key" in str(e).lower():
                 return "Sorry, I encountered an error: The OpenAI API key is missing or invalid. Please check your .env file or environment variables."
            print(f"Error calling LLM: {e}")
            return "Sorry, I encountered an error while trying to generate a response from the language model."

    def get_answer(self, query: str, top_k_contexts: int = 3) -> str:
        """
        Orchestrates retrieving contexts and generating an LLM response.

        Args:
            query: The user's query.
            top_k_contexts: The number of top contexts to retrieve.

        Returns:
            The final answer from the LLM or an error/info message.
        """
        print(f"\nProcessing query: '{query}'")
        print(f"Retrieving top {top_k_contexts} contexts...")
        contexts = self.retrieve_contexts(query, top_k=top_k_contexts)
        
        if not contexts:
            return "I could not find any relevant information in the documents to answer your question."
        
        print(f"Retrieved {len(contexts)} contexts:")
        for i, ctx in enumerate(contexts):
            print(f"  Context {i+1}: {ctx[:100]}...") # Print snippet of context

        return self.generate_llm_response(query, contexts)

if __name__ == '__main__':
    print("--- ChatLogic with LLM Integration Example ---")
    
    # INSTRUCTIONS FOR RUNNING THIS EXAMPLE:
    # 1. Make sure you have a .env file in the project's root directory.
    # 2. The .env file must contain your OpenAI API key:
    #    OPENAI_API_KEY="your_actual_openai_api_key_here"
    # 3. If the API key is missing or invalid, the LLM call will fail.

    # Check if OPENAI_API_KEY is set (primarily for user feedback here, ChatLogic handles loading)
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY is not set in your environment.")
        print("Please create a .env file in the root directory with OPENAI_API_KEY='your_key_here'")
        print("The LLM part of this example will likely fail without it.")
        # You could choose to exit here if the key is absolutely required for the demo to make sense.
        # For now, it will proceed and let ChatLogic handle the error.

    # Define the path for the example vector store
    example_index_path = "vector_db/example_llm_chat_index.faiss"
    example_index_directory = os.path.dirname(example_index_path)
    
    if not os.path.exists(example_index_directory):
        os.makedirs(example_index_directory, exist_ok=True)
        print(f"Created directory: {example_index_directory}")

    # 1. Initialize VectorStoreManager
    vsm = VectorStoreManager(index_path=example_index_path)
    print(f"VectorStoreManager initialized for index: {example_index_path}")

    # 2. Build or Load the Vector Store with Sample Data
    sample_texts_for_llm = [
        "The Eiffel Tower, located in Paris, France, was completed in 1889. It is one of the most recognizable landmarks in the world.",
        "The primary material used for the construction of the Eiffel Tower was wrought iron.",
        "Gustave Eiffel was the lead engineer for the Eiffel Tower project. His company specialized in metal structures like bridges.",
        "The tower is 330 meters tall and was the world's tallest man-made structure for 41 years.",
        "Initially, some artists and intellectuals criticized the design of the Eiffel Tower, but it has since become a global cultural icon."
    ]
    print("\n--- Building or Loading Vector Store for LLM Example ---")
    vsm.build_or_load_store(text_chunks=sample_texts_for_llm)

    if not vsm.faiss_index:
        print("Failed to build or load the vector store. Exiting LLM example.")
    else:
        print("Vector store is ready for LLM example.")
        
        # 3. Initialize ChatLogic with LLM
        # This will load OPENAI_API_KEY from .env if present
        chat_logic_instance = ChatLogic(vector_store_manager=vsm, llm_model_name="gpt-3.5-turbo")

        # 4. Get a full answer using the get_answer method
        query1 = "What material was used to build the Eiffel Tower?"
        print(f"\n--- Test Query 1: '{query1}' ---")
        answer1 = chat_logic_instance.get_answer(query=query1, top_k_contexts=2)
        print(f"\nLLM Answer for Query 1:\n{answer1}")

        query2 = "Who was the main engineer for the Eiffel Tower?"
        print(f"\n--- Test Query 2: '{query2}' ---")
        answer2 = chat_logic_instance.get_answer(query=query2, top_k_contexts=2)
        print(f"\nLLM Answer for Query 2:\n{answer2}")

        query3 = "What is the capital of Germany?" # This should not be in the context
        print(f"\n--- Test Query 3 (Out of Context): '{query3}' ---")
        answer3 = chat_logic_instance.get_answer(query=query3, top_k_contexts=2)
        print(f"\nLLM Answer for Query 3:\n{answer3}")
        
        query4 = "Tell me about Gustave Eiffel's company."
        print(f"\n--- Test Query 4: '{query4}' ---")
        answer4 = chat_logic_instance.get_answer(query=query4, top_k_contexts=1) # Only 1 context for this
        print(f"\nLLM Answer for Query 4:\n{answer4}")

    print("\n--- ChatLogic with LLM Integration Example End ---")
