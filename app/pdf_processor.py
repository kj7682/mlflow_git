# Import necessary libraries
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

# Define a function to extract text from a PDF file
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Loads a PDF file and extracts text from all pages.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A string containing the extracted text from the PDF.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Add empty string if page has no text
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Define a function to chunk text
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Chunks the extracted text into smaller pieces.

    Args:
        text: The text to be chunked.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        return []

# Define a function to generate embeddings for text chunks
def generate_embeddings(text_chunks: List[str], model_name: str = 'all-MiniLM-L6-v2') -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks using a pre-trained model.

    Args:
        text_chunks: A list of text chunks.
        model_name: The name of the sentence-transformer model to use.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(text_chunks, convert_to_tensor=False) # convert_to_tensor=False to get lists of floats
        return embeddings.tolist() # convert numpy array to list
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

if __name__ == '__main__':
    # This is an example of how to use the functions.
    # Create a dummy PDF file for testing (replace with an actual PDF path)
    # For this example, we'll skip actual PDF creation and use dummy text.
    sample_text = """This is a sample document. It has multiple sentences and paragraphs.
    The purpose of this document is to test the text processing functions.
    We will extract text, chunk it, and then generate embeddings.
    This is the second paragraph. It contains more information to make the document longer.
    Langchain's text splitters are useful for this task.
    Sentence transformers provide easy ways to get embeddings.
    This is the final paragraph. Ensure all functions work as expected."""

    print("Original Text:")
    print(sample_text)
    print("-" * 20)

    # Chunk the text
    text_chunks = chunk_text(sample_text)
    print(f"Number of chunks: {len(text_chunks)}")
    for i, chunk in enumerate(text_chunks):
        print(f"Chunk {i+1}: {chunk}")
    print("-" * 20)

    # Generate embeddings
    if text_chunks:
        embeddings = generate_embeddings(text_chunks)
        print(f"Number of embeddings: {len(embeddings)}")
        if embeddings:
            print(f"Dimension of embeddings: {len(embeddings[0])}")
            # print(f"First embedding: {embeddings[0][:10]}...") # Print first 10 dimensions of the first embedding
        else:
            print("No embeddings were generated.")
    else:
        print("No text chunks to generate embeddings from.")

    # Example with a placeholder PDF path (replace with a real PDF for actual testing)
    # This part will likely fail if a valid PDF is not at the specified path.
    pdf_path = "dummy.pdf" # Replace with a real PDF path to test extract_text_from_pdf
    print(f"\nAttempting to extract text from: {pdf_path} (This will fail if dummy.pdf doesn't exist or is not a valid PDF)")
    extracted_pdf_text = extract_text_from_pdf(pdf_path)
    if extracted_pdf_text:
        print(f"Extracted text length: {len(extracted_pdf_text)}")
        pdf_chunks = chunk_text(extracted_pdf_text)
        print(f"Number of chunks from PDF: {len(pdf_chunks)}")
        if pdf_chunks:
            pdf_embeddings = generate_embeddings(pdf_chunks)
            print(f"Number of embeddings from PDF: {len(pdf_embeddings)}")
            if pdf_embeddings:
                print(f"Dimension of PDF embeddings: {len(pdf_embeddings[0])}")
            else:
                print("No embeddings generated from PDF chunks.")
        else:
            print("No chunks generated from PDF text.")
    else:
        print(f"No text extracted from {pdf_path}. This is expected if the file doesn't exist or is not a valid PDF.")
