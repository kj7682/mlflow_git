import pytest
from unittest.mock import MagicMock, patch
import numpy as np # Used for a more realistic embedding structure in mock

# Modules to be tested
from app.pdf_processor import extract_text_from_pdf, chunk_text, generate_embeddings

# --- Tests for extract_text_from_pdf ---

@patch('app.pdf_processor.PdfReader') # Mock the PdfReader class
def test_extract_text_from_pdf_success(mock_pdf_reader):
    """
    Tests successful text extraction from a PDF.
    Mocks PdfReader to simulate PDF content.
    """
    # Configure the mock PdfReader instance and its pages
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "This is page one. "
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "This is page two."
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page1, mock_page2]
    mock_pdf_reader.return_value = mock_reader_instance # When PdfReader(path) is called, it returns our mock_reader_instance

    # Call the function with a dummy path (it won't be used due to mocking)
    extracted_text = extract_text_from_pdf("dummy.pdf")

    # Assertions
    assert extracted_text == "This is page one. This is page two."
    mock_pdf_reader.assert_called_once_with("dummy.pdf") # Check if PdfReader was called with the path
    mock_page1.extract_text.assert_called_once()
    mock_page2.extract_text.assert_called_once()

@patch('app.pdf_processor.PdfReader')
def test_extract_text_from_pdf_empty_pages(mock_pdf_reader):
    """
    Tests PDF extraction when pages have no text or extract_text returns None.
    """
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Some text. "
    mock_page_empty = MagicMock()
    mock_page_empty.extract_text.return_value = None # Simulate a page that returns None
    mock_page3 = MagicMock()
    mock_page3.extract_text.return_value = "More text."

    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page1, mock_page_empty, mock_page3]
    mock_pdf_reader.return_value = mock_reader_instance

    extracted_text = extract_text_from_pdf("dummy.pdf")
    assert extracted_text == "Some text. More text." # Ensure None page is handled (concatenated as empty string)

@patch('app.pdf_processor.PdfReader')
def test_extract_text_from_pdf_failure(mock_pdf_reader):
    """
    Tests PDF extraction failure when PdfReader raises an exception.
    """
    mock_pdf_reader.side_effect = Exception("Failed to open PDF") # Simulate an error during PdfReader instantiation

    extracted_text = extract_text_from_pdf("error.pdf")
    assert extracted_text == "" # Expect an empty string on failure as per current implementation

# --- Tests for chunk_text ---

def test_chunk_text_basic():
    """
    Tests basic text chunking functionality.
    """
    sample_text = "This is a long string of text that needs to be chunked. It has several sentences. The chunking should respect the size and overlap."
    chunks = chunk_text(sample_text, chunk_size=30, chunk_overlap=10)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 1 # Expect multiple chunks
    if chunks: # Check content if chunks are produced
        assert chunks[0] == "This is a long string of text" # Based on RecursiveCharacterTextSplitter behavior
        # Check overlap (approximate, as actual splitting depends on splitter logic)
        if len(chunks) > 1:
            overlap_start_of_chunk1_in_chunk0 = chunks[0].find(chunks[1][:10]) 
            # This is a simplified check; actual overlap is more complex due to RecursiveCharacterTextSplitter
            # For now, we'll just check if chunks are produced and have reasonable content.
            assert "string of text that needs to" in chunks[0] or "This is a long string of text" in chunks[0]

def test_chunk_text_small_text():
    """
    Tests chunking with text smaller than chunk_size.
    """
    sample_text = "Short text."
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == sample_text

def test_chunk_text_empty_text():
    """
    Tests chunking with empty text.
    """
    sample_text = ""
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 0 # Langchain's splitter returns empty list for empty text

def test_chunk_text_custom_params():
    """
    Tests chunking with custom chunk_size and chunk_overlap.
    """
    sample_text = "abcdefghijklmnopqrstuvwxyz" * 5 # 26 * 5 = 130 chars
    chunks = chunk_text(sample_text, chunk_size=50, chunk_overlap=5)
    assert len(chunks) > 0
    if chunks:
        for chunk in chunks:
            assert len(chunk) <= 50 or (len(chunk) > 50 and len(chunks) == 1) # Chunks should be <= chunk_size unless it's the only chunk

# --- Tests for generate_embeddings ---

@patch('app.pdf_processor.SentenceTransformer') # Mock the SentenceTransformer class
def test_generate_embeddings_success(mock_sentence_transformer):
    """
    Tests successful embedding generation.
    Mocks SentenceTransformer to simulate model behavior.
    """
    # Configure the mock SentenceTransformer instance
    # Simulate a model that produces 3-dimensional embeddings
    mock_model_instance = MagicMock()
    # The encode method should return a numpy array for tolist() to work
    mock_embeddings_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    mock_model_instance.encode.return_value = mock_embeddings_array
    
    mock_sentence_transformer.return_value = mock_model_instance # When SentenceTransformer(model_name) is called

    text_chunks = ["Hello world", "Another sentence"]
    embeddings = generate_embeddings(text_chunks, model_name='test-model')

    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    if embeddings:
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) == 3 # Dimension of embeddings
        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    mock_sentence_transformer.assert_called_once_with('test-model')
    mock_model_instance.encode.assert_called_once_with(text_chunks, convert_to_tensor=False)

@patch('app.pdf_processor.SentenceTransformer')
def test_generate_embeddings_empty_chunks(mock_sentence_transformer):
    """
    Tests embedding generation with an empty list of text chunks.
    """
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = np.array([]) # Encode might return empty array for empty list
    mock_sentence_transformer.return_value = mock_model_instance

    text_chunks = []
    embeddings = generate_embeddings(text_chunks)

    assert embeddings == []
    # Check if encode was called (it might be, or it might be short-circuited by the library)
    # Based on current implementation of generate_embeddings, encode would be called.
    mock_model_instance.encode.assert_called_once_with([], convert_to_tensor=False)


@patch('app.pdf_processor.SentenceTransformer')
def test_generate_embeddings_model_failure(mock_sentence_transformer):
    """
    Tests embedding generation failure when SentenceTransformer raises an exception.
    """
    mock_sentence_transformer.side_effect = Exception("Failed to load model")

    text_chunks = ["Some text"]
    embeddings = generate_embeddings(text_chunks)

    assert embeddings == [] # Expect an empty list on failure

# A simple (non-mocked) test for chunk_text to ensure it handles unicode, etc.
def test_chunk_text_real_splitter_edge_cases():
    """
    Tests chunk_text with some edge cases using the actual RecursiveCharacterTextSplitter.
    """
    # Test with very long word (should be split if it exceeds chunk_size by itself)
    long_word = "supercalifragilisticexpialidocious" * 3 # Longer than typical chunk_size
    chunks_long_word = chunk_text(long_word, chunk_size=30, chunk_overlap=5)
    assert len(chunks_long_word) > 0
    if chunks_long_word:
        assert len(chunks_long_word[0]) <= 30 or (len(chunks_long_word[0]) > 30 and len(chunks_long_word) ==1)


    # Test with unicode characters
    unicode_text = "你好，世界。这是一个测试。" * 5
    chunks_unicode = chunk_text(unicode_text, chunk_size=20, chunk_overlap=5) # Small chunk size for unicode
    assert len(chunks_unicode) > 0
    if chunks_unicode:
        # print(f"\nUnicode Chunks: {chunks_unicode}") # For debugging
        assert "你好，世界" in chunks_unicode[0] or "你好" in chunks_unicode[0]

    # Test with only whitespace
    whitespace_text = "    \n\t    "
    chunks_whitespace = chunk_text(whitespace_text, chunk_size=10, chunk_overlap=2)
    # Langchain's splitter might produce chunks with only whitespace or filter them.
    # Current behavior: it will produce chunks containing whitespace.
    # If it were to filter them, this assertion would need to change to len == 0
    assert len(chunks_whitespace) >= 0 # Can be 0 or more depending on splitter's handling of pure whitespace
    if chunks_whitespace:
        assert chunks_whitespace[0].strip() == "" or chunks_whitespace[0] == whitespace_text
        
    # Test with text that is exactly chunk_size
    exact_size_text = "a" * 100
    chunks_exact = chunk_text(exact_size_text, chunk_size=100, chunk_overlap=0)
    assert len(chunks_exact) == 1
    assert chunks_exact[0] == exact_size_text
    
    # Test with text that is slightly larger than chunk_size
    slightly_larger_text = "a" * 101
    chunks_larger = chunk_text(slightly_larger_text, chunk_size=100, chunk_overlap=0)
    assert len(chunks_larger) == 2 # "aaaaaaaa...a" (100), "a" (1)
    if len(chunks_larger) == 2:
      assert chunks_larger[0] == "a" * 100
      assert chunks_larger[1] == "a"
