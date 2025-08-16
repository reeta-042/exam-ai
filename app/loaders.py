

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str):
    """
    Loads a PDF using PyMuPDFLoader for reliable text extraction and then
    chunks it using a RecursiveCharacterTextSplitter.
    """
    # Using  PyMuPDFLoader
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    # Your chosen chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(pages)
    return chunks
    
