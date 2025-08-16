
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str):
    """
    Loads a PDF using PyMuPDFLoader for better text extraction and then
    chunks it using a RecursiveCharacterTextSplitter.
    """
    
    # This loader is often better at extracting clean text from complex PDFs.
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    # The chunking strategy remains the same.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(pages)
    return chunks
    
