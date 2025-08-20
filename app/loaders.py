

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str):
    """
    Loads a PDF using the fast and reliable PyMuPDFLoader and splits it into chunks.
    
    """
    print(f"--- Loading document with PyMuPDFLoader: {file_path} ---")
    
    # 1. Load the document using the lightweight loader
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    if not pages:
        print(f"--- PyMuPDFLoader returned no pages for {file_path}. ---")
        return []

    # 2. Split the document into chunks with our chosen strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(pages)
    
    print(f"--- Created {len(chunks)} chunks from the document. ---")
    return chunks
    
