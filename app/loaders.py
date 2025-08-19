
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str):
    """
    Loads ANY type of PDF using UnstructuredPDFLoader with an "auto" strategy.
    - For digital PDFs, it will extract text directly (fast and accurate).
    - For scanned/image-based PDFs, it will automatically use OCR.
    """
    
    loader = UnstructuredPDFLoader(
        file_path, mode="single", strategy="auto"
    )
    
    try:
        pages = loader.load()
    except Exception as e:
        # This is a fallback for extremely difficult PDFs.
        # If "auto" fails, we can force it to try OCR as a last resort.
        st.warning(f"Default 'auto' strategy failed: {e}. Forcing OCR as a fallback.")
        loader = UnstructuredPDFLoader(file_path, mode="single", strategy="ocr_only")
        pages = loader.load()


    # The rest of the function remains the same
    if not pages:
        st.warning("The loader was unable to extract any content from this document.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(pages)
    return chunks
    
