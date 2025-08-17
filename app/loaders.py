
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str):
    """
    Loads a PDF using UnstructuredPDFLoader for robust, layout-aware text
    extraction and then chunks it.
    """
    # The 'mode="single"' strategy is often best. It treats the whole
    # document as one big page, which allows the text splitter to find
    # the best semantic breaks.
    loader = UnstructuredPDFLoader(file_path, mode="single", strategy="fast")
    pages = loader.load()

    # The chunking strategy remains the same
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
        
    chunks = text_splitter.split_documents(pages)
    return chunks
    
