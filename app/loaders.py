# In app/loaders.py
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#ADD CACHED FUNCTION HERE ---
@st.cache_data(show_spinner=False)
def cached_chunk_pdf(_file_path: str):
    """
    Loads and chunks a PDF, wrapped in a Streamlit cache for data.
    """
    loader = PyMuPDFLoader(_file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    return text_splitter.split_documents(pages)
    
