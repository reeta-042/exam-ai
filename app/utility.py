

import streamlit as st
from app.loaders import load_and_chunk_pdf
from app.vectorbase import get_vectorstore, get_bm25_retriever

# Caching helpers
from streamlit.runtime.caching import cache_data, cache_resource



@cache_data(show_spinner=False)
def cached_chunk_pdf(file_path: str):
    """Loads and chunks a PDF, caching the result."""
    return load_and_chunk_pdf(file_path)

@cache_resource
def cached_get_vectorstore(_api_key, _index_name, _namespace):
    """Gets a vectorstore instance, caching the resource."""
    return get_vectorstore(_api_key, _index_name, _namespace)

def get_bm25_retriever_from_chunks(chunks):
    """Creates a BM25 retriever. Not cached to ensure freshness."""
    return get_bm25_retriever(chunks)
  
