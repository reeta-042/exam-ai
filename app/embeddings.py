
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_embedding_model():
    """
    Loads the FAST and EFFICIENT embedding model from Hugging Face.
    This function is cached so the model is only loaded once per session.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
