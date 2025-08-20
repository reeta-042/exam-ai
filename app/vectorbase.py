
import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_community.retrievers import BM25Retriever
import time
from app.embeddings import get_embedding_model


@st.cache_resource
def cached_get_vectorstore(api_key, index_name, namespace: str):
    """
    Loads an existing vectorstore from Pinecone, wrapped in a Streamlit cache.
    This ensures we only connect to the vectorstore once per session.
    """
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Get the embedding model from our central cached function
    embeddings = get_embedding_model()
    
    # Initialize the LangChainPinecone object
    return LangChainPinecone(
        index=index, 
        embedding=embeddings, 
        namespace=namespace
    )

def store_chunks(chunks, api_key, index_name, namespace: str):
    """
    Stores LangChain Document chunks in Pinecone using the cached embedding model.
    """
    # Get the embedding model from our central cached function
    embeddings = get_embedding_model()

    # Use the `from_documents` class method to efficiently create and upload.
    # This will create a new vectorstore instance for the upload process.
    vectorstore = LangChainPinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )

    # Efficient warm-up to wait for indexing to complete.
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    target_vector_count = len(chunks)
    for _ in range(10):
        stats = index.describe_index_stats()
        if namespace in stats.namespaces and stats.namespaces[namespace].vector_count >= target_vector_count:
            break
        time.sleep(1)

    return vectorstore

def get_bm25_retriever_from_chunks(chunks):
    """
    Initializes and returns a BM25Retriever from a list of documents.
    """
    bm25 = BM25Retriever.from_documents(documents=chunks)
    bm25.k = 5
    return bm25
    
