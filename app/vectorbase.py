#Importing necessary libraries from langchain
#Vector storage and embeddings(step 3)

import os
import pinecone
#from langchain.vectorstores import Pinecone
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever




def store_chunks(chunks, api_key, env, index_name):
    """
    Stores the given chunks in Pinecone with HuggingFace embeddings.
    Returns the vectorstore instance.
    """
    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment=env)
    index = pinecone.Index(index_name = index_name)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Wrap Pinecone with LangChain
    vectorstore = LangChainPinecone(index, embedding_function=embeddings.embed_query, text_key="text")

    # Convert chunks to LangChain documents
    from langchain.docstore.document import Document
    docs = [Document(page_content=chunk.page_content) for chunk in chunks]

    # Add documents to Pinecone
    vectorstore.add_documents(docs)

    return vectorstore


def get_bm25_retriever(chunks):
    """
    Initializes and returns a BM25Retriever from text chunks.
    Useful for keyword-based retrieval in hybrid search.
    """

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    return bm25


def get_vectorstore(api_key, env, index_name):
    """
    Loads the existing Pinecone index.
    """
    pinecone.init(api_key= api_key, environment= env)
    index = pinecone.Index(index_name = index_name)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = LangChainPinecone(index, embedding_function=embeddings.embed_query, text_key="text")

    return vectorstore
