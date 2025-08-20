from pinecone import Pinecone  # Official Pinecone SDK (v3+)
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
import time
from app.embeddings import get_advanced_embedding_model()


def store_chunks(chunks, api_key, index_name, namespace: str = ""):
    """
    Stores the given chunks in Pinecone with HuggingFace embeddings.
    Returns the vectorstore instance.
    """
    # Create Pinecone client
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Embedding model
    embeddings = get_advanced_embedding_model()

    # Wrap Pinecone with LangChain
    vectorstore = LangChainPinecone(index, embedding=embeddings, text_key="text", namespace=namespace)

    # Convert chunks to LangChain documents
    docs = [Document(page_content=chunk.page_content) for chunk in chunks]

    # Add documents to Pinecone
    vectorstore.add_documents(docs, namespace=namespace)

    # ✅ Warm-up: ensure docs are indexed
    for _ in range(5):  # retry up to ~5 seconds
        res = vectorstore.similarity_search("warmup", k=1, namespace=namespace)
        if res:
            break
        time.sleep(1)

    return vectorstore


def get_bm25_retriever(chunks):
    """
    Initializes and returns a BM25Retriever from text chunks.
    Useful for keyword-based retrieval in hybrid search.
    """
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    return bm25


def get_vectorstore(api_key, index_name, namespace: str = ""):
    """
    Loads the existing Pinecone index.
    """
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    embeddings = get_advanced_embedding_model()
    vectorstore = LangChainPinecone(index, embedding=embeddings, text_key="text", namespace=namespace)

    return vectorstore
