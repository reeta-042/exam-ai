from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from itertools import chain

def get_hybrid_retriever(vectorstore, all_chunks):
    """
    Creates a hybrid retriever that combines keyword (BM25) and semantic search.
    """
    # 1. Create the BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    
    # 2. Create the vectorstore retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    
    return bm25_retriever, vector_retriever


def get_advanced_reranking_retriever(vector_retriever, bm25_retriever):
    """
    Creates an advanced retriever that performs a hybrid search and then reranks the results.
    This is the most powerful setup for relevance.
    """
    # 1. Define the Reranker
    
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=5) # Return the top 5 most relevant docs

    # 2. Create a function to perform the hybrid search
    def hybrid_search(query):
        semantic_docs = vector_retriever.invoke(query)
        keyword_docs = bm25_retriever.invoke(query)
        
        # Combine and remove duplicates
        all_docs = list(chain(semantic_docs, keyword_docs))
        unique_docs = {doc.page_content: doc for doc in all_docs}
        return list(unique_docs.values())

    return compressor
    
