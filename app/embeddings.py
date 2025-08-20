# In new file: app/embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings

def get_advanced_embedding_model():
    """
    Loads a powerful, open-source embedding model from Hugging Face.
    This model is more robust and might handle garbled text better.
    """
    # Use a top-tier open-source model.
    model_name = "BAAI/bge-large-en-v1.5"
    
    # Set the model to run on the CPU.
    model_kwargs = {"device": "cpu"}
    
    # Use a specific instruction for retrieval queries.
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings
  
