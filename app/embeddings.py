from langchain_huggingface import HuggingFaceEmbeddings

def get_advanced_embedding_model():
    """
    Loads a FAST and EFFICIENT open-source embedding model from Hugging Face.
    This model is optimized for speed, making the user experience much better.
    """
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings
