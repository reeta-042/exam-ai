# Importing langchain libraries for loading pdfs ( Step 1)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


#Chunking pdf (Step 2)

def load_and_chunk_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=250
    )
    return text_splitter.split_documents(documents)
