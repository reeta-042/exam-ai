import os
import uuid
import streamlit as st
from itertools import chain

# --- KEY IMPORTS FOR ADVANCED RETRIEVAL ---
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_groq import ChatGroq
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Import functions from their respective files
from app.chain import build_llm_chain
from app.streamlit import upload_pdfs, save_uploaded_files
from app.utility import (
    cached_chunk_pdf,
    cached_get_vectorstore,
    get_bm25_retriever_from_chunks
)

# ------------------- PAGE CONFIGURATION -------------------
st.set_page_config(
    page_title="ExamAI 📄",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- API KEYS & CONSTANTS -------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] # Assumes you've added this to secrets
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- ADVANCED RETRIEVAL COMPONENTS (CACHED) ---
@st.cache_resource
def get_reranker():
    return HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def get_hyde_llm(_api_key):
    return ChatGroq(temperature=0, groq_api_key=_api_key, model_name="llama3-8b-8192")

reranker = get_reranker()
hyde_llm = get_hyde_llm(GROQ_API_KEY)

# ------------------- SIDEBAR FOR FILE UPLOADS -------------------
with st.sidebar:
    st.header("📚 Your Course Material")
    st.markdown("Upload your PDF files here. Once processed, you can ask questions in the main window.")
    
    uploaded_files, submitted = upload_pdfs()

    if submitted and uploaded_files:
        for key in st.session_state.keys():
            del st.session_state[key]
        
        namespace = f"session_{uuid.uuid4().hex}"
        st.session_state["namespace"] = namespace

        with st.spinner("📥 Ingesting and indexing your PDFs..."):
            file_paths = save_uploaded_files(uploaded_files)
            all_chunks = []
            for path in file_paths:
                chunks = cached_chunk_pdf(path)
                all_chunks.extend(chunks)
            st.session_state["all_chunks"] = all_chunks

            from app.vectorbase import store_chunks
            store_chunks(all_chunks, PINECONE_API_KEY, PINECONE_INDEX_NAME, namespace)
        
        st.success(f"✅ Uploaded {len(uploaded_files)} file(s) successfully!")
        st.rerun()

# ------------------- MAIN PAGE LAYOUT -------------------
st.title("💻 ExamAI: Chat with your Course Material")

session_active = "namespace" in st.session_state and "all_chunks" in st.session_state

if not session_active:
    st.info("Please upload your documents in the sidebar to begin your study session.")

st.subheader("...Ask Away...")
query = st.text_input(
    "What do you want to know?",
    placeholder="e.g., Compare and contrast top-down and bottom-up design...",
    label_visibility="collapsed",
    disabled=not session_active
)

# ------------------- QUERY PROCESSING & DISPLAY -------------------
if query and session_active:
    
    with st.spinner("Initializing retrieval engine..."):
        namespace = st.session_state["namespace"]
        all_chunks = st.session_state["all_chunks"]
        vectorstore = cached_get_vectorstore(PINECONE_API_KEY, PINECONE_INDEX_NAME, namespace)
        bm25_retriever = get_bm25_retriever_from_chunks(all_chunks)

    with st.spinner("🕵️‍♂️ Generating hypothetical answer & searching..."):
        # --- NEW: HYDE RETRIEVER LOGIC ---
        hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template="Please write a short, concise paragraph that provides a clear answer to the following question.\nQuestion: {question}\nAnswer:"
        )
        hyde_chain = LLMChain(llm=hyde_llm, prompt=hyde_prompt)
        hyde_retriever = HydeRetriever(vectorstore=vectorstore, llm_chain=hyde_chain)
        
        semantic_docs = hyde_retriever.invoke(query)
        keyword_docs = bm25_retriever.invoke(query)
        
        all_initial_docs = list(chain(semantic_docs, keyword_docs))
        unique_docs_map = {doc.page_content: doc for doc in all_initial_docs}
        unique_docs = list(unique_docs_map.values())

    if not unique_docs:
        st.error("I couldn't find any relevant information in the documents. Please try another question.")
    else:
        with st.spinner("📚 Reranking results for relevance..."):
            rerank_pairs = [(query, doc.page_content) for doc in unique_docs]
            scores = reranker.score(rerank_pairs)
            scored_docs = list(zip(scores, unique_docs))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            reranked_docs = [doc for score, doc in scored_docs[:5]]

        # --- The rest of the logic remains the same ---
        answer_tab, quiz_tab, context_tab = st.tabs(["💡 Answer", "📝 Quiz", "🔍 Retrieved Context"])

        input_data = {
            "context": "\n\n---\n\n".join([doc.page_content for doc in reranked_docs]),
            "question": query
        }
        answer_chain, followup_chain, quiz_chain = build_llm_chain(api_key=GOOGLE_API_KEY)

        with answer_tab:
            st.markdown("### Main Answer")
            with st.spinner("⌨️ Generating answer..."):
                answer = answer_chain.invoke(input_data)
                st.markdown(answer)
            
            st.markdown("### 🗨️ Follow-up Questions")
            with st.spinner("Thinking of more questions..."):
                followup = followup_chain.invoke(input_data)
                st.markdown(followup)

        with quiz_tab:
            with st.spinner("📝 Generating quiz..."):
                quiz_card = quiz_chain.invoke(input_data)
                if quiz_card:
                    for i, q in enumerate(quiz_card):
                        st.markdown(f"**Question {i+1}:** {q['question']}")
                        for label, opt in q["options"].items():
                            if opt:
                                st.markdown(f"- {label}. {opt}")
                        with st.expander("Show Answer"):
                            st.markdown(f"**✅ Correct Answer:** {q['answer']}")
                            if q["explanation"]:
                                st.markdown(f"**💡 Why?** {q['explanation']}")
                        st.markdown("---" if i < len(quiz_card) - 1 else "")
                else:
                    st.warning("⚠️ Quiz could not be generated for this topic.")

        with context_tab:
            st.markdown("These are the top 5 chunks found after advanced retrieval and reranking.")
            for i, doc in enumerate(reranked_docs):
                st.info(f"**Chunk {i+1}:**\n\n" + doc.page_content)
                
