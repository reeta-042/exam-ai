import os
import uuid
import streamlit as st
from app.loaders import load_and_chunk_pdf
from app.vectorbase import store_chunks, get_vectorstore, get_bm25_retriever
from app.chain import build_llm_chain, retrieve_hybrid_docs, rerank_documents
from app.streamlit import upload_pdfs, save_uploaded_files

# Caching helpers
from streamlit.runtime.caching import cache_data, cache_resource

# 🔑 API Keys
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Page setup
st.set_page_config(page_title="📄 Chat with your PDF and prep for your exams", layout="wide")
st.title("💻 ExamAI: Chat with your Course Material")

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# STEP 1: Upload PDFs
uploaded_files, submitted = upload_pdfs()


@cache_data(show_spinner=False)
def cached_chunk_pdf(file_path: str):
    return load_and_chunk_pdf(file_path)


@cache_resource
def cached_get_vectorstore(api_key, index_name, namespace):
    return get_vectorstore(api_key, index_name, namespace)

# --- CHANGE ---
# We need to cache the BM25 retriever as well, so it persists across reruns.
@cache_resource
def cached_get_bm25_retriever(_chunks):
    return get_bm25_retriever(_chunks)


# STEP 2: Store / Load Vectorstore
if submitted and uploaded_files:
    # Create a fresh namespace for this upload
    namespace = f"session_{uuid.uuid4().hex}"
    st.session_state["namespace"] = namespace

    file_paths = save_uploaded_files(uploaded_files)

    all_chunks = []
    for path in file_paths:
        chunks = cached_chunk_pdf(path)
        all_chunks.extend(chunks)

    # --- CHANGE --- Store all_chunks in session state to rebuild retriever later
    st.session_state["all_chunks"] = all_chunks

    with st.spinner("📥 Ingesting and indexing your PDFs..."):
        vectorstore = store_chunks(
            all_chunks,
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            namespace=namespace
        )
        # --- CHANGE --- Use the cached function to create and store the retriever
        bm25_retriever = cached_get_bm25_retriever(all_chunks)

    st.sidebar.success(f"✅ Uploaded {len(uploaded_files)} file(s) successfully!")

# --- CHANGE --- This block handles loading the retrievers for subsequent queries
if "namespace" in st.session_state:
    namespace = st.session_state["namespace"]
    vectorstore = cached_get_vectorstore(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace
    )
    # Rebuild the BM25 retriever from the stored chunks
    if "all_chunks" in st.session_state:
        bm25_retriever = cached_get_bm25_retriever(st.session_state["all_chunks"])
    else:
        # This is a fallback in case the session state for chunks is lost
        st.warning("⚠️ Session data for keyword search was lost. Please re-upload files for best results.")
        st.stop()
else:
    st.warning("⚠️ Please upload at least one PDF.")
    st.stop()


# STEP 3: User Query
st.subheader("...Ask Away...")
query = st.text_input("What do you want to know?")


# STEP 4: Containers
answer_container = st.empty()
followup_container = st.empty()
quiz_container = st.empty()


# STEP 5: Processing Query
if query:
    # --- START OF MAJOR CHANGES ---

    with st.spinner("🕵️‍♂️ Searching your documents..."):
        # Pass both the vectorstore and the bm25_retriever for a true hybrid search
        retrieved_docs = retrieve_hybrid_docs(query, vectorstore, bm25_retriever)

    # --- DEBUG CODE TO DISPLAY RESULTS ---
    with st.expander("🔍 Click to see retrieved document chunks"):
        if retrieved_docs:
            st.write(f"{len(retrieved_docs)} chunks retrieved from your documents:")
            for i, doc in enumerate(retrieved_docs):
                st.info(f"**Chunk {i+1}:**\n\n" + doc.page_content)
        else:
            st.warning("No relevant chunks were found for your query.")
    # --- END OF DEBUG CODE ---

    # --- Proceed only if documents were found ---
    if not retrieved_docs:
        st.error("I couldn't find any relevant information in the documents to answer this. Please try another question.")
    else:
        with st.spinner("📚 Reranking results for relevance..."):
            reranked_docs = rerank_documents(query, retrieved_docs)

        # Safety net: fallback if reranked_docs is empty (reranker failed)
        if not reranked_docs:
            reranked_docs = retrieved_docs

        # Build the chains
        answer_chain, followup_chain, quiz_chain = build_llm_chain(api_key=GOOGLE_API_KEY)

        # Prepare the input data for the chains
        input_data = {
            "context": "\n\n---\n\n".join([doc.page_content for doc in reranked_docs]),
            "question": query
        }

        # Invoke the chains and display results
        with st.spinner("⌨️ Generating answer..."):
            answer = answer_chain.invoke(input_data)
            answer_container.markdown(answer)

        with st.spinner("🤔 Generating follow-up questions..."):
            followup = followup_chain.invoke(input_data)
            followup_container.markdown(followup)

        with st.spinner("📝 Generating quiz..."):
            quiz_card = quiz_chain.invoke(input_data)

        # STEP 6: Render Quiz
        if quiz_card:
            quiz_box = quiz_container.container()
            with quiz_box:
                st.markdown("## 📝 Learn Through Quiz")
                for i, q in enumerate(quiz_card):
                    st.markdown(f"**Q{i+1}: {q['question']}**")
                    for label, opt in q["options"].items():
                        if opt:
                            st.markdown(f"- {label}. {opt}")
                    st.markdown(f"✅ **Correct Answer:** {q['answer']}")
                    if q["explanation"]:
                        st.markdown(f"💡 *Why?* {q['explanation']}")
                    st.markdown("---")
        else:
            st.warning("⚠️ Quiz could not be generated. Please check your prompt or context.")

    # --- END OF MAJOR CHANGES ---
                                             
