import os
import streamlit as st
from app.loaders import load_and_chunk_pdf
from app.vectorbase import store_chunks, get_vectorstore, get_bm25_retriever
from app.chain import build_llm_chain, retrieve_hybrid_docs, rerank_documents, format_quiz_card
from app.streamlit import upload_pdfs
#Loading my API KEYS
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
#PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Set Streamlit page configuration
st.set_page_config(page_title="📄 Chat with your PDF and prep for your exams", layout="wide")
st.title("💻ExamAI: Chat with your Course Material")

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# STEP 1: Upload PDF
pdf_file, submitted = upload_pdfs()

# STEP 2: Load + Index PDF if user submitted
if pdf_file and submitted:
    file_path = os.path.join(UPLOAD_DIR, pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.read())

    st.sidebar.success(f"Uploaded: {pdf_file.name}")

    with st.spinner("... Loading👀..."):
        chunks = load_and_chunk_pdf(file_path)
        st.success("✅ Course material loaded successfully!")

        vectorstore = store_chunks(
            chunks,
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME
        )

        bm25 = get_bm25_retriever(chunks)
else:
    try:
        chunks = []  # placeholder
        vectorstore = get_vectorstore(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME
        )
    except:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()
# STEP 3: User asks a question
st.subheader("....Ask away....🌚")
query = st.text_input("What do you want to know?")

if query:
    # STEP 4: Retrieve documents (Hybrid search)
    with st.spinner("🔍 Searching your course material..."):
        retrieved_docs = retrieve_hybrid_docs(query, vectorstore)

    # STEP 5: Apply reranker
    with st.spinner("📚 Reranking the most relevant chunks..."):
        reranked_docs = rerank_documents(query, retrieved_docs)

    # STEP 6: Build the chain
    answer_chain, followup_chain, quiz_chain = build_llm_chain(api_key=GOOGLE_API_KEY)

    # Prepare input for chains
    input_data = {
        "context": "\n\n".join([doc.page_content for doc in reranked_docs]),
        "question": query
    }

    # STEP 7: Invoke each chain sequentially
    st.markdown("### Detailed Answer with Follow-Up and Quiz ")

    answer_container = st.empty()
    followup_container = st.empty()
    quiz_container = st.empty()

    with st.spinner("⌨️ Generating answer..."):
        answer = answer_chain.invoke(input_data)
        answer_container.markdown(answer)

    with st.spinner("👀 Generating follow-up questions..."):
        followup = followup_chain.invoke(input_data)
        followup_container.markdown(followup)

    with st.spinner("🚶 Generating quiz..."):
        quiz = quiz_chain.invoke(input_data)
        #quiz_container.markdown(quiz)

    # STEP 8: Format and display quiz as a learning tool
    st.markdown("### 📘 Learn Through Quiz")
    quiz_card = format_quiz_card(quiz)

    for i, q in enumerate(quiz_card):
        st.markdown(f"**Q{i+1}: {q['question']}**")
        for label, opt in q["options"].items():
            st.markdown(f"- **{label}.** {opt}")
        st.markdown(f"✅ **Correct Answer:** {q['answer']}")
        if q["explanation"]:
            st.markdown(f"**Why?** {q['explanation']}")
        st.markdown("---")

    # STEP 9: Show retrieved chunks in the sidebar
    st.sidebar.subheader("🔍 Retrieved Chunks")
    if reranked_docs:
        for i, doc in enumerate(reranked_docs):
            st.sidebar.markdown(f"**Chunk {i+1}**")
            st.sidebar.caption(doc.page_content[:200])
    else:
        st.sidebar.info("No chunks retrieved yet.")
