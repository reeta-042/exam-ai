import os
import streamlit as st
from app.loaders import load_and_chunk_pdf
from app.vectorbase import store_chunks, get_vectorstore, get_bm25_retriever
from app.chain import build_llm_chain, retrieve_hybrid_docs, rerank_documents, format_quiz_card
from app.streamlit import upload_pdfs

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

    # Load, split, and embed PDF into chunks
    with st.spinner("... Loading👀..."):
        chunks = load_and_chunk_pdf(file_path)
        st.success("✅ Course material loaded successfully!")

        # Store in vectorstore
        vectorstore = store_chunks(chunks)
        bm25 = get_bm25_retriever(chunks)
else:
    try:
        # Fallback to existing vectorstore on app restart
        chunks = []
        vectorstore = get_vectorstore()
    except:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

# STEP 3: User asks a question
st.header("Ask away....🌚")
query = st.text_input("What do you want to know?")

if query:
    # STEP 4: Retrieve documents (Hybrid search)
    retrieved_docs = retrieve_hybrid_docs(query, vectorstore)

    # STEP 5: Apply reranker
    reranked_docs = rerank_documents(query, retrieved_docs)

    # STEP 6: Build the chain
    chain = build_llm_chain()

    # STEP 7: Stream response into Streamlit
    st.subheader("Detailed Answer with Follow-Up and Quiz 😌")

    # Create containers for each section
    answer_container = st.empty()
    followup_container = st.empty()
    quiz_container = st.empty()

    # Stream each part of the response
    results = chain.stream({"question": query, "docs": reranked_docs})

    # Buffer quiz text for formatting later
    quiz_text = ""

    with st.spinner("⌨️Generating answer..."):
        for chunk in results["answer"]:
            answer_container.markdown(chunk)

    with st.spinner("👀Generating follow-up questions..."):
        for chunk in results["followup"]:
            followup_container.markdown(chunk)

    with st.spinner("🚶Generating quiz..."):
        for chunk in results["quiz"]:
            quiz_text += chunk
            quiz_container.markdown(chunk)

    # STEP 8: Format and display quiz as a learning tool
    st.markdown("### 📘 Learn Through Quiz")
    quiz_card = format_quiz_card(quiz_text)

    for i, q in enumerate(quiz_card):
        st.markdown(f"**Q{i+1}: {q['question']}**")
        for opt in q["options"]:
            st.markdown(f"- {opt}")
        st.markdown(f"✅ **Correct Answer:** {q['answer']}")
        if q["explanation"]:
            st.markdown(f"**Why?** {q['explanation']}")
        st.markdown("---")

    # STEP 9: Show retrieved chunks in the sidebar
    st.sidebar.subheader("🔍 Retrieved Chunks")
    if reranked_docs:
        for i, doc in enumerate(reranked_docs):
            st.sidebar.markdown(f"**Chunk {i+1}**")
            st.sidebar.caption(doc.page_content[:400])
    else:
        st.sidebar.info("No chunks retrieved yet.")
