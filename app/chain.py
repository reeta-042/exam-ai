from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableMap, RunnableLambda
from langchain.schema.runnable import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from app.config import GOOGLE_API_KEY


def retrieve_hybrid_docs(query, vectorstore, top_k=5):
    """
    Performing hybrid documents retrieval (Vector + sparse)
    """
    semantic_docs = vectorstore.similarity_search(query, k=top_k)
    # keyword_docs = bm25_retriever.get_relevant_documents(query)

    # Deduplicate by page_content
    combined = {doc.page_content: doc for doc in semantic_docs}
    return list(combined.values())


def rerank_documents(query, docs, top_k=4):
    """
    Reranks documents using a pretrained CrossEncoder.
    """
    # Loading cross encoder model
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Loading the ranker
    reranker = CrossEncoderReranker(model=model)

    # Apply reranker
    top_docs = reranker.compress_documents(documents=docs, query=query)

    # Return top K documents
    return top_docs[:top_k]


def format_quiz_card(raw_quiz_text):
    questions = raw_quiz_text.strip().split("\n\n")
    quiz_card = []
    for q in questions:
        lines = q.strip().split("\n")
        if len(lines) >= 5:
            question_text = lines[0]
            options = lines[1:5]
            answer_line = next((line for line in lines if "Answer:" in line), None)
            explanation_line = next((line for line in lines if "Explanation:" in line), None)
            quiz_card.append({
                "question": question_text,
                "options": options,
                "answer": answer_line.replace("Answer:", "").strip() if answer_line else "Not found",
                "explanation": explanation_line.replace("Explanation:", "").strip() if explanation_line else ""
            })
    return quiz_card


def build_llm_chain(api_key):
    """
    Builds a streaming LLM chain using LangChain Runnables with Gemini 2.5 flash model.
    Includes answer generation, follow-up questions, and quiz creation.
    """

    # LLM Setup
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        model_kwargs={"streaming": True}
    )

    parser = StrOutputParser()

    # Answer Prompt
    answer_prompt = PromptTemplate.from_template("""
    You are a helpful undergraduate teaching AI assistant. Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:""")

    # Follow-Up Prompt
    followup_prompt = PromptTemplate.from_template("""
    You are a thoughtful AI tutor. Based on the context and the undergraduate question, provide relevant information in bullet points that will deepen their understanding and they can expect in exams.

    Context:
    {context}

    Question:
    {question}

    Follow-Up Questions:""")

    # Quiz Prompt
    quiz_prompt = PromptTemplate.from_template("""
    You are an AI tutor helping students prepare for exams. Based on the context and the original question, generate a quiz with 5 multiple-choice questions. Each question should have four options (A–D) and clearly indicate the correct answer, also provide a short explanation for each correct answer, the questions should range from beginner to advanced and the explanation should be simple for an undergraduate to understand.

    Context:
    {context}

    Original Question:
    {question}

    Quiz:""")

    # Chains
    answer_chain = answer_prompt | llm | parser
    followup_chain = followup_prompt | llm | parser
    quiz_chain = quiz_prompt | llm | parser

    # Parallel Chain: Inject context + question → run all chains
    parallel_chain = (
        RunnableMap({
            "context": lambda x: "\n\n".join([doc.page_content for doc in x["docs"]]),
            "question": lambda x: x["question"]
        })
        | RunnableParallel({
            "answer": answer_chain,
            "followup": followup_chain,
            "quiz": quiz_chain
        })
    )

    return parallel_chain
