from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_core.runnables import RunnableParallel, RunnableMap, RunnableLambda
#from langchain.schema.runnable import Runnable
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

import re

def parse_quiz(text):
    # Split the quiz into individual questions
    question_blocks = re.split(r'\n(?=Question:)', text.strip())
    quiz_data = []

    for block in question_blocks:
        question_match = re.search(r'Question:\s*(.*)', block)
        options = {
            'A': re.search(r'A\.\s*(.*)', block),
            'B': re.search(r'B\.\s*(.*)', block),
            'C': re.search(r'C\.\s*(.*)', block),
            'D': re.search(r'D\.\s*(.*)', block),
        }
        answer_match = re.search(r'Answer:\s*([A-D])', block)
        explanation_match = re.search(r'Explanation:\s*(.*)', block)

        if question_match and answer_match and explanation_match:
            quiz_data.append({
                'question': question_match.group(1).strip(),
                'options': {key: opt.group(1).strip() if opt else None for key, opt in options.items()},
                'answer': answer_match.group(1).strip(),
                'explanation': explanation_match.group(1).strip()
            })

    return quiz_data


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
You are an AI tutor helping students prepare for exams. Based on the context and the original question, generate a quiz with 5 multiple-choice questions.

Each question must follow this exact format:
- Question: [Your question here]
- A. [Option A]
- B. [Option B]
- C. [Option C]
- D. [Option D]
- Answer: [Correct option letter, e.g., C]
- Explanation: [Short explanation for why this is the correct answer]

Guidelines:
- Ensure the correct answer is clearly labeled with "Answer:" and the explanation starts with "Explanation:"
- Questions should range from beginner to advanced.
- Keep explanations simple and easy to understand for undergraduate students.

Context:
{context}

Original Question:
{question}

Quiz:
""")
    # Chains
    answer_chain = answer_prompt | llm | parser
    followup_chain = followup_prompt | llm | parser
    quiz_chain = quiz_prompt | llm | parser

    return answer_chain, followup_chain, quiz_chain
