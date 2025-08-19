from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
#from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import re




def format_quiz_card(text):
    """Parse quiz text into structured format."""
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
    Build chains for Answer, Follow-up, and Quiz generation.
    Ensures `context` and `question` are always passed into the prompt.
    """

    # LLM Setup
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        model_kwargs={"streaming": True}
    )
    parser = StrOutputParser()

    # --- Prompts ---
    answer_prompt = PromptTemplate.from_template("""
You are a helpful undergraduate teaching AI assistant. Answer the question based only on the context below.
Break down difficult words into simple explanations. Be clear and exam-focused.

Context:
{context}

Question:
{question}

Answer:""")

    followup_prompt = PromptTemplate.from_template("""
You are a thoughtful AI tutor. Based on the context and the question, 
provide  bullet point follow-ups that expand the student's understanding.

Context:
{context}

Question:
{question}

Follow-Up:""")

    quiz_prompt = PromptTemplate.from_template("""
You are an AI tutor helping students prep for exams. 
Generate 5 MCQs with this format:

Question: [text]
A. [option]
B. [option]
C. [option]
D. [option]
Answer: [letter]
Explanation: [short explanation]

Context:
{context}

Question:
{question}

Quiz:
""")

    # --- Chains (with explicit variable binding) ---
    # This ensures context & question always reach the LLM
    passthrough = RunnablePassthrough()

    answer_chain = (
        passthrough
        | answer_prompt
        | llm
        | parser
    )

    followup_chain = (
        passthrough
        | followup_prompt
        | llm
        | parser
    )

    quiz_chain = (
        passthrough
        | quiz_prompt
        | llm
        | parser
        | RunnableLambda(format_quiz_card)
    )

    return answer_chain, followup_chain, quiz_chain
