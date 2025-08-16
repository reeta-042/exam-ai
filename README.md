# Exam-prep.ai
# 🤖 ExamAI: Your Personal AI Study Assistant

ExamAI is an intelligent study application designed to help students prepare for exams by transforming their course materials into an interactive learning experience. Users can upload their PDF documents, and the application will leverage a sophisticated AI pipeline to answer questions, generate follow-up queries, and create quizzes based on the provided content.

This project is built to tackle the common challenge of studying from poorly formatted or complex academic documents, ensuring that students can get accurate and relevant information from their notes, no matter the format.

## Chosen Tech Stack

The application is built using a modern, powerful stack for AI and web development:

*   **Backend & AI Logic:**
    *   **Python:** The core programming language.
    *   **LangChain:** The primary framework for orchestrating the AI pipeline, including data loading, chunking, retrieval, and agentic chains.
    *   **Google Gemini:** The Large Language Model (LLM) used for generating answers, follow-up questions, and quizzes.
    *   **Pinecone:** A high-performance vector database used to store document embeddings for efficient semantic search.
    *   **Sentence-Transformers:** Used to create dense vector embeddings of the text chunks.
*   **Frontend:**
    *   **Streamlit:** A Python framework for rapidly building and deploying interactive web applications.
*   **Key Data Processing Libraries:**
    *   **`unstructured[local-inference]`:** A powerful library for parsing complex and poorly structured PDFs by understanding the document layout.
    *   **`opencv-python-headless`:** A dependency for computer vision tasks required by `unstructured`.

## Core Technical Decisions Explained

### Why `unstructured[local-inference]`?

During development, it became clear that standard PDF text extraction libraries (`PyPDFLoader`, `PyMuPDFLoader`) failed on certain real-world academic documents. These documents often have complex layouts, multi-column text, embedded tables, and inconsistent formatting, which caused the retrieval system to pull irrelevant chunks of text (e.g., SQL commands instead of a definition for "data").

**`unstructured[local-inference]`** was chosen to solve this critical problem. Unlike simpler loaders, it doesn't just read text streams; it uses computer vision and layout-parsing models to:
1.  **Understand the visual structure** of the page.
2.  **Intelligently identify** distinct elements like titles, paragraphs, and lists.
3.  **Extract clean, logically coherent text blocks.**

This ensures that the text fed into the chunking process is of the highest possible quality, which is the most critical step for accurate retrieval.

### Chunking Strategy: `chunk_size=1000`, `chunk_overlap=200`

The quality of retrieval is highly dependent on how the source material is chunked. The chosen strategy prioritizes creating chunks with **rich contextual information** while ensuring **semantic continuity**.

*   **`chunk_size=1000`:** This larger chunk size is deliberately chosen to provide the language model with more comprehensive context for each retrieved passage. Instead of just getting a single sentence or a small fact, the model receives a fuller paragraph or section. This is particularly effective for answering broader, more complex questions that require understanding the relationships between different ideas within the text. It allows for more nuanced and detailed responses.

*   **`chunk_overlap=200`:** A significant overlap of 20% is crucial for maintaining the integrity of the document's ideas. When the splitter is forced to break a paragraph or a long sentence, this large overlap ensures that the full context is preserved across adjacent chunks. This acts as a robust "safety net," preventing the loss of meaning that can occur at chunk boundaries.

This combination balances the need for rich context with the importance of continuity, optimizing the chunks for a powerful retrieval system that can handle both specific and broad queries.


## Setup and Installation

Follow these steps to set up and run the project locally.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Ensure you have the `requirements.txt` file from the project.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The installation for `unstructured[local-inference]` may take some time as it includes several machine learning libraries.*

4.  **Set Up API Keys:**
    The application requires API keys for Google Gemini and Pinecone. Store these securely. The app is configured to use Streamlit Secrets. Create a file at `.streamlit/secrets.toml` and add your keys:
    ```toml
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
    PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
    PINECONE_INDEX_NAME = "your-pinecone-index-name"
    ```

5.  **Run the Application:**
    ```bash
    streamlit run main.py
    ```
    The application should now be running and accessible in your web browser.

## Example Usage and Output

The user workflow is simple and intuitive:

1.  The user uploads a PDF document using the sidebar.
2.  The application processes and indexes the document.
3.  The user asks a question in the main input area.

**Example Query:**
> "What is the difference between a primary key and a foreign key?"

**Expected Output Format:**
The application will display the results in a clean, tabbed interface:

*   **💡 Answer Tab:**
    *   **Main Answer:** A direct, comprehensive answer generated by the AI based on the retrieved context from the document.
    *   **🗨️ Follow-up Questions:** A list of suggested questions to help the user explore the topic further.

*   **📝 Quiz Tab:**
    *   A multiple-choice quiz is generated based on the topic. Each question is presented clearly, with the answer hidden in an expandable section to encourage active recall.
    ```
    **Question 1:** Which type of key is used to uniquely identify a record in a table?
    - A. Foreign Key
    - B. Super Key
    - C. Primary Key
    
    [Show Answer]  <- This is an expandable button
    ```

*   **🔍 Retrieved Context Tab:**
    *   This tab shows the raw text chunks that were retrieved from the document and used as the context for generating the answer, providing transparency into the AI's process.

## Limitations and Known Issues

*   **Document Quality Dependency:** While `unstructured` is powerful, the application's performance is still highly dependent on the quality of the source PDF. Extremely degraded, scanned, or poorly formatted documents may still yield suboptimal results.
*   **Computational Cost:** The initial processing of a document, especially with `unstructured[local-inference]`, can be computationally intensive and may take time, particularly for large documents.
*   **Hallucination Risk:** Like all LLM-based systems, there is a small but inherent risk of the AI generating plausible but incorrect information (hallucinating), especially if the context it receives is ambiguous. The "Retrieved Context" tab is provided to help users verify the source of the information.

