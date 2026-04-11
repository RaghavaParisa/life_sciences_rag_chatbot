# 🧬 Life Sciences RAG Assistant

A **Retrieval-Augmented Generation (RAG)** system for life sciences
queries using **local open-source models**, ensuring **data privacy,
cost-efficiency, and offline capability**.

------------------------------------------------------------------------

# 🚀 Features

-   🔍 Hybrid Search (BM25 + FAISS)
-   🤖 Local LLM: Qwen2.5:3B (Ollama)
-   🧠 Semantic Embeddings (Sentence Transformers)
-   📊 Evaluation Metrics (F1, Faithfulness, Relevance)
-   🔐 Authentication (Admin/User roles)
-   📝 Audit Logging
-   🌐 Gradio UI

------------------------------------------------------------------------

# 🏗️ Architecture Diagram

            +-------------------+
            |     User Query    |
            +---------+---------+
                      |
                      v
            +-------------------+
            |   Gradio UI       |
            |    (app.py)       |
            +---------+---------+
                      |
                      v
            +-------------------+
            |   RAG Pipeline    |
            |    (rag.py)       |
            +---------+---------+
              |             |
              v             v
    +----------------+   +----------------+
    |  BM25 Search   |   |  FAISS Search  |
    +----------------+   +----------------+
              \             //
               \           //
                v         v
            +-------------------+
            | Hybrid Retriever  |
            +---------+---------+
                      |
                      v
            +-------------------+
            | Context Builder   |
            +---------+---------+
                      |
                      v
            +-------------------+
            |   Ollama LLM      |
            | (Qwen2.5:3B)      |
            +---------+---------+
                      |
                      v
            +-------------------+
            | Final Answer +    |
            | Citations         |
            +---------+---------+
                      |
                      v
            +-------------------+
            | Audit Logs        |
            +-------------------+

------------------------------------------------------------------------

# 🔄 Workflow

    1. User enters query
    2. Query sent to RAG pipeline
    3. Hybrid search retrieves relevant documents
    4. Context is constructed
    5. Prompt sent to local LLM (Ollama)
    6. Answer generated
    7. Sources appended
    8. Response shown in UI
    9. Interaction logged

------------------------------------------------------------------------

# ⚙️ Function-Level Breakdown

## app.py

-   `login()` → Authenticates user
-   `chat()` → Handles user query flow
-   `logout()` → Clears session

## rag.py

-   `init_hybrid()` → Initializes hybrid search
-   `retrieve(query)` → Fetches top relevant documents
-   `generate_answer(query, context, citations)` → Calls LLM and
    generates answer

## embeddings.py

-   `load_or_create_faiss()` → Loads or builds FAISS index

## ingestion.py

-   `load_documents()` → Reads PDFs/CSVs
-   `split_documents()` → Breaks into chunks
-   `create_embeddings()` → Generates embeddings

## hybrid_search.py

-   `HybridSearch.search()` → Combines BM25 + FAISS results

## evaluation.py

-   `evaluate()` → Runs test queries
-   `compute_token_metrics()` → Calculates precision/recall/F1
-   `compute_faithfulness()` → Checks grounding
-   `compute_answer_relevance()` → Checks semantic relevance

## auth.py

-   `authenticate()` → Validates credentials
-   `generate_token()` → Creates JWT

## audit.py

-   `log_interaction()` → Stores query, answer, sources

------------------------------------------------------------------------

# 📦 Installation

``` bash
git clone <repo-url>
cd clinicaltrails
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🧠 Model Setup

Install Ollama:

https://ollama.com

Pull model:

``` bash
ollama pull qwen2.5:3b
```

------------------------------------------------------------------------

# ▶️ Run Application

``` bash
python ingestion.py
python app.py
```

Open:

http://localhost:7860

------------------------------------------------------------------------

# 🔍 Evaluation

``` bash
python evaluation.py
```

## Metrics

-   F1 Score
-   Faithfulness
-   Answer Relevance
-   Accuracy Score

### Accuracy Formula

    accuracy_score = 0.3*F1 + 0.4*Faithfulness + 0.3*Relevance

------------------------------------------------------------------------

# 📁 Project Structure

    clinicaltrails/
    │
    ├── app.py
    ├── rag.py
    ├── ingestion.py
    ├── embeddings.py
    ├── hybrid_search.py
    ├── evaluation.py
    ├── auth.py
    ├── audit.py
    │
    ├── data/
    ├── evaluation_report.json
    └── audit_logs.jsonl

------------------------------------------------------------------------

# ⚠️ Troubleshooting

### Ollama Error

Ensure:

    "model": "qwen2.5:3b"

### No Data

Run:

    python ingestion.py

------------------------------------------------------------------------

# 🔐 Privacy

-   Fully local inference
-   No external APIs
-   Secure for sensitive data

------------------------------------------------------------------------

# 📈 Future Work

-   RAGAS integration
-   Re-ranking models
-   GPU optimization
-   Streaming responses

------------------------------------------------------------------------

# 👨‍💻 Author

Raghava Sai\
AI/ML Engineer \| RAG Systems
