import os
import requests
from hybrid_search import HybridSearch
from audit import log_interaction

session = requests.Session()
# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_MODEL = os.getenv("RAG_MODEL", "qwen2.5:3b")
OLLAMA_URL = "http://localhost:11434/api/generate"

print(f"DEBUG: Vectorless RAG using model={OLLAMA_MODEL}")

hybrid = None


# -----------------------------
# INIT
# -----------------------------
def init_hybrid(documents, index=None):
    global hybrid

    if not documents:
        print("⚠️ Empty documents passed to init_hybrid")
        hybrid = None
        return

    hybrid = HybridSearch(documents, index=index, embed_model=None)


def get_hybrid():
    global hybrid
    if hybrid is None:
        raise Exception("❌ Hybrid not initialized. Call init_hybrid() first.")
    return hybrid


def trim_text(text, max_chars=800):
    """
    Trim long content to reduce LLM input size.
    """
    return text[:max_chars]

# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve(query, top_k=4):
    global hybrid

    if hybrid is None:
        raise Exception("❌ Hybrid not initialized. Call init_hybrid() first.")

    results, scores = hybrid.search(query, top_k)

    if not results:
        return [], [], []

    unique_sources = set()
    diverse_results = []

    for doc in results:
        if doc["source"] not in unique_sources:
            diverse_results.append(doc)
            unique_sources.add(doc["source"])
        if len(diverse_results) == 3:
            break

    if len(diverse_results) < 2:
        diverse_results = results[:3]

    contexts = []
    citations = []

    for doc in diverse_results:
        trimmed_content = trim_text(doc['content'], max_chars=800)
        contexts.append(f"""
Source: {doc['source']}
Page: {doc.get('page', 'N/A')}

Content:
{trimmed_content}
""")

        source = doc["source"]
        page = doc.get("page")

        citations.append(f"{source} (Page {page})" if page else source)

    return contexts, citations, scores


# -----------------------------
# GENERATE
# -----------------------------
def generate_answer(query, context, citations, model=None):
    if not context:
        return "⚠️ No relevant documents found."

    # context_text = "\n".join(context)
    MAX_CONTEXT_CHARS = 2500

    context_text = ""
    for ctx in context:
        if len(context_text) + len(ctx) > MAX_CONTEXT_CHARS:
            break
        context_text += ctx + "\n"

    prompt = f"""
You are a strict Life Sciences RAG assistant.

Rules:
- Answer ONLY from provided context
- If answer not present → say "Not found in context"
- Do NOT hallucinate
- Be concise and accurate

=====================
CONTEXT:
{context_text}
=====================

QUESTION:
{query}

Answer:
"""

    # model_name = model or OLLAMA_MODEL
    if len(context_text) < 1000:
        model_name = "qwen2.5:3b"
    else:
        model_name = model or "qwen2.5:7b"

    try:
        response = session.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 200
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            answer = "❌ Error from model. Please Try again later."
        else:
            answer = response.json().get("response", "").strip()

    except Exception as e:
        print("❌ Ollama error:", e)
        answer = "❌ Model connection error"

    citation_text = "\n\nSources:\n" + "\n".join(set(citations))
    final_answer = answer

    log_interaction(
        user="default_user",
        query=query,
        answer=final_answer,
        sources=citations
    )

    return final_answer