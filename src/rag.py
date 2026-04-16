import os
import requests
from hybrid_search import HybridSearch
from audit import log_interaction

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


# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve(query, top_k=7):
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
        if len(diverse_results) == 5:
            break

    if len(diverse_results) < 3:
        diverse_results = results[:5]

    contexts = []
    citations = []

    for doc in diverse_results:
        contexts.append(f"""
Source: {doc['source']}
Page: {doc.get('page', 'N/A')}

Content:
{doc['content']}
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

    context_text = "\n".join(context)

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

    model_name = model or OLLAMA_MODEL

    try:
        response = requests.post(
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
            timeout=120
        )

        if response.status_code != 200:
            answer = "❌ Error from model. Please Try again later."
        else:
            answer = response.json().get("response", "").strip()

    except Exception as e:
        print("❌ Ollama error:", e)
        answer = "❌ Model connection error"

    citation_text = "\n\nSources:\n" + "\n".join(set(citations))
    final_answer = answer + citation_text

    log_interaction(
        user="default_user",
        query=query,
        answer=final_answer,
        sources=citations
    )

    return final_answer