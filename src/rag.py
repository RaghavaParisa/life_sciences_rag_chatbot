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
# INIT (BM25 ONLY)
# -----------------------------
def init_hybrid(documents, index=None):
    """
    Initialize HybridSearch WITHOUT embeddings (vectorless mode)
    """
    global hybrid
    hybrid = HybridSearch(documents, index=None, embed_model=None)


# -----------------------------
# RETRIEVE (BM25 ONLY)
# -----------------------------
def retrieve(query, top_k=7):
    """
    Pure keyword-based retrieval (vectorless)
    """
    results, scores = hybrid.search(query, top_k)

    unique_sources = set()
    diverse_results = []

    # Ensure diversity
    for doc in results:
        if doc["source"] not in unique_sources:
            diverse_results.append(doc)
            unique_sources.add(doc["source"])
        if len(diverse_results) == 5:
            break

    # Fallback
    if len(diverse_results) < 3:
        diverse_results = results[:5]

    contexts = []
    citations = []

    for doc in diverse_results:
        contexts.append(
            f"""
Source: {doc['source']}
Page: {doc.get('page', 'N/A')}

Content:
{doc['content']}
"""
        )

        source = doc["source"]
        page = doc.get("page")

        if page:
            citations.append(f"{source} (Page {page})")
        else:
            citations.append(source)

    return contexts, citations, scores


# -----------------------------
# GENERATE (OLLAMA)
# -----------------------------
def generate_answer(query, context, citations, model=None):
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
    print(f"DEBUG: Generating with Ollama model {model_name}")
    print(f"DEBUG: Prompt length = {len(prompt)}")

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,   # ✅ FIX: must be string
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,   # 🔥 lower = less hallucination
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_predict": 200
                }
            },
            timeout=120
        )

        if response.status_code != 200:
            print(f"❌ HTTP ERROR: {response.text}")
            answer = "Error generating response from local model."
        else:
            result = response.json()
            answer = result.get("response", "").strip()

    except Exception as e:
        print(f"❌ ERROR calling Ollama: {e}")
        answer = "Error generating response from local model."

    # -----------------------------
    # ADD CITATIONS
    # -----------------------------
    citation_text = "\n\nSources:\n" + "\n".join(set(citations))
    final_answer = answer + citation_text

    # -----------------------------
    # AUDIT LOG
    # -----------------------------
    log_interaction(
        user="default_user",
        query=query,
        answer=final_answer,
        sources=citations
    )

    return final_answer