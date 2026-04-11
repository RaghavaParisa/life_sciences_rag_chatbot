# import os
# import numpy as np
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from sentence_transformers import SentenceTransformer
# from hybrid_search import HybridSearch
# from audit import log_interaction

# # ✅ Use Qwen model
# LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# local_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"DEBUG: Provider=local, local_model={LOCAL_MODEL}, device={local_device}")

# # ✅ Load tokenizer
# local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, trust_remote_code=True)

# # Fix padding (VERY IMPORTANT for Qwen)
# if local_tokenizer.pad_token is None:
#     local_tokenizer.pad_token = local_tokenizer.eos_token

# # ✅ Load causal model
# local_model = AutoModelForCausalLM.from_pretrained(
#     LOCAL_MODEL,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
# ).to(local_device)

# MODEL = os.getenv("RAG_MODEL", LOCAL_MODEL)

# # Embeddings
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# hybrid = None


# def init_hybrid(documents, index):
#     global hybrid
#     hybrid = HybridSearch(documents, index, embed_model)


# def retrieve(query, top_k=7):
#     results, scores = hybrid.search(query, top_k)

#     unique_sources = set()
#     diverse_results = []

#     for doc in results:
#         if doc["source"] not in unique_sources:
#             diverse_results.append(doc)
#             unique_sources.add(doc["source"])
#         if len(diverse_results) == 5:
#             break

#     if len(diverse_results) < 3:
#         diverse_results = results[:5]

#     contexts = []
#     citations = []

#     for doc in diverse_results:
#         contexts.append(
#             f"""
# Source: {doc['source']}
# Page: {doc.get('page', 'N/A')}

# Content:
# {doc['content']}
# """
#         )

#         source = doc["source"]
#         page = doc.get("page")

#         if page:
#             citations.append(f"{source} (Page {page})")
#         else:
#             citations.append(source)

#     return contexts, citations, scores


# def generate_answer(query, context, citations, model=None):
#     context_text = "\n".join(context)

#     # ✅ Improved prompt (better for Qwen)
#     prompt = f"""
# You are a strict Life Sciences RAG assistant.

# Follow rules strictly:
# - Answer ONLY from provided context
# - If not in context → say "Not found in context"
# - Combine multiple sources if available
# - Do NOT hallucinate

# CONTEXT:
# {context_text}

# QUESTION:
# {query}

# Answer:
# """

#     local_model_name = model or MODEL
#     print(f"DEBUG: Generating with local model {local_model_name}")

#     tokenizer = local_tokenizer
#     model_obj = local_model

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=4096
#     ).to(local_device)

#     outputs = model_obj.generate(
#         **inputs,
#         max_new_tokens=300,
#         temperature=0.3,          # ✅ reduces hallucination
#         top_p=0.9,
#         do_sample=True,
#         repetition_penalty=1.1,
#         eos_token_id=tokenizer.eos_token_id
#     )

#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Remove prompt from output
#     answer = answer[len(prompt):].strip()

#     # Add citations
#     citation_text = "\n\nSources:\n" + "\n".join(set(citations))
#     final_answer = answer + citation_text

#     print("Calling audit logger...")
#     log_interaction(
#         user="default_user",
#         query=query,
#         answer=final_answer,
#         sources=citations
#     )

#     return final_answer


############################################################OLLAMA VERSION (SIMPLER, NO LOCAL MODEL LOADING)############################################################
# import os
# import requests
# from sentence_transformers import SentenceTransformer
# from hybrid_search import HybridSearch
# from audit import log_interaction

# # ✅ Ollama model name
# OLLAMA_MODEL = os.getenv("RAG_MODEL", "qwen2.5:3b")

# print(f"DEBUG: Using Ollama model={OLLAMA_MODEL}")

# # ✅ Embedding model (unchanged)
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# hybrid = None


# # -----------------------------
# # Initialize Hybrid Search
# # -----------------------------
# def init_hybrid(documents, index):
#     global hybrid
#     hybrid = HybridSearch(documents, index, embed_model)


# # -----------------------------
# # Retrieve Documents
# # -----------------------------
# def retrieve(query, top_k=7):
#     results, scores = hybrid.search(query, top_k)

#     unique_sources = set()
#     diverse_results = []

#     # Ensure diversity of sources
#     for doc in results:
#         if doc["source"] not in unique_sources:
#             diverse_results.append(doc)
#             unique_sources.add(doc["source"])
#         if len(diverse_results) == 5:
#             break

#     # Fallback if too few
#     if len(diverse_results) < 3:
#         diverse_results = results[:5]

#     contexts = []
#     citations = []

#     for doc in diverse_results:
#         contexts.append(
#             f"""
# Source: {doc['source']}
# Page: {doc.get('page', 'N/A')}

# Content:
# {doc['content']}
# """
#         )

#         source = doc["source"]
#         page = doc.get("page")

#         if page:
#             citations.append(f"{source} (Page {page})")
#         else:
#             citations.append(source)

#     return contexts, citations, scores


# # -----------------------------
# # Generate Answer (Ollama)
# # -----------------------------
# def generate_answer(query, context, citations, model=None):
#     context_text = "\n".join(context)

#     prompt = f"""
# You are a strict Life Sciences RAG assistant.

# Rules:
# - Answer ONLY from provided context
# - If answer not present → say "Not found in context"
# - Combine multiple sources if available
# - Do NOT hallucinate

# =====================
# CONTEXT:
# {context_text}
# =====================

# QUESTION:
# {query}

# Answer:
# """

#     model_name = model or OLLAMA_MODEL
#     print(f"DEBUG: Generating with Ollama model {model_name}")

#     try:
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={
#                 "model": model_name,
#                 "prompt": prompt,
#                 "stream": False,
#                 "options": {
#                     "temperature": 0.3,
#                     "top_p": 0.9,
#                     "repeat_penalty": 1.1,
#                     "num_predict": 200   # limits response length
#                 }
#             },
#             timeout=120
#         )

#         response.raise_for_status()
#         result = response.json()
#         answer = result.get("response", "").strip()

#     except Exception as e:
#         print(f"ERROR calling Ollama: {e}")
#         answer = "Error generating response from local model."

#     # -----------------------------
#     # Add citations
#     # -----------------------------
#     citation_text = "\n\nSources:\n" + "\n".join(set(citations))
#     final_answer = answer + citation_text

#     # -----------------------------
#     # Audit Logging
#     # -----------------------------
#     print("Calling audit logger...")
#     log_interaction(
#         user="default_user",
#         query=query,
#         answer=final_answer,
#         sources=citations
#     )

#     return final_answer


import os
import requests
from sentence_transformers import SentenceTransformer
from hybrid_search import HybridSearch
from audit import log_interaction

# -----------------------------
# Config
# -----------------------------
OLLAMA_MODEL = os.getenv("RAG_MODEL", "qwen2.5:3b")
OLLAMA_URL = "http://localhost:11434/api/generate"

print(f"DEBUG: Using Ollama model={OLLAMA_MODEL}")

# -----------------------------
# Embeddings
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

hybrid = None


# -----------------------------
# Init Hybrid Search
# -----------------------------
def init_hybrid(documents, index):
    global hybrid
    hybrid = HybridSearch(documents, index, embed_model)


# -----------------------------
# Retrieve
# -----------------------------
def retrieve(query, top_k=7):
    results, scores = hybrid.search(query, top_k)

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
# Generate Answer (Ollama)
# -----------------------------
def generate_answer(query, context, citations, model=None):
    MAX_CONTEXT_DOCS = 3
    MAX_CONTEXT_CHARS = 3000

    trimmed_contexts = [c[:1000] for c in context[:MAX_CONTEXT_DOCS]]
    context_text = "\n".join(trimmed_contexts)[:MAX_CONTEXT_CHARS]

    prompt = f"""
You are a strict Life Sciences RAG assistant.

Rules:
- Answer ONLY from provided context
- If answer not present → say "Not found in context"
- Combine multiple sources if available
- Do NOT hallucinate

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
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_predict": 200
                }
            },
            timeout=120
        )

        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()

    except requests.exceptions.HTTPError as e:
        print("❌ HTTP ERROR:", e.response.text)
        answer = "Error generating response from local model."

    except Exception as e:
        print("❌ GENERAL ERROR:", str(e))
        answer = "Error generating response from local model."

    citation_text = "\n\nSources:\n" + "\n".join(set(citations))
    final_answer = answer + citation_text

    log_interaction(
        user="default_user",
        query=query,
        answer=final_answer,
        sources=citations
    )

    return final_answer