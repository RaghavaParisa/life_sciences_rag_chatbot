import os
import re
import json
import time
from typing import List
import numpy as np
import requests

from embeddings import load_or_create_faiss
from rag import init_hybrid, retrieve, generate_answer
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore_score

# -----------------------------
# ENV
# -----------------------------
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "all-MiniLM-L6-v2")
embed_model = SentenceTransformer(MODEL_PATH)

DATA_DIR = os.path.join(BASE_DIR, "..", "data")
JSON_REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.json")

MODEL_NAME = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

EVAL_TASKS = [
    {
        "question": "What are the synonyms and the primary DrugBank ID for the drug Lepirudin?",
        "ground_truth": "[Leu1, Thr2]-63-desulfohirudin, Desulfatohirudin, Hirudin variant-1, Lepirudin, Lepirudin recombinant, R-hirudin",
        "expected_sources": ["drugbank vocabulary.csv"],
    },
    {
        "question": "Describe the purpose and the years of availability for the Mental Health Inventory: RAND Medical Outcomes Study found in the HSRR dataset",
        "ground_truth": "The study includes 37 measures that are part of the 116 core measures of functioning and well-being from the Medical Outcomes Study (MOS), a two-year study of patients with chronic conditions.",
        "expected_sources": ["drugbank vocabulary.csv"],
    },
    {
        "question": "Provide the DrugBank ID, common name, and one synonym for the entry starting with DB15629.",
        "ground_truth": "DrugBank ID: DB15629, Common Name: Garadacimab, Synonym: Immunoglobulin G4, anti-human blood-coagulation factor viia (human monoclonal CSL312 gamma4-chain, disulfide with human monoclonal CSL312 lambda-chain, dimer)",
        "expected_sources": ["drugbank vocabulary.csv"],
    },
    {
        "question": "What information does the Prescribed Drug Use dataset provide about MEPS?",
        "ground_truth": "MEPS collects information from survey participants about their prescription drugs and then are asked for permission to collect more detailed information from their pharmacies. At the pharmacies, data are collected on the type, dosage, and payment for each filled prescription. ... Prescribed drug expenditures in MEPS are defined as the sum of payments for care received, including out of pocket payments and payments made by private insurance, Medicaid, Medicare and other sources.",
        "expected_sources": ["Health_Services_and_Sciences_Research_Resources__HSRR__-_Archived_Data.csv"],
    },
    {
        "question": "What does B.R.I.D.G.E. TO DATA describe?",
        "ground_truth": "B.R.I.D.G.E. TO DATA is a unique non-profit online reference describing population healthcare databases for use in epidemiology and health outcomes research.",
        "expected_sources": ["Health_Services_and_Sciences_Research_Resources__HSRR__-_Archived_Data.csv"],
    },
    {
        "question": "What is the pooled mean EQ-5D-5L utility score for post-COVID HRQoL in India from the meta-analysis?",
        "ground_truth": "The pooled mean EQ-5D-5L utility score for post-COVID HRQoL in India from the meta-analysis was 0.65 (95% CI: 0.58-0.72).",
        "expected_sources": ["post_covid.pdf"],
    },
    {
        "question": "According to the post-COVID study, which group had poorer health-related quality of life?",
        "ground_truth": "Older adults, females, and patients with comorbidities had poorer health-related quality of life.",
        "expected_sources": ["post_covid.pdf"],
    },
    {
        "question": "Name three determinants consistently associated with impaired post-COVID HRQoL.",
        "ground_truth": "Older age, female sex, and presence of comorbidities were consistently associated with impaired post-COVID HRQoL.",
        "expected_sources": ["post_covid.pdf"],
    },
    {
        "question": " QODD assesses the quality of dying and death; versions include QODD - Version 1.0 (Significant Other after Death Interview) and QODD - Versions 3.2a (Family Member/Friend and Nursing After Death Self-Administered Questionnaires)",
        "ground_truth": "The Quality of Dying and Death (QODD) is a tool used to assess the quality of dying and death. It has different versions, including QODD - Version 1.0 (Significant Other after Death Interview) and QODD - Versions 3.2a (Family Member/Friend and Nursing After Death Self-Administered Questionnaires).",
        "expected_sources": ["Health_Services_and_Sciences_Research_Resources__HSRR__-_Archived_Data.csv"],
    },
    {
        "question": "Describe the EuroQol (EQ-5D) instrument and its primary purpose as listed in the HSRR data",
        "ground_truth": "The EuroQol (EQ-5D) is a standardized instrument used to measure health-related quality of life. Its primary purpose is to assess the health state of individuals across five dimensions: mobility, self-care, usual activities, pain/discomfort, and anxiety/depression.",
        "expected_sources": ["Health_Services_and_Sciences_Research_Resources__HSRR__-_Archived_Data.csv"],
    },
    {
        "question": "What causes age-related memory generalization in Drosophila according to the study?",
        "ground_truth": "Aberrant dopaminergic activity during memory consolidation; increased dopamine signaling from inability to inhibit glutamatergic activation leads to enhanced D2 receptor activation on engram cells.",
        "expected_sources": ["age_memory.pdf"],
    },
    {
        "question": " How do engram cells differ in reactivation between young and aged flies after spaced training?",
        "ground_truth": " In young flies, engram cells (c-Fos-positive) reactivate specifically with shock-paired odor (CS+); in aged flies, they reactivate similarly with both CS+ and unpaired/novel odors (CS-, novel), leading to memory generalization.",
        "expected_sources": ["age_memory.pdf"],
    },
]


# -----------------------------
# UTILS
# -----------------------------
def clean_answer(text: str) -> str:
    return text.split("Sources:")[0].strip() if text else ""


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9 ]+", "", text).strip()


# -----------------------------
# BERTScore
# -----------------------------
def compute_bertscore(prediction, reference):
    try:
        P, R, F1 = bertscore_score([prediction], [reference], lang="en")
        return float(F1[0])
    except:
        return 0.0


# -----------------------------
# EMBEDDING SIM
# -----------------------------
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -----------------------------
# FAITHFULNESS
# -----------------------------
def compute_faithfulness(answer, contexts):
    if not answer or not contexts:
        return 0.0

    context_text = " ".join(contexts[:3])

    a = embed_model.encode([answer])[0]
    c = embed_model.encode([context_text])[0]

    return cosine_sim(a, c)


# -----------------------------
# RELEVANCE
# -----------------------------
def compute_relevance(answer, question):
    if not answer:
        return 0.0

    a = embed_model.encode([answer])[0]
    q = embed_model.encode([question])[0]

    a = a / np.linalg.norm(a)
    q = q / np.linalg.norm(q)

    return float(np.dot(a, q))


# -----------------------------
# 🚨 HALLUCINATION SCORE (NEW)
# -----------------------------
def compute_groundedness(answer: str, contexts: List[str]):
    """
    Returns:
        1.0 → fully grounded (no hallucination)
        0.0 → fully hallucinated
    """
    if not answer or not contexts:
        return 0.0

    context_text = " ".join(contexts[:3])

    answer_sentences = re.split(r"[.?!]", answer)
    answer_sentences = [s.strip() for s in answer_sentences if s.strip()]

    grounded_count = 0

    context_emb = embed_model.encode([context_text])[0]

    for sent in answer_sentences:
        sent_emb = embed_model.encode([sent])[0]
        sim = cosine_sim(sent_emb, context_emb)

        if sim > 0.55:  # threshold
            grounded_count += 1

    return grounded_count / len(answer_sentences)


# -----------------------------
# LLM JUDGE
# -----------------------------
def llm_judge(question, answer, ground_truth, context):
    prompt = f"""
Evaluate answer quality.

Return JSON:
{{
"correctness": float,
"completeness": float,
"groundedness": float
}}

Q: {question}
GT: {ground_truth}
CTX: {context}
ANS: {answer}
"""

    try:
        res = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=60
        )

        txt = res.json()["response"]
        match = re.search(r"\{.*\}", txt, re.DOTALL)

        if match:
            return json.loads(match.group())

    except Exception as e:
        print("LLM Judge Error:", e)

    return {"correctness": 0, "completeness": 0, "groundedness": 0}


# -----------------------------
# MAIN EVALUATION
# -----------------------------
def evaluate():
    index, documents = load_or_create_faiss(DATA_DIR)
    init_hybrid(documents, index)

    results = []

    for task in EVAL_TASKS:
        question = task["question"]
        ground_truth = task["ground_truth"]

        print("\n======================")
        print("Q:", question)

        start = time.time()

        contexts, citations, _ = retrieve(question)
        raw = generate_answer(question, contexts, citations)
        answer = clean_answer(raw)

        latency = round(time.time() - start, 2)

        # -----------------------------
        # METRICS
        # -----------------------------
        bert_f1 = compute_bertscore(answer, ground_truth)
        faithfulness = compute_faithfulness(answer, contexts)
        relevance = compute_relevance(answer, question)
        groundness = compute_groundedness(answer, contexts)

        judge = llm_judge(
            question,
            answer,
            ground_truth,
            " ".join(contexts)
        )

        llm_score = (
            judge["correctness"] +
            judge["completeness"] +
            judge["groundedness"]
        ) / 3

        # -----------------------------
        # FINAL SCORE
        # -----------------------------
        accuracy = (
            0.30 * bert_f1 +
            0.25 * llm_score +
            0.20 * faithfulness +
            0.15 * relevance +
            0.10 * groundness   # 🔥 important
        )

        print("BERT:", round(bert_f1, 3))
        print("Faith:", round(faithfulness, 3))
        print("Relevance:", round(relevance, 3))
        print("Groundedness:", round(groundness, 3))
        print("LLM:", round(llm_score, 3))
        print("Final:", round(accuracy, 3))

        results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "accuracy": round(accuracy, 3),
            "bertscore": round(bert_f1, 3),
            "faithfulness": round(faithfulness, 3),
            "relevance": round(relevance, 3),
            "groundedness": round(groundedness, 3),
            "llm_score": round(llm_score, 3),
            "latency": latency,
        })

    output = {"results": results}

    with open(JSON_REPORT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ DONE")


if __name__ == "__main__":
    evaluate()