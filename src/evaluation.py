import os
import re
import json
import time
from typing import List
import numpy as np

from embeddings import load_or_create_faiss
from rag import init_hybrid, retrieve, generate_answer
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.txt")
JSON_REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.json")

USE_GRADIO_ANSWERS = True

MODEL_NAME = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

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
        "quesion": " How do engram cells differ in reactivation between young and aged flies after spaced training?",
        "ground_truth": " In young flies, engram cells (c-Fos-positive) reactivate specifically with shock-paired odor (CS+); in aged flies, they reactivate similarly with both CS+ and unpaired/novel odors (CS-, novel), leading to memory generalization.",
        "expected_sources": ["age_memory.pdf"],
    },
]
# -----------------------------
# TEXT UTILS
# -----------------------------
def clean_answer(text: str) -> str:
    if not text:
        return ""
    return text.split("Sources:")[0].strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9 ]+", "", text).strip()


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


# -----------------------------
# METRICS
# -----------------------------
def compute_token_metrics(prediction: str, reference: str):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    overlap = set(pred_tokens) & set(ref_tokens)

    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)

    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    if not answer.strip() or not contexts:
        return 0.0

    context_text = " ".join(contexts)

    answer_emb = embed_model.encode([answer])[0]
    context_emb = embed_model.encode([context_text])[0]

    return cosine_sim(answer_emb, context_emb)


def compute_answer_relevance(answer: str, question: str) -> float:
    if not answer.strip():
        return 0.0

    answer_emb = embed_model.encode([answer])[0]
    question_emb = embed_model.encode([question])[0]

    return cosine_sim(answer_emb, question_emb)


# -----------------------------
# LLM JUDGE (RAGAS STYLE)
# -----------------------------
def llm_judge(question, answer, ground_truth, context):
    prompt = f"""
You are an expert evaluator.

Evaluate the answer based on:
1. Correctness (0-1)
2. Completeness (0-1)
3. Groundedness (0-1)

Return JSON only:
{{
  "correctness": float,
  "completeness": float,
  "groundedness": float
}}

Question:
{question}

Ground Truth:
{ground_truth}

Context:
{context}

Answer:
{answer}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()["response"]

        # Extract JSON safely
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            return json.loads(match.group())

    except Exception as e:
        print("LLM Judge Error:", e)

    return {"correctness": 0, "completeness": 0, "groundedness": 0}


# -----------------------------
# GRADIO ANSWER
# -----------------------------
def get_gradio_answer(question):
    try:
        contexts, citations, _ = retrieve(question)
        return generate_answer(question, contexts, citations)
    except Exception as e:
        return f"Error: {str(e)}"


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate():
    index, documents = load_or_create_faiss(DATA_DIR)
    init_hybrid(documents, index)

    results = []
    totals = {
        "accuracy": 0,
        "f1": 0,
        "faithfulness": 0,
        "answer_relevance": 0,
        "llm_score": 0,
    }

    for task in EVAL_TASKS:
        question = task["question"]
        ground_truth = task["ground_truth"]

        print("\n==============================")
        print("QUESTION:", question)

        # Generate answer
        start = time.time()
        contexts, citations, _ = retrieve(question)
        eval_raw = generate_answer(question, contexts, citations)
        latency = round(time.time() - start, 2)

        eval_answer = clean_answer(eval_raw)

        # Gradio answer
        gradio_raw = get_gradio_answer(question)
        gradio_answer = clean_answer(gradio_raw)

        # Metrics
        precision, recall, f1 = compute_token_metrics(eval_answer, ground_truth)
        faithfulness = compute_faithfulness(eval_answer, contexts)
        relevance = compute_answer_relevance(eval_answer, question)

        # Accuracy score
        accuracy_score = (
            0.4 * f1 +
            0.3 * faithfulness +
            0.3 * relevance
        )

        if accuracy_score >= 0.75:
            accuracy_range = "0.8-1.0"
        elif accuracy_score >= 0.55:
            accuracy_range = "0.5-0.8"
        elif accuracy_score >= 0.35:
            accuracy_range = "0.3-0.5"
        else:
            accuracy_range = "0.0-0.3"

        # -----------------------------
        # LLM JUDGE
        # -----------------------------
        judge = llm_judge(
            question,
            eval_answer,
            ground_truth,
            " ".join(contexts)
        )

        llm_score = (
            judge["correctness"] +
            judge["completeness"] +
            judge["groundedness"]
        ) / 3

        # Print
        print("F1:", round(f1, 4))
        print("Faithfulness:", round(faithfulness, 4))
        print("Relevance:", round(relevance, 4))
        print("Accuracy Score:", round(accuracy_score, 4))
        print("LLM Score:", round(llm_score, 4))

        record = {
            "question": question,
            "evaluation_answer": eval_answer,
            "gradio_answer": gradio_answer,
            "accuracy_score": float(round(accuracy_score, 3)),
            "accuracy_range": accuracy_range,
            "f1": float(round(f1, 4)),
            "faithfulness": float(round(faithfulness, 4)),
            "answer_relevance": float(round(relevance, 4)),
            "llm_judge": judge,
            "llm_score": float(round(llm_score, 3)),
            "latency_sec": float(latency),
        }

        results.append(record)

        totals["accuracy"] += accuracy_score
        totals["f1"] += f1
        totals["faithfulness"] += faithfulness
        totals["answer_relevance"] += relevance
        totals["llm_score"] += llm_score

    # Summary
    n = len(results)

    summary = {
        "accuracy": round(totals["accuracy"] / n, 4),
        "f1": round(totals["f1"] / n, 4),
        "faithfulness": round(totals["faithfulness"] / n, 4),
        "answer_relevance": round(totals["answer_relevance"] / n, 4),
        "llm_score": round(totals["llm_score"] / n, 4),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    with open(JSON_REPORT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Evaluation complete")
    print("Saved to:", JSON_REPORT_PATH)


if __name__ == "__main__":
    evaluate()