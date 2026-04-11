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

# 🔥 Toggle Gradio comparison
USE_GRADIO_ANSWERS = True
# ✅ Use Ollama model
MODEL_NAME = "qwen2.5:3b"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

EVAL_TASKS = [
    {
        "question": "What are the synonyms and the primary DrugBank ID for the drug Lepirudin",
        "ground_truth": "[Leu1, Thr2]-63-desulfohirudin, Desulfatohirudin, Hirudin variant-1, Lepirudin, Lepirudin recombinant, R-hirudin",
        "expected_sources": ["drugbank vocabulary.csv"],
    },
    {
        "question": "Describe the purpose and the years of availability for the Mental Health Inventory: RAND Medical Outcomes Study found in the HSRR dataset",
        "ground_truth": "The study includes 37 measures that are part of the 116 core measures of functioning and well-being from the Medical Outcomes Study (MOS), a two-year study of patients with chronic conditions.",
        "expected_sources": ["drugbank vocabulary.csv"],
    },
    {
        "question": "Does the HSRR dataset have any resources related to 'Depressive Disorder'? If so, identify a resource and check if the drug Fluoxetine (mentioned in some HSRR descriptions) has a corresponding entry in the DrugBank vocabulary. Provide its DrugBank ID.",
        "ground_truth": "The \"Mental Health Inventory: RAND Medical Outcomes Study\" lists \"Depressive Disorder\" in its keywords. Additionally, \"Consumer Reports/Health\" mentions Fluoxetine in its description.",
        "expected_sources": ["Health_Services_and_Sciences_Research_Resources__HSRR__-_Archived_Data.csv", "drugbank vocabulary.csv"],
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
        "question": "What can you tell me about the age related memory issue?",
        "ground_truth": "Age-related memory issues are not just about \"forgetting,\" but specifically about decreased memory specificity caused by overactive dopaminergic neurons during the consolidation phase. This suggests that memory generalization is a regulated biological process that can potentially be manipulated or corrected.",
        "expected_sources": ["age related memory issues.pdf"],
    },
    {
        "question": "According to the post-COVID study, which group had poorer health-related quality of life?",
        "ground_truth": "Older adults, females, and patients with comorbidities had poorer health-related quality of life.",
        "expected_sources": ["post_covid.pdf"],
    },
]

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_answer(text: str) -> str:
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
    if not answer.strip():
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
    }

    for task in EVAL_TASKS:
        question = task["question"]
        ground_truth = task["ground_truth"]

        print("\n==============================")
        print("QUESTION:", question)

        # -----------------------------
        # Evaluation Answer
        # -----------------------------
        start = time.time()
        contexts, citations, _ = retrieve(question)
        eval_raw = generate_answer(question, contexts, citations)
        latency = round(time.time() - start, 2)

        eval_answer = clean_answer(eval_raw)

        # -----------------------------
        # Gradio Answer
        # -----------------------------
        gradio_raw = get_gradio_answer(question)
        gradio_answer = clean_answer(gradio_raw)

        print("EVAL ANSWER:", eval_answer[:150])
        print("GRADIO ANSWER:", gradio_answer[:150])

        # -----------------------------
        # METRICS (use evaluation answer)
        # -----------------------------
        precision, recall, f1 = compute_token_metrics(eval_answer, ground_truth)
        faithfulness = compute_faithfulness(eval_answer, contexts)
        relevance = compute_answer_relevance(eval_answer, question)

        # ✅ Improved accuracy
        # accuracy = int(f1 > 0.35 and faithfulness > 0.6 and relevance > 0.6)
        # -----------------------------
        # COMBINED ACCURACY SCORE
        # -----------------------------
        accuracy_score = (
            0.4 * f1 +
            0.3 * faithfulness +
            0.3 * relevance
        )

        # -----------------------------
        # RANGE-BASED ACCURACY
        # -----------------------------
        if accuracy_score > 0.75:
            accuracy_range = "0.8-1.0"
        elif accuracy_score > 0.5:
            accuracy_range = "0.5-0.7"
        else:
            accuracy_range = "0.0-0.5"

        # -----------------------------
        # PRINT METRICS
        # -----------------------------
        print("F1:", round(f1, 4))
        print("Faithfulness:", round(faithfulness, 4))
        print("Relevance:", round(relevance, 4))
        print("Accuracy Score:", round(accuracy_score, 4))
        print("Accuracy Range:", accuracy_range)

        # -----------------------------
        # STORE RESULT
        # -----------------------------
        record = {
            "question": question,
            "ground_truth": ground_truth,
            "evaluation_answer": eval_answer,
            "gradio_answer": gradio_answer,
            "accuracy_score": float(round(accuracy_score, 3)),
            "accuracy_range": accuracy_range,   # ✅ FIX: missing comma added
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1": float(round(f1, 4)),
            "faithfulness": float(round(faithfulness, 4)),
            "answer_relevance": float(round(relevance, 4)),
            "latency_sec": float(latency),
        }

        results.append(record)

        # -----------------------------
        # TOTALS (IMPORTANT FIX)
        # -----------------------------
        totals["accuracy"] += accuracy_score 
        totals["f1"] += f1
        totals["faithfulness"] += faithfulness
        totals["answer_relevance"] += relevance

    # -----------------------------
    # SUMMARY
    # -----------------------------
    n = len(results)

    summary = {
        "accuracy": float(round(totals["accuracy"] / n, 4)),
        "f1": float(round(totals["f1"] / n, 4)),
        "faithfulness": float(round(totals["faithfulness"] / n, 4)),
        "answer_relevance": float(round(totals["answer_relevance"] / n, 4)),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    # Save JSON
    with open(JSON_REPORT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Save text report
    with open(REPORT_PATH, "w") as f:
        f.write("RAG Evaluation Report\n\n")

        for r in results:
            f.write("====================================\n")
            f.write(f"Question: {r['question']}\n\n")
            f.write(f"Evaluation Answer:\n{r['evaluation_answer']}\n\n")
            f.write(f"Gradio Answer:\n{r['gradio_answer']}\n\n")
            f.write(f"Accuracy: {r['accuracy_score']}\n")
            f.write(f"F1: {r['f1']}\n")
            f.write(f"Faithfulness: {r['faithfulness']}\n")
            f.write(f"Answer Relevance: {r['answer_relevance']}\n\n")

    print("\n✅ Evaluation complete")
    print("Saved to:", JSON_REPORT_PATH)


if __name__ == "__main__":
    evaluate()