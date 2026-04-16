from rank_bm25 import BM25Okapi
import numpy as np
import re


class HybridSearch:
    def __init__(self, documents, index=None, embed_model=None):
        self.documents = documents
        self.index = index
        self.embed_model = embed_model

        if not documents:
            print("⚠️ WARNING: No documents provided to HybridSearch")
            self.texts = []
            self.bm25 = None
            return

        # ✅ Clean + tokenize
        self.texts = [self._clean_text(doc["content"]) for doc in documents]
        tokenized = [text.split() for text in self.texts]

        self.bm25 = BM25Okapi(tokenized)

    # -----------------------------
    # Text Cleaning
    # -----------------------------
    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # -----------------------------
    # BM25 Search
    # -----------------------------
    def bm25_search(self, query, top_k):
        if self.bm25 is None:
            return [], []

        query = self._clean_text(query)
        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = [self.documents[i] for i in top_indices]
        scores = [float(scores[i]) for i in top_indices]

        return results, scores

    # -----------------------------
    # Vector Search
    # -----------------------------
    def vector_search(self, query, top_k):
        q_embed = self.embed_model.encode([query]).astype("float32")

        distances, indices = self.index.search(q_embed, top_k)

        results = [self.documents[i] for i in indices[0]]
        scores = [float(distances[0][i]) for i in range(len(indices[0]))]

        return results, scores

    # -----------------------------
    # Hybrid Search
    # -----------------------------
    def search(self, query, top_k=5):

        # ✅ FIX 1: check BM25 first
        if self.bm25 is None:
            print("⚠️ BM25 not initialized — no documents")
            return [], []

        bm25_results, bm25_scores = self.bm25_search(query, top_k)

        # ✅ VECTORLESS MODE
        if self.embed_model is None or self.index is None:
            return bm25_results, bm25_scores

        vector_results, vector_scores = self.vector_search(query, top_k)

        # -----------------------------
        # Combine scores
        # -----------------------------
        combined_dict = {}

        # BM25 weight
        for doc, score in zip(bm25_results, bm25_scores):
            combined_dict[doc["content"]] = 0.7 * score

        # Vector weight
        for doc, score in zip(vector_results, vector_scores):
            key = doc["content"]
            if key in combined_dict:
                combined_dict[key] += 0.3 * score
            else:
                combined_dict[key] = 0.3 * score

        # Sort
        sorted_docs = sorted(
            combined_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        final_results = []
        final_scores = []

        for content, score in sorted_docs:
            for doc in self.documents:
                if doc["content"] == content:
                    final_results.append(doc)
                    final_scores.append(float(score))
                    break

        return final_results, final_scores