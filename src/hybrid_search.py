import numpy as np
import re
from rank_bm25 import BM25Okapi


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

        # ✅ Clean once (good)
        self.texts = [self._clean_text(doc["content"]) for doc in documents]
        tokenized = [text.split() for text in self.texts]

        self.bm25 = BM25Okapi(tokenized)

        # ✅ Pre-map content → doc (CRITICAL optimization)
        self.doc_lookup = {doc["content"]: doc for doc in documents}

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

        # ✅ Faster than full sort
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [self.documents[i] for i in top_indices]
        scores = [float(scores[i]) for i in top_indices]

        return results, scores

    # -----------------------------
    # Vector Search
    # -----------------------------
    def vector_search(self, query, top_k):
        # ✅ Batch-safe + faster
        q_embed = self.embed_model.encode([query], batch_size=1).astype("float32")

        distances, indices = self.index.search(q_embed, top_k)

        results = [self.documents[i] for i in indices[0]]
        scores = [float(distances[0][i]) for i in range(len(indices[0]))]

        return results, scores

    # -----------------------------
    # Hybrid Search
    # -----------------------------
    def search(self, query, top_k=4):

        if self.bm25 is None:
            print("⚠️ BM25 not initialized — no documents")
            return [], []

        # ✅ Reduce candidate size for speed
        candidate_k = max(top_k, 4)

        bm25_results, bm25_scores = self.bm25_search(query, candidate_k)

        # VECTORLESS MODE (fast path)
        if self.embed_model is None or self.index is None:
            return bm25_results[:top_k], bm25_scores[:top_k]

        vector_results, vector_scores = self.vector_search(query, candidate_k)

        # -----------------------------
        # Normalize scores (IMPORTANT)
        # -----------------------------
        def normalize(scores):
            if not scores:
                return scores
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [1.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        bm25_scores = normalize(bm25_scores)
        vector_scores = normalize(vector_scores)

        # -----------------------------
        # Combine
        # -----------------------------
        combined_dict = {}

        for doc, score in zip(bm25_results, bm25_scores):
            combined_dict[doc["content"]] = 0.7 * score

        for doc, score in zip(vector_results, vector_scores):
            key = doc["content"]
            combined_dict[key] = combined_dict.get(key, 0) + (0.3 * score)

        # ✅ Sort top_k only
        sorted_docs = sorted(combined_dict.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        final_results = []
        final_scores = []

        # ✅ FAST lookup instead of nested loop
        for content, score in sorted_docs:
            doc = self.doc_lookup.get(content)
            if doc:
                final_results.append(doc)
                final_scores.append(float(score))

        return final_results, final_scores
