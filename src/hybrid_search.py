from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:
    def __init__(self, documents, index, embed_model):
        self.documents = documents
        self.index = index
        self.embed_model = embed_model

        # Prepare BM25 corpus
        self.texts = [doc["content"] for doc in documents]
        tokenized = [text.split() for text in self.texts]

        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=3):
        # BM25 scores
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # ✅ Normalize BM25 scores (0–1)
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
            np.max(bm25_scores) - np.min(bm25_scores) + 1e-8
        )

        # FAISS scores
        q_embed = self.embed_model.encode([query]).astype("float32")
        faiss_k = 20  # or 30
        distances, indices = self.index.search(q_embed, faiss_k)

        faiss_scores = np.zeros(len(self.documents))

        raw_scores = []
        for i, idx in enumerate(indices[0]):
            score = distances[0][i]  # already cosine similarity
            faiss_scores[idx] = score
            raw_scores.append(score)

        # ✅ Normalize FAISS scores (only for retrieved docs)
        if raw_scores:
            min_f, max_f = min(raw_scores), max(raw_scores)
            for i, idx in enumerate(indices[0]):
                faiss_scores[idx] = (faiss_scores[idx] - min_f) / (max_f - min_f + 1e-8)

        # Combine scores
        combined = 0.7 * bm25_scores + 0.3 * faiss_scores

        top_indices = np.argsort(combined)[-top_k:][::-1]

        # ✅ ADD DEBUG PRINTS HERE
        print("\n Retrieved Documents:")
        for i in top_indices:
            print("SOURCE:", self.documents[i].get("source", "N/A"))
            print("CONTENT:", self.documents[i]["content"][:200])
            print("SCORE:", combined[i])
            print("-" * 50)
            
        print("\n Top 10 FAISS Results:")
        for i, idx in enumerate(indices[0][:10]):
            print(self.documents[idx]["source"], "->", distances[0][i])

        results = [self.documents[i] for i in top_indices]
        scores = [combined[i] for i in top_indices]

        return results, scores