"""Document Recommendation System (Phase 2)."""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from utils import build_corpus_catalog


class DocumentRecommender:
    def __init__(self, n_components: int = 25):
        self.catalog = build_corpus_catalog()
        self.texts = [d["text"] for d in self.catalog]
        self.names = [d["name"] for d in self.catalog]
        self.labels = [d["category"] for d in self.catalog]

        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

        k = min(n_components, self.tfidf_matrix.shape[0] - 1, self.tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=max(2, k), random_state=42)
        self.semantic_embeddings = normalize(self.svd.fit_transform(self.tfidf_matrix))

        tfidf_sim = cosine_similarity(self.tfidf_matrix)
        lsa_sim = cosine_similarity(self.semantic_embeddings)
        concept_sim = self._compute_concept_similarity()
        self.doc_similarity = 0.50 * tfidf_sim + 0.35 * lsa_sim + 0.15 * concept_sim

    def _compute_concept_similarity(self) -> np.ndarray:
        n = len(self.catalog)
        sim = np.zeros((n, n), dtype=float)
        for i in range(n):
            ci = set(self.catalog[i]["concepts"])
            for j in range(n):
                cj = set(self.catalog[j]["concepts"])
                if ci and cj:
                    sim[i, j] = len(ci & cj) / len(ci | cj)
        return sim

    def _embed_query(self, query: str):
        q_tfidf = self.vectorizer.transform([query])
        q_dense = normalize(self.svd.transform(q_tfidf))
        return q_tfidf, q_dense

    def recommend_by_document(self, doc_name: str, top_k: int = 5) -> list[dict]:
        if doc_name not in self.names:
            return []
        idx = self.names.index(doc_name)
        scores = self.doc_similarity[idx].copy()
        scores[idx] = -1.0
        ranked = np.argsort(scores)[::-1][:top_k]

        ref_concepts = set(self.catalog[idx]["concepts"])
        recs = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            shared = sorted(ref_concepts & set(self.catalog[i]["concepts"]))
            reason = f"Shared concepts: {', '.join(shared)}" if shared else f"Semantic similarity in {self.catalog[i]['category_display']}"
            recs.append({**self.catalog[i], "score": round(float(scores[i]), 4), "reason": reason})
        return recs

    def recommend_by_context(self, query_history=None, viewed_docs=None, top_k: int = 5) -> list[dict]:
        query_history = [q.strip() for q in (query_history or []) if q.strip()]
        viewed_docs = [d for d in (viewed_docs or []) if d in self.names]

        if not query_history and not viewed_docs:
            seen, fallback = set(), []
            for doc in self.catalog:
                if doc["category"] not in seen:
                    seen.add(doc["category"])
                    fallback.append({**doc, "score": 0.50, "reason": f"Featured starter guide for {doc['category_display']}"})
                if len(fallback) >= top_k:
                    break
            return fallback

        combined = np.zeros(len(self.catalog), dtype=float)
        if query_history:
            weights = np.linspace(0.5, 1.0, len(query_history))
            weights /= weights.sum()
            for w, q in zip(weights, query_history):
                qt, qd = self._embed_query(q)
                combined += w * (0.60 * cosine_similarity(qt, self.tfidf_matrix).ravel() + 0.40 * cosine_similarity(qd, self.semantic_embeddings).ravel())

        if viewed_docs:
            v_scores = np.zeros(len(self.catalog), dtype=float)
            for d_name in viewed_docs:
                idx = self.names.index(d_name)
                v_scores += self.doc_similarity[idx]
                v_scores[idx] -= 0.40
            v_scores /= max(1, len(viewed_docs))
            wq = 0.65 if query_history else 0.0
            combined = wq * combined + (1.0 - wq) * v_scores

        ranked = np.argsort(combined)[::-1][:top_k]
        results = []
        for i in ranked:
            if combined[i] <= 0:
                continue
            parts = []
            if query_history:
                parts.append(f"Matches search intent ('{query_history[-1][:32]}')")
            if viewed_docs:
                parts.append(f"Related to {len(viewed_docs)} viewed doc(s)")
            results.append({**self.catalog[i], "score": round(float(combined[i]), 4), "reason": " + ".join(parts)})
        return results