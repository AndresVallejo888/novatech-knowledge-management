"""Natural Language Knowledge Assistant / Chatbot (Phase 2)."""
from __future__ import annotations
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recommender import DocumentRecommender
from utils import load_ontology_triples

DOMAIN_SYNONYMS = {
    "breach": "incident response security breach containment gdpr",
    "hack": "incident response security breach access control",
    "attack": "incident response security breach threat",
    "pto": "vacation leave annual days time off",
    "holiday": "vacation leave days",
    "wfh": "remote work policy home vpn",
    "home": "remote work policy",
    "salary": "compensation benefits pay",
    "pay": "compensation benefits",
    "money": "budget financial revenue expenses",
    "cost": "budget financial expenses",
    "deploy": "ci/cd pipeline docker kubernetes cloud infrastructure",
    "server": "cloud infrastructure aws kubernetes",
    "login": "password reset authentication access control intranet",
    "forgot": "password reset instructions",
    "client": "customer satisfaction survey crm",
    "new hire": "onboarding recruitment orientation",
}


class KnowledgeChatbot:
    def __init__(self, recommender: DocumentRecommender | None = None):
        self.recommender = recommender or DocumentRecommender()
        self.triples = load_ontology_triples()

    def expand_query(self, query: str) -> tuple[str, list[str]]:
        lower = query.lower()
        expansions = [extra for trigger, extra in DOMAIN_SYNONYMS.items() if re.search(rf"\b{re.escape(trigger)}\b", lower)]
        matched_concepts = set()
        for _, row in self.triples.iterrows():
            subj, obj = str(row["subject"]), str(row["object"])
            if subj.lower() in lower:
                matched_concepts.add(obj)
            if obj.lower() in lower:
                matched_concepts.add(subj)
        if matched_concepts:
            expansions.append(" ".join(sorted(matched_concepts)))
        return f"{query} {' '.join(expansions)}".strip(), sorted(matched_concepts)

    def retrieve(self, query: str, top_k: int = 3) -> tuple[list[dict], list[str]]:
        expanded_query, ontology_concepts = self.expand_query(query)
        q_tfidf, q_dense = self.recommender._embed_query(expanded_query)
        scores = 0.65 * cosine_similarity(q_tfidf, self.recommender.tfidf_matrix).ravel() + 0.35 * cosine_similarity(q_dense, self.recommender.semantic_embeddings).ravel()
        ranked = np.argsort(scores)[::-1][:top_k]
        hits = [{**self.recommender.catalog[idx], "score": round(float(scores[idx]), 4)} for idx in ranked if scores[idx] > 0.03]
        return hits, ontology_concepts

    def answer(self, query: str, top_k: int = 3) -> dict:
        q_clean = query.strip()
        if not q_clean:
            return {"answer": "Please enter a question about NovaTech documents.", "sources": [], "ontology_concepts": [], "confidence": 0.0}
        hits, ontology_concepts = self.retrieve(q_clean, top_k=top_k)
        if not hits:
            return {"answer": "No matching documents found. Try keywords like *vacation*, *password*, *AWS*, or *incident response*.", "sources": [], "ontology_concepts": ontology_concepts, "confidence": 0.0}

        primary = hits[0]
        sents = [s.strip() for s in primary["text"].split(".") if s.strip()]
        core_fact = ". ".join(sents[1:]) + "." if len(sents) > 1 else primary["text"]
        lines = [
            f"Based on **{primary['title']}** (`{primary['category_display']}`), here is what NovaTech's documentation states:",
            "",
            f"> {core_fact}",
            "",
        ]
        if len(hits) > 1:
            lines.append("**Additional relevant context from the repository:**")
            for h in hits[1:]:
                hs = [s.strip() for s in h["text"].split(".") if s.strip()]
                snip = (hs[1] + ".") if len(hs) > 1 else h["text"]
                lines.append(f"- **{h['title']}** (`{h['name']}.txt`, relevance `{h['score']:.2f}`): {snip}")
        if ontology_concepts:
            lines.append("")
            lines.append(f"*Ontology expansion applied across linked domain concepts:* **{', '.join(ontology_concepts)}**.")
        return {"answer": "\n".join(lines), "sources": hits, "ontology_concepts": ontology_concepts, "confidence": primary["score"]}