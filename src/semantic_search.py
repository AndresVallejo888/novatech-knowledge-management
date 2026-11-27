"""Semantic (vector-space) search: TF-IDF + cosine similarity.

Run:  python src/semantic_search.py                -> demo queries
      python src/semantic_search.py "your query"   -> custom query
"""
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_documents


class SemanticSearchEngine:
    def __init__(self):
        self.texts, self.labels, self.names = load_documents()
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)   # documents -> TF-IDF vectors

    def search(self, query, top_k=5):
        q_vec = self.vectorizer.transform([query])                     # query -> TF-IDF vector
        scores = cosine_similarity(q_vec, self.doc_matrix).ravel()     # cosine similarity
        ranked = scores.argsort()[::-1][:top_k]
        return [(self.names[i], self.labels[i], float(scores[i]), self.texts[i])
                for i in ranked if scores[i] > 0]

    def show(self, query, top_k=5):
        print(f'\nQuery: "{query}"')
        print("-" * 78)
        results = self.search(query, top_k)
        if not results:
            print("  No relevant documents found.")
        for rank, (name, cat, score, text) in enumerate(results, 1):
            print(f"  {rank}. [{score:.3f}] {name}  ({cat})")
            print(f"       {text[:95]}...")


if __name__ == "__main__":
    engine = SemanticSearchEngine()
    print(f"Index built: {len(engine.names)} documents, {len(engine.vectorizer.vocabulary_)} terms")
    queries = sys.argv[1:] or [
        "How many days of vacation do employees get?",
        "How do I reset my forgotten password?",
        "What are the risks and budget status of the project?",
        "How is our infrastructure deployed in the cloud?",
        "What should we do after a data breach?",
    ]
    for q in queries:
        engine.show(q)
