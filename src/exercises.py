"""Practical exercises supporting the project (classification tests, retrieval analysis).

Exercise A: most discriminative TF-IDF terms per category.
Exercise B: effect of preprocessing choices on 5-fold CV accuracy.
Exercise C: retrieval check - does the top-1 result match the expected document?
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils import load_documents
from semantic_search import SemanticSearchEngine

texts, labels, names = load_documents()
labels_arr = np.array(labels)

print("=== Exercise A: top TF-IDF terms per category ===")
vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(texts)
terms = np.array(vec.get_feature_names_out())
for cat in sorted(set(labels)):
    mean = np.asarray(X[labels_arr == cat].mean(axis=0)).ravel()
    print(f"  {cat:<26}", ", ".join(terms[mean.argsort()[::-1][:6]]))

print("\n=== Exercise B: preprocessing ablation (Naive Bayes, 5-fold CV) ===")
cv = StratifiedKFold(5, shuffle=True, random_state=42)
configs = {
    "unigrams, no stop-word removal": dict(),
    "unigrams + stop-word removal": dict(stop_words="english"),
    "uni+bigrams + stop-word removal": dict(stop_words="english", ngram_range=(1, 2)),
}
for name, kw in configs.items():
    pipe = Pipeline([("t", TfidfVectorizer(**kw)), ("c", MultinomialNB(alpha=0.1))])
    s = cross_val_score(pipe, texts, labels, cv=cv)
    print(f"  {name:<36} accuracy = {s.mean():.2%}")

print("\n=== Exercise C: retrieval check (expected document must appear in top-3) ===")
engine = SemanticSearchEngine()
tests = [
    ("How many days of vacation do employees get?", "vacation_and_leave_policy"),
    ("How do I reset my forgotten password?", "password_reset_instructions"),
    ("What are the risks and budget status of the project?", "project_atlas_status_report"),
    ("How is our infrastructure deployed in the cloud?", "cloud_infrastructure_setup"),
    ("What should we do after a data breach?", "incident_response_plan"),
    ("remote work and VPN rules", "remote_work_policy"),
    ("how to submit an expense receipt", "expense_reporting_tool_manual"),
    ("ISO 27001 internal audit", "iso27001_audit_checklist"),
    ("Docker images and automated tests deployment", "cicd_pipeline_guide"),
    ("customer survey satisfaction score", "client_satisfaction_survey_results"),
]
top1 = top3 = 0
for q, expected in tests:
    res = [r[0] for r in engine.search(q, 3)]
    top1 += res[:1] == [expected]; hit = expected in res; top3 += hit
    pos = res.index(expected) + 1 if hit else "-"
    print(f"  rank={pos}  {q}")
print(f"\n  Top-1 accuracy: {top1}/{len(tests)} | Top-3 accuracy: {top3}/{len(tests)}")
