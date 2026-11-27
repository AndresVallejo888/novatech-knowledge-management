"""Phase 2 System Evaluation Suite."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score
from utils import load_documents, build_corpus_catalog
from classifier import build_pipeline
from recommender import DocumentRecommender
from chatbot import KnowledgeChatbot
from knowledge_graph import build_expanded_knowledge_graph, graph_statistics


def run_full_evaluation() -> dict:
    texts, labels, _ = load_documents()
    catalog = build_corpus_catalog()

    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.25, stratify=labels, random_state=42)
    pipe = build_pipeline("nb").fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    cls_report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    cv_scores = cross_validate(build_pipeline("nb"), texts, labels, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"])

    bot = KnowledgeChatbot()
    queries = [
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
    top1, top3, mrr, details = 0, 0, [], []
    for q, exp in queries:
        hits, concepts = bot.retrieve(q, top_k=5)
        names = [h["name"] for h in hits]
        rank = names.index(exp) + 1 if exp in names else 0
        if rank == 1: top1 += 1
        if 1 <= rank <= 3: top3 += 1
        mrr.append(1.0 / rank if rank > 0 else 0.0)
        details.append({"query": q, "expected": exp, "top_hit": names[0] if names else "None", "rank": rank, "expanded_concepts": concepts})

    rec = DocumentRecommender()
    coh, sims = [], []
    for doc in catalog:
        r = rec.recommend_by_document(doc["name"], top_k=3)
        if r:
            coh.append(sum(1 for x in r if x["category"] == doc["category"]) / len(r))
            sims.append(np.mean([x["score"] for x in r]))

    return {
        "classification": {
            "test_accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
            "test_precision_macro": round(float(cls_report["macro avg"]["precision"]), 4),
            "test_recall_macro": round(float(cls_report["macro avg"]["recall"]), 4),
            "test_f1_macro": round(float(cls_report["macro avg"]["f1-score"]), 4),
            "cv5_accuracy": round(float(np.mean(cv_scores["test_accuracy"])), 4),
            "cv5_precision_macro": round(float(np.mean(cv_scores["test_precision_macro"])), 4),
            "cv5_recall_macro": round(float(np.mean(cv_scores["test_recall_macro"])), 4),
            "cv5_f1_macro": round(float(np.mean(cv_scores["test_f1_macro"])), 4),
        },
        "retrieval_chatbot": {"total_queries": len(queries), "top1_accuracy": round(top1 / len(queries), 4),
                              "top3_accuracy": round(top3 / len(queries), 4), "mrr": round(float(np.mean(mrr)), 4), "details": details},
        "recommendation": {"avg_category_coherence_at_3": round(float(np.mean(coh)), 4), "avg_top3_hybrid_similarity": round(float(np.mean(sims)), 4)},
        "knowledge_graph": graph_statistics(build_expanded_knowledge_graph(include_documents=True)),
    }