"""Shared helpers: loading the document corpus, metadata, and ontology concept links."""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "documents"
ONTOLOGY_PATH = ROOT_DIR / "docs" / "ontology_triples.csv"

CATEGORY_LABELS = {
    "hr_policy": "HR Policy",
    "project_report": "Project Report",
    "security_compliance": "Security & Compliance",
    "technical_documentation": "Technical Documentation",
    "user_manual": "User Manual",
}

CONCEPT_KEYWORDS = {
    "Microservices": ["microservice", "microservices", "api gateway", "rest", "message queue"],
    "Cloud Infrastructure": ["cloud", "aws", "terraform", "kubernetes", "vpc", "autoscaler", "infrastructure"],
    "CI/CD": ["ci/cd", "pipeline", "docker", "deployment", "build", "git", "github actions"],
    "Testing": ["test", "testing", "unit", "integration", "qa", "coverage"],
    "Recruitment": ["recruit", "hiring", "interview", "candidate", "resume", "job posting"],
    "Leave & Vacation": ["vacation", "leave", "sick", "pto", "holiday", "time off"],
    "Performance Review": ["performance", "evaluation", "review", "appraisal", "feedback", "kpi"],
    "Risk": ["risk", "mitigation", "threat", "delay", "overrun", "vulnerability"],
    "Budget": ["budget", "cost", "revenue", "financial", "expense", "margin", "sales"],
    "Milestone": ["milestone", "sprint", "schedule", "deadline", "retrospective", "status"],
    "Helpdesk Ticket": ["ticket", "helpdesk", "support", "issue", "request", "portal"],
    "Onboarding": ["onboarding", "new employee", "orientation", "first week", "training"],
    "GDPR": ["gdpr", "personal data", "consent", "privacy", "data protection", "retention"],
    "Incident Response": ["incident", "breach", "contain", "containment", "response team", "alert"],
    "Access Control": ["password", "access control", "authentication", "mfa", "multi-factor", "token", "encryption"],
}


def load_documents(docs_dir=DOCS_DIR):
    texts, labels, names = [], [], []
    for path in sorted(Path(docs_dir).rglob("*.txt")):
        texts.append(path.read_text(encoding="utf-8").strip())
        labels.append(path.parent.name)
        names.append(path.stem)
    return texts, labels, names


def load_ontology_triples(ontology_path=ONTOLOGY_PATH):
    if Path(ontology_path).exists():
        return pd.read_csv(ontology_path)
    return pd.DataFrame(columns=["subject", "relation", "object"])


def extract_document_concepts(text, category=None):
    lower = text.lower()
    matched = [c for c, kws in CONCEPT_KEYWORDS.items() if any(kw in lower for kw in kws)]
    if not matched and category:
        fallback = {
            "technical_documentation": ["Cloud Infrastructure"],
            "hr_policy": ["Onboarding"],
            "project_report": ["Milestone"],
            "user_manual": ["Helpdesk Ticket"],
            "security_compliance": ["Access Control"],
        }
        matched = fallback.get(category, [])
    return matched


def build_corpus_catalog(docs_dir=DOCS_DIR):
    texts, labels, names = load_documents(docs_dir)
    catalog = []
    for idx, (text, label, name) in enumerate(zip(texts, labels, names)):
        title = text.split(".")[0].strip() if "." in text else name.replace("_", " ").title()
        concepts = extract_document_concepts(text, label)
        catalog.append({
            "doc_id": idx,
            "name": name,
            "title": title,
            "category": label,
            "category_display": CATEGORY_LABELS.get(label, label),
            "concepts": concepts,
            "word_count": len(text.split()),
            "text": text,
        })
    return catalog