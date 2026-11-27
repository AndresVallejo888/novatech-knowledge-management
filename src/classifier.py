"""Automated document classification (TF-IDF + Naive Bayes / Logistic Regression).

Run:  python src/classifier.py
"""
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import load_documents

MODEL_PATH = Path(__file__).resolve().parent.parent / "outputs" / "classifier.joblib"


def build_pipeline(model="nb"):
    """TF-IDF vectorizer followed by a classifier."""
    clf = MultinomialNB(alpha=0.1) if model == "nb" else LogisticRegression(max_iter=1000, C=10)
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("clf", clf),
    ])


def train_and_evaluate():
    texts, labels, names = load_documents()
    print(f"[1] Loaded {len(texts)} documents in {len(set(labels))} categories")
    X_tr, X_te, y_tr, y_te, n_tr, n_te = train_test_split(
        texts, labels, names, test_size=0.25, stratify=labels, random_state=42)
    print(f"[2] Train/test split: {len(X_tr)} train, {len(X_te)} test\n")

    best_name, best_acc, best_pipe = None, -1, None
    for key, title in [("nb", "Multinomial Naive Bayes"), ("lr", "Logistic Regression")]:
        pipe = build_pipeline(key).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, pred)
        cv = cross_val_score(build_pipeline(key), texts, labels,
                             cv=StratifiedKFold(5, shuffle=True, random_state=42))
        print(f"=== {title} ===")
        print(f"Test accuracy: {acc:.2%} | 5-fold CV accuracy: {cv.mean():.2%} (+/- {cv.std():.2%})")
        if acc > best_acc:
            best_name, best_acc, best_pipe, best_pred = title, acc, pipe, pred
    print(f"\n[3] Selected model: {best_name}\n")

    print("Classification report (test set):")
    print(classification_report(y_te, best_pred, zero_division=0))
    cats = sorted(set(labels))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"Order: {cats}")
    print(confusion_matrix(y_te, best_pred, labels=cats))

    print("\nPer-document predictions (test set):")
    for n, t, p in zip(n_te, y_te, best_pred):
        print(f"  {'OK ' if t == p else 'ERR'} {n:<38} true={t:<26} pred={p}")

    # Final model trained on the full corpus and saved for reuse
    final = build_pipeline("nb" if "Naive" in best_name else "lr").fit(texts, labels)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"\nModel trained on all documents and saved to {MODEL_PATH.name}")
    return final


def classify_new(model, new_docs):
    """Classify unseen text and return (category, confidence)."""
    probs = model.predict_proba(new_docs)
    preds = model.predict(new_docs)
    return [(p, row.max()) for p, row in zip(preds, probs)]


if __name__ == "__main__":
    model = train_and_evaluate()
    unseen = [
        "The Kubernetes cluster autoscaler uses Docker containers and REST endpoints for deployment.",
        "Employees must request annual leave and the HR department approves vacation days and benefits.",
        "The milestone is delayed, the project budget is exceeded and sales revenue grew this quarter.",
        "Click the Save button, select the menu option and upload your file to the portal.",
        "All confidential data must be encrypted and security incidents reported to the compliance audit team.",
    ]
    print("\n[4] Classification of NEW unseen documents:")
    for doc, (cat, conf) in zip(unseen, classify_new(model, unseen)):
        print(f"  -> {cat} ({conf:.1%}) | \"{doc[:70]}...\"")
