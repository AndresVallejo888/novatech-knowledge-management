# NovaTech Solutions - AI-Powered Knowledge Management System (Phase 1 Prototype)

Prototype of a knowledge management system for a fictional software consulting company.
It demonstrates (1) **automated document classification** and (2) **semantic search**
using TF-IDF, Naive Bayes / Logistic Regression and cosine similarity (Scikit-learn).

## Repository structure
```
documents/            40 sample documents in 5 categories (folder name = label)
src/utils.py          corpus loader
src/classifier.py     TF-IDF + Naive Bayes / Logistic Regression classifier
src/semantic_search.py TF-IDF + cosine-similarity search engine
src/exercises.py      supporting experiments (top terms, ablation, retrieval check)
src/diagrams.py       generates lifecycle map and ontology graph
src/render_evidence.py renders console outputs as screenshots
docs/                 diagrams, ontology triples (CSV), screenshots
outputs/              saved console outputs and trained model
```

## Run
```bash
pip install -r requirements.txt
python src/classifier.py                      # train, evaluate, classify new documents
python src/semantic_search.py                 # demo queries
python src/semantic_search.py "how do I reset my password?"   # your own query
python src/exercises.py
```

## Results (sample corpus)
- Classification: 100% accuracy on the 10-document test set; 87.5% 5-fold CV (Naive Bayes).
- Retrieval: expected document ranked #1 in 9/10 test queries and in the top 3 in 10/10.

## Limitation
TF-IDF matches vocabulary, not meaning (e.g. "data breach" vs "security incident").
Future phases: word embeddings / transformers, ontology-driven query expansion.
