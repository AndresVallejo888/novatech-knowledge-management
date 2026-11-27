# NovaTech Solutions — AI-Powered Knowledge Management System (Phase 2)

Complete, integrated Knowledge Management System (KMS) for **NovaTech Solutions**, built with Python, Scikit-learn, NetworkX, PyVis, and Streamlit.

## Integrated Features
1. **Automated Document Classification (`src/classifier.py`)**: TF-IDF + Multinomial Naive Bayes with live ingestion and automatic ontology concept tagging.
2. **Semantic & Context-Aware Recommendation System (`src/recommender.py`)**: Hybrid similarity (TF-IDF + LSA/SVD embeddings + ontology overlap) and session query/reading history.
3. **Natural Language Knowledge Chatbot (`src/chatbot.py`)**: Retrieval-Augmented Generation (RAG) assistant with ontology-driven query expansion.
4. **Interactive Knowledge Graph Explorer (`src/knowledge_graph.py`)**: 4-layer organizational graph (*Knowledge Areas -> Document Types -> Domain Concepts -> Documents*) with NetworkX and PyVis.
5. **Unified Web Interface (`app.py`)**: Streamlit app integrating all modules + live evaluation metrics (`src/evaluate.py`).

## Quick Start (Local Execution)
```bash
pip install -r requirements.txt
streamlit run app.py