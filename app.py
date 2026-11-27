"""NovaTech Solutions - AI-Powered Knowledge Management System (Phase 2)."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import build_corpus_catalog, CATEGORY_LABELS, DOCS_DIR, extract_document_concepts
from classifier import build_pipeline
from recommender import DocumentRecommender
from chatbot import KnowledgeChatbot
from knowledge_graph import build_expanded_knowledge_graph, graph_statistics, render_pyvis_html
from evaluate import run_full_evaluation

st.set_page_config(page_title="NovaTech KMS | AI Knowledge Management", page_icon="🧠", layout="wide")

@st.cache_resource
def load_engines():
    catalog = build_corpus_catalog()
    clf = build_pipeline("nb").fit([d["text"] for d in catalog], [d["category"] for d in catalog])
    rec = DocumentRecommender()
    bot = KnowledgeChatbot(recommender=rec)
    return catalog, clf, rec, bot

catalog, clf, recommender, chatbot = load_engines()

if "query_history" not in st.session_state: st.session_state.query_history = []
if "viewed_docs" not in st.session_state: st.session_state.viewed_docs = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "Welcome to **NovaTech Solutions KMS**! Ask me anything about company policies, cloud/DevOps architecture, security compliance, project reports, or manuals.", "sources": []}]

with st.sidebar:
    st.title("🧠 NovaTech KMS")
    st.caption("Phase 2 Integrated Prototype")
    st.divider()
    page = st.radio("System Modules", ["💬 AI Knowledge Chatbot", "🎯 Smart Recommendations", "📂 Automated Classification", "🕸️ Knowledge Graph Explorer", "📊 System Evaluation & Metrics"])
    st.divider()
    st.write(f"**Indexed Documents:** {len(catalog)}")
    st.write(f"**Queries in Session:** {len(st.session_state.query_history)}")
    st.write(f"**Viewed Docs:** {len(st.session_state.viewed_docs)}")
    if st.button("Reset Session History", use_container_width=True):
        st.session_state.query_history, st.session_state.viewed_docs = [], []
        st.rerun()

if page == "💬 AI Knowledge Chatbot":
    st.header("💬 Natural Language Knowledge Assistant")
    cols = st.columns(4)
    samples = ["What should we do after a data breach?", "How many days of vacation do employees get?",
               "How is our infrastructure deployed in the cloud?", "What are the risks and budget status of Project Atlas?"]
    selected_sample = next((q for col, q in zip(cols, samples) if col.button(q, use_container_width=True)), None)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View Retrieved Source Documents"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['title']}** (`{s['name']}.txt` | `{s['category_display']}` | Score: `{s['score']:.3f}`)")
                        st.caption(s["text"])

    user_prompt = st.chat_input("Ask a question about NovaTech documents...") or selected_sample
    if user_prompt:
        st.session_state.query_history.append(user_prompt)
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"): st.markdown(user_prompt)
        res = chatbot.answer(user_prompt, top_k=3)
        for src in res["sources"]:
            if src["name"] not in st.session_state.viewed_docs: st.session_state.viewed_docs.append(src["name"])
        st.session_state.chat_messages.append({"role": "assistant", "content": res["answer"], "sources": res["sources"]})
        with st.chat_message("assistant"):
            st.markdown(res["answer"])
            if res["sources"]:
                with st.expander("View Retrieved Source Documents"):
                    for s in res["sources"]:
                        st.markdown(f"**{s['title']}** (`{s['name']}.txt` | `{s['category_display']}` | Score: `{s['score']:.3f}`)")
                        st.caption(s["text"])

elif page == "🎯 Smart Recommendations":
    st.header("🎯 Context & Semantic Document Recommendation System")
    t1, t2 = st.tabs(["📄 Similar to a Selected Document", "🕒 Personalized by Query & Reading History"])
    with t1:
        doc_names = [d["name"] for d in catalog]
        sel = st.selectbox("Select a reference document:", doc_names)
        if sel and sel not in st.session_state.viewed_docs: st.session_state.viewed_docs.append(sel)
        ref = next(d for d in catalog if d["name"] == sel)
        with st.container(border=True):
            st.subheader(f"Reference: {ref['title']}")
            st.caption(f"Category: **{ref['category_display']}** | Concepts: {', '.join(ref['concepts'])}")
            st.write(ref["text"])
        for rank, r in enumerate(recommender.recommend_by_document(sel, top_k=5), 1):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{rank}. {r['title']}** (`{r['name']}.txt`)")
                c1.caption(f"{r['category_display']} • {r['reason']}")
                c2.metric("Similarity", f"{r['score']:.2f}")
                st.write(r["text"])
    with t2:
        cq = st.text_input("Add search intent to session:", placeholder="e.g., AWS Terraform deployment")
        if st.button("Add Context") and cq.strip():
            st.session_state.query_history.append(cq.strip())
            st.rerun()
        st.info(f"**Queries:** {st.session_state.query_history[-5:] or ['None']} | **Viewed:** {st.session_state.viewed_docs[-5:] or ['None']}")
        for rank, r in enumerate(recommender.recommend_by_context(st.session_state.query_history, st.session_state.viewed_docs, top_k=5), 1):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{rank}. {r['title']}** (`{r['name']}.txt`)")
                c1.caption(f"{r['category_display']} • {r['reason']}")
                c2.metric("Relevance", f"{r['score']:.2f}")
                st.write(r["text"])

elif page == "📂 Automated Classification":
    st.header("📂 Automated Document Classification & Repository Ingestion")
    samples = {
        "Custom text...": "",
        "Cloud / DevOps": "The Kubernetes cluster autoscaler provisions Docker containers on AWS using Terraform pipelines and REST API endpoints.",
        "HR Leave Policy": "Employees must submit annual vacation and sick leave requests through the HR portal at least two weeks in advance.",
        "Security Incident": "All confidential customer records must be encrypted and any unauthorized access or security incident reported immediately.",
    }
    preset = st.selectbox("Sample text:", list(samples.keys()))
    txt = st.text_area("Document text:", value=samples[preset], height=130)
    b1, b2 = st.columns(2)
    if b1.button("🔍 Classify", type="primary", use_container_width=True) and txt.strip():
        pred = clf.predict([txt])[0]
        probs = clf.predict_proba([txt])[0]
        st.success(f"**Predicted Category:** {CATEGORY_LABELS.get(pred, pred)} — **Confidence:** {probs.max():.1%}")
        st.markdown(f"**Extracted Ontology Concepts:** {', '.join(extract_document_concepts(txt, pred))}")
        st.bar_chart(pd.DataFrame({"Category": [CATEGORY_LABELS.get(c, c) for c in clf.classes_], "Probability": probs}).set_index("Category"))

elif page == "🕸️ Knowledge Graph Explorer":
    st.header("🕸️ Interactive Organizational Knowledge Graph")
    c1, c2 = st.columns([2, 1])
    cat = c1.selectbox("Filter by Category:", ["All"] + list(CATEGORY_LABELS.keys()))
    show_docs = c2.checkbox("Show Document Nodes (40 docs)", value=True)
    G = build_expanded_knowledge_graph(include_documents=show_docs, category_filter=cat)
    stats = graph_statistics(G)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nodes", stats["total_nodes"])
    m2.metric("Edges", stats["total_edges"])
    m3.metric("Density", stats["density"])
    m4.metric("Top Hub", stats["top_hubs"][0][0] if stats["top_hubs"] else "N/A")
    components.html(render_pyvis_html(include_documents=show_docs, category_filter=cat, height="600px"), height=620, scrolling=True)

elif page == "📊 System Evaluation & Metrics":
    st.header("📊 System Evaluation & Performance Metrics")
    ev = run_full_evaluation()
    c, r, rec = ev["classification"], ev["retrieval_chatbot"], ev["recommendation"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test F1-Score", f"{c['test_f1_macro']:.1%}", f"5-Fold CV: {c['cv5_accuracy']:.1%}")
    k2.metric("Chatbot Top-1", f"{r['top1_accuracy']:.1%}", "Up from 90.0% in Phase 1")
    k3.metric("Chatbot Top-3 / MRR", f"{r['top3_accuracy']:.1%}", f"MRR: {r['mrr']:.3f}")
    k4.metric("Rec. Coherence @3", f"{rec['avg_category_coherence_at_3']:.1%}", f"Sim: {rec['avg_top3_hybrid_similarity']:.3f}")
    st.dataframe(pd.DataFrame(r["details"]), use_container_width=True)