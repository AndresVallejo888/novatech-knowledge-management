"""Interactive Knowledge Graph Visualization (Phase 2)."""
from __future__ import annotations
import json
import networkx as nx
from utils import load_ontology_triples, build_corpus_catalog, CATEGORY_LABELS

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

NODE_COLORS = {
    "Knowledge Area": "#2E86AB",
    "Document Type": "#E08E2B",
    "Domain Concept": "#3FA796",
    "Document": "#8E6BBF",
}


def build_expanded_knowledge_graph(include_documents: bool = True, category_filter: str | None = None) -> nx.DiGraph:
    G = nx.DiGraph()
    triples = load_ontology_triples()
    catalog = build_corpus_catalog()

    for _, row in triples.iterrows():
        subj, rel, obj = str(row["subject"]).strip(), str(row["relation"]).strip(), str(row["object"]).strip()
        if rel == "produces":
            G.add_node(subj, layer="Knowledge Area", color=NODE_COLORS["Knowledge Area"], size=28, title=f"Knowledge Area: {subj}")
            G.add_node(obj, layer="Document Type", color=NODE_COLORS["Document Type"], size=24, title=f"Document Category: {obj}")
        elif rel == "describes":
            G.add_node(subj, layer="Document Type", color=NODE_COLORS["Document Type"], size=24, title=f"Document Category: {subj}")
            G.add_node(obj, layer="Domain Concept", color=NODE_COLORS["Domain Concept"], size=20, title=f"Domain Concept: {obj}")
        else:
            for node in (subj, obj):
                if node not in G:
                    G.add_node(node, layer="Domain Concept", color=NODE_COLORS["Domain Concept"], size=20, title=f"Domain Concept: {node}")
        G.add_edge(subj, obj, label=rel, title=f"{subj} --[{rel}]--> {obj}")

    if include_documents:
        for doc in catalog:
            if category_filter and category_filter != "All" and doc["category"] != category_filter:
                continue
            doc_node = f"{doc['name']}.txt"
            cat_display = CATEGORY_LABELS.get(doc["category"], doc["category"])
            onto_cat = "Security Policy" if doc["category"] == "security_compliance" else cat_display
            G.add_node(doc_node, layer="Document", color=NODE_COLORS["Document"], size=14,
                       title=f"Document: {doc['title']} | Category: {cat_display} | Concepts: {', '.join(doc['concepts'])}")
            if onto_cat in G:
                G.add_edge(doc_node, onto_cat, label="classified_as", title=f"{doc_node} classified as {onto_cat}")
            for concept in doc["concepts"]:
                if concept not in G:
                    G.add_node(concept, layer="Domain Concept", color=NODE_COLORS["Domain Concept"], size=20, title=f"Domain Concept: {concept}")
                G.add_edge(doc_node, concept, label="mentions", title=f"{doc_node} mentions {concept}")
    return G


def graph_statistics(G: nx.DiGraph) -> dict:
    layers = {}
    for _, data in G.nodes(data=True):
        lyr = data.get("layer", "Other")
        layers[lyr] = layers.get(lyr, 0) + 1
    centrality = nx.degree_centrality(G)
    top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "layer_counts": layers,
        "top_hubs": [(n, round(c, 3)) for n, c in top_hubs],
    }


def render_pyvis_html(include_documents: bool = True, category_filter: str | None = None, height: str = "620px") -> str:
    G = build_expanded_knowledge_graph(include_documents=include_documents, category_filter=category_filter)
    if HAS_PYVIS:
        net = Network(height=height, width="100%", directed=True, bgcolor="#0E1117", font_color="#FAFAFA")
        net.from_nx(G)
        return net.generate_html()
    nodes_js = [{"id": n, "label": n, "color": d.get("color", "#8E6BBF"), "size": d.get("size", 16), "title": d.get("title", n)} for n, d in G.nodes(data=True)]
    edges_js = [{"from": u, "to": v, "label": d.get("label", ""), "title": d.get("title", ""), "arrows": "to"} for u, v, d in G.edges(data=True)]
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>body{{margin:0;background:#0E1117;}}#mynetwork{{width:100%;height:{height};}}</style></head><body><div id="mynetwork"></div>
<script>new vis.Network(document.getElementById("mynetwork"),{{nodes:new vis.DataSet({json.dumps(nodes_js)}),edges:new vis.DataSet({json.dumps(edges_js)})}},{{nodes:{{font:{{color:"#FAFAFA"}}}},edges:{{font:{{color:"#B0B8C4",strokeWidth:0}}}}}});</script></body></html>"""