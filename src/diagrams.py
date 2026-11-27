"""Generates the knowledge lifecycle map and the ontology graph as PNG images."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import networkx as nx

OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)


def lifecycle():
    stages = [
        ("1. Creation /\nAcquisition", "Engineers, consultants\nand HR author new\nknowledge; lessons\nlearned captured", "#2E86AB"),
        ("2. Storage /\nDocumentation", "Files saved in the\nDocument Repository\nwith metadata and\nversion control", "#3FA796"),
        ("3. Organization /\nClassification", "AI classifier (TF-IDF +\nNaive Bayes) assigns\ncategory; ontology\nlinks concepts", "#8E6BBF"),
        ("4. Sharing", "Semantic search,\nintranet portal and\nteam wikis give\naccess to all staff", "#E08E2B"),
        ("5. Application", "Staff apply knowledge\nin projects, support\nand decisions", "#D1495B"),
        ("6. Update /\nImprovement", "Feedback, audits and\nretrospectives trigger\nreview and retraining", "#5C6B73"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.6); ax.axis("off")
    pos = [(1.9, 4.9), (5.5, 4.9), (9.1, 4.9), (9.1, 1.9), (5.5, 1.9), (1.9, 1.9)]
    for (title, desc, color), (x, y) in zip(stages, pos):
        ax.add_patch(FancyBboxPatch((x - 1.5, y - 0.95), 3.0, 1.9, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=color, ec="none", alpha=0.95))
        ax.text(x, y + 0.5, title, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x, y - 0.35, desc, ha="center", va="center", color="white", fontsize=7.8)
    def arrow(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=2, color="#333",
                                     connectionstyle=f"arc3,rad={rad}"))
    arrow((3.45, 4.9), (3.95, 4.9)); arrow((7.05, 4.9), (7.55, 4.9))
    arrow((9.1, 3.9), (9.1, 2.95)); arrow((7.55, 1.9), (7.05, 1.9)); arrow((3.95, 1.9), (3.45, 1.9))
    arrow((1.9, 2.95), (1.9, 3.9), rad=0)     # stage 6 -> back to stage 1 (continuous loop)
    ax.text(5.5, 3.4, "NovaTech Solutions\nKnowledge Lifecycle", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#222")
    ax.text(5.5, 0.35, "Continuous loop: improvements from stage 6 feed new knowledge creation in stage 1",
            ha="center", fontsize=9, style="italic", color="#444")
    fig.savefig(OUT / "lifecycle_map.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ontology():
    areas = ["Engineering", "Human Resources", "Project Management", "Support & Training", "Security & Compliance"]
    doc_types = ["Technical Documentation", "HR Policy", "Project Report", "User Manual", "Security Policy"]
    concepts = {
        "Engineering": ["Microservices", "Cloud Infrastructure", "CI/CD", "Testing"],
        "Human Resources": ["Recruitment", "Leave & Vacation", "Performance Review"],
        "Project Management": ["Risk", "Budget", "Milestone"],
        "Support & Training": ["Helpdesk Ticket", "Onboarding"],
        "Security & Compliance": ["GDPR", "Incident Response", "Access Control"],
    }
    cross = [("CI/CD", "Testing", "includes"), ("Cloud Infrastructure", "Access Control", "depends_on"),
             ("Recruitment", "Onboarding", "followed_by"), ("Performance Review", "Onboarding", "related_to"),
             ("Budget", "Milestone", "tracked_by"), ("Risk", "Incident Response", "related_to"),
             ("Helpdesk Ticket", "Incident Response", "may_trigger"), ("Onboarding", "Access Control", "requires")]
    # --- layered left-to-right layout
    all_c = [c for a in areas for c in concepts[a]]
    n = len(all_c)
    cy = {c: n - i for i, c in enumerate(all_c)}
    dy, ay = {}, {}
    for a, d in zip(areas, doc_types):
        ys = [cy[c] for c in concepts[a]]
        dy[d] = ay[a] = sum(ys) / len(ys)
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(-0.5, 12.2); ax.set_ylim(-0.2, n + 1.6); ax.axis("off")
    X = {"area": 0.9, "doc": 4.0, "concept": 7.0}
    def box(x, y, text, color, shape="round"):
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.5" if shape == "round" else "square,pad=0.5", fc=color, ec="none"))
    def edge(p, q, rad=0.0, color="#777", lw=1.4, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=lw, color=color,
                                     connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=22, shrinkB=22, zorder=1))
    for a, d in zip(areas, doc_types):
        edge((X["area"], ay[a]), (X["doc"], dy[d]))
        ax.text((X["area"] + X["doc"]) / 2, (ay[a] + dy[d]) / 2 + 0.22, "produces", fontsize=7, color="#555", ha="center")
        for c in concepts[a]:
            edge((X["doc"], dy[d]), (X["concept"], cy[c]))
        ax.text((X["doc"] + X["concept"]) / 2, dy[d] + 0.25, "describes", fontsize=7, color="#555", ha="center")
    # concept-to-concept relations: elbow connectors with one lane each
    for i, (u, v, r) in enumerate(cross):
        yu, yv = cy[u], cy[v]
        xi = 8.55 + 0.42 * i
        ax.plot([X["concept"] + 1.25, xi, xi], [yu, yu, (yu + yv) / 2], color="#B03A2E", lw=1.5, zorder=1)
        ax.plot([xi, xi], [(yu + yv) / 2, yv], color="#B03A2E", lw=1.5, zorder=1)
        ax.annotate("", xy=(X["concept"] + 1.25, yv), xytext=(xi, yv),
                    arrowprops=dict(arrowstyle="-|>", color="#B03A2E", lw=1.5), zorder=1)
        ax.text(xi + 0.06, (yu + yv) / 2, r, fontsize=7, color="#B03A2E", rotation=90, va="center", ha="left",
                bbox=dict(fc="white", ec="none", pad=0.5), zorder=2)
    for a in areas: box(X["area"], ay[a], a.replace(" & ", "\n& ").replace(" ", "\n", 1) if len(a) > 14 and "&" not in a else a, "#2E86AB")
    for d in doc_types: box(X["doc"], dy[d], d.replace(" ", "\n", 1), "#E08E2B", "square")
    for c in all_c: box(X["concept"], cy[c], c, "#3FA796")
    for x, t in [(X["area"], "KNOWLEDGE AREAS"), (X["doc"], "DOCUMENT TYPES"), (X["concept"], "DOMAIN CONCEPTS")]:
        ax.text(x, n + 1.1, t, ha="center", fontsize=10, fontweight="bold", color="#222")
    ax.text(10.0, n + 1.1, "CONCEPT\nRELATIONS", ha="center", fontsize=10, fontweight="bold", color="#B03A2E")
    ax.set_title("Ontology of Organizational Knowledge - NovaTech Solutions", fontsize=14, fontweight="bold", pad=12)
    fig.savefig(OUT / "ontology_graph.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    triples = [(a, "produces", d) for a, d in zip(areas, doc_types)]
    triples += [(d, "describes", c) for a, d in zip(areas, doc_types) for c in concepts[a]]
    triples += [(u, r, v) for u, v, r in cross]
    with open(OUT / "ontology_triples.csv", "w") as f:
        f.write("subject,relation,object\n")
        for t in triples: f.write(",".join(x.replace(",", "") for x in t) + "\n")
    print("Triples:", len(triples))


if __name__ == "__main__":
    lifecycle(); ontology(); print("Diagrams saved in", OUT)
