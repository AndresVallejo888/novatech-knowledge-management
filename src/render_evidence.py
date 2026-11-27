"""Renders the saved console outputs as terminal-style images for the report (evidence)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT, DOCS = ROOT / "outputs", ROOT / "docs"


def render(lines, path, title, width=12.5):
    h = 0.5 + 0.215 * len(lines)
    fig = plt.figure(figsize=(width, h), facecolor="#1e1e1e")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor("#1e1e1e")
    ax.set_xlim(0, 1); ax.set_ylim(0, len(lines) + 2)
    ax.text(0.01, len(lines) + 1.2, f"$ {title}", color="#7ee787", family="monospace", fontsize=9.5, fontweight="bold")
    for i, ln in enumerate(lines):
        color = "#f85149" if ln.strip().startswith("ERR") else "#d4d4d4"
        if ln.startswith(("[", "===", "Query")): color = "#79c0ff"
        ax.text(0.01, len(lines) - i, ln, color=color, family="monospace", fontsize=8.6)
    fig.savefig(path, dpi=140, facecolor="#1e1e1e"); plt.close(fig)


cls = (OUT / "classification_output.txt").read_text().splitlines()
split = next(i for i, l in enumerate(cls) if l.startswith("Per-document"))
render(cls[:split], DOCS / "screenshot_classification_1.png", "python src/classifier.py   (metrics)")
render(cls[split:], DOCS / "screenshot_classification_2.png", "python src/classifier.py   (predictions on test set + new documents)")
srch = (OUT / "search_output.txt").read_text().splitlines()
mid = next(i for i, l in enumerate(srch) if "reset my forgotten" in l) - 1
mid2 = next(i for i, l in enumerate(srch) if "infrastructure deployed" in l) - 1
render(srch[:mid2], DOCS / "screenshot_search_1.png", "python src/semantic_search.py   (queries 1-3)")
render(srch[mid2:], DOCS / "screenshot_search_2.png", "python src/semantic_search.py   (queries 4-5)")
print("ok")
