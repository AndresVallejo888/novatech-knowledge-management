"""Shared helpers: loading the document corpus."""
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "documents"


def load_documents(docs_dir=DOCS_DIR):
    """Load every .txt file. The sub-folder name is the category label.

    Returns three parallel lists: texts, labels, filenames.
    """
    texts, labels, names = [], [], []
    for path in sorted(Path(docs_dir).rglob("*.txt")):
        texts.append(path.read_text(encoding="utf-8").strip())
        labels.append(path.parent.name)
        names.append(path.stem)
    return texts, labels, names
