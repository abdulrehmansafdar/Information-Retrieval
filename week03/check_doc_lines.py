"""Check that every file:line reference in the week03 docs points at what it claims.

A documentation page with stale line numbers is worse than no page, so the
references are checked rather than trusted. Run after any edit to a .py file:

    python check_doc_lines.py

Two checks, because one is not enough: every `file.py:NN` in every page must
resolve to a non-blank line of code, AND every row of CLAIMS must find the stated
text on that exact line. The first alone would pass a reference to the wrong
function; the second is what catches a line that moved after an edit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs" / "understanding"

# (filename, line, substring the line must contain)
CLAIMS: list[tuple[str, int, str]] = [
    # step 0 - the collection
    ("collection.py", 67, "def tokenize"),
    ("collection.py", 86, "class Document"),
    ("collection.py", 182, "def load_documents"),
    # step 1 - the taxonomy
    ("taxonomy.py", 84, "RULES: tuple[Rule, ...] = ("),
    ("taxonomy.py", 115, "def classify(document: Document)"),
    ("taxonomy.py", 194, "def breadcrumb(self, node_id: str)"),
    ("taxonomy.py", 203, "def documents_in(self, node_id: str)"),
    ("taxonomy.py", 236, "def build_tree(documents"),
    # step 2 - the hypertext graph
    ("hypertext.py", 83, "def related_scores(documents"),
    ("hypertext.py", 129, "class HypertextGraph"),
    ("hypertext.py", 180, "def shortest_path(self, start: str, goal: str)"),
    ("hypertext.py", 203, "def related(self, doc_id: str"),
    ("hypertext.py", 215, "def graph_stats(self) -> dict"),
    ("hypertext.py", 234, "def build_graph(documents"),
    # step 3 - the browsing engine
    ("browsing.py", 39, "BROWSING_MODELS = ("),
    ("browsing.py", 61, "class Filters:"),
    ("browsing.py", 155, "class NavigationState:"),
    ("browsing.py", 168, "def visit(self, node_id: str)"),
    ("browsing.py", 254, 'def children(self, node_id: str = "root")'),
    ("browsing.py", 261, 'def documents_in(self, node_id: str = "root",'),
    ("browsing.py", 270, "def home(self, state: NavigationState | None = None)"),
    ("browsing.py", 284, "def search(self, query: str, filters: Filters | None = None,"),
    ("browsing.py", 343, "def preview(self, doc_id: str, k: int = 3)"),
    ("browsing.py", 394, "def _passes(document: Document, filters: Filters"),
    ("browsing.py", 408, "def get_engine() -> BrowseEngine:"),
]

REFERENCE = re.compile(r"`([a-z_]+\.py):(\d+)`")


def main() -> int:
    problems: list[str] = []
    cache: dict[str, list[str]] = {}

    def lines_of(filename: str) -> list[str]:
        if filename not in cache:
            path = ROOT / filename
            cache[filename] = (path.read_text(encoding="utf-8").splitlines()
                               if path.exists() else [])
        return cache[filename]

    pages = sorted(DOCS.glob("*.md"))
    if not pages:
        problems.append("no documentation pages found")

    referenced: set[tuple[str, int]] = set()
    for page in pages:
        for filename, number in REFERENCE.findall(page.read_text(encoding="utf-8")):
            referenced.add((filename, int(number)))

    for filename, number in sorted(referenced):
        rows = lines_of(filename)
        where = f"{filename}:{number}"
        if not rows:
            problems.append(f"{where} - {filename} does not exist")
        elif not 1 <= number <= len(rows):
            problems.append(f"{where} - outside the file ({len(rows)} lines)")
        elif not rows[number - 1].strip():
            problems.append(f"{where} - blank line")
        elif rows[number - 1].strip().startswith(("#", '"""', "'''")):
            problems.append(f"{where} - points at prose, not code")

    for filename, number, needle in CLAIMS:
        rows = lines_of(filename)
        if not rows:
            problems.append(f"{filename} does not exist")
            continue
        if number > len(rows):
            problems.append(f"{filename}:{number} outside the file "
                            f"({len(rows)} lines) - claimed to contain {needle!r}")
            continue
        if needle not in rows[number - 1]:
            problems.append(
                f"{filename}:{number} does NOT contain {needle!r}\n"
                f"        actual: {rows[number - 1].strip()[:90]!r}")

    print(f"{len(pages)} pages, {len(referenced)} file:line references, "
          f"{len(CLAIMS)} content claims")
    for problem in problems:
        print(f"  BAD {problem}")
    if problems:
        print(f"\n{len(problems)} problem(s)")
        return 1
    print("every reference resolves to code, and every claim matches that line")
    return 0


if __name__ == "__main__":
    sys.exit(main())
