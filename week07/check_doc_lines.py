"""Check that every file:line reference in the week07 docs points at what it claims.

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
    ("term_matrix.py", 28, "WEIGHTINGS ="),
    ("term_matrix.py", 31, "class TermDocumentMatrix:"),
    ("term_matrix.py", 65, "def document_vector(self"),
    ("term_matrix.py", 69, "def query_vector(self"),
    ("vector_cosine.py", 26, "def dot_product(q"),
    ("vector_cosine.py", 36, "def vector_length(v"),
    ("vector_cosine.py", 44, "def cosine_similarity(q"),
    ("vector_cosine.py", 98, "def cosine_steps(q"),
    ("gvsm_model.py", 22, "class GvsmModel:"),
    ("gvsm_model.py", 34, "def rank(self"),
    ("lsi_model.py", 50, "def __init__(self, matrix"),
    ("lsi_model.py", 66, "def __init__(self, factors"),
    ("lsi_model.py", 82, "def energy_kept(self)"),
    ("lsi_model.py", 89, "def query_vector(self"),
    ("lsi_model.py", 93, "def rank(self"),
    ("lsi_model.py", 109, "def concepts(self"),
    ("semantic_search.py", 26, "DEFAULT_K = 15"),
    ("semantic_eval.py", 76, "def evaluate(settings"),
    ("slide_examples.py", 43, "def slide_matrix()"),
    ("slide_examples.py", 54, "def space_exploration()"),
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

    claimed = {(filename, number) for filename, number, _ in CLAIMS}
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
        elif (filename, number) not in claimed:
            problems.append(f"{where} - cited in the docs but has no row in CLAIMS")

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
