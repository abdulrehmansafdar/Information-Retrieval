"""Check that every file:line reference in the week06 docs points at what it claims.

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
    # step 0 - the shared corpus and its stems
    ("articles.py", 64, "def tokenize(text: str)"),
    ("articles.py", 79, "class Article:"),
    ("articles.py", 104, "def parse_article(path"),
    ("articles.py", 170, "def get_articles()"),
    # step 1 - membership and partial matching
    ("membership.py", 42, "TITLE_WEIGHT = 3"),
    ("membership.py", 48, "def trigrams(word: str)"),
    ("membership.py", 54, "def dice(a: str, b: str)"),
    ("membership.py", 62, "class FuzzyIndex:"),
    ("membership.py", 87, "def membership(self, stem: str"),
    ("membership.py", 101, "def expansions(self, stem: str"),
    ("membership.py", 119, "def best_match(self, stem: str"),
    # step 2 - the query language
    ("query_parser.py", 50, "class Term:"),
    ("query_parser.py", 75, "def to_text(node: Node"),
    ("query_parser.py", 159, "def expr(self) -> Node:"),
    ("query_parser.py", 173, "def and_expr(self) -> Node:"),
    ("query_parser.py", 226, "def parse_query(query: str"),
    # step 3 - the operators
    ("fuzzy_logic.py", 45, "def conjunction(values: list[float]"),
    ("fuzzy_logic.py", 53, "def disjunction(values: list[float]"),
    ("fuzzy_logic.py", 69, "def evaluate(node: Node, leaf: Leaf"),
    # step 4 - the engine
    ("fuzzy_search.py", 38, "class Settings:"),
    ("fuzzy_search.py", 155, "def search(self, query: str"),
    ("fuzzy_search.py", 185, "def crisp_set(self, query: str"),
    ("fuzzy_search.py", 191, "def set_view(self, a: str"),
    ("fuzzy_search.py", 226, "def get_engine()"),
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
