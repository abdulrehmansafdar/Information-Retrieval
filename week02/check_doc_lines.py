"""Check that every file:line reference in the docs points at what it claims.

A documentation page with stale line numbers is worse than no page, so the
references are checked rather than trusted. Run after any edit to a .py file:

    python check_doc_lines.py

The first version of this script only checked that a referenced line existed and
was not a comment, which let seven wrong references through: the line existed,
it was code, and it was the WRONG code. So the claims below say what the line
must contain, and a reference that satisfies only the weak check still fails.

Add one row to CLAIMS per significant claim. The point is not the coverage
count; it is that every number in the docs can be re-checked in one command
after an edit, instead of by hand.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs" / "understanding"

# Every claim in formulas.md, checked against the live source.
# (filename, line, substring the line must contain)
CLAIMS: list[tuple[str, int, str]] = [
    # step 1 - preprocessing
    ("preprocess.py", 206, "def _build_phrase_lookup"),
    ("preprocess.py", 216, "def tokenize"),
    ("preprocess.py", 227, "def stem_token"),
    ("preprocess.py", 259, "def preprocess_with_positions"),
    ("preprocess.py", 314, "drop_stop_words and"),
    ("preprocess.py", 243, "def _ensure_phrase_lookup"),
    ("preprocess.py", 328, "class Token"),
    ("preprocess.py", 336, "class Region"),
    # step 2 - the positional index
    ("structured_index.py", 92, "def number_of_documents"),
    ("structured_index.py", 105, "def average_document_length"),
    ("structured_index.py", 131, "def positions_of"),
    ("structured_index.py", 166, "def enclosing"),
    ("structured_index.py", 274, "def build_index"),
    ("structured_index.py", 285, "sorted(grouped[term])"),
    ("structured_index.py", 287, "len(places)"),
    ("structured_index.py", 311, "def check_invariants"),
    # step 3 - probability ranking
    ("ranking.py", 94, "number_of_documents"),
    ("ranking.py", 97, "len(self.index.lookup(term))"),
    ("ranking.py", 101, "def _idf_value"),
    ("ranking.py", 234, "if tf <= 0"),
    ("ranking.py", 238, "def score_paper"),
    ("ranking.py", 248, "total += idf"),
    ("ranking.py", 308, "def length_norm"),
    ("ranking.py", 312, "def saturation"),
    ("ranking.py", 320, "(tf * (self.k1 + 1.0))"),
    ("ranking.py", 327, "def score_paper"),
    ("ranking.py", 333, "term_frequency_in"),
    ("ranking.py", 336, "idf_for(term)"),
    ("ranking.py", 338, "contribution = idf * sat"),
    ("ranking.py", 339, "total += contribution"),
    ("ranking.py", 373, "def _rank"),
    # step 4 - fields
    ("fields.py", 50, "DEFAULT_WEIGHTS"),
    ("fields.py", 352, "def __post_init__"),
    ("fields.py", 365, "for term in present"),
    ("fields.py", 375, "self.idf[name] = {"),
    ("fields.py", 376, "math.log(1.0 + (holders - df + 0.5)"),
    ("fields.py", 394, "return length / average"),
    ("fields.py", 427, "def _bm25"),
    ("fields.py", 511, "def restrict"),
    ("fields.py", 571, "def filter"),
    ("fields.py", 628, "def weighted"),
    ("fields.py", 657, "total += weight * part"),
    ("fields.py", 447, "def _prepare"),
    # the BM25F call chain, line by line. These used to be checked only for
    # "is this a line of code", which happily passed fields.py:217 - a real
    # line, the wrong one. Every arrow in the diagram now names its own code.
    ("fields.py", 356, "holders = 0"),
    ("fields.py", 430, "counts = self._terms_in"),
    ("fields.py", 438, "kn = (1.0 - b) + b * norm"),
    ("fields.py", 441, "sat = (tf * (k1 + 1.0))"),
    ("fields.py", 442, "stats.idf_for(name, term) * sat"),
    ("fields.py", 562, '"+".join(dict.fromkeys(contributing))'),
    ("fields.py", 899, "if not found"),
    # step 5 - the list models
    ("lists.py", 118, "def collapse"),
    ("lists.py", 134, "ordered = sorted"),
    ("lists.py", 139, "match.position <= end"),
    ("lists.py", 142, "spans.append((start, end))"),
    ("lists.py", 149, "def subtract"),
    ("lists.py", 205, "def _region_score"),
    ("lists.py", 228, "def overlapped"),
    ("lists.py", 246, "Match(term=token.term"),
    ("lists.py", 279, "def non_overlapped"),
    # step 6 - the proximal node model
    ("proximity.py", 162, "def innermost_component"),
    ("proximity.py", 280, "def _op_in"),
    ("proximity.py", 295, "def _op_beginin"),
    ("proximity.py", 306, "def _op_child"),
    ("proximity.py", 354, "def _op_with"),
    ("proximity.py", 368, "counts[node.region_id] += 1"),
    ("proximity.py", 370, "count >= k"),
    ("proximity.py", 372, "def _op_parent"),
    ("proximity.py", 394, "def _op_near"),
    ("proximity.py", 426, "def _in_context"),
    ("proximity.py", 434, "def looks_like_constructor"),
    ("proximity.py", 330, "preprocess_query(raw)"),
    # step 7 - evaluation
    ("evaluate.py", 121, "def precision_at_k"),
    ("evaluate.py", 131, "def recall_at_k"),
    ("evaluate.py", 142, "def reciprocal_rank"),
    ("evaluate.py", 150, "def average_precision"),
    # the front ends
    ("engine.py", 277, "def search"),
    ("engine.py", 363, "def _bim"),
    ("engine.py", 370, "def _bm25"),
    ("engine.py", 378, "def _fields"),
    ("engine.py", 385, "def _bm25f"),
    ("engine.py", 408, "def _overlapped"),
    ("engine.py", 414, "def _non_overlapped"),
    ("engine.py", 443, "def _proximal"),
    ("main.py", 34, "if args.console"),
    ("main.py", 42, "subprocess.call"),
]

REFERENCE = re.compile(r"`([a-z_]+\.py):(\d+)`")


def main() -> int:
    problems: list[str] = []
    cache: dict[str, list[str]] = {}

    def lines_of(filename: str) -> list[str]:
        if filename not in cache:
            path = ROOT / filename
            if not path.exists():
                cache[filename] = []
            else:
                cache[filename] = path.read_text(encoding="utf-8").splitlines()
        return cache[filename]

    pages = sorted(DOCS.glob("*.md"))
    if not pages:
        problems.append("no documentation pages found")

    # every file:line in every page must at least resolve to code
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

    # and every claim must hold
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
                f"        actual: {rows[number - 1].strip()[:90]!r}"
            )

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