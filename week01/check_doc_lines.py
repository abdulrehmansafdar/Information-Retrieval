"""Check the file:line references in week01's formulas.md.

week02 has check_doc_lines.py for its own docs; this is the same idea for
week01, which was written before that script existed. Run from week01/.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DOC = ROOT / "docs" / "understanding" / "formulas.md"

# (filename, line, substring the line must contain)
CLAIMS = [
    ("preprocess.py", 143, "text.lower()"),
    ("preprocess.py", 145, "wordpunct_tokenize"),
    ("preprocess.py", 146, "NON_ALNUM_RE.sub"),
    ("preprocess.py", 152, "def stem_token"),
    ("preprocess.py", 163, "def extract_phrases"),
    ("preprocess.py", 211, "drop_stop_words and"),
    ("preprocess.py", 214, "unigrams + phrases"),
    ("preprocess.py", 250, "preprocess_text(document.title)"),
    ("index.py", 47, "def number_of_documents"),
    ("index.py", 51, "def total_postings"),
    ("index.py", 55, "def build_index"),
    ("index.py", 60, "postings: dict"),
    ("index.py", 64, "Counter(document.terms)"),
    ("index.py", 65, "postings.setdefault(term, []).append"),
    ("index.py", 70, "sorted(postings)"),
    ("index.py", 71, "doc_lengths="),
    ("index.py", 75, "def lookup"),
    ("index.py", 84, "def documents_containing"),
    ("index.py", 86, "for posting in lookup(index, term)"),
    ("index.py", 89, "def document_frequency"),
    ("idf.py", 46, "def term_frequencies"),
    ("idf.py", 64, "def idf_for"),
    ("idf.py", 69, "return self.idf[term]"),
    ("idf.py", 70, "math.log(self.number_of_documents / 1)"),
    ("idf.py", 72, "def df_for"),
    ("idf.py", 88, "def build_idf_model"),
    ("idf.py", 90, "n = index.number_of_documents"),
    ("idf.py", 92, "len(index.postings[term])"),
    ("idf.py", 94, "math.log(n / df)"),
    ("tfidf.py", 49, "def weight"),
    ("tfidf.py", 53, "def document_terms"),
    ("tfidf.py", 60, "def document_total"),
    ("tfidf.py", 68, "def term_weight"),
    ("tfidf.py", 71, "1.0 + math.log(tf)"),
    ("tfidf.py", 73, "scaled = float(tf)"),
    ("tfidf.py", 74, "return scaled * self.idf_model.idf_for(term)"),
    ("tfidf.py", 77, "def build_tfidf_model"),
    ("tfidf.py", 84, "document.terms.count(term)"),
    ("tfidf.py", 85, "model.term_weight(term, tf)"),
    ("vsm.py", 49, "def document_vector"),
    ("vsm.py", 54, "def magnitude"),
    ("vsm.py", 56, "math.sqrt"),
    ("vsm.py", 59, "def dot_product"),
    ("vsm.py", 63, "sum(weight * b.get(term, 0.0)"),
    ("vsm.py", 74, "def norm"),
    ("vsm.py", 82, "def build_query_vector"),
    ("vsm.py", 93, "preprocess_query(query)"),
    ("vsm.py", 94, "Counter(found)"),
    ("vsm.py", 101, "idf_model.idf[term] if weight_by_idf"),
    ("vsm.py", 102, "count * weight"),
    ("vsm.py", 120, "def score_documents"),
    ("vsm.py", 126, "score = dot_product"),
    ("vsm.py", 127, "if score <= 0"),
    ("vsm.py", 139, "results.sort"),
    ("cosine.py", 52, "def cosine_similarity"),
    ("cosine.py", 59, "length_a, length_b = magnitude"),
    ("cosine.py", 60, "if length_a == 0.0"),
    ("cosine.py", 62, "return dot_product(a, b) / (length_a * length_b)"),
    ("cosine.py", 75, "def rank_documents"),
    ("cosine.py", 89, "document_vector"),
    ("cosine.py", 90, "score = cosine_similarity"),
    ("cosine.py", 91, "if score <= 0.0"),
    ("cosine.py", 100, "results.sort"),
    ("cosine.py", 101, "for position, result in enumerate"),
    ("boolean_search.py", 80, "def terms"),
    ("boolean_search.py", 118, "def scan"),
    ("boolean_search.py", 217, "def parse_query"),
    ("boolean_search.py", 227, "def evaluate"),
    ("boolean_search.py", 236, "documents_containing(index, term)"),
    ("boolean_search.py", 239, "everything - evaluate"),
    ("boolean_search.py", 241, "evaluate(node.left, index) &"),
    ("boolean_search.py", 243, "evaluate(node.left, index) |"),
    ("boolean_search.py", 247, "def search"),
    ("engine.py", 136, "def search"),
    ("engine.py", 159, "def _search_boolean"),
    ("engine.py", 176, "def _search_ranked"),
    ("engine.py", 211, "def similarity"),
    ("engine.py", 236, "def corpus_stats"),
    ("main.py", 33, "from console_ui import main as console_main"),
    ("main.py", 35, "console_main(remaining)"),
    ("main.py", 40, "subprocess.call"),
]

REFERENCE = re.compile(r"`([a-z_]+\.py):(\d+)`")


def main() -> int:
    if not DOC.exists():
        print(f"  {DOC} does not exist")
        return 1
    problems = []
    cache = {}

    def rows_of(filename):
        if filename not in cache:
            path = ROOT / filename
            cache[filename] = (path.read_text(encoding="utf-8").splitlines()
                               if path.exists() else [])
        return cache[filename]

    referenced = {(f, int(n)) for f, n in
                  REFERENCE.findall(DOC.read_text(encoding="utf-8"))}
    for filename, number in sorted(referenced):
        rows = rows_of(filename)
        if not rows:
            problems.append(f"{filename}:{number} - {filename} missing")
        elif not 1 <= number <= len(rows):
            problems.append(f"{filename}:{number} - outside the file")
        elif not rows[number - 1].strip():
            problems.append(f"{filename}:{number} - blank line")

    for filename, number, needle in CLAIMS:
        rows = rows_of(filename)
        if not rows:
            problems.append(f"{filename} missing")
        elif number > len(rows):
            problems.append(f"{filename}:{number} outside the file")
        elif needle not in rows[number - 1]:
            problems.append(f"{filename}:{number} does NOT contain {needle!r}\n"
                            f"        actual: {rows[number - 1].strip()[:80]!r}")

    print(f"{len(referenced)} references, {len(CLAIMS)} claims")
    for problem in problems:
        print(f"  BAD {problem}")
    if problems:
        print(f"\n{len(problems)} problem(s)")
        return 1
    print("every reference and claim checks out")
    return 0


if __name__ == "__main__":
    sys.exit(main())