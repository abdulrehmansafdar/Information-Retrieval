"""Step 7 - Vector Space Model.

Everything so far produced numbers about single documents. The vector space
model turns those numbers into a way to answer a question none of them could:

    "Which document is ABOUT this query?"

A document becomes a vector of TF-IDF weights, one coordinate per vocabulary
term. A query becomes a vector in the same space. The score is the dot product:

    score(q, d) = sum over terms t of  w(t,q) * w(t,d)

Terms the document does not contain simply contribute nothing, so documents
missing part of the query still score.

This is a different retrieval model from Boolean. A Boolean query is a logic
formula with AND/OR/NOT; a VSM query is a bag of weights and there is no way to
express 'must contain'. Every document gets a number, and the numbers decide
the order.

The dot product has a known flaw, fixed in step 8 (cosine). It is built here
anyway so the fix can be shown on real numbers: for 122 of the 608 terms in this
corpus, ranking by dot product and ranking by cosine give DIFFERENT orders.

Run:  python vsm.py
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from idf import IdfModel, build_idf_model
from index import InvertedIndex, build_index
from preprocess import preprocess_query
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------


def document_vector(doc_id: str, model) -> dict[str, float]:
    """A document as a vector of TF-IDF weights, indexed by vocabulary term."""
    return model.document_terms(doc_id)


def magnitude(vector: dict[str, float]) -> float:
    """Euclidean length of a vector: sqrt(sum of squared weights)."""
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def dot_product(a: dict[str, float], b: dict[str, float]) -> float:
    """Sum of products over shared terms. Missing coordinates count as zero."""
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())


@dataclass
class QueryVector:
    """A query expressed in the same vector space as the documents."""

    original: str
    terms: dict[str, float]
    unknown_terms: list[str] = field(default_factory=list)

    def norm(self) -> float:
        return magnitude(self.terms)

    def describe(self) -> str:
        return ", ".join(f"{t} ({w:.4f})" for t, w in
                         sorted(self.terms.items(), key=lambda kv: (-kv[1], kv[0])))


def build_query_vector(query: str, idf_model: IdfModel,
                       weight_by_idf: bool = True) -> QueryVector:
    """Turn raw query text into a vector.

    The query goes through the same preprocessing as the documents, so
    'Machine Learning' becomes the phrase term 'machin learn' and matches the
    same index entry.

    weight_by_idf=True  each query term is weighted by its idf, like a document.
    weight_by_idf=False binary: every query term counts 1, ignoring idf.
    """
    found = preprocess_query(query)
    counts = Counter(found)
    terms: dict[str, float] = {}
    unknown: list[str] = []
    for term, count in counts.items():
        if term not in idf_model.idf:
            unknown.append(term)
            continue
        weight = idf_model.idf[term] if weight_by_idf else 1.0
        terms[term] = terms.get(term, 0.0) + count * weight
    return QueryVector(original=query, terms=terms, unknown_terms=unknown)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@dataclass
class ScoredDocument:
    doc_id: str
    title: str
    score: float
    length: int
    matched_terms: list[tuple[str, float]] = field(default_factory=list)


def score_documents(query: QueryVector, index: InvertedIndex, model,
                    top: int | None = None) -> list[ScoredDocument]:
    """Score every document against a query vector and sort by dot product."""
    results = []
    for document in index.documents:
        vector = document_vector(document.doc_id, model)
        score = dot_product(query.terms, vector)
        if score <= 0:
            continue
        matched = sorted(((t, query.terms[t] * vector[t])
                          for t in query.terms if vector.get(t, 0.0) > 0),
                         key=lambda kv: -kv[1])
        results.append(ScoredDocument(
            doc_id=document.doc_id,
            title=document.title,
            score=score,
            length=len(document.terms),
            matched_terms=matched,
        ))
    results.sort(key=lambda r: (-r.score, r.doc_id))
    return results[:top] if top else results


# ---------------------------------------------------------------------------
# console output
# ---------------------------------------------------------------------------

CONSOLE = Console()

ACCENT = "bold cyan"
WARM = "bold yellow"
DIM = "grey58"

# The Windows console is cp437 and stdout is cp1252, so block characters must not
# be hard-coded or they print as '?'. Pick them from the detected encoding.
_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"


def bar(fraction: float, width: int = 18) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(
        BAR_OFF * (width - filled), style="grey30"
    )


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def section_idea() -> None:
    heading(1, "THE IDEA")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("document vector", "one coordinate per vocabulary term, holding"
                                    " that term's TF-IDF weight in the document")
    body.add_row("query vector", "the same shape, built from the query text")
    body.add_row("score", "[bold]score(q,d) = SUM over t of w(t,q) x w(t,d)[/]")
    body.add_row("missing terms", "a document that lacks a query term simply gets"
                                  " no contribution from it - no error, no penalty"
                                  " beyond the smaller score")
    CONSOLE.print(Panel(body, title="vector space model", title_align="left",
                        border_style="grey35", padding=(0, 1)))

    contrast = Table.grid(padding=(0, 2))
    contrast.add_column(style=DIM, no_wrap=True)
    contrast.add_column(style="white", overflow="fold")
    contrast.add_row("vs Boolean", "a Boolean query is a logic formula; AND, OR"
                                   " and NOT are exact and a document either"
                                   " satisfies them or does not.")
    contrast.add_row("", "a VSM query is a bag of weights. There is no way to say"
                         " 'must contain', and every matching document gets a"
                         " graded score instead of a yes or no.")
    CONSOLE.print(Panel(contrast, border_style="cyan", padding=(0, 1)))


def section_query_vector(index: InvertedIndex, idf_model: IdfModel,
                         query: str) -> None:
    heading(2, f"BUILDING THE QUERY VECTOR - {query!r}")
    vector = build_query_vector(query, idf_model)
    binary = build_query_vector(query, idf_model, weight_by_idf=False)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("query term", style="bold white", no_wrap=True)
    table.add_column("df", justify="right", style=DIM, no_wrap=True)
    table.add_column("idf weight", justify="right", style=WARM, no_wrap=True)
    table.add_column("binary weight", justify="right", style=DIM, no_wrap=True)
    table.add_column("documents", style=DIM, overflow="fold")
    for term, weight in sorted(vector.terms.items(), key=lambda kv: (-kv[1], kv[0])):
        postings = index.postings.get(term, [])
        table.add_row(term, str(len(postings)), f"{weight:.4f}", "1",
                      ", ".join(p.doc_id for p in postings))
    CONSOLE.print(table)

    notes = Table.grid(padding=(0, 2))
    notes.add_column(style=DIM, no_wrap=True)
    notes.add_column(style="white", overflow="fold")
    notes.add_row("query length", f"|q| = {vector.norm():.4f} (Euclidean length of "
                                  f"the query vector)")
    if vector.unknown_terms:
        notes.add_row("dropped", f"{vector.unknown_terms} - not in the vocabulary,"
                                  f" so they cannot match any document")
    notes.add_row("idf on queries?", "weight_by_idf=True reuses the document idf, so"
                                     " a query term that appears in every document"
                                     " contributes little. False gives every query"
                                     " term weight 1.")
    CONSOLE.print(Panel(notes, padding=(0, 1)))
    CONSOLE.print(f"  [grey58]binary form would be: {binary.describe() or '-'}[/]\n")


def section_scoring(index: InvertedIndex, model, query: str) -> None:
    heading(3, f"SCORING BY DOT PRODUCT - {query!r}")
    vector = build_query_vector(query, model.idf_model)
    results = score_documents(vector, index, model, top=10)
    if not results:
        CONSOLE.print("  [grey58]no document matches any query term[/]\n")
        return
    largest = results[0].score

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style=DIM, no_wrap=True)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    table.add_column("len", justify="right", style=DIM, no_wrap=True)
    table.add_column("score", justify="right", style=WARM, no_wrap=True)
    table.add_column("dot product", width=18)
    for position, result in enumerate(results, 1):
        table.add_row(str(position), result.doc_id, result.title,
                      str(result.length), f"{result.score:.3f}",
                      bar(result.score / largest, 16))
    CONSOLE.print(table)

    detail = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    detail.add_column("doc", style=ACCENT, no_wrap=True)
    detail.add_column("score", justify="right", style=WARM, no_wrap=True)
    detail.add_column("built from", style=DIM, overflow="fold")
    for result in results[:4]:
        parts = "  ".join(f"{t} x {w:.3f}" for t, w in result.matched_terms)
        detail.add_row(result.doc_id, f"{result.score:.3f}", parts)
    CONSOLE.print(detail)
    CONSOLE.print(
        f"  [grey58]{len(score_documents(vector, index, model))} of"
        f" {index.number_of_documents} documents score above zero."
        f" A VSM query returns a graded list, not a set.[/]\n"
    )


def section_dot_flaw(index: InvertedIndex, model) -> None:
    heading(4, "THE FLAW IN THE DOT PRODUCT")
    idf_model = model.idf_model

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("problem", "the dot product has no notion of vector length. A"
                            " document that is large simply has more to contribute,"
                            " whether or not the query is what it is about.")
    body.add_row("worse", "two documents with identical weight for a query term get"
                          " an identical score, and there is no way to break the"
                          " tie.")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))

    term = "abl"
    qv = build_query_vector(term, idf_model)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("len", justify="right", style=DIM, no_wrap=True)
    table.add_column("tf-idf", justify="right", no_wrap=True)
    table.add_column("|d|", justify="right", style=DIM, no_wrap=True)
    table.add_column("dot product", justify="right", style=WARM, no_wrap=True)
    table.add_column("verdict", style=DIM, no_wrap=True)
    for result in score_documents(qv, index, model):
        vector = document_vector(result.doc_id, model)
        table.add_row(result.doc_id, str(result.length),
                      f"{vector.get(term, 0.0):.3f}", f"{magnitude(vector):.3f}",
                      f"{result.score:.3f}", "[red]tied[/]")
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]Query {term!r} gives doc01 and doc12 the same tf-idf, so the"
        f" dot product ties them\n  at exactly {score_documents(qv, index, model)[0].score:.3f}."
        f" Note doc12 is the shorter document yet has the\n  LARGER vector"
        f" magnitude, because its weight is spread over more repeated terms."
        f" Length alone is not\n  the problem - concentration is.[/]\n"
    )

    disagreements = 0
    for candidate in index.vocabulary:
        probe = build_query_vector(candidate, idf_model)
        results = score_documents(probe, index, model)
        if len(results) < 2:
            continue
        # The cosine formula is inlined here rather than imported, because
        # cosine.py imports this module and a reverse import would be circular.
        query_length = probe.norm()
        by_cosine = sorted(
            results,
            key=lambda r: (
                -dot_product(probe.terms, document_vector(r.doc_id, model))
                / (query_length * magnitude(document_vector(r.doc_id, model))),
                r.doc_id,
            ),
        )
        if [r.doc_id for r in results][:2] != [r.doc_id for r in by_cosine][:2]:
            disagreements += 1
    CONSOLE.print(
        f"  [bold]{disagreements} of {len(index.vocabulary)}[/] vocabulary terms"
        f" would rank differently once length is\n  taken into account. That is"
        f" step 8: cosine similarity. Cosine divides the dot\n  product by both"
        f" magnitudes, turning the score into an angle and removing the"
        f" length effect.\n"
    )


def section_summary(index: InvertedIndex, model) -> None:
    heading(5, "STATE OF PLAY")
    rows = [
        ("Boolean queries", "done in step 4 - exact, no ranking"),
        ("inverted index", "done in step 3 - 1,401 postings"),
        ("TF-IDF weights", "done in step 6 - 1,401 weights"),
        ("vector space model", "done in step 7 - this file"),
        ("cosine similarity", "step 8 - removes the length flaw shown above"),
        ("ranked results", "step 8 - orders by cosine, highest first"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style="bold white")
    table.add_column(style=DIM)
    for label, value in rows:
        table.add_row(label, value)
    CONSOLE.print(table)
    CONSOLE.print()


def main() -> None:
    index = build_index()
    idf_model = build_idf_model(index)
    from tfidf import build_tfidf_model
    model = build_tfidf_model(index, idf_model)

    CONSOLE.rule("[bold cyan]Step 7 - Vector Space Model[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} documents  {SEP}  "
        f"{len(idf_model.idf):,} terms  {SEP}  "
        f"score(q,d) = SUM w(t,q) x w(t,d)[/]\n"
    )

    section_idea()
    section_query_vector(index, idf_model, "machine learning healthcare")
    section_scoring(index, model, "machine learning healthcare")
    section_scoring(index, model, "medical imaging")
    section_dot_flaw(index, model)
    section_summary(index, model)


if __name__ == "__main__":
    main()