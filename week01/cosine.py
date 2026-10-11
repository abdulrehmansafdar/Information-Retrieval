"""Step 8 - Cosine similarity and ranked results.

Step 7 left a specific problem: the dot product scores a document on how much
weight it carries, not on how much of that weight answers the query. Two
documents with the same weight for the query term tied at exactly 10.604, with
nothing to separate them.

Cosine similarity divides the dot product by both magnitudes:

    cos(q, d) = (q . d) / (|q| x |d|)

The result is the cosine of the angle between the two vectors, between 0 and 1:

    1.0  the same direction, so every weighted term in d also appears in q
    0.0  no shared terms
    0.3  partial agreement

Dividing by |d| is what removes the length effect. A long document no longer
wins simply by being long; it wins by being *concentrated* on the query. For 122
of the 608 terms in this corpus, cosine and dot product produce different
rankings.

This file also delivers the ranked result list, which is the last of the five
required features.

Run:  python cosine.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from idf import build_idf_model
from index import InvertedIndex, build_index
from tfidf import build_tfidf_model
from vsm import (
    QueryVector,
    build_query_vector,
    document_vector,
    dot_product,
    magnitude,
    score_documents,
)
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Angle between two weighted vectors, in the range 0.0 to 1.0.

    Returns 0.0 if either vector has zero magnitude, which is the only case where
    the division is undefined. It arises for an empty query, never for a real
    document.
    """
    length_a, length_b = magnitude(a), magnitude(b)
    if length_a == 0.0 or length_b == 0.0:
        return 0.0
    return dot_product(a, b) / (length_a * length_b)


@dataclass
class RankedResult:
    rank: int
    doc_id: str
    title: str
    score: float
    length: int
    matched_terms: list[tuple[str, float]] = field(default_factory=list)


def rank_documents(query: str | QueryVector, index: InvertedIndex, model,
                   top: int | None = None) -> list[RankedResult]:
    """Score every document by cosine similarity and return them best first.

    Documents with no shared query term score 0.0 and are dropped, so the result
    is the relevant set rather than all 20 documents.
    """
    vector = query if isinstance(query, QueryVector) else build_query_vector(
        query, model.idf_model
    )
    by_id = {d.doc_id: d for d in index.documents}

    results = []
    for document in index.documents:
        doc_vector = document_vector(document.doc_id, model)
        score = cosine_similarity(vector.terms, doc_vector)
        if score <= 0.0:
            continue
        matched = sorted(((t, vector.terms[t] * doc_vector[t])
                          for t in vector.terms if doc_vector.get(t, 0.0) > 0),
                         key=lambda kv: -kv[1])
        results.append(RankedResult(
            rank=0, doc_id=document.doc_id, title=document.title,
            score=score, length=len(document.terms), matched_terms=matched,
        ))
    results.sort(key=lambda r: (-r.score, r.doc_id))
    for position, result in enumerate(results, 1):
        result.rank = position
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
ARROW = "→" if _UNICODE else "->"


def bar(fraction: float, width: int = 20) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(
        BAR_OFF * (width - filled), style="grey30"
    )


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def ranked_table(results: list[RankedResult], width: int = 20) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style=DIM, no_wrap=True)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    table.add_column("cosine", justify="right", style=WARM, no_wrap=True)
    table.add_column("relevance", width=width)
    for result in results:
        table.add_row(str(result.rank), result.doc_id, result.title,
                      f"{result.score:.4f}", bar(result.score, width - 2))
    return table


def section_formula() -> None:
    heading(1, "THE FORMULA")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("step 7", "score = q . d            (the dot product)")
    body.add_row("step 8", "[bold]cos(q,d) = (q . d) / (|q| x |d|)[/]")
    body.add_row("why", "dividing by |d| cancels out how much total weight a"
                        " document carries, leaving only the direction")
    body.add_row("range", "1.0 identical direction, 0.0 nothing in common")
    CONSOLE.print(Panel(body, title="cosine similarity", title_align="left",
                        border_style="grey35", padding=(0, 1)))


def section_tie_broken(index: InvertedIndex, model) -> None:
    heading(2, "THE TIE IS BROKEN")
    term = "abl"
    vector = build_query_vector(term, model.idf_model)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("len", justify="right", style=DIM, no_wrap=True)
    table.add_column("tf-idf", justify="right", no_wrap=True)
    table.add_column("|d|", justify="right", style=DIM, no_wrap=True)
    table.add_column("dot", justify="right", style=DIM, no_wrap=True)
    table.add_column("cosine", justify="right", style=WARM, no_wrap=True)
    for result in score_documents(vector, index, model):
        doc_vector = document_vector(result.doc_id, model)
        cosine = cosine_similarity(vector.terms, doc_vector)
        table.add_row(result.doc_id, str(result.length),
                      f"{doc_vector.get(term, 0.0):.3f}",
                      f"{magnitude(doc_vector):.3f}",
                      f"{result.score:.3f}", f"{cosine:.4f}")
    CONSOLE.print(table)

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("step 7 said", "doc01 and doc12 tie at 10.604. No ranking"
                                " possible.")
    body.add_row("step 8 says", "doc01 scores 0.2762 and doc12 only 0.2075, so"
                                " doc01 is the better answer.")
    body.add_row("why not length", "doc12 is the SHORTER document (79 against 83)"
                                 " and still loses. Its weight is spread thinner"
                                 " across repeated terms, giving it the larger"
                                 " magnitude (22.19 against 16.68).")
    body.add_row("lesson", "cosine rewards concentration, not brevity.")
    CONSOLE.print(Panel(body, border_style="cyan", padding=(0, 1)))


def section_how_much(index: InvertedIndex, model) -> None:
    heading(3, "HOW MUCH COSINE CHANGES")
    disagreements = 0
    examples = []
    for candidate in index.vocabulary:
        vector = build_query_vector(candidate, model.idf_model)
        by_dot = score_documents(vector, index, model)
        if len(by_dot) < 2:
            continue
        by_cos = rank_documents(vector, index, model)
        if [r.doc_id for r in by_dot][:2] != [r.doc_id for r in by_cos][:2]:
            disagreements += 1
            if len(examples) < 3:
                examples.append((candidate, by_dot[:2], by_cos[:2]))

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("query", style="bold white", no_wrap=True)
    table.add_column("by dot product", style=DIM, overflow="fold")
    table.add_column("by cosine", style=WARM, overflow="fold")
    for candidate, dot_top, cos_top in examples:
        table.add_row(candidate,
                      ", ".join(r.doc_id for r in dot_top),
                      ", ".join(r.doc_id for r in cos_top))
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [bold]{disagreements} of {len(index.vocabulary)}[/] terms"
        f" ( {disagreements / len(index.vocabulary):.0%} ) produce a"
        f" different top-2 order under\n  cosine than under the raw dot"
        f" product. Normalising is not a cosmetic adjustment.\n"
    )


def section_ranked_results(index: InvertedIndex, model) -> None:
    heading(4, "RANKED RESULTS")
    queries = [
        "machine learning healthcare",
        "medical imaging",
        "neural network",
        "security attack anomaly",
    ]
    for query in queries:
        vector = build_query_vector(query, model.idf_model)
        results = rank_documents(vector, index, model)
        CONSOLE.print(f"  [bold white]{query!r}[/]  [grey58]{len(results)} of"
                      f" {index.number_of_documents} documents score above zero[/]")
        CONSOLE.print(ranked_table(results, 14))
        if results:
            best = results[0]
            parts = ", ".join(t for t, _ in best.matched_terms)
            CONSOLE.print(f"    [grey58]top hit matched: {parts}[/]")
        CONSOLE.print()


def section_boolean_vs_ranked(index: InvertedIndex, model) -> None:
    heading(5, "BOOLEAN AND RANKED, SIDE BY SIDE")
    from boolean_search import search as boolean_search

    boolean_query = "machine learning AND healthcare"
    ranked_query = "machine learning healthcare"

    boolean_hits = boolean_search(index, boolean_query)
    ranked = rank_documents(ranked_query, index, model)

    boolean_ids = {d.doc_id for d in boolean_hits}
    ranked_ids = {r.doc_id for r in ranked}

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("model", style=ACCENT, no_wrap=True)
    table.add_column("query", style="white", overflow="fold")
    table.add_column("hits", justify="right", no_wrap=True)
    table.add_column("what it means", style=DIM, overflow="fold")
    table.add_row("Boolean", boolean_query, str(len(boolean_ids)),
                  "must contain BOTH terms - exact, no order, no partial credit")
    table.add_row("ranked", ranked_query, str(len(ranked_ids)),
                  "graded match - returns everything related, best first")
    CONSOLE.print(table)

    overlap = sorted(boolean_ids & ranked_ids)
    ranked_only = sorted(ranked_ids - boolean_ids)
    comparison = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    comparison.add_column("group", style=ACCENT, no_wrap=True)
    comparison.add_column("n", justify="right", style=DIM, no_wrap=True)
    comparison.add_column("documents", style=DIM, overflow="fold")
    comparison.add_row("both models", str(len(overlap)), ", ".join(overlap))
    comparison.add_row("ranked only", str(len(ranked_only)), ", ".join(ranked_only))
    comparison.add_row("Boolean only", str(len(boolean_ids - ranked_ids)),
                       ", ".join(sorted(boolean_ids - ranked_ids)) or "- none -")
    CONSOLE.print(comparison)

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("agree", f"both put the two healthcare documents first, so the"
                          f" models do not contradict each other on the obvious"
                          f" case.")
    body.add_row("differ", f"Boolean returns nothing for a document that matches"
                           f" only ONE query term. Ranked retrieval still returns"
                           f" {len(ranked_only)} such documents, with lower scores.")
    body.add_row("neither wins", "Boolean cannot rank, so it cannot tell a good"
                                 " partial match from a poor one. Ranked retrieval"
                                 " can, but cannot express 'must contain'. A real"
                                 " engine uses both: Boolean to filter, ranking"
                                 " to order.")
    CONSOLE.print(Panel(body, padding=(0, 1)))


def section_properties(index: InvertedIndex, model) -> None:
    heading(6, "PROPERTIES")
    vector = build_query_vector("machine learning healthcare", model.idf_model)
    first = rank_documents(vector, index, model)[0]
    doc_vector = document_vector(first.doc_id, model)

    scores = [r.score for r in rank_documents(vector, index, model)]
    checks = [
        ("all scores within [0, 1]",
         "yes" if all(0.0 <= s <= 1.0 for s in scores) else "NO"),
        ("self-similarity is exactly 1.0",
         f"{cosine_similarity(doc_vector, doc_vector):.6f}"),
        ("symmetric (cos(a,b) == cos(b,a))",
         "yes" if abs(cosine_similarity(vector.terms, doc_vector)
                      - cosine_similarity(doc_vector, vector.terms)) < 1e-15 else "NO"),
        ("cosine of an empty vector", f"{cosine_similarity({}, doc_vector):.1f}"
                                      "  (guarded against division by zero)"),
        ("highest score", f"{max(scores):.4f}  ({first.doc_id})"),
        ("lowest returned score", f"{min(scores):.4f}"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    for label, value in checks:
        table.add_row(label, value)
    CONSOLE.print(table)
    CONSOLE.print()


def main() -> None:
    index = build_index()
    idf_model = build_idf_model(index)
    model = build_tfidf_model(index, idf_model)

    CONSOLE.rule("[bold cyan]Step 8 - Cosine Similarity and Ranked Results[/]",
                 style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} documents  {SEP}  "
        f"{len(idf_model.idf):,} terms  {SEP}  "
        f"cos(q,d) = (q.d) / (|q| x |d|)[/]\n"
    )

    section_formula()
    section_tie_broken(index, model)
    section_how_much(index, model)
    section_ranked_results(index, model)
    section_boolean_vs_ranked(index, model)
    section_properties(index, model)


if __name__ == "__main__":
    main()