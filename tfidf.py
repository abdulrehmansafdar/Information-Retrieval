"""Step 6 - TF-IDF weighting.

Step 5 produced two numbers and refused to multiply them. This step does.

    tfidf(t, d) = tf(t, d) x idf(t)

The effect is not cosmetic. In this corpus 11 of the 20 documents have a
*different* most-important term once weighting is applied:

    doc03  by tf     ->  'use'      (tf=4, idf=0.16, weight 0.65)
    doc03  by tf-idf ->  'clinic'   (tf=2, idf=2.30, weight 4.61)

Raw frequency promotes the word that appears in 17 of 20 documents. Weighting
promotes the word that identifies the document.

What this is NOT
----------------
tfidf(t, d) is a weight for one term inside one document. It is NOT a document
score and documents are NOT comparable yet: a longer document accumulates a
larger total. Fixing that needs a length norm, which is the vector space model
in step 7 and cosine in step 8. Neither is computed here.

Run:  python tfidf.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from index import InvertedIndex, build_index
from idf import IdfModel, build_idf_model
from preprocess import Document
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class TfidfModel:
    """Term weights for every document, built from the postings and the IDF table."""

    idf_model: IdfModel
    sublinear: bool = False
    weights: dict[str, dict[str, float]] = field(default_factory=dict)

    def weight(self, doc_id: str, term: str) -> float:
        """TF-IDF weight of one term in one document. 0.0 if absent from either."""
        return self.weights.get(doc_id, {}).get(term, 0.0)

    def document_terms(self, doc_id: str) -> dict[str, float]:
        return dict(self.weights.get(doc_id, {}))

    def top_terms(self, doc_id: str, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.document_terms(doc_id).items(),
                      key=lambda kv: (-kv[1], kv[0]))[:n]

    def document_total(self, doc_id: str) -> float:
        """Sum of a document's weights.

        NOT a relevance score. It grows with document length, which is exactly
        why the vector space model divides by a length norm later.
        """
        return sum(self.document_terms(doc_id).values())

    def term_weight(self, term: str, tf: int) -> float:
        """The formula itself, exposed for comparison."""
        if self.sublinear:
            scaled = 1.0 + math.log(tf) if tf > 0 else 0.0
        else:
            scaled = float(tf)
        return scaled * self.idf_model.idf_for(term)


def build_tfidf_model(index: InvertedIndex, idf_model: IdfModel,
                      sublinear: bool = False) -> TfidfModel:
    """Weight every (document, term) posting with TF-IDF."""
    model = TfidfModel(idf_model=idf_model, sublinear=sublinear)
    for document in index.documents:
        weights: dict[str, float] = {}
        for term in set(document.terms):
            tf = document.terms.count(term)
            weights[term] = model.term_weight(term, tf)
        model.weights[document.doc_id] = weights
    return model


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


def bar(fraction: float, width: int = 16) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(
        BAR_OFF * (width - filled), style="grey30"
    )


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def section_formula(model: TfidfModel, index: InvertedIndex) -> None:
    heading(1, "THE FORMULA")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("tf(t,d)", "from step 3 - the count inside each posting")
    body.add_row("idf(t)", "from step 5 - log(N / df), one value per term")
    body.add_row("tfidf", "[bold]tf(t,d) x idf(t)[/]  - one weight per "
                          "(document, term) pair")
    body.add_row("result", f"{index.total_postings} weights over "
                          f"{len(model.weights)} documents")
    CONSOLE.print(Panel(body, title="tf-idf", title_align="left",
                        border_style="grey35", padding=(0, 1)))

    limits = Table.grid(padding=(0, 2))
    limits.add_column(style=DIM, no_wrap=True)
    limits.add_column(style="white", overflow="fold")
    limits.add_row("not a score", "a weight describes one term inside one document.")
    limits.add_row("not comparable", "documents are still not comparable to each"
                                    " other: a longer document collects a larger"
                                    " total weight.")
    limits.add_row("fix comes later", "dividing by a length norm is the vector"
                                      " space model (step 7) and cosine (step 8).")
    CONSOLE.print(Panel(limits, border_style="yellow", padding=(0, 1)))


def _competition_ranks(values: dict[str, float]) -> dict[str, int]:
    """Rank items by value, giving equal values the same rank (1, 2, 2, 4...).

    Six terms with tf=2 are all rank 2 if one term has tf=4. Alphabetical
    tie-breaking would invent a false ordering.
    """
    distinct = sorted({v for v in values.values()}, reverse=True)
    return {key: distinct.index(value) + 1 for key, value in values.items()}


def _leader_set(values: dict[str, float]) -> frozenset[str]:
    return frozenset(k for k, v in values.items() if v == max(values.values()))


def _first_leader(values: dict[str, float]) -> str:
    """Alphabetically first of the leading terms, for a single-term comparison."""
    return sorted(_leader_set(values))[0]


def _top_display(values: dict[str, float]) -> str:
    """Highest value, ties shown honestly instead of being silently ordered."""
    winners = sorted(_leader_set(values))
    if len(winners) == 1:
        return winners[0]
    return f"{winners[0]} (+{len(winners) - 1} tied)"


def section_reordering(index: InvertedIndex, model: TfidfModel,
                       doc_id: str = "doc03") -> None:
    heading(2, f"THE REORDERING - {doc_id}")
    document = next(d for d in index.documents if d.doc_id == doc_id)
    weights = model.document_terms(doc_id)

    tf_values = {t: float(document.terms.count(t)) for t in set(document.terms)}
    tf_rank = _competition_ranks(tf_values)
    weight_rank = _competition_ranks(weights)

    by_tf = sorted(tf_values.items(), key=lambda kv: (-kv[1], kv[0]))
    by_weight = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = sorted({t for t, _ in by_weight[:5]} | {t for t, _ in by_tf[:2]},
                   key=lambda t: weight_rank[t])

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("tf", justify="right", no_wrap=True)
    table.add_column("idf", justify="right", style=DIM, no_wrap=True)
    table.add_column("tf-idf", justify="right", style=WARM, no_wrap=True)
    table.add_column("rank", justify="center", no_wrap=True)
    table.add_column("moved", justify="right", style=DIM, no_wrap=True)

    for term in shown:
        moved = tf_rank[term] - weight_rank[term]
        if moved > 0:
            marker = f"[green]+{moved}[/]"
        elif moved < 0:
            marker = f"[red]{moved}[/]"
        else:
            marker = "[grey50]0[/]"
        table.add_row(
            term,
            str(int(tf_values[term])),
            f"{model.idf_model.idf_for(term):.3f}",
            f"{weights[term]:.3f}",
            f"{tf_rank[term]} [grey50]->[/] {weight_rank[term]}",
            marker,
        )
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]rank is competition style, so equal tf values share a rank."
        " 'moved' is the change in\n  rank: positive means the term rose once"
        " weighting replaced raw frequency.[/]\n"
    )

    top_tf = by_tf[0]
    top_weight = by_weight[0]
    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("tf says", f"{_top_display(tf_values)} lead(s) on frequency"
                             f" alone at tf={int(top_tf[1])}")
    body.add_row("", f"but it appears in"
                     f" {model.idf_model.df_for(top_tf[0])} of"
                     f" {model.idf_model.number_of_documents} documents, so its"
                     f" weight is only {weights[top_tf[0]]:.3f}")
    body.add_row("tf-idf says", f"{_top_display(weights)} lead(s) with"
                                f" {top_weight[1]:.3f}  -  "
                                f"{top_weight[1] / weights[top_tf[0]]:.1f}x higher")
    CONSOLE.print(Panel(body, border_style="cyan", padding=(0, 1)))
    CONSOLE.print()


def section_all_documents(index: InvertedIndex, model: TfidfModel) -> None:
    heading(3, "HOW MANY DOCUMENTS CHANGE THEIR MOST IMPORTANT TERM")
    changed = 0
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("top by tf", style="white", no_wrap=True)
    table.add_column("top by tf-idf", style="white", no_wrap=True)
    table.add_column("change", style=DIM, no_wrap=True)

    for document in index.documents:
        weights = model.document_terms(document.doc_id)
        tf_values = {t: float(document.terms.count(t)) for t in set(document.terms)}
        # compare the leading term itself, not the size of the tie group
        if _first_leader(tf_values) == _first_leader(weights):
            note = "[grey50]same[/]"
        else:
            changed += 1
            note = "[green]changed[/]"
        table.add_row(document.doc_id, _top_display(tf_values),
                      _top_display(weights), note)
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [bold]{changed} of {index.number_of_documents}[/] documents get a"
        f" different most-important term once TF-IDF\n  replaces raw frequency."
        f" TF-IDF is not a small adjustment to TF; it reorders.\n"
    )


def section_sublinear(index: InvertedIndex, model: TfidfModel) -> None:
    heading(4, "SUBLINEAR TF - AND WHY IT BARELY MATTERS HERE")
    plain = TfidfModel(idf_model=model.idf_model, sublinear=False)
    linear_model = build_tfidf_model(index, model.idf_model, sublinear=False)
    log_model = build_tfidf_model(index, model.idf_model, sublinear=True)
    document = next(d for d in index.documents if d.doc_id == "doc03")

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("tf", justify="right", no_wrap=True)
    table.add_column("tf x idf", justify="right", style=WARM, no_wrap=True)
    table.add_column("(1+log tf) x idf", justify="right", style=ACCENT, no_wrap=True)

    for term, weight in linear_model.top_terms("doc03", 8):
        log_weight = log_model.weight("doc03", term)
        table.add_row(term, str(document.terms.count(term)),
                      f"{weight:.3f}", f"{log_weight:.3f}")
    CONSOLE.print(table)

    max_tf = max(max(d.terms.count(t) for t in set(d.terms))
                 for d in index.documents)
    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("the idea", "raw tf rewards repetition linearly: a term used 10"
                             " times scores 10x one used once. Sublinear tf,"
                             " (1 + log tf), damps that.")
    body.add_row("here", f"the highest tf anywhere in this corpus is {max_tf}, so"
                         f" there is very little repetition to damp. The top-8"
                         f" order is essentially unchanged.")
    body.add_row("choice", "the plain definition tf x idf is used everywhere else"
                           " in this project, because on this corpus the"
                           " alternative would add complexity and change nothing.")
    CONSOLE.print(Panel(body, padding=(0, 1)))


def section_hapax_promotion(index: InvertedIndex, model: TfidfModel,
                            doc_id: str = "doc03") -> None:
    heading(5, "A REAL WEAKNESS - RARE IS NOT THE SAME AS IMPORTANT")
    document = next(d for d in index.documents if d.doc_id == doc_id)
    weights = model.document_terms(doc_id)

    hapax = sorted(((t, w) for t, w in weights.items()
                    if model.idf_model.df_for(t) == 1),
                   key=lambda kv: (-kv[1], kv[0]))
    topic = sorted(((t, w) for t, w in weights.items()
                    if model.idf_model.df_for(t) > 1),
                   key=lambda kv: (-kv[1], kv[0]))

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("kind", style=ACCENT, no_wrap=True)
    table.add_column("best examples", style="white", overflow="fold")
    table.add_column("best weight", justify="right", style=WARM, no_wrap=True)
    table.add_row(f"df = 1 ({len(hapax)} terms)", ", ".join(t for t, _ in hapax[:8]),
                  f"{hapax[0][1]:.3f}")
    table.add_row(f"df > 1 ({len(topic)} terms)", ", ".join(t for t, _ in topic[:8]),
                  f"{topic[0][1]:.3f}")
    CONSOLE.print(table)

    above = [t for t, w in hapax if w > 2.0]
    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("what you see", f"all {len(hapax)} of {doc_id}'s hapax terms score"
                                 f" above 2.0, because tf=1 multiplied by the"
                                 f" maximum idf gives the maximum weight a term"
                                 f" can get.")
    body.add_row("the awkward part", f"words like {', '.join(above[:4])} are"
                                     f" grammatical but topically empty. TF-IDF"
                                     f" cannot tell 'rare' from 'important'.")
    body.add_row("not fixed here", "step 7's length norm and step 8's cosine"
                                   " reduce the influence of long tails of unique"
                                   " words, but no weighting scheme fully solves"
                                   " this.")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))
    CONSOLE.print()


def section_document_totals(index: InvertedIndex, model: TfidfModel) -> None:
    heading(6, "DOCUMENT TOTALS - AND WHY THEY ARE NOT SCORES")
    rows = [(model.document_total(d.doc_id), len(d.terms), d.doc_id)
            for d in index.documents]
    rows.sort(reverse=True)
    largest = rows[0][0]

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    table.add_column("sum of tf-idf", justify="right", style=WARM, no_wrap=True)
    table.add_column("total", width=20)
    for total, length, doc_id in rows:
        table.add_row(doc_id, str(length), f"{total:.2f}",
                      bar(total / largest, 18))
    CONSOLE.print(table)

    longest = max(index.documents, key=lambda d: len(d.terms))
    biggest = next(d for d in index.documents if d.doc_id == rows[0][2])
    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("watch", f"the longest document ({longest.doc_id}, "
                          f"{len(longest.terms)} terms, total"
                          f" {model.document_total(longest.doc_id):.2f}) is NOT the"
                          f" highest total. {biggest.doc_id} is, at"
                          f" {model.document_total(biggest.doc_id):.2f}.")
    body.add_row("why", "a document about a rare topic collects high-idf terms."
                        " Length is only one of two influences on the total.")
    body.add_row("still not a score", "these totals must not be used to rank"
                                      " documents. Dividing by a length norm is"
                                      " the vector space model, step 7.")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))


def section_properties(index: InvertedIndex, model: TfidfModel) -> None:
    heading(7, "PROPERTIES")
    all_weights = [w for terms in model.weights.values() for w in terms.values()]
    total_weights = sum(model.document_total(d.doc_id) for d in index.documents)
    checks = [
        ("(document, term) pairs weighted",
         f"{index.total_postings:,} (equal to the postings count)"),
        ("documents covered", f"{len(model.weights)}"),
        ("all weights positive", "yes" if all(w > 0 for w in all_weights) else "no"),
        ("zero weights", f"{sum(1 for w in all_weights if w == 0)}"),
        ("lowest weight", f"{min(all_weights):.4f}  (a high-df term at tf=1)"),
        ("highest weight", f"{max(all_weights):.4f}"),
        ("sum of all weights", f"{total_weights:,.2f}"),
        ("sublinear tf in use", "no - plain tf x idf"),
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

    CONSOLE.rule("[bold cyan]Step 6 - TF-IDF Weighting[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} documents  {SEP}  "
        f"{len(idf_model.idf):,} terms  {SEP}  "
        f"{index.total_postings:,} weights  {SEP}  "
        f"tfidf(t,d) = tf(t,d) x idf(t)[/]\n"
    )

    section_formula(model, index)
    section_reordering(index, model)
    section_all_documents(index, model)
    section_sublinear(index, model)
    section_hapax_promotion(index, model)
    section_document_totals(index, model)
    section_properties(index, model)


if __name__ == "__main__":
    main()