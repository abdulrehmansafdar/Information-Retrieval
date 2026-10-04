"""Presentation helpers for the pipeline walkthrough.

This module is deliberately separate from engine.py: the backend returns data,
while this module turns a few representative results into Rich renderables for
the console and Streamlit front ends.
"""

from __future__ import annotations

from io import StringIO

from engine import corpus_stats, get_engine, search
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_UNICODE = "utf" in (Console().options.encoding or "").lower()
_BAR_ON, _BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")


def _bar(fraction: float, width: int = 12) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(_BAR_ON * filled, style="cyan") + Text(
        _BAR_OFF * (width - filled), style="grey30"
    )


def demo_sections() -> list[tuple[str, object]]:
    """Return (title, Rich renderable) pairs with real corpus measurements."""
    built = get_engine()
    stats = corpus_stats()
    sections: list[tuple[str, object]] = []

    overview = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                     show_header=False)
    overview.add_column(style="grey58")
    overview.add_column(style="bold white")
    overview.add_row("1 preprocessing",
                     f"{stats['total_terms']:,} terms from {stats['documents']}"
                     f" documents, {stats['vocabulary']:,} distinct")
    overview.add_row("2 inverted index",
                     f"{stats['postings']:,} postings, {stats['hapax']:,} terms"
                     f" in exactly one document")
    overview.add_row("3 boolean",
                     "AND / OR / NOT with parentheses, parsed to a tree first")
    overview.add_row("4 tf, df, idf",
                     f"idf = log(N/df), range {stats['idf_min']:.4f} to"
                     f" {stats['idf_max']:.4f}")
    overview.add_row("5 tf-idf",
                     f"{stats['postings']:,} weights, one per (document, term)")
    overview.add_row("6 vector space",
                     "score(q,d) = SUM over t of w(t,q) x w(t,d)")
    overview.add_row("7 cosine",
                     "cos(q,d) = (q.d) / (|q| x |d|), range 0 to 1")
    sections.append(("pipeline", Panel(
        overview, title="mini search engine", title_align="left",
        border_style="grey35", padding=(0, 1))))

    weighting = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    weighting.add_column("term", style="bold white", no_wrap=True)
    weighting.add_column("df", justify="right", style="bold cyan", no_wrap=True)
    weighting.add_column("idf", justify="right", style="bold yellow", no_wrap=True)
    weighting.add_column("verdict", style="grey58")
    for term, df in stats["top_by_df"][:8]:
        verdict = ("worthless" if df >= 15 else "weak" if df >= 9
                   else "useful" if df >= 4 else "very useful")
        weighting.add_row(term, str(df),
                          f"{built.idf_model.idf_for(term):.4f}", verdict)
    sections.append(("term weighting, most frequent = least useful", weighting))

    ranked = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    ranked.add_column("#", justify="right", style="grey58", no_wrap=True)
    ranked.add_column("doc", style="bold cyan", no_wrap=True)
    ranked.add_column("title", style="white")
    ranked.add_column("cosine", justify="right", style="bold yellow", no_wrap=True)
    ranked.add_column("AND all", justify="center", no_wrap=True)
    example = search("machine learning healthcare")
    largest = max((hit.score for hit in example.hits), default=0.0)
    for hit in example.hits:
        ranked.add_row(
            str(hit.rank), hit.doc_id, hit.title, f"{hit.score:.4f}",
            "yes" if hit.satisfies_and else "no",
            _bar(hit.score / largest) if largest else "")
    sections.append(("ranked results for 'machine learning healthcare'", ranked))

    comparison = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    comparison.add_column("model", style="bold cyan", no_wrap=True)
    comparison.add_column("hits", justify="right", style="bold yellow", no_wrap=True)
    comparison.add_column("meaning", style="grey58", overflow="fold")
    comparison.add_row("ranked", str(example.count),
                       "graded cosine match, ordered best first")
    comparison.add_row("boolean AND", str(len(example.boolean_ids)),
                       "must contain every term, no order, no partial credit")
    sections.append(("boolean and ranked compared", comparison))
    return sections


def renderable_to_text(renderable, width: int = 94) -> str:
    """ASCII-safe plain-text form for the Streamlit pipeline tab."""
    buffer = StringIO()
    Console(file=buffer, width=width, no_color=True,
            legacy_windows=False).print(renderable)
    replacements = str.maketrans({
        "│": "|", "┃": "|", "║": "|",
        "─": "-", "━": "-", "═": "=",
        "┌": "+", "┐": "+", "└": "+", "┘": "+",
        "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
        "╭": "+", "╮": "+", "╰": "+", "╯": "+",
        "█": "#", "░": ".", "·": "|", "→": "->", "—": "-", "–": "-",
    })
    return buffer.getvalue().translate(replacements)
