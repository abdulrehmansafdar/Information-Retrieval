"""Interactive ranked search - the finished mini search engine.

    python search.py                      interactive prompt
    python search.py "medical imaging"    one query and exit
    python search.py --top 5 "neural network"

Two retrieval models are available on the same corpus and the same index:

    plain text          ranked retrieval - VSM query vector, cosine similarity,
                        documents ordered best first
    boolean: A AND B    Boolean retrieval - exact set from step 4, no ranking
    both: <query>       run a plain query and its AND-ed equivalent side by side
    top N <query>       limit the result list
    terms <docNN>       inspect one document's weights
    stats               corpus summary
    help / quit

This is the deliverable for the fifth required feature: display ranked results.
"""

from __future__ import annotations

import sys

from boolean_search import BooleanSyntaxError, search as boolean_search
from cosine import RankedResult, cosine_similarity, rank_documents
from idf import build_idf_model
from index import build_index, documents_containing
from preprocess import get_stop_words, preprocess_query
from tfidf import build_tfidf_model
from vsm import build_query_vector, document_vector
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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

INDEX = build_index()
IDF_MODEL = build_idf_model(INDEX)
TFIDF_MODEL = build_tfidf_model(INDEX, IDF_MODEL)
BY_ID = {d.doc_id: d for d in INDEX.documents}


def bar(fraction: float, width: int = 18) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(
        BAR_OFF * (width - filled), style="grey30"
    )


def results_table(results: list[RankedResult]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style=DIM, no_wrap=True)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    table.add_column("cosine", justify="right", style=WARM, no_wrap=True)
    table.add_column("relevance", width=18)
    for result in results:
        table.add_row(str(result.rank), result.doc_id, result.title,
                      f"{result.score:.4f}", bar(result.score, 16))
    return table


def do_ranked(query: str, top: int | None = None) -> list[RankedResult]:
    vector = build_query_vector(query, IDF_MODEL)
    results = rank_documents(vector, INDEX, TFIDF_MODEL)

    CONSOLE.print(f"  [bold white]{query!r}[/]  [grey58]{len(results)} of"
                  f" {INDEX.number_of_documents} documents match[/]")
    if vector.unknown_terms:
        CONSOLE.print(f"  [yellow]not in the vocabulary, ignored:"
                      f" {', '.join(vector.unknown_terms)}[/]")
    if not results:
        CONSOLE.print("  [grey58]nothing matched[/]\n")
        return results

    shown = results[:top] if top else results
    CONSOLE.print(results_table(shown))
    if top and len(results) > top:
        CONSOLE.print(f"  [grey58]showing {top} of {len(results)}[/]")

    best = shown[0]
    parts = ", ".join(f"{t} ({w:.2f})" for t, w in best.matched_terms)
    CONSOLE.print(f"  [grey58]best match: {parts}[/]\n")
    return results


def do_boolean(query: str) -> None:
    try:
        documents = boolean_search(INDEX, query)
    except BooleanSyntaxError as error:
        CONSOLE.print(f"  [red]query error:[/] {error}\n")
        return
    CONSOLE.print(f"  [bold white]boolean:[/] [bold white]{query!r}[/]"
                  f"  [grey58]{len(documents)} of"
                  f" {INDEX.number_of_documents} documents[/]")
    if not documents:
        CONSOLE.print("  [grey58]nothing matched[/]\n")
        return
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    for document in documents:
        table.add_row(document.doc_id, document.title, str(len(document.terms)))
    CONSOLE.print(table)
    CONSOLE.print(f"  [grey58]Boolean returns a set, so there is no order[/]\n")


def do_both(query: str, top: int | None = None) -> None:
    """Run the plain query and the AND-ed equivalent together.

    The Boolean form is built from the *preprocessed* query terms, not the raw
    words. Splitting raw words would turn 'machine learning healthcare' into
    'machine AND learning AND healthcare' and lose the phrase, returning
    nothing.
    """
    terms = [t for t in preprocess_query(query) if t not in get_stop_words()]
    if not terms:
        CONSOLE.print("  [red]no searchable terms in that query[/]\n")
        return
    boolean_query = " AND ".join(terms)

    ranked = rank_documents(query, INDEX, TFIDF_MODEL)
    boolean_ids = {d.doc_id for d in boolean_search(INDEX, boolean_query)}

    CONSOLE.print(f"  [bold white]{query!r}[/]  [grey58]-> terms"
                  f" {terms}[/]")
    CONSOLE.print(f"  [grey58]ranked {len(ranked)} documents;"
                  f" boolean: {boolean_query!r} gives {len(boolean_ids)}[/]\n")

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style=DIM, no_wrap=True)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    table.add_column("cosine", justify="right", style=WARM, no_wrap=True)
    table.add_column("AND all", justify="center", no_wrap=True)
    shown = ranked[:top] if top else ranked
    for result in shown:
        mark = "[green]yes[/]" if result.doc_id in boolean_ids else "[grey50]no[/]"
        table.add_row(str(result.rank), result.doc_id, result.title,
                      f"{result.score:.4f}", mark)
    CONSOLE.print(table)

    only = sorted({r.doc_id for r in ranked} - boolean_ids)
    if only:
        CONSOLE.print(f"  [grey58]partial matches, dropped by AND:"
                      f" {', '.join(only)}[/]")
    CONSOLE.print()


def do_terms(doc_id: str) -> None:
    document = BY_ID.get(doc_id)
    if document is None:
        CONSOLE.print(f"  [red]no such document:[/] {doc_id}\n")
        return
    vector = document_vector(doc_id, TFIDF_MODEL)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("tf", justify="right", style=DIM, no_wrap=True)
    table.add_column("idf", justify="right", style=DIM, no_wrap=True)
    table.add_column("tf-idf", justify="right", style=WARM, no_wrap=True)
    for term, weight in sorted(vector.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
        table.add_row(term, str(document.terms.count(term)),
                      f"{IDF_MODEL.idf_for(term):.4f}", f"{weight:.4f}")
    CONSOLE.print(f"  [bold white]{doc_id}[/]  {document.title}"
                  f"  [grey58]{len(document.terms)} terms,"
                  f" {len(vector)} distinct[/]")
    CONSOLE.print(table)
    CONSOLE.print()


def do_stats() -> None:
    rows = [
        ("documents", str(INDEX.number_of_documents)),
        ("vocabulary terms", f"{len(INDEX.vocabulary):,}"),
        ("postings", f"{INDEX.total_postings:,}"),
        ("total terms", f"{sum(len(d.terms) for d in INDEX.documents):,}"),
        ("tf-idf weights", f"{INDEX.total_postings:,}"),
        ("preprocessing", "lowercase, tokenise, stem, phrases, stop words"),
        ("boolean model", "AND / OR / NOT with parentheses"),
        ("ranked model", "TF-IDF vectors, cosine similarity"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=ACCENT)
    table.add_column(style="white")
    for label, value in rows:
        table.add_row(label, value)
    CONSOLE.print(Panel(table, title="corpus", title_align="left",
                        border_style="grey35", padding=(0, 1)))


HELP = [
    ("<query>", "ranked search using TF-IDF and cosine"),
    ("boolean: <query>", "Boolean search, e.g. boolean: healthcare AND diagnosis"),
    ("both: <query>", "ranked results with the AND-ed equivalent alongside"),
    ("top N <query>", "limit results, e.g. top 3 neural network"),
    ("terms docNN", "show a document's highest-weight terms"),
    ("stats", "corpus summary"),
    ("help, quit", "this list / exit"),
]


def print_help() -> None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=ACCENT, no_wrap=True)
    table.add_column(style="white")
    for label, description in HELP:
        table.add_row(label, description)
    CONSOLE.print(Panel(table, title="commands", title_align="left",
                        border_style="grey35", padding=(0, 1)))


def handle(line: str, top: int | None = None) -> None:
    line = line.strip()
    if not line:
        return
    lowered = line.lower()
    if lowered in {"quit", "exit", "q"}:
        raise SystemExit(0)
    if lowered == "help":
        print_help()
        return
    if lowered == "stats":
        do_stats()
        return

    if lowered.startswith("boolean:"):
        do_boolean(line.split(":", 1)[1].strip())
        return
    if lowered.startswith("both:"):
        do_both(line.split(":", 1)[1].strip(), top)
        return
    if lowered.startswith("terms"):
        do_terms(line.split(maxsplit=1)[1].strip() if " " in line else "")
        return

    limit = top
    if lowered.startswith("top "):
        parts = line.split(maxsplit=2)
        if len(parts) < 3 or not parts[1].isdigit():
            CONSOLE.print("  [red]usage:[/] top N <query>\n")
            return
        limit = int(parts[1])
        line = parts[2]
    do_ranked(line, limit)


def main() -> None:
    arguments = sys.argv[1:]
    top: int | None = None
    if arguments and arguments[0] in {"-n", "--top"} and len(arguments) >= 2:
        top = int(arguments[1])
        arguments = arguments[2:]

    CONSOLE.rule("[bold cyan]Mini Search Engine[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{INDEX.number_of_documents} documents  {SEP}  "
        f"{len(INDEX.vocabulary):,} terms  {SEP}  ranked = TF-IDF + cosine  {SEP}  "
        f"boolean = AND/OR/NOT[/]\n"
    )

    if arguments:
        handle(" ".join(arguments), top)
        return

    print_help()
    while True:
        try:
            line = CONSOLE.input("\n[bold cyan]search[/] > ")
        except (EOFError, KeyboardInterrupt):
            CONSOLE.print()
            return
        try:
            handle(line, top)
        except SystemExit:
            CONSOLE.print("\n[grey58]bye[/]\n")
            return


if __name__ == "__main__":
    main()