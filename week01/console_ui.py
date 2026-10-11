"""Console front end for the mini search engine.

    python main.py --console                interactive prompt
    python main.py --console "medical imaging"
    python main.py --console --demo
    python main.py --console --json "query"

The retrieval logic lives in engine.py, which the Streamlit app also uses.
This file only handles input and output.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import sys

import engine as ir
from demo import demo_sections
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


def bar(fraction: float, width: int = 18) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(
        BAR_OFF * (width - filled), style="grey30"
    )


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def results_table(response: ir.SearchResponse) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style=DIM, no_wrap=True)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    if response.mode == "ranked":
        table.add_column("cosine", justify="right", style=WARM, no_wrap=True)
        table.add_column("AND all", justify="center", no_wrap=True)
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)

    for hit in response.hits:
        row = [str(hit.rank), hit.doc_id, hit.title]
        if response.mode == "ranked":
            mark = "[green]yes[/]" if hit.satisfies_and else "[grey50]no[/]"
            row += [f"{hit.score:.4f}", mark]
        row.append(str(hit.length))
        table.add_row(*row)
    return table


def render_response(response: ir.SearchResponse) -> None:
    CONSOLE.print(f"  [bold white]{response.query!r}[/]  [grey58]mode:"
                  f" {response.mode}[/]")
    if response.terms:
        CONSOLE.print(f"  [grey58]terms: {response.terms}[/]")
    if response.unknown_terms:
        CONSOLE.print(f"  [yellow]not in the vocabulary, ignored:"
                      f" {', '.join(response.unknown_terms)}[/]")
    if not response.ok:
        CONSOLE.print(f"  [red]{response.error}[/]\n")
        return
    if response.boolean_error:
        CONSOLE.print(f"  [yellow]the strict AND reading could not be"
                      f" evaluated: {response.boolean_error}[/]")
    if not response.hits:
        CONSOLE.print("  [grey58]nothing matched[/]\n")
        return
    CONSOLE.print(results_table(response))
    total = ir.get_engine().number_of_documents
    if response.mode == "ranked":
        CONSOLE.print(f"  [grey58]{response.count} of {total} documents"
                      f"{SEP}strict AND matches: {len(response.boolean_ids)}[/]\n")
    else:
        CONSOLE.print(f"  [grey58]{response.count} of {total} documents"
                      f" satisfy the Boolean expression[/]\n")


# ---------------------------------------------------------------------------
# output formats for other programs
# ---------------------------------------------------------------------------


def to_json(response: ir.SearchResponse) -> str:
    payload = {
        "query": response.query,
        "mode": response.mode,
        "terms": response.terms,
        "unknown_terms": response.unknown_terms,
        "boolean_ids": response.boolean_ids,
        "error": response.error,
        "count": response.count,
        "hits": [
            {
                "rank": h.rank,
                "doc_id": h.doc_id,
                "title": h.title,
                "score": round(h.score, 6),
                "length": h.length,
                "satisfies_and": h.satisfies_and,
                "matched_terms": [{"term": t, "weight": round(w, 6)}
                                  for t, w in h.matched_terms],
            }
            for h in response.hits
        ],
    }
    return json.dumps(payload, indent=2)


def render_to(console: Console, response: ir.SearchResponse) -> None:
    """Render a response to any Console, so it can be captured for export."""
    console.print(f"[bold white]{response.query!r}[/]  [grey58]mode:"
                  f" {response.mode}[/]")
    if response.terms:
        console.print(f"  [grey58]terms: {response.terms}[/]")
    if response.unknown_terms:
        console.print(f"  [yellow]not in the vocabulary, ignored:"
                      f" {', '.join(response.unknown_terms)}[/]")
    if not response.ok:
        console.print(f"  [red]{response.error}[/]")
        return
    if response.boolean_error:
        console.print(f"  [yellow]the strict AND reading could not be"
                      f" evaluated: {response.boolean_error}[/]")
    if not response.hits:
        console.print("  [grey58]nothing matched[/]")
        return
    console.print(results_table(response))
    total = ir.get_engine().number_of_documents
    if response.mode == "ranked":
        console.print(f"  [grey58]{response.count} of {total} documents"
                      f" | strict AND matches: {len(response.boolean_ids)}[/]")
    else:
        console.print(f"  [grey58]{response.count} of {total} documents"
                      f" satisfy the Boolean expression[/]")


def to_html(response: ir.SearchResponse) -> str:
    """A standalone HTML page.

    rich's export_html() already returns a complete document with DOCTYPE, head,
    style and body, so wrapping it again would produce two doctypes. Only the
    title is inserted. Note that rich exports as styled <pre><code> rather than
    an HTML <table>, which is a faithful copy of what the terminal shows.
    """
    capture = Console(record=True, width=100, file=io.StringIO())
    render_to(capture, response)
    rendered_html = capture.export_html(inline_styles=True)
    safe_title = html.escape(response.query)
    return rendered_html.replace("<head>", f"<head><title>{safe_title}</title>", 1)


def to_text(response: ir.SearchResponse) -> str:
    lines = [f"query : {response.query}", f"mode  : {response.mode}"]
    if response.terms:
        lines.append(f"terms : {', '.join(response.terms)}")
    if response.error:
        lines.append(f"error : {response.error}")
        return "\n".join(lines)
    if not response.hits:
        lines.append("no matches")
        return "\n".join(lines)
    lines.append(f"{'#':>3}  {'doc':<7}  {'cosine':>8}  {'AND':<4}  title")
    for hit in response.hits:
        cosine = f"{hit.score:>8.4f}" if response.mode == "ranked" else f"{'-':>8}"
        mark = ("yes" if hit.satisfies_and else "no") if response.mode == "ranked" else "-"
        lines.append(f"{hit.rank:>3}  {hit.doc_id:<7}  {cosine}  {mark:<4}  {hit.title}")
    if response.mode == "ranked":
        lines.append(f"\n{response.count} ranked results, strict AND matches:"
                     f" {len(response.boolean_ids)}")
    else:
        lines.append(f"\n{response.count} Boolean results")
    return "\n".join(lines)


def _write_machine(text: str) -> None:
    """Write machine-readable output as UTF-8 bytes.

    print() would fail on this machine because stdout is cp1252 and rich's HTML
    export contains box-drawing characters that cp1252 cannot represent.
    """
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# corpus statistics
# ---------------------------------------------------------------------------


def render_stats() -> None:
    stats = ir.corpus_stats()
    low, high = stats["terms_per_document"]
    rows = [
        ("documents", str(stats["documents"])),
        ("vocabulary terms", f"{stats['vocabulary']:,}"),
        ("postings", f"{stats['postings']:,}"),
        ("total terms", f"{stats['total_terms']:,}"),
        ("terms per document", f"{low} to {high}  (ratio {high / low:.2f}x)"),
        ("hapax terms (df = 1)", f"{stats['hapax']:,}"),
        ("idf range", f"{stats['idf_min']:.4f} to {stats['idf_max']:.4f}"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=ACCENT)
    table.add_column(style="white")
    for label, value in rows:
        table.add_row(label, value)
    CONSOLE.print(Panel(table, title="corpus", title_align="left",
                        border_style="grey35", padding=(0, 1)))

    top = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    top.add_column("most frequent terms", style="bold white", no_wrap=True)
    top.add_column("df", justify="right", style=ACCENT, no_wrap=True)
    top.add_column("idf", justify="right", style=WARM, no_wrap=True)
    top.add_column("share", width=18)
    largest = stats["top_by_df"][0][1]
    for term, df in stats["top_by_df"]:
        idf = ir.get_engine().idf_model.idf_for(term)
        top.add_row(term, str(df), f"{idf:.4f}",
                    bar(df / largest, 16))
    CONSOLE.print(top)


# ---------------------------------------------------------------------------
# pipeline demo
# ---------------------------------------------------------------------------


def render_demo() -> None:
    for number, (title, renderable) in enumerate(demo_sections(), 1):
        heading(number, title.upper())
        CONSOLE.print(renderable)
        CONSOLE.print()


# ---------------------------------------------------------------------------
# interactive
# ---------------------------------------------------------------------------


HELP = [
    ("<query>", "ranked search, or Boolean if you use AND/OR/NOT"),
    ("boolean: <query>", "force Boolean retrieval"),
    ("ranked: <query>", "force cosine ranking"),
    ("both: <query>", "ranked results with the strict AND reading"),
    ("stats", "corpus summary"),
    ("demo", "pipeline walkthrough"),
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


def handle(line: str, default_mode: str = "auto",
           top: int | None = None) -> bool:
    """Returns False when the user asked to quit."""
    line = line.strip()
    if not line:
        return True
    lowered = line.lower()
    if lowered in {"quit", "exit", "q"}:
        return False
    if lowered == "help":
        print_help()
        return True
    if lowered == "stats":
        render_stats()
        return True
    if lowered == "demo":
        render_demo()
        return True

    mode = default_mode
    for prefix, forced in (("boolean:", "boolean"), ("ranked:", "ranked"),
                           ("both:", "both")):
        if lowered.startswith(prefix):
            mode = "both" if forced == "both" else forced
            line = line.split(":", 1)[1].strip()
            break

    response = ir.search(line, top=top, mode="auto" if mode == "both" else mode)
    render_response(response)
    if mode == "both" and response.ok and response.hits:
        CONSOLE.print(f"  [grey58]strict AND reading:"
                      f" {len(response.boolean_ids)} documents[/]\n")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Mini search engine: Boolean retrieval and vector ranking.")
    parser.add_argument("query", nargs="*", help="query to run")
    parser.add_argument("--demo", action="store_true",
                        help="print the pipeline walkthrough")
    parser.add_argument("--stats", action="store_true",
                        help="print corpus statistics")
    parser.add_argument("--top", type=int, default=None,
                        help="limit the number of results")
    parser.add_argument("--mode", choices=["auto", "ranked", "boolean"],
                        default="auto", help="force a retrieval model")
    parser.add_argument("--text", action="store_true",
                        help="plain text output, for piping")
    parser.add_argument("--html", action="store_true",
                        help="HTML output, for a web page")
    parser.add_argument("--json", action="store_true",
                        help="JSON output, for another program")
    arguments = parser.parse_args(argv)

    if arguments.demo:
        render_demo()
        return
    if arguments.stats:
        render_stats()
        return

    query = " ".join(arguments.query).strip()

    if arguments.json or arguments.html or arguments.text:
        if not query:
            print("error: those formats need a query", file=sys.stderr)
            print('example: python main.py --json "medical imaging"',
                  file=sys.stderr)
            raise SystemExit(2)
        response = ir.search(query, top=arguments.top, mode=arguments.mode)
        if arguments.json:
            _write_machine(to_json(response))
        elif arguments.html:
            _write_machine(to_html(response))
        else:
            _write_machine(to_text(response))
        return

    built = ir.get_engine()
    CONSOLE.rule("[bold cyan]Mini Search Engine[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{built.number_of_documents} documents  {SEP}  "
        f"{len(built.vocabulary):,} terms  {SEP}  "
        f"{built.total_postings:,} postings  {SEP}  "
        f"boolean = AND/OR/NOT  {SEP}  ranked = TF-IDF + cosine[/]\n"
    )

    if query:
        handle(query, arguments.mode, arguments.top)
        return

    print_help()
    while True:
        try:
            line = CONSOLE.input("\n[bold cyan]search[/] > ")
        except (EOFError, KeyboardInterrupt):
            CONSOLE.print()
            return
        if not handle(line, arguments.mode, arguments.top):
            CONSOLE.print("\n[grey58]bye[/]\n")
            return


if __name__ == "__main__":
    main()
