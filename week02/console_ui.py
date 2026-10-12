"""Console front end for the week02 search engine.

    python main.py --console                       interactive prompt
    python main.py --console "proximity window"    one search
    python main.py --console --stats
    python main.py --console --json "term weighting"
    python main.py --console --text --top 3 "mean average precision"

Every retrieval decision is made by engine.py, which app.py also uses. This file
only handles input and output, which is why the two front ends can never
disagree about an answer.

Commands at the prompt
----------------------
    <query>                 search with the current mode
    bm25: <query>           force BM25
    bim: <query>            force the Binary Independence Model
    tfidf: <query>          force the week01 cosine baseline
    bool: <query>           force exact AND
    field: <query>          force field-restricted search
    title:search            field prefixes work directly, no prefix needed
    both: <query>           overlapped AND non-overlapped, side by side
    prox: <query>           force a proximal node operator
    top N <query>           limit the result list
    paper <p04>             every region of one paper, with term frequencies
    stats                   corpus summary
    modes                   what each mode does
    help, quit              this list / exit
"""

from __future__ import annotations

import argparse
import html
import io
import json
import sys

import engine as ir
from evaluate import evaluate_all
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CONSOLE = Console()

ACCENT = "bold cyan"
WARM = "bold yellow"
DIM = "grey58"

# The Windows console here is cp437 and stdout is cp1252, so block characters
# must not be hard-coded or they print as '?'. Pick them from the encoding.
_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"
ARROW = "→" if _UNICODE else "->"

REGION_MODES = {"overlapped", "non-overlapped", "proximal", "fields", "bm25f"}


def bar(fraction: float, width: int = 16) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(BAR_OFF * (width - filled),
                                                      style="grey30")


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def results_table(response: ir.SearchResponse, width: int = 14) -> Table:
    """One table shape per kind of answer, not one per mode.

    A region result's answer IS the region, so the table shows the region id and
    its label instead of the paper title. Carrying both is what pushed the
    table past the console width and wrapped every paper name onto four lines.
    """
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style=DIM, no_wrap=True)
    table.add_column("paper", style=ACCENT, no_wrap=True)
    table.add_column("score", justify="right", style=WARM, no_wrap=True)

    if response.mode in {"overlapped", "non-overlapped", "proximal"}:
        table.add_column("region", style=ACCENT, no_wrap=True)
        table.add_column("label", style="white", overflow="fold", ratio=1)
        table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    else:
        table.add_column("title", style="white", overflow="fold", ratio=2)
        table.add_column("field", style=ACCENT, overflow="fold", ratio=1)
        table.add_column("terms", justify="right", style=DIM, no_wrap=True)

    largest = max((h.score for h in response.hits), default=0.0)
    # No bar when every score is 0.0: that is the proximal model, which returns
    # a set of nodes rather than a ranking. A full-width bar there would imply
    # an ordering the model never produced.
    if largest > 0.0:
        table.add_column("score", width=width)

    # 'matched/total' is measured AFTER preprocessing, so a two-word query that
    # loses one word to the phrase list reads "1/1" - a complete match. Put the
    # dropped words in the denominator or the column lies.
    dropped = len(response.unknown)
    for hit in response.hits:
        wanted = hit.total_terms + dropped
        matched = f"{hit.matched}/{wanted}" if wanted else "-"
        if response.mode in {"overlapped", "non-overlapped", "proximal"}:
            detail, label = hit.region_id, hit.field
            if response.mode == "proximal":
                label = f"{hit.field} ({hit.length} terms)"
        else:
            detail, label = hit.field or "-", hit.title
        row = [str(hit.rank), hit.doc_id, f"{hit.score:.4f}",
               detail, label, matched]
        if largest > 0.0:
            row.append(bar(hit.score / largest, width - 2))
        table.add_row(*row)
    return table


def render_response(response: ir.SearchResponse) -> None:
    """Print one response to the terminal."""
    console = CONSOLE
    console.print(f"  [bold white]{response.query!r}[/]  [grey58]mode:"
                  f" {response.mode}[/]  [grey58]"
                  f"{ir.MODE_LABELS.get(response.mode, '')}[/]")
    if response.terms:
        console.print(f"  [grey58]terms: {', '.join(response.terms)}[/]")
    if response.unknown:
        engine = ir.get_engine()
        console.print(f"  [yellow]not in the vocabulary, ignored:"
                      f" {', '.join(response.unknown)}[/]")
        hint = ir.phrase_hint(response, engine)
        if hint:
            console.print(f"  [yellow]{hint}[/]")
            console.print("  [yellow]so those words were never searched -"
                          " only the rest of the query was[/]")
    if response.error:
        console.print(f"  [red]{response.error}[/]\n")
        return
    for note in response.notes:
        console.print(f"  [grey58]{note}[/]")
    if not response.hits:
        console.print("  [grey58]nothing matched[/]\n")
        return

    misrouted = ir.misrouted_field_prefix(response)
    if misrouted:
        console.print(f"  [yellow]{misrouted}[/]")

    console.print(results_table(response))
    stats = ir.get_engine()
    console.print(f"  [grey58]{response.count} results in"
                  f" {len(set(h.doc_id for h in response.hits))} of"
                  f" {engine_total()} papers[/]")
    coverage = ir.coverage_note(response)
    if coverage:
        console.print(f"  [grey58]{coverage}[/]")
    if response.parameters:
        parts = ", ".join(f"{k}={v}" for k, v in response.parameters.items())
        console.print(f"  [grey58]{parts}[/]")
    console.print()

    for hit in response.hits:
        if not hit.snippet:
            continue
        snippet = " ".join(hit.snippet.split())
        if len(snippet) > 150:
            snippet = snippet[:147] + "..."
        console.print(f"  [bold cyan]{hit.region_id}[/]  [grey58]{hit.note}[/]")
        console.print(f"  [white]{snippet}[/]")
    if any(h.snippet for h in response.hits):
        console.print()


def engine_total() -> int:
    return ir.get_engine().number_of_documents


def render_both(query: str, hierarchy: str, top: int | None) -> None:
    """Overlapped and non-overlapped, one after the other, with the difference."""
    strict = ir.search(query, mode="overlapped", top=top, hierarchy=hierarchy)
    loose = ir.search(query, mode="non-overlapped", top=top, hierarchy=hierarchy)

    console = CONSOLE
    console.print(f"  [bold white]{query!r}[/]  [grey58]hierarchy: {hierarchy}[/]\n")
    for label, response in (("OVERLAPPED - all terms in ONE region", strict),
                            ("NON-OVERLAPPED - terms in separate regions", loose)):
        console.print(f"  [bold cyan]{label}[/]  [grey58]{response.count}"
                      f" regions in {len(response.papers)} papers[/]")
        if response.hits:
            console.print(results_table(response))
        else:
            console.print("    [grey58]no region qualifies[/]")
        console.print()

    only_loose = sorted(set(loose.papers) - set(strict.papers))
    only_strict = sorted(set(strict.papers) - set(loose.papers))
    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("both", ", ".join(sorted(set(strict.papers) & set(loose.papers)))
                 or "none")
    body.add_row("non-overlapped only", ", ".join(only_loose) or "none")
    body.add_row("overlapped only", ", ".join(only_strict) or "none")
    body.add_row("precision", f"{strict.count} regions hold every query term")
    body.add_row("coverage", f"{loose.count} regions across"
                            f" {len(loose.papers)} papers")
    console.print(Panel(body, border_style="yellow", padding=(0, 1)))
    console.print()


def render_stats() -> None:
    stats = ir.corpus_stats()
    low, high = stats["length_min"], stats["length_max"]
    rows = [
        ("papers", f"{stats['papers']}"),
        ("vocabulary terms", f"{stats['vocabulary']:,}"),
        ("postings", f"{stats['postings']:,}"),
        ("stored positions", f"{stats['positions']:,}"),
        ("total terms", f"{stats['total_terms']:,}"),
        ("paper length", f"{low} to {high} terms "
                         f"(ratio {high / low:.2f}x, average {stats['avg_length']:.1f})"),
        ("idf range", f"{stats['idf_min']:.4f} to {stats['idf_max']:.4f}"),
        ("sections", f"{stats['sections']}"),
        ("paragraphs", f"{stats['paragraphs']}"),
        ("sentences", f"{stats['sentences']}"),
        ("judged queries", str(stats["judged_queries"])),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=ACCENT, no_wrap=True)
    table.add_column(style="bold white")
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
    table_idf = ir.get_engine().idf_table
    for term, df in stats["top_by_df"]:
        top.add_row(term, str(df), f"{table_idf.idf_for(term):.4f}",
                    bar(df / largest, 16))
    CONSOLE.print(top)
    CONSOLE.print(
        "  [grey58]A term in every paper gets an idf near zero and can no"
        " longer separate them.\n  That is the whole reason idf exists.[/]\n"
    )


def render_modes() -> None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("mode", style=ACCENT, no_wrap=True)
    table.add_column("what it does", style="white", overflow="fold")
    for mode, label in ir.MODE_LABELS.items():
        table.add_row(mode, label)
    CONSOLE.print(table)
    CONSOLE.print()


def render_paper(doc_id: str) -> None:
    """Every region of one paper, with the terms each one is about."""
    index = ir.get_index()
    paper = next((p for p in index.papers if p.doc_id == doc_id), None)
    if paper is None:
        CONSOLE.print(f"  [red]no such paper:[/] {doc_id}\n")
        return

    from collections import Counter

    console = CONSOLE
    console.print(f"  [bold white]{paper.doc_id}[/]  [bold white]{paper.title}[/]")
    console.print(f"  [grey58]{paper.year}  {SEP}  {paper.venue}  {SEP}  "
                  f"{paper.length} terms[/]\n")

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("kind", style=ACCENT, no_wrap=True)
    table.add_column("region", style="white", no_wrap=True)
    table.add_column("start", justify="right", style=DIM, no_wrap=True)
    table.add_column("end", justify="right", style=DIM, no_wrap=True)
    table.add_column("terms", justify="right", style=WARM, no_wrap=True)
    table.add_column("heaviest terms", style=DIM, overflow="fold")
    for kind in ("section", "paragraph", "sentence"):
        for region in sorted(index.regions_of_kind(paper.doc_id, kind),
                             key=lambda r: r.start):
            counts = Counter(t.term for t in paper.tokens
                             if region.contains(t.position))
            heaviest = ", ".join(f"{t} x{n}"
                                 for t, n in counts.most_common(6))
            table.add_row(kind, region.region_id, str(region.start),
                          str(region.end), str(region.length), heaviest)
    console.print(table)
    console.print()


def render_evaluation() -> None:
    scores = evaluate_all(ir.get_judgements(), ir.evaluation_models())
    from evaluate import SHORT_NAMES

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("model", style=ACCENT, no_wrap=True)
    table.add_column("P@1", justify="right", style=WARM, no_wrap=True)
    table.add_column("P@5", justify="right", style=WARM, no_wrap=True)
    table.add_column("R@10", justify="right", style=WARM, no_wrap=True)
    table.add_column("MRR", justify="right", style=WARM, no_wrap=True)
    table.add_column("MAP", justify="right", style=WARM, no_wrap=True)
    table.add_column("nohit", justify="right", style=DIM, no_wrap=True)
    for score in scores:
        table.add_row(SHORT_NAMES.get(score.model, score.model),
                      f"{score.mean('precision', 1):.3f}",
                      f"{score.mean('precision', 5):.3f}",
                      f"{score.mean('recall', 10):.3f}",
                      f"{score.mrr:.3f}", f"{score.map:.3f}",
                      str(len(score.queries_with_no_hit)))
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]12 hand-made queries over 20 papers. This is a demonstration,"
        "\n  not a benchmark - see evaluate.py for what these numbers do and"
        "\n  do not support.[/]\n"
    )


# ---------------------------------------------------------------------------
# machine readable output
# ---------------------------------------------------------------------------


def to_json(response: ir.SearchResponse) -> str:
    payload = {
        "query": response.query,
        "mode": response.mode,
        "terms": response.terms,
        "unknown": response.unknown,
        "papers": response.papers,
        "notes": response.notes,
        "parameters": {k: str(v) for k, v in response.parameters.items()},
        "error": response.error,
        "count": response.count,
        "hits": [
            {
                "rank": h.rank, "doc_id": h.doc_id, "title": h.title,
                "score": round(h.score, 6), "length": h.length,
                "matched": h.matched, "total_terms": h.total_terms,
                "field": h.field, "region_id": h.region_id,
                "snippet": h.snippet[:400], "note": h.note,
            }
            for h in response.hits
        ],
    }
    return json.dumps(payload, indent=2)


def to_text(response: ir.SearchResponse) -> str:
    lines = [f"query : {response.query}", f"mode  : {response.mode}",
             f"terms : {', '.join(response.terms)}"]
    if response.unknown:
        lines.append(f"unknown: {', '.join(response.unknown)}")
    if response.error:
        lines.append(f"error : {response.error}")
        return "\n".join(lines)
    if not response.hits:
        lines.append("no matches")
        return "\n".join(lines)
    lines.append(f"{'#':>3}  {'paper':<6}  {'score':>9}  "
                 f"{'region / field':<26}  matched  title")
    for hit in response.hits:
        detail = hit.region_id or hit.field or "-"
        lines.append(f"{hit.rank:>3}  {hit.doc_id:<6}  {hit.score:>9.4f}  "
                     f"{detail:<26}  {hit.matched:>3}/{hit.total_terms:<3}    "
                     f"{hit.title}")
    lines.append(f"\n{response.count} results")
    return "\n".join(lines)


def to_html(response: ir.SearchResponse) -> str:
    """A standalone HTML page.

    rich's export_html() already returns a complete document with DOCTYPE, head,
    style and body, so wrapping it again would produce two doctypes. Only the
    title is inserted.
    """
    buffer = io.StringIO()
    capture = Console(record=True, width=110, file=buffer)
    capture.print(renderable_for(response))
    document = capture.export_html(inline_styles=True)
    return document.replace("<head>", f"<head><title>"
                                     f"{html.escape(response.query)}</title>", 1)


def renderable_for(response: ir.SearchResponse):
    """The Rich renderable behind to_html, so both formats agree."""
    table = results_table(response, 12)
    panel = Table.grid(padding=(0, 2))
    panel.add_column(style=DIM)
    panel.add_column(style="white")
    panel.add_row("query", response.query)
    panel.add_row("mode", response.mode)
    panel.add_row("terms", ", ".join(response.terms))
    panel.add_row("results", str(response.count))
    if response.error:
        panel.add_row("error", response.error)
    return Panel(panel, title="week02 search", title_align="left",
                 border_style="grey35")


def _write_machine(text: str) -> None:
    """Write machine-readable output as UTF-8 bytes.

    print() would fail on this machine because stdout is cp1252 and rich's HTML
    export contains box-drawing characters that cp1252 cannot represent.
    """
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# interactive
# ---------------------------------------------------------------------------


HELP = [
    ("<query>", "search with the current mode"),
    ("bm25: / bim: / tfidf:", "force a ranking model"),
    ("bool: <query>", "force exact AND"),
    ("field: <query>", "force field-restricted search"),
    ("prox: <query>", "force a proximal node operator"),
    ("title:search", "field prefixes work directly, no prefix needed"),
    ("both: <query>", "overlapped and non-overlapped, side by side"),
    ("top N <query>", "limit the result list"),
    ("paper <p04>", "every region of one paper"),
    ("stats / modes / eval", "corpus summary / mode list / effectiveness"),
    ("help, quit", "this list / exit"),
]


def print_help() -> None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=ACCENT, overflow="fold")
    table.add_column(style="white", overflow="fold")
    for label, description in HELP:
        table.add_row(label, description)
    CONSOLE.print(Panel(table, title="commands", title_align="left",
                        border_style="grey35", padding=(0, 1)))


PREFIXES = {
    "bm25:": "bm25", "bim:": "bim", "tfidf:": "tfidf",
    "bool:": "boolean", "field:": "fields", "f:": "fields",
    "ovl:": "overlapped", "non-ovl:": "non-overlapped",
    "prox:": "proximal", "bm25f:": "bm25f",
}


def handle(line: str, default_mode: str = "bm25", top: int | None = None,
           hierarchy: str = "paragraph") -> bool:
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
    if lowered == "modes":
        render_modes()
        return True
    if lowered in {"eval", "evaluation"}:
        render_evaluation()
        return True
    if lowered.startswith("paper"):
        render_paper(line.split(":", 1)[1].strip() if ":" in line
                     else line.split(maxsplit=1)[-1].strip())
        return True

    mode = default_mode
    for prefix, forced in PREFIXES.items():
        if lowered.startswith(prefix):
            mode = forced
            line = line.split(":", 1)[1].strip()
            break

    if lowered.startswith("both:"):
        render_both(line.split(":", 1)[1].strip(), hierarchy, top)
        return True

    limit = top
    if lowered.startswith("top "):
        parts = line.split(maxsplit=2)
        if len(parts) < 3 or not parts[1].isdigit():
            CONSOLE.print("  [red]usage:[/] top N <query>\n")
            return True
        limit = int(parts[1])
        line = parts[2]

    if not line.strip():
        CONSOLE.print("  [red]empty query[/]\n")
        return True

    render_response(ir.search(line, mode=mode, top=limit, hierarchy=hierarchy))
    return True


def header() -> None:
    stats = ir.corpus_stats()
    CONSOLE.rule("[bold cyan]Week 02 - Structured Search Engine[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{stats['papers']} papers  {SEP}  "
        f"{stats['vocabulary']:,} terms  {SEP}  "
        f"{stats['postings']:,} postings  {SEP}  "
        f"{stats['sections']} sections  {SEP}  "
        f"{stats['paragraphs']} paragraphs  {SEP}  "
        f"{stats['sentences']} sentences[/]\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Week02 engine: probability ranking, fields, lists, nodes.")
    parser.add_argument("query", nargs="*", help="query to run")
    parser.add_argument("--stats", action="store_true",
                        help="print corpus statistics")
    parser.add_argument("--modes", action="store_true",
                        help="explain every retrieval mode")
    parser.add_argument("--eval", action="store_true",
                        help="print the effectiveness table")
    parser.add_argument("--paper", metavar="p04",
                        help="print every region of one paper")
    parser.add_argument("--top", type=int, default=None,
                        help="limit the number of results")
    parser.add_argument("--mode", choices=ir.MODES, default="bm25",
                        help="retrieval model to use")
    parser.add_argument("--hierarchy", default="paragraph",
                        choices=("section", "paragraph", "sentence"),
                        help="which region hierarchy the list models use")
    parser.add_argument("--text", action="store_true",
                        help="plain text output, for piping")
    parser.add_argument("--json", action="store_true",
                        help="JSON output, for another program")
    parser.add_argument("--html", action="store_true",
                        help="HTML output, for a web page")
    arguments = parser.parse_args(argv)

    if arguments.stats:
        header()
        render_stats()
        return
    if arguments.modes:
        render_modes()
        return
    if arguments.eval:
        render_evaluation()
        return
    if arguments.paper:
        render_paper(arguments.paper)
        return

    query = " ".join(arguments.query).strip()

    if arguments.json or arguments.html or arguments.text:
        if not query:
            print("error: those formats need a query", file=sys.stderr)
            print('example: python main.py --json "term weighting"',
                  file=sys.stderr)
            raise SystemExit(2)
        response = ir.search(query, mode=arguments.mode, top=arguments.top,
                             hierarchy=arguments.hierarchy)
        if arguments.json:
            _write_machine(to_json(response))
        elif arguments.html:
            _write_machine(to_html(response))
        else:
            _write_machine(to_text(response))
        return

    header()
    print_help()

    if query:
        handle(query, arguments.mode, arguments.top, arguments.hierarchy)
        return

    while True:
        try:
            line = CONSOLE.input("\n[bold cyan]search[/] > ")
        except (EOFError, KeyboardInterrupt):
            CONSOLE.print()
            return
        if not handle(line, arguments.mode, arguments.top, arguments.hierarchy):
            CONSOLE.print("\n[grey58]bye[/]\n")
            return


if __name__ == "__main__":
    main()