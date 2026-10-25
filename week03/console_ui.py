"""Console front end for the week03 browsing engine.

    python main.py --console                     interactive browser
    python main.py --console --home              the home page
    python main.py --console --tree              the category hierarchy
    python main.py --console --find "proximity"  one search
    python main.py --console --doc p04           one document preview
    python main.py --console --graph             H = (N, L) statistics
    python main.py --console --models            the three browsing models
    python main.py --console --stats             corpus summary

Every decision is made by browsing.py, which app.py also uses, so the terminal
and the browser can never disagree about an answer. This file only handles input
and output.

Commands at the prompt
----------------------
    home                    the home page: domains, counts, recently viewed
    tree [node]             the hierarchy, or one node's children
    cd <node_id>            move to a category (e.g. cd root/ir)
    up                      move to the parent category
    ls                      the documents under the current category
    open <p04>              open a document (alias for preview)
    preview <p04>           the full document page: sections and links
    find <query>            search the collection
    filter year=2021        set facets: year=, venue=, domain=, subtopic=
    clearfilters            drop every facet
    related <p04>           the related documents of one document
    links <p04>             every link leaving one document
    path <p04> <p18>        shortest navigation path between two documents
    graph                   |N|, |L| and the link breakdown
    history / recent        the back/forward stack / recently viewed
    back / forward          walk the navigation history
    models                  the three browsing models
    stats                   corpus summary
    help, quit              this list / exit
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

import browsing as browse
from browsing import BrowseEngine, Filters, NavigationState
from hypertext import document_node
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CONSOLE = Console()

ACCENT = "bold cyan"
WARM = "bold yellow"
DIM = "grey58"

# The Windows console here is cp437 and stdout is cp1252, so non-ASCII glyphs
# must be picked from the encoding or they print as '?'.
_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
SEP = "·" if _UNICODE else "|"
ARROW = "→" if _UNICODE else "->"


@dataclass
class Session:
    """Where the browser currently stands."""

    node: str = "root"
    filters: Filters = field(default_factory=Filters)
    nav: NavigationState = field(default_factory=NavigationState)


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def engine() -> BrowseEngine:
    return browse.get_engine()


def crumb(node_id: str) -> Text:
    """The breadcrumb, e.g. Repository / Information Retrieval / Ranking."""
    text = Text()
    for position, node in enumerate(engine().breadcrumb(node_id)):
        if position:
            text.append(f"  {SEP}  ", style="grey35")
        text.append(node.name, style=ACCENT if position else "bold white")
    return text


def documents_table(summaries) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("year", justify="right", style=DIM, no_wrap=True)
    table.add_column("title", style="white", overflow="fold", ratio=2)
    table.add_column("category", style=WARM, overflow="fold", ratio=1)
    for summary in summaries:
        table.add_row(summary.doc_id, str(summary.year), summary.title,
                      summary.category)
    return table


def show_home(session: Session) -> None:
    home = engine().home(session.nav)
    stats = home["stats"]
    CONSOLE.rule("[bold cyan]Week 03 - FYP Browsing Architecture[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{stats['documents']} documents  {SEP}  "
        f"{stats['sections']} sections  {SEP}  "
        f"{stats['words']:,} words  {SEP}  "
        f"{stats['vocabulary']:,} distinct terms  {SEP}  "
        f"{len(stats['years'])} years[/]\n")

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("domain", style=ACCENT)
    table.add_column("docs", justify="right", style=WARM, no_wrap=True)
    table.add_column("node id", style=DIM, no_wrap=True)
    for domain in home["domains"]:
        table.add_row(domain["name"], str(domain["count"]), domain["node_id"])
    CONSOLE.print(table)

    if home["recent"]:
        CONSOLE.print("  [grey58]recently viewed:[/]  " + "  |  ".join(
            f"{r['label']}" for r in home["recent"]))
    CONSOLE.print("\n  [grey58]type `tree` to walk the hierarchy, `find "
                  "<query>` to search, `help` for everything.[/]\n")


def show_tree(node_id: str | None = None, session: Session | None = None) -> None:
    node_id = node_id or (session.node if session else "root")
    taxonomy = engine().taxonomy
    if node_id not in taxonomy:
        CONSOLE.print(f"  [red]no such category:[/] {node_id}\n")
        return
    root = taxonomy.find(node_id)
    CONSOLE.print(crumb(node_id))
    CONSOLE.print()

    # A compact ASCII tree, indented by depth below the chosen node.
    def render(category, indent: int) -> None:
        marker = "" if category.kind == "root" else f"[{WARM}]{category.count:>2}[/]  "
        style = ACCENT if category.kind != "subtopic" else "white"
        CONSOLE.print("  " + "   " * indent + marker +
                      f"[{style}]{category.name}[/]  [grey35]{category.node_id}[/]")
        for child in taxonomy.find(category.node_id).children:
            render(taxonomy.find(child), indent + 1)

    render(root, 0)
    CONSOLE.print()


def show_listing(session: Session) -> None:
    node = engine().taxonomy.find(session.node)
    summaries = engine().documents_in(session.node, session.filters)
    CONSOLE.print(crumb(session.node))
    if session.filters.active:
        CONSOLE.print(f"  [yellow]filters:[/] {session.filters.describe()}")
    if not summaries:
        CONSOLE.print("  [grey58]no documents match here[/]\n")
        return
    CONSOLE.print(documents_table(summaries))
    CONSOLE.print(f"  [grey58]{len(summaries)} documents in "
                  f"'{node.name}'[/]\n")


def show_search(session: Session, query: str) -> None:
    response = engine().search(query, session.filters)
    CONSOLE.print(f"  [bold white]{query!r}[/]"
                  + (f"  [yellow]filters: {response.filters.describe()}[/]"
                     if response.filters.active else ""))
    if response.unknown:
        CONSOLE.print(f"  [yellow]not in the collection, ignored:"
                      f" {', '.join(response.unknown)}[/]")
    if response.error:
        CONSOLE.print(f"  [red]{response.error}[/]\n")
        return
    if not response.hits:
        CONSOLE.print("  [grey58]nothing matched[/]\n")
        return
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style=DIM, no_wrap=True)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("score", justify="right", style=WARM, no_wrap=True)
    table.add_column("title", style="white", overflow="fold", ratio=2)
    table.add_column("category", style=DIM, overflow="fold", ratio=1)
    for hit in response.hits:
        table.add_row(str(hit.rank), hit.doc_id, f"{hit.score:.3f}",
                      hit.title, hit.category)
    CONSOLE.print(table)
    CONSOLE.print(f"  [grey58]{response.count} results in "
                  f"{response.considered} candidate documents[/]\n")


def show_preview(doc_id: str, session: Session | None = None) -> None:
    doc_id = doc_id.strip()
    if doc_id not in engine().by_id:
        CONSOLE.print(f"  [red]no such document:[/] {doc_id}\n")
        return
    preview = engine().preview(doc_id)
    if session is not None:
        session.node = document_node(doc_id)
        session.nav.visit(session.node)

    CONSOLE.print(crumb(engine().taxonomy.leaf_of(doc_id)))
    CONSOLE.print()
    CONSOLE.print(f"  [bold white]{preview.doc_id}[/]  "
                  f"[bold white]{preview.title}[/]")
    CONSOLE.print(f"  [grey58]{preview.year}  {SEP}  {preview.venue}  {SEP}  "
                  f"{preview.category}[/]\n")

    for name, text in preview.sections:
        snippet = " ".join(text.split())
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        CONSOLE.print(f"  [bold cyan]{name}[/]")
        CONSOLE.print(f"  [white]{snippet}[/]\n")

    links = Table.grid(padding=(0, 2))
    links.add_column(style=DIM, no_wrap=True)
    links.add_column(style="white", overflow="fold")
    links.add_row("related", ", ".join(
        f"{d} ({s:.2f})" for d, _, s in preview.related) or "none")
    links.add_row("same venue", ", ".join(
        f"{d}" for d, _ in preview.peers_venue) or "none")
    links.add_row("same year", ", ".join(
        f"{d}" for d, _ in preview.peers_year) or "none")
    links.add_row("internal", ", ".join(preview.internal) or "none")
    CONSOLE.print(Panel(links, title="links", title_align="left",
                        border_style="grey35", padding=(0, 1)))
    CONSOLE.print()


def show_related(doc_id: str) -> None:
    if doc_id not in engine().by_id:
        CONSOLE.print(f"  [red]no such document:[/] {doc_id}\n")
        return
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("score", justify="right", style=WARM, no_wrap=True)
    table.add_column("title", style="white", overflow="fold")
    for other, title, score in engine().related(doc_id, 5):
        table.add_row(other, f"{score:.3f}", title)
    CONSOLE.print(f"  [bold white]related to {doc_id}[/] "
                  f"[grey58](cosine over title + abstract)[/]")
    CONSOLE.print(table)
    CONSOLE.print()


def show_links(doc_id: str) -> None:
    if doc_id not in engine().by_id:
        CONSOLE.print(f"  [red]no such document:[/] {doc_id}\n")
        return
    node = document_node(doc_id)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("kind", style=ACCENT, no_wrap=True)
    table.add_column("target", style="white", overflow="fold")
    table.add_column("weight", justify="right", style=DIM, no_wrap=True)
    for link in engine().graph.out_links(node):
        table.add_row(link.kind, link.target,
                      f"{link.weight:.3f}" if link.weight else "")
    CONSOLE.print(f"  [bold white]{doc_id}[/]  "
                  f"[grey58]out-links (degree {engine().graph.degree(node)})[/]")
    CONSOLE.print(table)
    CONSOLE.print()


def show_path(start: str, goal: str) -> None:
    try:
        path = engine().path_between(start, goal)
    except KeyError as error:
        CONSOLE.print(f"  [red]{error}[/]\n")
        return
    if not path:
        CONSOLE.print(f"  [grey58]no path from {start} to {goal}[/]\n")
        return
    CONSOLE.print(f"  [bold white]{start}[/] {ARROW} [bold white]{goal}[/]  "
                  f"[grey58]{len(path) - 1} move(s)[/]")
    for label, kind, target in engine().describe_path(path):
        CONSOLE.print(f"    {label}  [yellow]-[{kind}]-[/]  {target}")
    CONSOLE.print()


def show_graph() -> None:
    stats = engine().graph.graph_stats()
    table = Table.grid(padding=(0, 2))
    table.add_column(style=DIM, no_wrap=True)
    table.add_column(style="bold white")
    table.add_row("nodes  |N|", str(stats["nodes"]))
    table.add_row("links  |L|", str(stats["links"]))
    table.add_row("repository", str(stats["repository"]))
    table.add_row("categories", str(stats["categories"]))
    table.add_row("documents", str(stats["documents"]))
    table.add_row("sections", str(stats["sections"]))
    CONSOLE.print(Panel(table, title="H = (N, L)", title_align="left",
                        border_style="grey35", padding=(0, 1)))

    kinds = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    kinds.add_column("link kind", style=ACCENT, no_wrap=True)
    kinds.add_column("count", justify="right", style=WARM, no_wrap=True)
    kinds.add_column("meaning", style="white", overflow="fold")
    meanings = {
        "hierarchy": "repository -> domain -> subtopic -> document",
        "internal": "document -> its sections, sections in reading order",
        "related": "document -> document, by cosine over title + abstract",
        "shared-venue": "document -> document, same publication venue",
        "shared-year": "document -> document, same year",
    }
    for kind, count in stats["by_kind"].items():
        kinds.add_row(kind, str(count), meanings.get(kind, ""))
    CONSOLE.print(kinds)
    CONSOLE.print()


def show_models() -> None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("model", style=ACCENT, no_wrap=True)
    table.add_column("idea", style="white", overflow="fold")
    table.add_column("best for", style=DIM, overflow="fold")
    table.add_column("here", style=WARM, overflow="fold")
    for model in browse.BROWSING_MODELS:
        table.add_row(model["name"], model["idea"], model["best_for"],
                      model["here"])
    CONSOLE.print(table)
    CONSOLE.print()


def show_stats() -> None:
    stats = engine().home()["stats"]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=ACCENT, no_wrap=True)
    table.add_column(style="bold white")
    table.add_row("documents", str(stats["documents"]))
    table.add_row("sections", str(stats["sections"]))
    table.add_row("words", f"{stats['words']:,}")
    table.add_row("distinct terms", f"{stats['vocabulary']:,}")
    table.add_row("years", f"{stats['years'][0]}-{stats['years'][-1]}")
    table.add_row("venues", str(len(stats["venues"])))
    CONSOLE.print(Panel(table, title="collection", title_align="left",
                        border_style="grey35", padding=(0, 1)))
    CONSOLE.print()


def show_history(session: Session) -> None:
    if not session.nav.history:
        CONSOLE.print("  [grey58]nothing visited yet[/]\n")
        return
    lines = Table.grid(padding=(0, 2))
    lines.add_column(style=DIM, no_wrap=True)
    lines.add_column(style="white")
    for position, node in enumerate(session.nav.history):
        marker = ARROW if position == session.nav.position else " "
        lines.add_row(f"{marker} {position}", engine().label(node))
    CONSOLE.print(Panel(lines, title="navigation history", title_align="left",
                        border_style="grey35", padding=(0, 1)))
    if session.nav.recent():
        CONSOLE.print("  [grey58]recent:[/]  " + "  |  ".join(
            engine().label(n) for n in session.nav.recent()))
    CONSOLE.print()


# ---------------------------------------------------------------------------
# machine-readable output
# ---------------------------------------------------------------------------


def search_json(response: browse.SearchResponse) -> str:
    return json.dumps({
        "query": response.query,
        "filters": response.filters.describe(),
        "count": response.count,
        "considered": response.considered,
        "terms": response.terms,
        "unknown": response.unknown,
        "error": response.error,
        "hits": [{"rank": h.rank, "doc_id": h.doc_id, "title": h.title,
                  "score": round(h.score, 6), "category": h.category}
                 for h in response.hits],
    }, indent=2)


def search_text(response: browse.SearchResponse) -> str:
    if response.error:
        return f"error: {response.error}"
    lines = [f"query : {response.query}", f"filters: {response.filters.describe()}",
             f"{response.count} results of {response.considered} candidates"]
    for hit in response.hits:
        lines.append(f"{hit.rank:>3}  {hit.doc_id:<6}  {hit.score:>8.3f}  "
                     f"{hit.title}")
    return "\n".join(lines)


def _write_machine(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------


HELP = [
    ("home", "the home page: domains, counts, recently viewed"),
    ("tree [node]", "the hierarchy, or one node's children"),
    ("cd <node_id>", "move to a category (e.g. cd root/ir)"),
    ("up", "move to the parent category"),
    ("ls", "the documents under the current category"),
    ("open / preview <p04>", "the full document page: sections and links"),
    ("find <query>", "search the collection"),
    ("filter year=2021", "set facets: year=, venue=, domain=, subtopic="),
    ("clearfilters", "drop every facet"),
    ("related <p04>", "the related documents of one document"),
    ("links <p04>", "every link leaving one document"),
    ("path <p04> <p18>", "shortest navigation path between two documents"),
    ("graph", "|N|, |L| and the link breakdown"),
    ("history / recent", "the back/forward stack / recently viewed"),
    ("back / forward", "walk the navigation history"),
    ("models / stats", "the three browsing models / corpus summary"),
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


def parse_filters(text: str, current: Filters) -> tuple[Filters, str]:
    """Read `year=2021 domain=root/ir` into a Filters. Returns (filters, error)."""
    year = current.year
    venue, domain, subtopic = current.venue, current.domain, current.subtopic
    for token in text.split():
        key, _, value = token.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if key == "year":
            if not value.isdigit():
                return current, f"year must be a number, not {value!r}"
            year = int(value)
        elif key == "venue":
            venue = value
        elif key == "domain":
            domain = value if value.startswith("root/") else f"root/{value}"
            if domain not in engine().taxonomy:
                return current, f"no such domain: {domain}"
        elif key == "subtopic":
            subtopic = value
            if subtopic and subtopic not in engine().taxonomy:
                return current, f"no such category: {subtopic}"
        else:
            return current, f"unknown facet {key!r} - use year, venue, domain, subtopic"
    return Filters(domain, subtopic, year, venue), ""


def handle(line: str, session: Session) -> bool:
    """Returns False when the user asked to quit."""
    line = line.strip()
    if not line:
        return True
    parts = line.split(maxsplit=1)
    command, rest = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")

    if command in {"quit", "exit", "q"}:
        return False
    if command == "help":
        print_help()
    elif command == "home":
        show_home(session)
    elif command == "tree":
        show_tree(rest or session.node, session)
    elif command == "cd":
        if not rest:
            CONSOLE.print("  [red]usage:[/] cd <node_id>\n")
        elif rest not in engine().taxonomy:
            CONSOLE.print(f"  [red]no such category:[/] {rest}\n")
        else:
            session.node = rest
            session.nav.visit(rest)
            show_listing(session)
    elif command == "up":
        parent = engine().taxonomy.find(session.node).parent_id
        if not parent:
            CONSOLE.print("  [grey58]already at the repository root[/]\n")
        else:
            session.node = parent
            session.nav.visit(parent)
            show_listing(session)
    elif command == "ls":
        show_listing(session)
    elif command in {"open", "preview"}:
        if not rest:
            CONSOLE.print("  [red]usage:[/] open <p04>\n")
        else:
            show_preview(rest, session)
    elif command in {"find", "search"}:
        if not rest:
            CONSOLE.print("  [red]usage:[/] find <query>\n")
        else:
            show_search(session, rest)
    elif command == "filter":
        filters, error = parse_filters(rest, session.filters)
        if error:
            CONSOLE.print(f"  [red]{error}[/]\n")
        else:
            session.filters = filters
            show_listing(session)
    elif command in {"clearfilters", "nofilter"}:
        session.filters = Filters()
        show_listing(session)
    elif command == "related":
        show_related(rest)
    elif command == "links":
        show_links(rest)
    elif command == "path":
        bits = rest.split()
        if len(bits) != 2:
            CONSOLE.print("  [red]usage:[/] path <p04> <p18>\n")
        else:
            show_path(bits[0], bits[1])
    elif command == "graph":
        show_graph()
    elif command == "models":
        show_models()
    elif command == "stats":
        show_stats()
    elif command == "history":
        show_history(session)
    elif command == "recent":
        if session.nav.recent():
            CONSOLE.print("  [grey58]recent:[/]  " + "  |  ".join(
                engine().label(n) for n in session.nav.recent(6)) + "\n")
        else:
            CONSOLE.print("  [grey58]nothing visited yet[/]\n")
    elif command == "back":
        node = session.nav.back()
        session.node = node
        CONSOLE.print(f"  [grey58]back to[/] {engine().label(node)}\n")
    elif command == "forward":
        node = session.nav.forward()
        session.node = node
        CONSOLE.print(f"  [grey58]forward to[/] {engine().label(node)}\n")
    else:
        CONSOLE.print(f"  [red]unknown command:[/] {command}.  "
                      f"Type [bold]help[/] for the list.\n")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Week03 browsing engine: categories, search, links, paths.")
    parser.add_argument("--home", action="store_true", help="the home page")
    parser.add_argument("--tree", action="store_true", help="the hierarchy")
    parser.add_argument("--find", metavar="QUERY", help="one search")
    parser.add_argument("--filter", default="", metavar="FACETS",
                        help="facets for --find/--doc, e.g. 'year=2021'")
    parser.add_argument("--doc", metavar="p04", help="one document preview")
    parser.add_argument("--graph", action="store_true", help="graph statistics")
    parser.add_argument("--models", action="store_true",
                        help="the three browsing models")
    parser.add_argument("--stats", action="store_true", help="corpus summary")
    parser.add_argument("--text", action="store_true", help="plain text, for piping")
    parser.add_argument("--json", action="store_true", help="JSON, for another program")
    arguments = parser.parse_args(argv)

    session = Session()
    if arguments.filter:
        session.filters, error = parse_filters(arguments.filter, session.filters)
        if error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(2)

    if arguments.home:
        show_home(session)
        return
    if arguments.tree:
        show_tree(None, session)
        return
    if arguments.graph:
        show_graph()
        return
    if arguments.models:
        show_models()
        return
    if arguments.stats:
        show_stats()
        return
    if arguments.doc:
        show_preview(arguments.doc, session)
        return
    if arguments.find:
        response = engine().search(arguments.find, session.filters)
        if arguments.json:
            _write_machine(search_json(response))
        elif arguments.text:
            _write_machine(search_text(response))
        else:
            show_search(session, arguments.find)
        return

    show_home(session)
    print_help()
    while True:
        try:
            line = CONSOLE.input("\n[bold cyan]browse[/] > ")
        except (EOFError, KeyboardInterrupt):
            CONSOLE.print()
            return
        if not handle(line, session):
            CONSOLE.print("\n[grey58]bye[/]\n")
            return


if __name__ == "__main__":
    main()
