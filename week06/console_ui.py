"""The terminal interface. It calls fuzzy_search.py and prints what it returns.

    python main.py --console                                  type queries at a prompt
    python main.py --console --query "hypertension AND NOT diabetes" --mode extended
    python main.py --console --query "hypertenion" --explain  show the working
    python main.py --console --query "asthma OR COPD" --compare
    python main.py --console --membership hypertension        mu under all 3 functions
    python main.py --console --sets diabetes hypertension kidney

No ranking, matching or set logic lives here. Every number comes from the engine.
"""

from __future__ import annotations

import argparse
import math
import sys

from rich.console import Console
from rich.table import Table

from fuzzy_search import MODES, Settings, get_engine
from fuzzy_logic import plain_english
from membership import FUNCTIONS
from query_parser import QueryError

console = Console(highlight=False)
SEP = "[grey58]" + "-" * 72 + "[/]"


def settings_from(args: argparse.Namespace) -> Settings:
    p = math.inf if args.p.lower() in ("inf", "infinity") else float(args.p)
    return Settings(mode=args.mode, implicit=args.implicit, function=args.function,
                    partial=not args.no_partial, tau=args.tau, p=p, alpha=args.alpha)


def show_response(response, explain_working: bool = False) -> bool:
    """Print one response. Returns False when the query could not be read."""
    if not response.ok:
        console.print(f"[red]{response.message}[/]")
        return False
    s = response.settings
    console.print(f"[bold]parsed as[/]  {response.parsed}")
    console.print(f"[grey58]mode {s.mode}  |  membership {s.function}  |  "
                  f"partial {'on' if s.partial else 'off'}  |  "
                  f"implicit {s.implicit}"
                  + (f"  |  p = {s.p:g}" if s.mode == "extended" else "")
                  + (f"  |  alpha {s.alpha:g}" if s.mode == "extended" else "")
                  + "[/]")
    if not response.hits:
        console.print("[yellow]No article matches. Try a shorter query, or switch on "
                      "partial matching.[/]")
        return True

    table = Table(title=f"{response.count} of {response.considered} articles",
                  title_style="bold", header_style="bold", border_style="grey58")
    table.add_column("#", justify="right", style="grey58")
    table.add_column("score", justify="right")
    table.add_column("id", style="grey58")
    table.add_column("article")
    table.add_column("matched as", style="grey58")
    for rank, hit in enumerate(response.hits[:10], start=1):
        matched = [m.matched_as for m in hit.matches
                   if m.matched_as and m.matched_as != m.stem]
        table.add_row(str(rank), f"{hit.score:.2f}", hit.doc_id,
                      f"{hit.title}  [grey58]{hit.code}[/]",
                      ", ".join(matched) or "exact")
    console.print(table)
    if response.count > 10:
        console.print(f"[grey58]showing the top 10 of {response.count}[/]")

    if explain_working:
        for hit in response.hits[:3]:
            console.print(f"\n[bold]{hit.doc_id}  {hit.title}[/]")
            for line in plain_english(hit.working):
                console.print(line, markup=False)
    return True


def show_compare(engine, query: str, args: argparse.Namespace) -> None:
    results = engine.compare(query, settings_from(args))
    table = Table(title=f"'{query}' in each mode", title_style="bold",
                  header_style="bold", border_style="grey58")
    table.add_column("mode")
    table.add_column("results", justify="right")
    table.add_column("top three")
    for mode in MODES:
        response = results[mode]
        if not response.ok:
            table.add_row(mode, "-", f"[red]{response.message}[/]")
            continue
        top = ", ".join(f"{h.doc_id} {h.score:.2f}" for h in response.hits[:3])
        table.add_row(mode, str(response.count), top or "none")
    console.print(table)


def show_membership(engine, word: str) -> None:
    rows = {fn: engine.membership_table(word, fn) for fn in FUNCTIONS}
    table = Table(title=f"mu for '{word}' (top 8 articles)", title_style="bold",
                  header_style="bold", border_style="grey58")
    table.add_column("article")
    for fn in FUNCTIONS:
        table.add_column(fn, justify="right")
    for doc_id, title, _ in rows["log"][:8]:
        values = {fn: next(mu for d, _, mu in rows[fn] if d == doc_id) for fn in FUNCTIONS}
        table.add_row(f"{doc_id}  {title}", *(f"{values[fn]:.2f}" for fn in FUNCTIONS))
    console.print(table)
    matches = engine.partial_matches(word)
    if matches:
        console.print("partial matches: " + ", ".join(f"{v} {s:.2f}" for v, s in matches))


def show_sets(engine, a: str, b: str, c: str) -> None:
    view = engine.set_view(a, b, c)
    console.print(f"A = {a}: {sorted(view['A'])}")
    console.print(f"B = {b}: {sorted(view['B'])}")
    console.print(f"C = {c}: {sorted(view['C'])}")
    console.print(f"A u B: {sorted(view['union'])}")
    console.print(f"(A u B) - C: {sorted(view['result'])}")
    verdict = "[green]equal[/]" if view["equal"] else "[red]different[/]"
    console.print(f"query '(({a}) OR ({b})) AND NOT ({c})' gives the same set: {verdict}")


def repl(engine, settings: Settings) -> None:
    console.print("[bold]Medical article search[/]  "
                  "[grey58]type a query, or 'quit'[/]")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if query.lower() in ("quit", "exit", ""):
            return
        show_response(engine.search(query, settings))
        console.print(SEP)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 6 fuzzy and extended Boolean search")
    parser.add_argument("--query", help="run one query and exit")
    parser.add_argument("--mode", choices=MODES, default="fuzzy")
    parser.add_argument("--implicit", choices=("AND", "OR"), default="AND")
    parser.add_argument("--function", choices=FUNCTIONS, default="log")
    parser.add_argument("--no-partial", action="store_true",
                        help="exact words only, no partial matching")
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--p", default="2", help="p-norm exponent, or 'inf'")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--explain", action="store_true", help="show the working")
    parser.add_argument("--compare", action="store_true",
                        help="run the query in all three modes")
    parser.add_argument("--membership", metavar="WORD",
                        help="show a word's membership under all three functions")
    parser.add_argument("--sets", nargs=3, metavar=("A", "B", "C"),
                        help="check (A OR B) AND NOT C against (A u B) - C")
    args = parser.parse_args(argv)

    engine = get_engine()
    settings = settings_from(args)
    try:
        if args.membership:
            show_membership(engine, args.membership)
        elif args.sets:
            show_sets(engine, *args.sets)
        elif args.query and args.compare:
            show_compare(engine, args.query, args)
        elif args.query:
            if not show_response(engine.search(args.query, settings), args.explain):
                return 1
        else:
            repl(engine, settings)
    except (QueryError, ValueError) as error:
        console.print(f"[red]{error}[/]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
