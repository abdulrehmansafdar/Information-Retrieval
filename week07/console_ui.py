"""The terminal interface. It calls semantic_search.py and prints what it returns.

    python main.py --console                                  type queries at a prompt
    python main.py --console --query "wheezing"               LSI, the top articles
    python main.py --console --query "wheezing" --model gvsm  the plain vector model
    python main.py --console --query "wheezing" --compare     both, side by side
    python main.py --console --query "wheezing" --steps m10   the worked cosine
    python main.py --console --slide                          the lecture's examples
    python main.py --console --matrix slide                   A (slide, space or corpus)
    python main.py --console --svd slide                      U, Sigma_k and V_k^T
    python main.py --console --concepts                       the hidden themes
    python main.py --console --eval                           GVSM against LSI, measured

No ranking or matrix logic lives here. Every number comes from the engine.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from clinical_docs import tokenize
from lsi_model import LsiModel, SvdFactors
from semantic_eval import evaluate
from semantic_search import DEFAULT_K, DEFAULT_TOP, MODELS, Settings, get_engine
from slide_examples import (FRUIT_QUERY, SPACE_DOCUMENTS, SPACE_K, SPACE_QUERY,
                            fruits_and_vegetables, slide_matrix, space_exploration,
                            space_matrix)
from term_matrix import WEIGHTINGS, TermDocumentMatrix

console = Console(highlight=False)
SEP = "[grey58]" + "-" * 72 + "[/]"
ROW_LIMIT = 12
COLUMN_LIMIT = 10


def show_response(response) -> bool:
    """Print one response. Returns False when the query could not be answered."""
    if not response.ok:
        console.print(f"[red]{response.message}[/]")
        return False
    s = response.settings
    console.print(f"[bold]query terms[/]  {', '.join(response.terms)}"
                  + (f"  [yellow](not in the collection: {', '.join(response.unknown)})[/]"
                     if response.unknown else ""))
    console.print(f"[grey58]model {s.model}  |  weighting {s.weighting}"
                  + (f"  |  k = {s.k}" if s.model == "lsi" else "") + "[/]")
    if not response.hits:
        console.print("[yellow]No article scores above 0.[/]")
        return True
    table = Table(title=f"top {len(response.hits)} of {response.count} articles scoring above 0",
                  title_style="bold", header_style="bold", border_style="grey58")
    table.add_column("#", justify="right", style="grey58")
    table.add_column("cosine", justify="right")
    table.add_column("id", style="grey58")
    table.add_column("article")
    table.add_column("shared words")
    for rank, hit in enumerate(response.hits, start=1):
        shared = ", ".join(hit.shared_terms) or "[cyan]none: matched on meaning[/]"
        table.add_row(str(rank), f"{hit.score:.3f}", hit.doc_id,
                      f"{hit.title}  [grey58]{hit.specialty}[/]", shared)
    console.print(table)
    return True


def show_compare(engine, query: str, settings: Settings) -> None:
    results = engine.compare(query, settings)
    table = Table(title=f"'{query}' under both models", title_style="bold",
                  header_style="bold", border_style="grey58")
    table.add_column("#", justify="right", style="grey58")
    table.add_column("GVSM")
    table.add_column(f"LSI (k = {settings.k})")
    gvsm, lsi = results["gvsm"], results["lsi"]
    if not gvsm.ok:
        console.print(f"[red]{gvsm.message}[/]")
        return

    def cell(hits, i: int) -> str:
        if i >= len(hits):
            return ""
        hit = hits[i]
        mark = "" if hit.shared_terms else "  [cyan]no shared word[/]"
        return f"{hit.score:.3f}  {hit.doc_id} {hit.title}{mark}"

    for i in range(max(len(gvsm.hits), len(lsi.hits))):
        table.add_row(str(i + 1), cell(gvsm.hits, i), cell(lsi.hits, i))
    console.print(table)
    console.print(f"Articles scoring above 0: GVSM {gvsm.count}, LSI {lsi.count}.")


def show_steps(engine, query: str, doc_id: str, settings: Settings) -> bool:
    if doc_id not in engine.by_id:
        console.print(f"[red]No article called {doc_id!r}.[/]")
        return False
    response = engine.search(query, settings)
    if not response.ok:
        console.print(f"[red]{response.message}[/]")
        return False
    steps = engine.steps(query, doc_id, settings)
    console.print(f"[bold]'{query}' against {doc_id} {engine.by_id[doc_id].title}[/]  "
                  f"[grey58]model {settings.model}, weighting {settings.weighting}[/]")
    for name, working in steps.lines():
        console.print(f"[bold]{name}[/]\n  {working}", markup=True)
    return True


def show_matrix_table(title: str, rows: list[str], columns: list[str], values) -> None:
    table = Table(title=title, title_style="bold", header_style="bold",
                  border_style="grey58")
    table.add_column("")
    for column in columns[:COLUMN_LIMIT]:
        table.add_column(column, justify="right")
    for label, row in list(zip(rows, values))[:ROW_LIMIT]:
        table.add_row(label, *(f"{v:.2f}" for v in list(row)[:COLUMN_LIMIT]))
    console.print(table)
    if len(rows) > ROW_LIMIT or len(columns) > COLUMN_LIMIT:
        console.print(f"[grey58]showing {min(len(rows), ROW_LIMIT)} of {len(rows)} rows and "
                      f"{min(len(columns), COLUMN_LIMIT)} of {len(columns)} columns[/]")


def matrix_source(engine, name: str, settings: Settings) -> tuple[TermDocumentMatrix, int]:
    """The matrix a --matrix or --svd flag asks for, and the k to cut it to."""
    if name == "slide":
        return slide_matrix(), min(settings.k, 2)
    if name == "space":
        return space_matrix(), SPACE_K
    return engine.matrix(settings.weighting), settings.k


def show_matrix(engine, name: str, settings: Settings) -> None:
    matrix, _ = matrix_source(engine, name, settings)
    m, n = matrix.shape
    show_matrix_table(f"A: {m} terms (rows) x {n} documents (columns), "
                      f"{matrix.weighting} weights",
                      list(matrix.terms), list(matrix.doc_ids), matrix.values.tolist())


def show_svd(engine, name: str, settings: Settings) -> None:
    matrix, k = matrix_source(engine, name, settings)
    model = LsiModel(SvdFactors(matrix), k)
    concepts = [f"c{c + 1}" for c in range(k)]
    console.print(f"[bold]A ~ U Sigma_k V_k^T[/] with k = {k}, keeping "
                  f"{model.energy_kept():.1%} of A's squared weight")
    show_matrix_table("U: term-concept matrix", list(matrix.terms), concepts, model.u_k.tolist())
    show_matrix_table("Sigma_k: singular values (concept strength)", concepts, concepts,
                      [[model.sigma_k[i] if i == j else 0.0 for j in range(k)]
                       for i in range(k)])
    show_matrix_table("V_k^T: document-concept matrix", concepts, list(matrix.doc_ids),
                      model.vt_k.tolist())


def show_slide_examples() -> None:
    console.print(f"[bold]Slides 6 to 8: '{FRUIT_QUERY}'[/]")
    model = fruits_and_vegetables()
    terms = tokenize(FRUIT_QUERY)
    console.print(f"Q = {model.query_vector(terms)}   D1 = {model.document_vector('D1')}")
    for name, working in model.steps(terms, "D1").lines():
        console.print(f"[bold]{name}[/]\n  {working}")
    console.print(SEP)
    console.print(f"[bold]Slides 4, 10, 13: '{SPACE_QUERY}'[/]  [grey58]LSI k = {SPACE_K}[/]")
    table = Table(header_style="bold", border_style="grey58")
    table.add_column("document")
    table.add_column("text")
    table.add_column("GVSM", justify="right")
    table.add_column("LSI", justify="right")
    for doc, scores in space_exploration().items():
        table.add_row(doc, SPACE_DOCUMENTS[doc], f"{scores['gvsm']:.2f}", f"{scores['lsi']:.2f}")
    console.print(table)


def show_concepts(engine, settings: Settings, how_many: int = 6) -> None:
    model = engine.lsi(settings.weighting, settings.k)
    table = Table(title=f"the first {how_many} of {settings.k} hidden concepts",
                  title_style="bold", header_style="bold", border_style="grey58")
    table.add_column("concept", justify="right")
    table.add_column("strength", justify="right")
    table.add_column("terms that load on it most")
    table.add_column("articles")
    for concept in model.concepts()[:how_many]:
        table.add_row(str(concept.number), f"{concept.strength:.2f}",
                      ", ".join(t for t, _ in concept.terms),
                      ", ".join(d for d, _ in concept.documents[:4]))
    console.print(table)


def show_eval(settings: Settings) -> None:
    table = Table(title="20 judged queries, relevance = same specialty", title_style="bold",
                  header_style="bold", border_style="grey58")
    table.add_column("model")
    table.add_column("mean average precision", justify="right")
    table.add_column("recall at R", justify="right")
    table.add_column("relevant, no shared word", justify="right")
    for s in (Settings("gvsm", settings.weighting),
              Settings("lsi", settings.weighting, settings.k)):
        result = evaluate(s)
        table.add_row(result.label, f"{result.mean_average_precision:.3f}",
                      f"{result.mean_recall_at_r:.3f}", str(result.no_word_hits))
    console.print(table)


def repl(engine, settings: Settings) -> None:
    console.print("[bold]Semantic article search[/]  [grey58]type a query, or 'quit'[/]")
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
    parser = argparse.ArgumentParser(description="Week 7 semantic ranking with GVSM and LSI")
    parser.add_argument("--query", help="run one query and exit")
    parser.add_argument("--model", choices=MODELS, default="lsi")
    parser.add_argument("--weighting", choices=WEIGHTINGS, default="tfidf")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="LSI concepts to keep")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="results to show")
    parser.add_argument("--compare", action="store_true", help="GVSM and LSI side by side")
    parser.add_argument("--steps", metavar="DOC", help="the worked cosine for one article")
    parser.add_argument("--slide", action="store_true", help="the lecture's worked examples")
    parser.add_argument("--matrix", choices=("slide", "space", "corpus"),
                        help="print a term-document matrix")
    parser.add_argument("--svd", choices=("slide", "space", "corpus"),
                        help="print U, Sigma_k and V_k^T")
    parser.add_argument("--concepts", action="store_true", help="the hidden themes")
    parser.add_argument("--eval", action="store_true", help="GVSM against LSI, measured")
    args = parser.parse_args(argv)

    # the working uses the root sign, which the old Windows console code page lacks
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    engine = get_engine()
    settings = Settings(args.model, args.weighting, args.k, args.top)
    try:
        if args.slide:
            show_slide_examples()
        elif args.matrix:
            show_matrix(engine, args.matrix, settings)
        elif args.svd:
            show_svd(engine, args.svd, settings)
        elif args.concepts:
            show_concepts(engine, settings)
        elif args.eval:
            show_eval(settings)
        elif args.query and args.steps:
            if not show_steps(engine, args.query, args.steps, settings):
                return 1
        elif args.query and args.compare:
            show_compare(engine, args.query, settings)
        elif args.query:
            if not show_response(engine.search(args.query, settings)):
                return 1
        else:
            repl(engine, settings)
    except ValueError as error:
        console.print(f"[red]{error}[/]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
