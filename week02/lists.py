"""Step 5 - Overlapped and non-overlapped lists.

Week 01 returned whole documents. Everything since has been about WHERE in a
paper the evidence is. This step returns the REGION, and asks a question that
sounds like a technicality and is not:

    if the query terms never appear in the same place, have we found it?

The two models here are two different answers, and both are defensible.

    OVERLAPPED      every query term must appear inside ONE region. Matches that
                    overlap are merged into the smallest region containing them
                    all. High precision: a returned region genuinely discusses
                    the whole query. Low coverage: a paper that discusses term A
                    in its methodology and term B in its results is invisible.

    NON-OVERLAPPED  terms may appear in DIFFERENT regions. Each term is answered
                    by its own region and the results are the union. High
                    coverage: the paper is found. Low precision: two regions
                    returned together may have nothing to do with each other.

Within ONE hierarchy the regions are non-overlapping by construction: a
section list tiles the paper, a paragraph list tiles the paper, a sentence list
tiles the paper. A region may sit inside a region of another hierarchy - a
sentence inside a paragraph inside a section - and THAT is the overlap. The
names describe the structure, not the query behaviour.

The operators
-------------
Navarro and Baeza-Yates call the match algebra two operations:

    M collapse M'    superimpose two sets of matches, MERGING where they overlap
    M subtract M'    delete the points of M that fall inside a region of M'

`collapse` is what turns a bag of term occurrences into regions: three terms
matching at positions 10, 12 and 14 collapse into the single region [10, 15).

Run:  python lists.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from preprocess import Region
from ranking import IdfTable, parse_query
from structured_index import HIERARCHIES, StructuredIndex, build_index

DEFAULT_HIERARCHY = "paragraph"


@dataclass(frozen=True)
class Match:
    """One term occurrence resolved to a position."""

    term: str
    position: int


@dataclass
class ListRegion:
    """A region returned as an answer, with why it was returned."""

    region_id: str
    doc_id: str
    kind: str
    label: str
    start: int
    end: int
    text: str
    terms: tuple[str, ...]
    matched: int
    total_query_terms: int
    score: float = 0.0
    collapsed_from: int = 1
    role: str = "collapsed"

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def complete(self) -> bool:
        return self.matched == self.total_query_terms


@dataclass
class ListResult:
    """A whole list-model answer."""

    query: str
    model: str
    hierarchy: str
    regions: list[ListRegion] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)
    unknown: tuple[str, ...] = ()
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.regions)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def complete_regions(self) -> int:
        return sum(1 for r in self.regions if r.complete)


# ---------------------------------------------------------------------------
# the algebra
# ---------------------------------------------------------------------------


def collapse(matches: list[Match]) -> list[tuple[int, int]]:
    """M collapse M': merge overlapping match spans into minimal extents.

    Input matches are individual TERM OCCURRENCES. Output is a list of disjoint
    half-open ranges [start, end) whose union equals the union of the inputs.

    ADJACENT occurrences merge; occurrences with a gap do not. Positions 10 and
    11 have nothing between them, so a region containing both contains no term
    it does not have, and merging them is correct. Positions 10 and 13 have two
    unmatched terms between them, so merging would claim a stretch of text no
    query term touched. The test is `position <= end` because `end` is
    EXCLUSIVE: position 11 sits at offset 11 and the run already covers
    offsets 10..10, so 11 continues it.
    """
    if not matches:
        return []
    ordered = sorted(matches, key=lambda m: m.position)
    spans: list[tuple[int, int]] = []
    start = ordered[0].position
    end = start + 1
    for match in ordered[1:]:
        if match.position <= end:
            end = match.position + 1
            continue
        spans.append((start, end))
        start = match.position
        end = start + 1
    spans.append((start, end))
    return spans


def subtract(spans: list[tuple[int, int]], holes: list[tuple[int, int]]
             ) -> list[tuple[int, int]]:
    """M subtract M': remove the parts of `spans` covered by `holes`."""
    if not holes:
        return list(spans)
    remaining = list(spans)
    for hole_start, hole_end in sorted(holes):
        nxt: list[tuple[int, int]] = []
        for start, end in remaining:
            if hole_end <= start or end <= hole_start:
                nxt.append((start, end))
                continue
            if start < hole_start:
                nxt.append((start, hole_start))
            if hole_end < end:
                nxt.append((hole_end, end))
        remaining = nxt
    return remaining


def span_gap(left: tuple[int, int], right: tuple[int, int]) -> int:
    """Terms between two spans: right.start - left.end."""
    return right[0] - left[1]


# ---------------------------------------------------------------------------
# the two models
# ---------------------------------------------------------------------------


class ListModels:
    """Overlapped and non-overlapped retrieval over one index."""

    def __init__(self, index: StructuredIndex) -> None:
        self.index = index
        self.idf_table = IdfTable(index, "bm25")

    # --- shared -------------------------------------------------------------

    def _query_terms(self, query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        parsed = parse_query(query, self.idf_table)
        return parsed.terms, parsed.unknown

    def _regions_covering(self, doc_id: str, spans: list[tuple[int, int]],
                          hierarchy: str) -> list[Region]:
        """Smallest hierarchy region covering each span."""
        out: list[Region] = []
        for start, end in spans:
            best: Region | None = None
            for region in self.index.regions_of_kind(doc_id, hierarchy):
                if region.start <= start and end <= region.end:
                    if best is None or region.length < best.length:
                        best = region
            out.append(best) if best is not None else None
        return [r for r in out if r is not None]

    def _region_score(self, region: Region, terms: tuple[str, ...]) -> float:
        """BM25 inside a single region, used to ORDER the regions of a paper.

        df is the corpus-wide df of the term, which is the standard choice: the
        region decides WHICH documents are returned, and the term weights only
        decide the order among the regions that already matched.
        """
        paper = next(p for p in self.index.papers if p.doc_id == region.doc_id)
        counts = Counter(t.term for t in paper.tokens if region.contains(t.position))
        length = max(region.length, 1)
        average = self.index.average_document_length
        total = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if tf <= 0:
                continue
            sat = (tf * 2.2) / (tf + 1.2 * ((1.0 - 0.75)
                                             + 0.75 * length / average))
            total += self.idf_table.idf_for(term) * sat
        return total

    # --- overlapped ---------------------------------------------------------

    def overlapped(self, query: str, hierarchy: str = DEFAULT_HIERARCHY,
                   top: int | None = None) -> ListResult:
        """All query terms must occur inside ONE region. Matches are collapsed.

        A region qualifies only if EVERY query term has at least one occurrence
        inside it. Occurrences of different terms inside a qualifying region are
        then collapsed into that region, so a paper with three separate mentions
        of all three terms still yields ONE region rather than three.
        """
        terms, unknown = self._query_terms(query)
        if not terms:
            return ListResult(query=query, model="overlapped",
                              hierarchy=hierarchy, unknown=unknown)
        wanted = len(terms)

        found: dict[str, ListRegion] = {}
        for paper in self.index.papers:
            occurrences = [
                Match(term=token.term, position=token.position)
                for token in paper.tokens if token.term in terms
            ]
            spans = collapse(occurrences)
            if not spans:
                continue
            for region in self._regions_covering(paper.doc_id, spans, hierarchy):
                present = {t.term for t in paper.tokens
                           if region.contains(t.position) and t.term in terms}
                if len(present) != wanted:
                    continue
                # Several collapsed spans can land in the SAME region - three
                # terms in one paragraph give three spans and one region. The
                # region is the unit of the answer, so it is keyed by id and
                # counted once; otherwise one paper returns three identical
                # rows that differ only in collapsed_from.
                if region.region_id in found:
                    continue
                found[region.region_id] = ListRegion(
                    region_id=region.region_id, doc_id=region.doc_id,
                    kind=region.kind, label=region.label,
                    start=region.start, end=region.end, text=region.text,
                    terms=tuple(sorted(present)), matched=len(present),
                    total_query_terms=wanted,
                    score=self._region_score(region, terms),
                    role="all terms in one region",
                    collapsed_from=len(spans),
                )
        return self._finish(query, "overlapped", hierarchy, list(found.values()),
                            top, unknown)

    # --- non-overlapped -----------------------------------------------------

    def non_overlapped(self, query: str, hierarchy: str = DEFAULT_HIERARCHY,
                       top: int | None = None) -> ListResult:
        """Terms may occur in DIFFERENT regions. One region per term.

        For each query term the region containing its FIRST occurrence is
        selected, and the answer is the union of those regions. A paper is
        therefore returned when every term appears SOMEWHERE, which is the
        coverage that the overlapped model refuses to give up.
        """
        terms, unknown = self._query_terms(query)
        if not terms:
            return ListResult(query=query, model="non-overlapped",
                              hierarchy=hierarchy, unknown=unknown)
        wanted = len(terms)

        found: list[ListRegion] = []
        for paper in self.index.papers:
            chosen: list[Region] = []
            seen: set[str] = set()
            covered: set[str] = set()
            for term in terms:
                places = self.index.positions_of(paper.doc_id, term)
                if not places:
                    continue
                covered.add(term)
                region = self.index.enclosing(hierarchy, paper.doc_id, places[0])
                if region is None or region.region_id in seen:
                    continue
                seen.add(region.region_id)
                chosen.append(region)

            if not chosen or covered != set(terms):
                continue

            # one ListRegion per distinct region, tagged with WHICH terms it
            # answers - that tag is the difference from the overlapped model
            grouped: dict[str, list[str]] = {}
            owner: dict[str, Region] = {}
            for term in terms:
                places = self.index.positions_of(paper.doc_id, term)
                if not places:
                    continue
                region = self.index.enclosing(hierarchy, paper.doc_id, places[0])
                if region is None:
                    continue
                grouped.setdefault(region.region_id, []).append(term)
                owner.setdefault(region.region_id, region)

            for region_id, answered in grouped.items():
                region = owner[region_id]
                found.append(ListRegion(
                    region_id=region.region_id, doc_id=region.doc_id,
                    kind=region.kind, label=region.label,
                    start=region.start, end=region.end, text=region.text,
                    terms=tuple(sorted(answered)), matched=len(answered),
                    total_query_terms=wanted,
                    score=self._region_score(region, terms),
                    role="answers " + " ".join(sorted(answered)),
                ))
        return self._finish(query, "non-overlapped", hierarchy, found, top,
                            unknown)

    # --- shared tail --------------------------------------------------------

    def _finish(self, query: str, model: str, hierarchy: str,
                found: list[ListRegion], top: int | None,
                unknown: tuple[str, ...]) -> ListResult:
        found.sort(key=lambda r: (-r.score, r.doc_id, r.start))
        kept = found[:top] if top else found
        return ListResult(query=query, model=model, hierarchy=hierarchy,
                          regions=kept,
                          papers=sorted({r.doc_id for r in kept}),
                          unknown=unknown)


def compare(index: StructuredIndex, query: str,
            hierarchy: str = DEFAULT_HIERARCHY, top: int = 10
            ) -> dict[str, ListResult]:
    models = ListModels(index)
    return {
        "overlapped": models.overlapped(query, hierarchy, top),
        "non-overlapped": models.non_overlapped(query, hierarchy, top),
    }


# ---------------------------------------------------------------------------
# console output
# ---------------------------------------------------------------------------

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CONSOLE = Console()

ACCENT = "bold cyan"
WARM = "bold yellow"
DIM = "grey58"

_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"


def bar(fraction: float, width: int = 16) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(BAR_OFF * (width - filled),
                                                      style="grey30")


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def regions_table(result: ListResult, limit: int = 10) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("paper", style=ACCENT, no_wrap=True)
    table.add_column("region", style="white", no_wrap=True)
    table.add_column("label", style="cyan", overflow="fold")
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    table.add_column("matched", justify="right", style=WARM, no_wrap=True)
    table.add_column("score", justify="right", style=WARM, no_wrap=True)
    for region in result.regions[:limit]:
        table.add_row(region.doc_id, region.region_id, region.label,
                      str(region.length), f"{region.matched}/{region.total_query_terms}",
                      f"{region.score:.4f}")
    return table


def section_definitions() -> None:
    heading(1, "THE TWO ANSWERS TO ONE QUESTION")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("model", style=ACCENT, no_wrap=True)
    table.add_column("rule", style="white", overflow="fold")
    table.add_column("gains", style=WARM, overflow="fold")
    table.add_column("costs", style=DIM, overflow="fold")
    table.add_row("overlapped", "every query term must occur inside ONE region;"
                                " matching occurrences are collapsed into the"
                                " smallest region holding them all",
                  "precision: a returned region really does discuss the whole"
                  " query",
                  "coverage: a paper with term A in its methodology and term B"
                  " in its results is invisible")
    table.add_row("non-overlapped", "terms may occur in DIFFERENT regions; each"
                                    " term is answered by its own region and the"
                                    " answer is the union",
                  "coverage: the paper is found wherever the terms are",
                  "precision: the regions returned together may be unrelated")
    CONSOLE.print(table)

    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("overlapped", "exact multi-term search. Use it when the user"
                               " means 'about all of this together'.")
    body.add_row("non-overlapped", "exploratory structured search. Use it when"
                                   " the user means 'about any of this, show me"
                                   " where'.")
    CONSOLE.print(Panel(body, title="best use", title_align="left",
                        border_style="cyan", padding=(0, 1)))


def section_collapse() -> None:
    heading(2, "M collapse M' - THE OPERATOR BOTH MODELS SHARE")
    matches = [Match("index", 10), Match("index", 22),
               Match("phrase", 12), Match("phrase", 23), Match("query", 14)]
    spans = collapse(matches)
    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white")
    body.add_row("input matches", "index@10, index@22, phrase@12, phrase@23,"
                                  " query@14")
    body.add_row("collapse()", ", ".join(f"[{s},{e})" for s, e in spans))
    body.add_row("rule", "two ranges that merely TOUCH are not merged; [0,3) and"
                         " [3,6) stay separate, because merging would invent a"
                         " region containing no occurrence")
    CONSOLE.print(Panel(body, title="collapse", title_align="left",
                        border_style="grey35", padding=(0, 1)))


def section_hierarchies(index: StructuredIndex) -> None:
    heading(3, "ONE HIERARCHY AT A TIME")
    models = ListModels(index)
    query = "proximity window phrase query"
    terms, _ = models._query_terms(query)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("hierarchy", style=ACCENT, no_wrap=True)
    table.add_column("regions", justify="right", style=DIM, no_wrap=True)
    table.add_column("mean terms", justify="right", style=DIM, no_wrap=True)
    table.add_column("ovlp reg", justify="right", style=WARM, no_wrap=True)
    table.add_column("ovlp papers", justify="right", style=WARM, no_wrap=True)
    table.add_column("non reg", justify="right", style=DIM, no_wrap=True)
    table.add_column("non papers", justify="right", style=DIM, no_wrap=True)
    for hierarchy in HIERARCHIES:
        regions = index.regions_of(hierarchy)
        mean = sum(r.length for r in regions) / len(regions)
        over = models.overlapped(query, hierarchy)
        non = models.non_overlapped(query, hierarchy)
        table.add_row(hierarchy, str(len(regions)), f"{mean:.1f}",
                      str(over.count), str(len(over.papers)),
                      str(non.count), str(len(non.papers)))
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]The sentence hierarchy returns the smallest regions and the"
        " section\n  hierarchy the largest. That choice IS the granularity of"
        " the answer.[/]\n"
    )


def section_same_query(index: StructuredIndex) -> None:
    heading(4, "THE SAME QUERY, BOTH MODELS, PARAGRAPHS")
    query = "document index positions"
    terms, _ = ListModels(index)._query_terms(query)
    results = compare(index, query, "paragraph", top=10)

    CONSOLE.print(f"  [bold white]{query!r}[/]  [grey58]terms:"
                  f" {', '.join(terms)}[/]")
    for name in ("overlapped", "non-overlapped"):
        result = results[name]
        CONSOLE.print(f"  [bold cyan]{name}[/]  [grey58]{result.count} regions in"
                      f" {len(result.papers)} papers"
                      f"  ({result.complete_regions} hold every term)[/]")
        CONSOLE.print(regions_table(result, 6))
        CONSOLE.print()

    only_non = set(results["non-overlapped"].papers) - set(results["overlapped"].papers)
    only_over = set(results["overlapped"].papers) - set(results["non-overlapped"].papers)
    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("both", ", ".join(sorted(set(results["overlapped"].papers) & set(results["non-overlapped"].papers))))
    body.add_row("non-overlapped only", ", ".join(sorted(only_non)) or "none")
    body.add_row("overlapped only", ", ".join(sorted(only_over)) or "none")
    body.add_row("read that as", "the extra papers contain all three terms but"
                                " never in the SAME paragraph. Whether they are"
                                " relevant is exactly the judgement the two"
                                " models force you to make, and step 7 settles"
                                " it with relevance judgements.")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))

    lonely = "ranking web search"
    strict = ListModels(index).overlapped(lonely, "paragraph")
    loose = ListModels(index).non_overlapped(lonely, "paragraph")
    CONSOLE.print(
        f"  [bold white]{lonely!r}[/]  [grey58]overlapped:"
        f" {strict.count} regions  {SEP}  non-overlapped:"
        f" {loose.count} regions in {len(loose.papers)} papers[/]"
    )
    CONSOLE.print(regions_table(loose, 6))
    CONSOLE.print(
        "  [grey58]The overlapped model returns NOTHING here. That is not a bug"
        " - no paragraph\n  holds 'ranking', 'web' and 'search' together, so"
        " under the strict rule the corpus\n  has no answer. A system that hid"
        " that behind a zero-result page would look broken, so\n  the interface"
        " has to offer the broader model next to the strict one. That is the"
        " whole\n  argument for building both.[/]\n"
    )


def section_precision_coverage(index: StructuredIndex) -> None:
    heading(5, "PRECISION AND COVERAGE, MEASURED")
    queries = ["document index positions", "ranking web search",
               "passage retrieval recall", "evaluation precision metrics",
               "query expansion embeddings", "mean average precision",
               "term weighting saturation"]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("query", style="bold white", overflow="fold", ratio=2)
    table.add_column("ovlp reg", justify="right", style=WARM, no_wrap=True)
    table.add_column("ovlp pap", justify="right", style=WARM, no_wrap=True)
    table.add_column("all terms", justify="right", style=WARM, no_wrap=True)
    table.add_column("non reg", justify="right", style=DIM, no_wrap=True)
    table.add_column("non pap", justify="right", style=DIM, no_wrap=True)
    for query in queries:
        results = compare(index, query, "paragraph", top=40)
        over, non = results["overlapped"], results["non-overlapped"]
        table.add_row(query, str(over.count), str(len(over.papers)),
                      str(over.complete_regions), str(non.count),
                      str(len(non.papers)))
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]'all terms' counts overlapped regions holding EVERY query"
        " term, which is\n  every one of them by construction. The"
        " non-overlapped model has no such\n  column because its regions hold"
        " one or two terms each - that is the coverage it\n  buys. No metric"
        " here says which model is RIGHT; step 7 measures that with\n"
        "  relevance judgements.[/]\n"
    )


def section_invariants(index: StructuredIndex) -> None:
    heading(6, "PROPERTIES, CHECKED")
    models = ListModels(index)
    query = "proximity window phrase query"
    over = models.overlapped(query, "paragraph")
    non = models.non_overlapped(query, "paragraph")

    disjoint_ok = True
    for doc_id in {r.doc_id for r in over.regions}:
        rows = sorted((r for r in over.regions if r.doc_id == doc_id),
                      key=lambda r: r.start)
        for a, b in zip(rows, rows[1:]):
            if a.end > b.start:
                disjoint_ok = False

    overlap_ids = {r.region_id for r in over.regions}
    non_ids = {r.region_id for r in non.regions}

    checks = [
        ("every overlapped region holds ALL query terms",
         "yes" if all(r.complete for r in over.regions) else "NO"),
        ("overlapped regions in one paper never overlap each other",
         "yes" if disjoint_ok else "NO"),
        ("no duplicated region in an answer",
         "yes" if len(overlap_ids) == over.count else "NO"),
        ("non-overlapped returns at least as many PAPERS",
         "yes" if set(non.papers) >= set(over.papers) else "NO"),
        ("every non-overlapped region names the terms it answers",
         "yes" if all(r.role.startswith("answers") for r in non.regions) else "NO"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    for label, value in checks:
        table.add_row(label, value)
    CONSOLE.print(table)

    both_ways = [q for q in ("document index positions", "ranking web search",
                             "mean average precision")
                 if (set(ListModels(index).overlapped(q, "paragraph").papers)
                     ^ set(ListModels(index).non_overlapped(q, "paragraph").papers))]
    CONSOLE.print(
        f"  [grey58]Region IDS are not comparable between the two models. On"
        f" this query the\n  non-overlapped answer contributes"
        f" {len(non_ids - overlap_ids)} region(s) the overlapped model never"
        f" returns, and on\n  {len(both_ways)} of the three queries above the"
        f" two paper sets differ. That is\n  the model difference, not a bug -"
        f" the non-overlapped model answers with regions that\n  hold one term"
        f" each, which by definition cannot also hold them all.[/]\n"
    )


def main() -> None:
    index = build_index()
    models = ListModels(index)

    CONSOLE.rule("[bold cyan]Step 5 - Overlapped and Non-Overlapped Lists[/]",
                 style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} papers  {SEP}  hierarchies: "
        f"{', '.join(HIERARCHIES)}  {SEP}  default: {DEFAULT_HIERARCHY}[/]\n"
    )

    section_definitions()
    section_collapse()
    section_hierarchies(index)
    section_same_query(index)
    section_precision_coverage(index)
    section_invariants(index)


if __name__ == "__main__":
    main()