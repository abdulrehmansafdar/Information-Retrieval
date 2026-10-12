"""Step 2 - The positional index and the three region hierarchies.

Week 01's inverted index answered one question: which documents contain this
term. Step 3 of week 02 needs to answer two more:

    WHERE did it occur        -> proximity, phrase and window operators
    WHICH region contains it  -> the overlapped / non-overlapped list models
                                and the proximal node model

So a posting here carries three things instead of one:

    (doc_id, tf, [positions])

That third element is the whole difference between this file and week01/index.py,
and everything downstream in week 02 is built on it.

The three hierarchies
--------------------
A hierarchy is one flat, non-overlapping view of the text:

    'section'    one region per named field   - 4 per paper
    'paragraph'  one region per blank-line block - 5 per paper
    'sentence'   one region per sentence       - about 19 per paper

Within a hierarchy the regions never overlap: they tile the paper from position 0
to its length. ACROSS hierarchies they do overlap, and that is the point - a
sentence lies inside a paragraph, which lies inside a section. The three views
together form the hierarchy tree

    paper
      section        title | abstract | methodology | results
        paragraph    blank-line block inside one field
          sentence   one sentence inside one paragraph

which is exactly the tree the proximal node model walks.

An invariant worth stating, because every model here depends on it:
for every paper, the section regions tile [0, length), and so do the paragraph
regions and the sentence regions. `check_invariants()` proves it rather than
trusting it.

Run:  python structured_index.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from preprocess import (
    Paper,
    Region,
    Token,
    load_papers,
    preprocess_with_positions,
    split_sentences,
)

HIERARCHIES = ("section", "paragraph", "sentence")

# Which hierarchy is nested inside which. Used to walk upward, so a model can
# ask "what is the smallest paragraph containing position 51?".
PARENT_OF = {"section": "paper", "paragraph": "section", "sentence": "paragraph"}


@dataclass(frozen=True)
class Posting:
    """One 'term t occurs in document d, tf times, at these positions' entry."""

    doc_id: str
    tf: int
    positions: tuple[int, ...]

    def __str__(self) -> str:
        return f"{self.doc_id}:{self.tf}"


@dataclass
class StructuredIndex:
    """The whole built corpus. Treat as read-only."""

    papers: list[Paper]
    postings: dict[str, list[Posting]]
    vocabulary: list[str]
    doc_lengths: dict[str, int]
    regions: dict[str, list[Region]]
    by_id: dict[str, Region] = field(default_factory=dict)

    # --- corpus-level properties -------------------------------------------

    @property
    def number_of_documents(self) -> int:
        return len(self.papers)

    @property
    def total_postings(self) -> int:
        return sum(len(p) for p in self.postings.values())

    @property
    def total_positions(self) -> int:
        return sum(len(post.positions) for plist in self.postings.values()
                   for post in plist)

    @property
    def average_document_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    @property
    def document_frequency(self) -> dict[str, int]:
        return {term: len(self.postings[term]) for term in self.vocabulary}

    # --- lookups -----------------------------------------------------------

    def lookup(self, term: str) -> list[Posting]:
        """Postings for a term. An unknown term returns [], never raises.

        Every model in this week treats an unknown term as matching nothing, so
        raising here would turn a clean miss into a crash.
        """
        return self.postings.get(term, [])

    def documents_containing(self, term: str) -> set[str]:
        return {post.doc_id for post in self.lookup(term)}

    def term_frequency(self, term: str) -> int:
        """Corpus-wide count, the sum of tf over every posting. NOT df."""
        return sum(post.tf for post in self.lookup(term))

    def positions_of(self, doc_id: str, term: str) -> list[int]:
        """Ascending positions of `term` in one document."""
        for post in self.lookup(term):
            if post.doc_id == doc_id:
                return list(post.positions)
        return []

    def term_frequency_in(self, doc_id: str, term: str) -> int:
        return len(self.positions_of(doc_id, term))

    # --- regions -----------------------------------------------------------

    def regions_of(self, kind: str) -> list[Region]:
        if kind not in self.regions:
            raise KeyError(f"unknown hierarchy {kind!r}; "
                           f"expected one of {HIERARCHIES}")
        return self.regions[kind]

    def regions_of_kind(self, doc_id: str, kind: str) -> list[Region]:
        return [r for r in self.regions_of(kind) if r.doc_id == doc_id]

    def field_region(self, doc_id: str, name: str) -> Region | None:
        paper = next((p for p in self.papers if p.doc_id == doc_id), None)
        return None if paper is None else paper.fields.get(name)

    def field_at(self, doc_id: str, position: int) -> str:
        """Name of the named field holding a position, e.g. 'methodology'."""
        paper = next((p for p in self.papers if p.doc_id == doc_id), None)
        if paper is None:
            return ""
        for name, region in paper.fields.items():
            if region.contains(position):
                return name
        return ""

    def enclosing(self, kind: str, doc_id: str, position: int) -> Region | None:
        """Smallest region of `kind` containing a position.

        This is the single lookup the proximal node model depends on. Called
        once per position, it is O(regions of that document), which is why the
        model evaluates queries bottom-up: match the content first, then walk up,
        rather than walking the whole tree for every query.
        """
        best: Region | None = None
        for region in self.regions_of_kind(doc_id, kind):
            if not region.contains(position):
                continue
            if best is None or region.length < best.length:
                best = region
        return best

    def ancestors(self, region: Region) -> list[Region]:
        """A region, then its section, then the paper - outward, never downward.

        `P in Q` and the bottom-up evaluation both need this direction only.
        """
        chain = [region]
        current = region
        while PARENT_OF[current.kind] in HIERARCHIES:
            parent_kind = PARENT_OF[current.kind]
            found = next(
                (r for r in self.regions_of_kind(current.doc_id, parent_kind)
                 if r.contains(current.start) and r.contains(current.end - 1)),
                None,
            )
            if found is None:
                break
            chain.append(found)
            current = found
        return chain


def _regions_for(paper: Paper) -> dict[str, list[Region]]:
    """Build the three hierarchies for one paper.

    Re-running preprocess_with_positions on each block is exact rather than
    approximate: parse_paper ran the same function over the same block texts in
    the same order, so the term counts line up and the regions tile the paper.
    `check_invariants` proves that instead of trusting it.
    """
    sections = sorted(paper.fields.values(), key=lambda r: r.start)

    paragraphs: list[Region] = []
    sentences: list[Region] = []

    def add_paragraphs(section: Region, blocks: list[str]) -> None:
        """Tile one field with paragraph regions, then tile each paragraph with
        sentence regions.

        The two offsets are tracked separately on purpose. A sentence's range is
        measured in the PAPER's coordinates, so it starts at its paragraph's
        start, not at the running offset after that paragraph has been added.
        """
        offset = section.start
        for number, block in enumerate(blocks, 1):
            terms, _ = preprocess_with_positions(block)
            if not terms:
                continue
            paragraph_id = f"{paper.doc_id}:{section.label}:p{number}"
            start = offset
            paragraphs.append(Region(
                region_id=paragraph_id,
                doc_id=paper.doc_id,
                kind="paragraph",
                label=f"{section.label} paragraph {number}",
                start=start,
                end=start + len(terms),
                text=block,
                parent_id=section.region_id,
                depth=1,
            ))
            offset = start + len(terms)

            sentence_offset = start
            for index, sentence in enumerate(split_sentences(block), 1):
                sterms, _ = preprocess_with_positions(sentence)
                if not sterms:
                    continue
                sentences.append(Region(
                    region_id=f"{paragraph_id}:s{index}",
                    doc_id=paper.doc_id,
                    kind="sentence",
                    label=f"{section.label} p{number} sentence {index}",
                    start=sentence_offset,
                    end=sentence_offset + len(sterms),
                    text=sentence,
                    parent_id=paragraph_id,
                    depth=2,
                ))
                sentence_offset += len(sterms)

    for section in sections:
        if section.label == "title":
            # The title is one region. It still gets an entry in all three
            # hierarchies, otherwise every model would have to special-case the
            # one field that has no internal structure.
            add_paragraphs(section, [section.text])
            continue
        add_paragraphs(section, paper.paragraphs.get(section.label, [section.text]))

    return {"section": sections, "paragraph": paragraphs, "sentence": sentences}


def build_index(papers: list[Paper] | None = None) -> StructuredIndex:
    """Build the positional index and all three region hierarchies."""
    if papers is None:
        papers = load_papers()

    postings: dict[str, list[Posting]] = {}
    for paper in papers:
        grouped: dict[str, list[int]] = {}
        for token in paper.tokens:
            grouped.setdefault(token.term, []).append(token.position)
        for term in sorted(grouped):
            places = tuple(sorted(grouped[term]))
            postings.setdefault(term, []).append(
                Posting(paper.doc_id, len(places), places)
            )

    regions: dict[str, list[Region]] = {k: [] for k in HIERARCHIES}
    for paper in papers:
        built = _regions_for(paper)
        for kind in HIERARCHIES:
            regions[kind].extend(built[kind])

    index = StructuredIndex(
        papers=papers,
        postings=postings,
        vocabulary=sorted(postings),
        doc_lengths={p.doc_id: p.length for p in papers},
        regions=regions,
        # A paragraph's parent is a SECTION, and a sentence's parent is a
        # paragraph, so all three kinds must be in by_id - not just the
        # finer two, or every parent lookup fails.
        by_id={r.region_id: r
               for kind in HIERARCHIES for r in regions[kind]},
    )
    return index


def check_invariants(index: StructuredIndex) -> list[str]:
    """Prove the properties every model here assumes. Returns problems found.

    A model built on regions that do not tile their document will still run and
    will still produce numbers, so these are checked rather than assumed.
    """
    problems: list[str] = []
    by_id = {p.doc_id: p for p in index.papers}

    for paper in index.papers:
        positions = [t.position for t in paper.tokens]
        if positions != list(range(paper.length)):
            problems.append(f"{paper.doc_id}: positions are not dense 0..n-1")

        if sum(r.length for r in paper.fields.values()) != paper.length:
            problems.append(f"{paper.doc_id}: fields do not sum to the length")

        for kind in HIERARCHIES:
            rows = sorted(index.regions_of_kind(paper.doc_id, kind),
                          key=lambda r: r.start)
            if not rows:
                problems.append(f"{paper.doc_id}: no {kind} regions")
                continue
            if rows[0].start != 0:
                problems.append(
                    f"{paper.doc_id}/{kind}: starts at {rows[0].start}, not 0")
            if rows[-1].end != paper.length:
                problems.append(
                    f"{paper.doc_id}/{kind}: ends at {rows[-1].end}, "
                    f"not {paper.length}")
            for before, after in zip(rows, rows[1:]):
                if before.end != after.start:
                    problems.append(
                        f"{paper.doc_id}/{kind}: gap or overlap between "
                        f"{before.region_id} and {after.region_id}")

    # a child region must lie inside its parent
    for kind in ("paragraph", "sentence"):
        for region in index.regions_of(kind):
            parent = index.by_id.get(region.parent_id)
            if parent is None:
                problems.append(f"{region.region_id}: missing parent")
            elif region.start < parent.start or region.end > parent.end:
                problems.append(f"{region.region_id}: not inside {parent.region_id}")

    # positions stored in postings must agree with the document's own tokens
    for term in index.vocabulary:
        for post in index.lookup(term):
            actual = [t.position for t in by_id[post.doc_id].tokens
                      if t.term == term]
            if list(post.positions) != actual:
                problems.append(f"{term}@{post.doc_id}: posting positions "
                                f"disagree with the document")
                break
        if problems:
            break

    return problems


def region_summary(index: StructuredIndex, doc_id: str) -> list[dict]:
    """Plain rows for the console and the Streamlit structure tab."""
    paper = next(p for p in index.papers if p.doc_id == doc_id)
    rows = []
    for kind in HIERARCHIES:
        for region in sorted(index.regions_of_kind(doc_id, kind),
                             key=lambda r: r.start):
            hits = sorted(Counter(
                t.term for t in paper.tokens if region.contains(t.position)
            ).items(), key=lambda kv: -kv[1])[:6]
            rows.append({
                "kind": kind,
                "region": region.region_id,
                "start": region.start,
                "end": region.end,
                "terms": region.length,
                "parent": region.parent_id or "-",
                "heaviest": ", ".join(f"{t} x{c}" for t, c in hits) or "-",
            })
    return rows


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


def bar(fraction: float, width: int = 18) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(BAR_OFF * (width - filled),
                                                      style="grey30")


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def section_posting_difference() -> None:
    heading(1, "ONE POSTING, THREE FACTS")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("week 01", "Posting(doc_id, tf)                      -> index.py:29")
    body.add_row("week 02", "Posting(doc_id, tf, positions)          -> structured_index.py:47")
    body.add_row("tf", "how many times the term occurs in the document")
    body.add_row("positions", "WHERE it occurs, ascending, measured from the start")
    body.add_row("why it matters", "a proximity window cannot be evaluated without positions,")
    body.add_row("", "and a region model cannot return a region without them")
    CONSOLE.print(Panel(body, title="the only structural change this week",
                        title_align="left", border_style="grey35", padding=(0, 1)))


def section_overview(index: StructuredIndex) -> None:
    heading(2, "OVERVIEW")
    df = index.document_frequency
    lengths = sorted(df.values())
    rows = [
        ("papers", str(index.number_of_documents)),
        ("vocabulary", f"{len(index.vocabulary):,}"),
        ("postings", f"{index.total_postings:,}"),
        ("stored positions", f"{index.total_positions:,}"),
        ("document length", f"{min(index.doc_lengths.values())} to "
                            f"{max(index.doc_lengths.values())} terms, "
                            f"average {index.average_document_length:.1f}"),
        ("document frequency", f"min {lengths[0]}  {SEP}  median "
                               f"{lengths[len(lengths) // 2]}  {SEP}  max {lengths[-1]}"),
        ("sections", f"{len(index.regions_of('section')):,}"),
        ("paragraphs", f"{len(index.regions_of('paragraph')):,}"),
        ("sentences", f"{len(index.regions_of('sentence')):,}"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    for label, value in rows:
        table.add_row(label, value)
    CONSOLE.print(table)


def section_hierarchies(index: StructuredIndex, doc_id: str = "p01") -> None:
    heading(3, f"THE THREE HIERARCHIES - {doc_id}")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("kind", style=ACCENT, no_wrap=True)
    table.add_column("region", style="white", no_wrap=True)
    table.add_column("start", justify="right", style=DIM, no_wrap=True)
    table.add_column("end", justify="right", style=DIM, no_wrap=True)
    table.add_column("terms", justify="right", style=WARM, no_wrap=True)
    table.add_column("share", width=16)
    paper = next(p for p in index.papers if p.doc_id == doc_id)
    for kind in HIERARCHIES:
        for region in sorted(index.regions_of_kind(doc_id, kind),
                             key=lambda r: r.start):
            table.add_row(kind, region.region_id, str(region.start),
                          str(region.end), str(region.length),
                          bar(region.length / paper.length, 14))
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]Read the start column down within one kind: they are contiguous"
        f"\n  and they tile 0 to {paper.length}. Read across kinds and they nest,"
        f" which is\n  what makes a hierarchy.[/]\n"
    )


def section_positions(index: StructuredIndex, doc_id: str = "p04",
                      term: str = "proxim oper") -> None:
    heading(4, f"POSITIONS IN ONE PAPER - '{term}' in {doc_id}")
    places = index.positions_of(doc_id, term)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("position", justify="right", style=DIM, no_wrap=True)
    table.add_column("field", style=ACCENT, no_wrap=True)
    table.add_column("section", style="white", no_wrap=True)
    table.add_column("sentence", style=DIM, overflow="fold")
    for position in places:
        sentence = index.enclosing("sentence", doc_id, position)
        table.add_row(
            str(position),
            index.field_at(doc_id, position),
            sentence.label if sentence else "?",
            (sentence.text[:54] + "...") if sentence and len(sentence.text) > 54
            else (sentence.text if sentence else ""),
        )
    CONSOLE.print(table)

    if len(places) > 1:
        gaps = [b - a for a, b in zip(places, places[1:])]
        CONSOLE.print(
            f"  [grey58]tf = {len(places)}, distances between occurrences:"
            f" {gaps} terms.\n  A window of {max(gaps) + 1} terms would contain all of"
            f" them; a window of\n  {min(gaps) + 1} would contain only the closest pair."
            f" That single number is the whole\n  proximity parameter.[/]\n"
        )


def section_invariants(index: StructuredIndex) -> None:
    heading(5, "INVARIANTS, CHECKED NOT ASSUMED")
    problems = check_invariants(index)
    checks = [
        ("positions dense, 0..n-1, no gaps", "yes"),
        ("fields sum to the document length", "yes"),
        ("each hierarchy tiles 0..length", "yes"),
        ("every child region inside its parent", "yes"),
        ("posting positions agree with the document", "yes"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    for label, value in checks:
        table.add_row(label, value if not problems else "SEE BELOW")
    CONSOLE.print(table)
    if problems:
        CONSOLE.print(f"  [red]{len(problems)} problem(s):[/]")
        for problem in problems[:10]:
            CONSOLE.print(f"    [red]{problem}[/]")
    else:
        CONSOLE.print("  [green]check_invariants() found nothing.[/]\n")


def main() -> None:
    index = build_index()

    CONSOLE.rule("[bold cyan]Step 2 - Positional Index and Region Hierarchies[/]",
                 style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} papers  {SEP}  "
        f"{len(index.vocabulary):,} terms  {SEP}  "
        f"{index.total_postings:,} postings  {SEP}  "
        f"{index.total_positions:,} positions[/]\n"
    )

    section_posting_difference()
    section_overview(index)
    section_hierarchies(index)
    section_positions(index)
    section_invariants(index)


if __name__ == "__main__":
    main()