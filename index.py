"""Step 3 - Inverted index for the university IR mini search engine.

An inverted index is a dictionary that flips the usual direction of a corpus.

    forward  (how the documents are stored) : document -> [term, term, term]
    inverted (how this index stores it)     : term     -> [document, document]

With it, "which documents contain this term?" is a single dictionary lookup
instead of opening every document and reading it.

Term frequency (tf) is recorded inside each posting because counting occurrences
is a free by-product of the single pass over the corpus. Step 4 analyses it;
nothing here searches, scores or ranks anything.

Run:  python index.py            summary + samples
      python index.py --full     every posting list
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from preprocess import Document, build_documents


@dataclass(frozen=True)
class Posting:
    """One 'term t occurs in document d, tf times' entry."""

    doc_id: str
    tf: int

    def __str__(self) -> str:
        return f"{self.doc_id}:{self.tf}"


@dataclass
class InvertedIndex:
    documents: list[Document]
    postings: dict[str, list[Posting]]
    vocabulary: list[str]
    doc_lengths: dict[str, int]

    @property
    def number_of_documents(self) -> int:
        return len(self.documents)

    @property
    def total_postings(self) -> int:
        return sum(len(p) for p in self.postings.values())


def build_index(documents: list[Document] | None = None) -> InvertedIndex:
    """Build the inverted index. With no argument, load and preprocess the corpus."""
    if documents is None:
        documents = build_documents()

    postings: dict[str, list[Posting]] = {}
    for document in documents:
        # Counter counts the repeats, set removes the duplicate terms so a
        # document can never appear twice in one posting list
        for term, tf in sorted(Counter(document.terms).items()):
            postings.setdefault(term, []).append(Posting(document.doc_id, tf))

    return InvertedIndex(
        documents=documents,
        postings=postings,
        vocabulary=sorted(postings),
        doc_lengths={d.doc_id: len(d.terms) for d in documents},
    )


def lookup(index: InvertedIndex, term: str) -> list[Posting]:
    """Postings for a term, or an empty list if the term is not in the corpus.

    An unknown term must never raise: step 7 needs a missing term to mean
    'appears in no documents' so that NOT works.
    """
    return index.postings.get(term, [])


def documents_containing(index: InvertedIndex, term: str) -> set[str]:
    """Set of document ids containing a term. Used by the Boolean step."""
    return {posting.doc_id for posting in lookup(index, term)}


def document_frequency(index: InvertedIndex, term: str) -> int:
    """How many documents contain a term. Step 5 turns this into IDF."""
    return len(lookup(index, term))


# ---------------------------------------------------------------------------
# console output
# ---------------------------------------------------------------------------

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROBE_TERMS = [
    'machin learn',
    'machin',
    'learn',
    'healthcar',
    'autonom',
    'blockchain',
]

LARGEST_LISTS = 10




CONSOLE = Console()

ACCENT = "bold cyan"
DIM = "grey58"

# The Windows console here is cp437 and stdout is cp1252, so block characters
# must not be hard-coded or they print as '?'. Pick them from the encoding.
_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"


def bar(fraction: float, width: int = 24) -> Text:
    """A horizontal bar, used to show postings-list length at a glance."""
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(BAR_OFF * (width - filled),
                                                      style="grey30")


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def section_overview(index: InvertedIndex) -> None:
    heading(1, "OVERVIEW")
    postings = index.total_postings
    vocabulary = len(index.vocabulary)
    occurrences = sum(index.doc_lengths.values())
    lengths = sorted(len(p) for p in index.postings.values())

    rows = [
        ("documents", str(index.number_of_documents)),
        ("vocabulary size (unique terms)", str(vocabulary)),
        ("total postings", str(postings)),
        ("avg postings per term", f"{postings / vocabulary:.2f}"),
        ("postings per term",
         f"min {lengths[0]}  {SEP}  median {lengths[len(lengths) // 2]}  "
         f"{SEP}  max {lengths[-1]}"),
        ("term occurrences", f"{occurrences:,}  ->  {occurrences - postings} repeats "
                             f"folded into tf"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    for label, value in rows:
        table.add_row(label, value)
    CONSOLE.print(table)


def section_distribution(index: InvertedIndex) -> None:
    heading(2, "POSTINGS-LENGTH DISTRIBUTION")
    buckets = [
        ("1 document", "hapax", lambda n: n == 1),
        ("2-3 documents", "", lambda n: 2 <= n <= 3),
        ("4-5 documents", "", lambda n: 4 <= n <= 5),
        ("6-10 documents", "", lambda n: 6 <= n <= 10),
        ("11-20 documents", "", lambda n: n >= 11),
    ]
    total = len(index.vocabulary)
    largest = max(
        sum(1 for term in index.vocabulary if test(len(index.postings[term])))
        for _, _, test in buckets
    )

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("postings length", style="white")
    table.add_column("terms", justify="right", style="bold cyan")
    table.add_column("share", justify="right", style=DIM)
    table.add_column("distribution", width=26)

    for label, note, test in buckets:
        count = sum(1 for term in index.vocabulary if test(len(index.postings[term])))
        name = f"{label}  [grey50]{note}[/]" if note else label
        table.add_row(name, str(count), f"{count / total:.0%}",
                      bar(count / largest, 24))
    CONSOLE.print(table)

    hapax = sum(1 for term in index.vocabulary if len(index.postings[term]) == 1)
    CONSOLE.print(
        f"  [bold]{hapax} of {total} terms ({hapax / total:.0%})[/] appear in exactly "
        f"one document. These get the [bold]highest IDF[/] in step 5.\n"
    )


def section_largest_lists(index: InvertedIndex, top: int = LARGEST_LISTS) -> None:
    heading(3, f"{top} LARGEST POSTING LISTS")
    ranked = sorted(index.postings.items(), key=lambda item: (-len(item[1]), item[0]))

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("docs", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("coverage", width=26)
    for term, postings in ranked[:top]:
        table.add_row(term, str(len(postings)),
                      bar(len(postings) / index.number_of_documents, 24))
    CONSOLE.print(table)


def section_lookups(index: InvertedIndex) -> None:
    heading(4, "WORKED LOOKUPS")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("docs", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("documents containing it", style=DIM)

    for term in PROBE_TERMS:
        postings = lookup(index, term)
        if not postings:
            table.add_row(term, "0",
                          "[grey50]not in the corpus - clean miss, no error[/]")
            continue
        table.add_row(term, str(len(postings)),
                      ", ".join(p.doc_id for p in postings))
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]'machin learn' is its own index entry, while 'machin' and 'learn'"
        "\n  remain searchable on their own.[/]\n"
    )


def section_reverse_view(index: InvertedIndex, doc_id: str = "doc09") -> None:
    heading(5, f"REVERSE VIEW - {doc_id}")
    document = next(d for d in index.documents if d.doc_id == doc_id)
    repeated = sorted({t for t in document.terms if document.terms.count(t) > 1})

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM)
    body.add_column(style="white")
    body.add_row("title", document.title)
    body.add_row("length", f"{index.doc_lengths[doc_id]} terms, "
                            f"{len(set(document.terms))} distinct")
    body.add_row("repeated", "  ".join(
        f"[bold cyan]{t}[/][grey50]({document.terms.count(t)})[/]" for t in repeated
    ))
    CONSOLE.print(Panel(body, title=doc_id, title_align="left",
                        border_style="grey35", padding=(0, 1)))


def section_full_dump(index: InvertedIndex) -> None:
    heading(6, f"FULL DUMP - ALL {len(index.vocabulary)} TERMS")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="white", no_wrap=True)
    table.add_column("docs", justify="right", style=ACCENT, no_wrap=True)
    table.add_column("postings", style=DIM)
    for term in index.vocabulary:
        postings = index.postings[term]
        table.add_row(term, str(len(postings)),
                      " ".join(str(p) for p in postings))
    CONSOLE.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and display the inverted index.")
    parser.add_argument("--full", action="store_true",
                        help="print every posting list, not just the samples")
    args = parser.parse_args()

    index = build_index()

    CONSOLE.rule("[bold cyan]Step 3 - Inverted Index[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} documents  {SEP}  "
        f"{len(index.vocabulary):,} terms  {SEP}  "
        f"{index.total_postings:,} postings[/]\n"
    )

    section_overview(index)
    section_distribution(index)
    section_largest_lists(index)
    section_lookups(index)
    section_reverse_view(index)

    if args.full:
        section_full_dump(index)


if __name__ == "__main__":
    main()