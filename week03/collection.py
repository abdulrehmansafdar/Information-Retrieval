"""Step 0 - the collection this module browses.

week01 indexed a flat list of documents and week02 indexed a structured one.
Neither is what a *browser* needs: a browser needs a collection of things it can
list, group, preview and link together. So this module loads the same curated
20-paper corpus into the smallest structure that supports those four operations:

    Document   one paper: its metadata and its named prose sections
    Section    one '## Heading' block inside a paper (Abstract, Methodology, ...)

Deliberately lighter than week02/preprocess.py. A browser previews raw text and
links documents by topic; it does not need stems, positions or a positional
index. Search here is a plain TF-IDF cosine over the words as written, which is
honest about what a browsing module's search box is: a way in, not a ranking
model. week02's BM25 remains the place to go for ranked retrieval.

File format (identical to week02/data/papers)
---------------------------------------------
    Title: Deep Learning for Medical Image Segmentation
    Year: 2023
    Venue: Journal of Applied Informatics

    ## Abstract
    one paragraph

    ## Methodology
    a paragraph

    Blank lines separate paragraphs. Headings start a section. The heading text
    is kept verbatim so a preview can show it.

Run:  python collection.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

COLLECTION_DIR = Path(__file__).resolve().parent / "data" / "papers"

META_RE = re.compile(r"^(Title|Year|Venue):\s*(.+?)\s*$")
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
WORD_RE = re.compile(r"[a-z0-9]+")

# A built-in list rather than an nltk download. Browsing tolerates an imperfect
# stop list, and a module that fails to start because a corpus was not
# downloaded is a worse trade than a slightly noisier search box.
STOP_WORDS = frozenset(
    """
    a an the and or but if while of to in on at by for with from into over after
    before between under again further then once here there all any both each few
    more most other some such no nor not only own same so than too very can will
    just should now is are was were be been being have has had having do does did
    doing would could shall may might must i you he she it we they them his her
    its their our your my me him us what which who whom whose when where why how
    as because until about against among throughout during this that these those
    we our they their it its also often usually still thus however within
    upon whether rather said almost whenever along one every many much like
    non even paper study report reports
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words, with stop words and one-character tokens dropped."""
    return [w for w in WORD_RE.findall(text.lower())
            if len(w) > 1 and w not in STOP_WORDS]


@dataclass(frozen=True)
class Section:
    """One '## Heading' block. `name` is the heading text, verbatim."""

    name: str
    text: str

    @property
    def words(self) -> int:
        return len(WORD_RE.findall(self.text.lower()))


@dataclass(frozen=True)
class Document:
    """One paper, ready to list, group, preview and link."""

    doc_id: str
    title: str
    year: int
    venue: str
    sections: tuple[Section, ...]
    topic_terms: tuple[str, ...]
    terms: tuple[str, ...]

    @property
    def section_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sections)

    def section(self, name: str) -> Section | None:
        wanted = name.strip().lower()
        return next((s for s in self.sections if s.name.lower() == wanted), None)

    @property
    def abstract(self) -> str:
        found = self.section("abstract")
        return found.text if found else (self.sections[0].text if self.sections
                                         else "")

    @property
    def words(self) -> int:
        return sum(s.words for s in self.sections)


class CollectionFormatError(ValueError):
    """A paper file is unreadable. Raised with a filename, never bare."""


def parse_document(path: Path | str) -> Document:
    """Read one paper file into a Document."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()

    lines = text.splitlines()
    meta: dict[str, str] = {}
    cursor = 0
    for cursor, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            cursor += 1
            break
        match = META_RE.match(stripped)
        if not match:
            break
        meta[match.group(1).lower()] = match.group(2)
    else:
        cursor = len(lines)

    if "title" not in meta:
        raise CollectionFormatError(f"{path.name}: no 'Title:' line")

    # Everything from the first heading on becomes a named section.
    sections: list[Section] = []
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            body = " ".join(buffer).strip()
            sections.append(Section(name=current, text=body))

    for line in lines[cursor:]:
        heading = HEADING_RE.match(line.strip())
        if heading:
            flush()
            current = heading.group(1).strip()
            buffer = []
            continue
        if line.strip():
            buffer.append(line.strip())
    flush()

    if not sections:
        raise CollectionFormatError(f"{path.name}: no '## Heading' sections")

    topic = tokenize(meta["title"] + " " + " ".join(
        s.text for s in sections if s.name.lower() == "abstract"))
    everything = tokenize(meta["title"] + " " + " ".join(s.text for s in sections))

    return Document(
        doc_id=path.stem,
        title=meta["title"],
        year=int(meta.get("year") or 0),
        venue=meta.get("venue", ""),
        sections=tuple(sections),
        topic_terms=tuple(topic),
        terms=tuple(everything),
    )


def load_documents(directory: Path | str = COLLECTION_DIR) -> list[Document]:
    """Load every paper in the collection, sorted by id."""
    return [parse_document(p) for p in sorted(Path(directory).glob("*.txt"))]


@lru_cache(maxsize=1)
def get_documents() -> tuple[Document, ...]:
    """The loaded collection. Built once for the whole process."""
    return tuple(load_documents())


def corpus_stats() -> dict:
    """Numbers for the home page and the console header."""
    documents = get_documents()
    years = sorted({d.year for d in documents})
    venues = sorted({d.venue for d in documents})
    return {
        "documents": len(documents),
        "sections": sum(len(d.sections) for d in documents),
        "section_names": sorted({s.name for d in documents for s in d.sections}),
        "words": sum(d.words for d in documents),
        "vocabulary": len({t for d in documents for t in d.terms}),
        "years": years,
        "venues": venues,
    }


def main() -> None:
    documents = get_documents()
    stats = corpus_stats()
    print(f"{stats['documents']} documents | {stats['sections']} sections | "
          f"{stats['words']:,} words | {stats['vocabulary']:,} distinct terms")
    print(f"years {stats['years'][0]}-{stats['years'][-1]} | "
          f"{len(stats['venues'])} venues | "
          f"sections: {', '.join(stats['section_names'])}")
    for document in documents:
        print(f"  {document.doc_id}  {document.year}  "
              f"{document.title[:56]:56s}  {document.words:>4} words")


if __name__ == "__main__":
    main()
