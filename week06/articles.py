"""Step 0 - the shared corpus this module searches.

The clinical articles live once, at the repository root, in corpus/articles. Every
module from week 6 on reads them from there and never keeps its own copy. This
file is the only place that knows the folder layout.

    Article    one clinical article: metadata and its named sections
    Section    one '## Heading' block (Overview, Symptoms, Diagnosis, Treatment)

Text is turned into stems: lower case, stop words removed, Snowball stemmed. So
"hypertension", "hypertensive" and "hypertensions" share one stem. Stemming is
what lets a query word match an article word that is spelled differently, and it
is the first of the two kinds of partial matching this module does (the second,
in membership.py, compares stems that are spelled differently).

File format
-----------
    Title: Hypertension
    Specialty: Cardiology
    Code: I10
    Updated: 2025

    ## Overview
    one paragraph

Run:  python articles.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from nltk.stem import SnowballStemmer

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "articles"

META_RE = re.compile(r"^(Title|Specialty|Code|Updated):\s*(.+?)\s*$")
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
WORD_RE = re.compile(r"[a-z0-9]+")
REQUIRED = ("title", "specialty", "code", "updated")

# Built in rather than downloaded from nltk, so the module starts without a
# corpus download. Words that carry meaning in a medical search are NOT listed.
STOP_WORDS = frozenset(
    """
    a an the and or but if while of to in on at by for with from into over after
    before between under again further then once here there all any both each few
    more most other some such no nor not only own same so than too very can will
    just should now is are was were be been being have has had having do does did
    doing would could shall may might must i you he she it we they them his her
    its their our your my me him us what which who whom whose when where why how
    as because until about against among throughout during this that these those
    also often usually still thus however within upon whether rather said almost
    one every many much like even
    """.split()
)

_stemmer = SnowballStemmer("english")


def tokenize(text: str) -> list[str]:
    """Stems of the words in `text`, with stop words and one-character tokens dropped."""
    return [_stemmer.stem(w) for w in WORD_RE.findall(text.lower())
            if len(w) > 1 and w not in STOP_WORDS]


@dataclass(frozen=True)
class Section:
    """One '## Heading' block. `name` is the heading text, verbatim."""

    name: str
    text: str


@dataclass(frozen=True)
class Article:
    """One clinical article, ready to search, rank and preview."""

    doc_id: str
    title: str
    specialty: str
    code: str
    updated: int
    sections: tuple[Section, ...]
    title_terms: tuple[str, ...]
    body_terms: tuple[str, ...]

    @property
    def section_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sections)

    def section(self, name: str) -> Section | None:
        wanted = name.strip().lower()
        return next((s for s in self.sections if s.name.lower() == wanted), None)


class CorpusFormatError(ValueError):
    """An article file is unreadable. Always raised with the file name."""


def parse_article(path: Path | str) -> Article:
    """Read one article file into an Article."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    meta: dict[str, str] = {}
    cursor = 0
    for cursor, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            break
        match = META_RE.match(stripped)
        if not match:
            raise CorpusFormatError(f"{path.name}: unexpected line {stripped[:40]!r}")
        meta[match.group(1).lower()] = match.group(2)
    else:
        cursor = len(lines)

    missing = [key for key in REQUIRED if key not in meta]
    if missing:
        raise CorpusFormatError(f"{path.name}: missing {', '.join(missing)}")
    if not meta["updated"].isdigit():
        raise CorpusFormatError(f"{path.name}: 'Updated' must be a year")

    sections: list[Section] = []
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            sections.append(Section(name=current, text=" ".join(buffer).strip()))

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
        raise CorpusFormatError(f"{path.name}: no '## Heading' sections")

    return Article(
        doc_id=path.stem,
        title=meta["title"],
        specialty=meta["specialty"],
        code=meta["code"],
        updated=int(meta["updated"]),
        sections=tuple(sections),
        title_terms=tuple(tokenize(meta["title"])),
        # the specialty is searchable too, so "cardiology" finds the cardiology articles
        body_terms=tuple(tokenize(meta["specialty"])
                         + [t for s in sections for t in tokenize(s.text)]),
    )


def load_articles(directory: Path | str = CORPUS_DIR) -> list[Article]:
    """Every article in the shared corpus, sorted by id."""
    return [parse_article(p) for p in sorted(Path(directory).glob("*.txt"))]


@lru_cache(maxsize=1)
def get_articles() -> tuple[Article, ...]:
    """The loaded corpus. Built once for the whole process."""
    return tuple(load_articles())


def corpus_stats() -> dict:
    """Numbers for the home page, the console header and the docs."""
    articles = get_articles()
    return {
        "articles": len(articles),
        "sections": sum(len(a.sections) for a in articles),
        "specialties": len({a.specialty for a in articles}),
        "section_names": sorted({s.name for a in articles for s in a.sections}),
        "vocabulary": len({t for a in articles for t in a.title_terms + a.body_terms}),
    }


def main() -> None:
    stats = corpus_stats()
    print(f"{stats['articles']} articles | {stats['sections']} sections | "
          f"{stats['specialties']} specialties | {stats['vocabulary']} stems")
    print(f"sections: {', '.join(stats['section_names'])}")
    for article in get_articles():
        print(f"  {article.doc_id}  {article.code:5s} {article.title}")


if __name__ == "__main__":
    main()
