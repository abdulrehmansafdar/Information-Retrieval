"""Step 0 - the document collection: the shared clinical corpus.

The articles live once, at the repository root, in corpus/articles. This module reads
them from there and never keeps a copy. Each week imports no code from another week,
so the small loader is repeated here rather than imported from week06.

A document is its title and its four sections. The specialty is NOT indexed: it is
kept as a label, and semantic_eval.py uses it as the independent answer key for
"was this article relevant?".

Run:  python clinical_docs.py
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

# The same stop list as week 6, so the two modules read the corpus the same way.
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
class Article:
    """One clinical article. `terms` is every stem of its title and sections, in order."""

    doc_id: str
    title: str
    specialty: str
    code: str
    sections: tuple[tuple[str, str], ...]
    terms: tuple[str, ...]


class CorpusFormatError(ValueError):
    """An article file is unreadable. Always raised with the file name."""


def parse_article(path: Path | str) -> Article:
    """Read one article file into an Article."""
    path = Path(path)
    meta: dict[str, str] = {}
    sections: list[tuple[str, str]] = []
    in_header = True
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if in_header:
            if not stripped:
                in_header = False
                continue
            match = META_RE.match(stripped)
            if not match:
                raise CorpusFormatError(f"{path.name}: unexpected line {stripped[:40]!r}")
            meta[match.group(1).lower()] = match.group(2)
            continue
        heading = HEADING_RE.match(stripped)
        if heading:
            sections.append((heading.group(1), ""))
        elif stripped and sections:
            name, text = sections[-1]
            sections[-1] = (name, f"{text} {stripped}".strip())

    missing = [key for key in REQUIRED if key not in meta]
    if missing:
        raise CorpusFormatError(f"{path.name}: missing {', '.join(missing)}")
    if not sections:
        raise CorpusFormatError(f"{path.name}: no '## Heading' sections")

    terms = tokenize(meta["title"]) + [t for _, text in sections for t in tokenize(text)]
    return Article(doc_id=path.stem, title=meta["title"], specialty=meta["specialty"],
                   code=meta["code"], sections=tuple(sections), terms=tuple(terms))


@lru_cache(maxsize=1)
def get_articles() -> tuple[Article, ...]:
    """Every article in the shared corpus, sorted by id. Read once per process."""
    return tuple(parse_article(p) for p in sorted(CORPUS_DIR.glob("*.txt")))


def main() -> None:
    articles = get_articles()
    vocabulary = {t for a in articles for t in a.terms}
    print(f"{len(articles)} articles | {len(vocabulary)} terms")
    for article in articles:
        print(f"  {article.doc_id}  {article.specialty:20s} {article.title}")


if __name__ == "__main__":
    main()
