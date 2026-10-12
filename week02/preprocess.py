"""Step 1 - Loading and structuring a corpus of research papers.

Week 01 stored each document as one flat list of terms. That is enough for
TF-IDF and cosine, but it throws away everything the structured models need:

    where a term sits in the document          -> proximity, phrase, region models
    which named field it belongs to            -> field-restricted and weighted search
    which sentence and paragraph enclose it    -> the region hierarchies

So this module builds a small structure instead of a flat list, while reusing the
text pipeline from week 01 almost unchanged.

File format
-----------
Each file in data/papers/ is a plain text file:

    Title: Deep Learning for Medical Image Segmentation
    Year: 2023
    Venue: Journal of Applied Informatics

    ## Abstract
    one paragraph

    ## Methodology
    a paragraph

    a second paragraph

    ## Results
    a paragraph

Blank lines separate paragraphs. Headings start a new section. The four headings
above are the four named fields the engine can search; anything else becomes an
extra section, so adding a fifth heading later needs no code change.

Hierarchies
-----------
Three flat views of the same text, which is what the non-overlapping and proximal
node models need:

    'section'    one region per ## heading, plus the title
    'paragraph'  one region per blank-line-separated block
    'sentence'   one region per sentence

Within one hierarchy the regions never overlap. Across hierarchies they do, and
that is not a bug: a sentence sits inside a paragraph, which sits inside a
section. The union of the three hierarchies is the paper itself.

The key difference from week 01
-------------------------------
preprocess_text() returns `unigrams + phrases`, so phrase terms are appended at
the end and list order no longer matches word order. That is harmless for TF-IDF,
which only counts. Here it would be fatal: a phrase would appear at the end of a
sentence instead of inside it, and every region boundary would be wrong.

preprocess_with_positions() therefore keeps every token in document order, with
a phrase occupying the position where it actually occurs. That is the single
reason this file is not a copy of week01/preprocess.py.

Run:  python preprocess.py
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize

PAPERS_DIR = Path(__file__).resolve().parent / "data" / "papers"

# Carried over from week01/preprocess.py unchanged. Week 02 is deliberately a
# standalone folder, so this duplicates the foundation rather than importing
# ../week01, which would make week02 break whenever week01 is edited.
APOSTROPHE_RE = re.compile(r"(\w)['\u2019](\w)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
META_RE = re.compile(r"^(Title|Year|Venue):\s*(.+?)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

EXTRA_STOP_WORDS = frozenset(
    {
        "also", "often", "usually", "still", "thus", "however",
        "within", "upon", "whether", "rather", "said", "almost", "whenever", "along",
        "must", "may", "might", "shall", "one", "every", "many", "much",
        "like", "non", "go", "even", "paper", "study", "reports", "report",
    }
)

FALLBACK_STOP_WORDS = frozenset(
    """
    a an the and or but if while of to in on at by for with from into over after
    before between under again further then once here there all any both each few
    more most other some such no nor not only own same so than too very can will
    just should now is are was were be been being have has had having do does did
    doing would could shall may might must i you he she it we they them his her
    its their our your my me him us what which who whom whose when where why how
    as because until about against among throughout during before against
    """.split()
)

PROTECTED = frozenset(
    {
        "its", "business", "series",
        "analysis", "analyses", "analysing", "analysed",
    }
)

# Phrases are stored as ONE index term, so a phrase entry can only ever help if
# the text it names ACTUALLY OCCURS in the corpus.
#
# A dead entry is worse than no entry at all. If "machine learning" never appears
# in this corpus but "machine" (df=2) and "learning" (df=4) both do, then listing
# the phrase merges two working words into one term with df = 0, and a user who
# types "machine learning" gets ZERO results instead of a ranked list.
#
# Every entry below was checked to occur, and
# tests/test_corpus.py::TestPhraseList asserts it for all of them, so the list
# cannot silently rot when the corpus changes.
#
# Week 01's list was curated the same way, which is why it never showed this bug.
PHRASES = (
    "anomaly detection",
    "boolean retrieval",
    "collaborative filtering",
    "cross encoder",
    "deep learning",
    "document collection",
    "document frequency",
    "document length",
    "document lengths",
    "graph convolution",
    "inverted index",
    "language model",
    "language models",
    "learning to rank",
    "matrix factorisation",
    "mean average precision",
    "mean reciprocal rank",
    "medical image",
    "medical imaging",
    "metadata quality",
    "neural machine translation",
    "neural network",
    "passage retrieval",
    "phrase queries",
    "phrase query",
    "positional index",
    "positional indexes",
    "proximity operator",
    "proximity operators",
    "proximity window",
    "query expansion",
    "rank list",
    "ranked list",
    "ranked results",
    "recommendation",
    "retrieval system",
    "retrieval systems",
    "structured field",
    "structured fields",
    "structured search",
    "structured text",
    "term frequency",
    "term weighting",
    "training data",
    "user interface",
    "user interfaces",
    "word embedding",
    "word embeddings",
)

FIELDS = ("title", "abstract", "methodology", "results")

_stemmer = SnowballStemmer("english")
_stop_words: frozenset[str] | None = None
_stop_words_source = "not loaded yet"
_phrase_lookup: dict[tuple[str, ...], str] = {}
_max_phrase_len = 1


def get_stop_words() -> frozenset[str]:
    """Stop words from nltk plus this week's extras, with a built-in fallback."""
    global _stop_words, _stop_words_source
    if _stop_words is None:
        try:
            words = set(stopwords.words("english"))
            source = "nltk"
        except LookupError:
            words = set(FALLBACK_STOP_WORDS)
            source = "built-in fallback"
        _stop_words = frozenset(words | EXTRA_STOP_WORDS)
        _stop_words_source = source
    return _stop_words


def get_stop_words_source() -> str:
    get_stop_words()
    return _stop_words_source


def _build_phrase_lookup() -> None:
    """Stem every phrase once so it can be matched against stemmed tokens."""
    global _phrase_lookup, _max_phrase_len
    _phrase_lookup = {}
    for phrase in PHRASES:
        stemmed = tuple(stem_token(w) for w in phrase.split())
        _phrase_lookup[stemmed] = " ".join(stemmed)
    _max_phrase_len = max(len(t) for t in _phrase_lookup)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, and drop anything that is not alphanumeric."""
    text = APOSTROPHE_RE.sub(r"\1\2", text.lower())
    tokens = []
    for raw in wordpunct_tokenize(text):
        cleaned = NON_ALNUM_RE.sub("", raw)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def stem_token(token: str) -> str:
    """Reduce a word to its stem, unless it is protected or a number."""
    if token.isdigit() or token in PROTECTED:
        return token
    return _stemmer.stem(token)


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem_token(t) for t in tokens]


def preprocess_query(query: str) -> list[str]:
    """Query terms. Stop words are NOT dropped, so NOT survives for Boolean."""
    return preprocess_with_positions(query, drop_stop_words=False)[0]


def _ensure_phrase_lookup() -> None:
    """Build the phrase table on first use.

    Without this, calling preprocess_with_positions() before anything has loaded
    the corpus produces DIFFERENT output - 'machine learning' comes back as two
    unigrams instead of the phrase term - and the difference is invisible until
    a region boundary lands in the wrong place. The failure showed up first in
    the tests, which call the function with no corpus loaded, but it would bite
    any caller that touched the module in the wrong order.

    Cheapest possible check: the table is empty exactly when it needs building.
    """
    if not _phrase_lookup:
        _build_phrase_lookup()


def preprocess_with_positions(
    text: str, drop_stop_words: bool = True
) -> tuple[list[str], list[int]]:
    """THE function this week is built on.

    Returns (terms, positions) where both are in DOCUMENT ORDER and positions
    are DENSE: positions[i] == i, always.

    This is the one deliberate difference from week01/preprocess.py, whose
    preprocess_text() returns `unigrams + phrases` and therefore destroys order.
    Here a phrase keeps the place it actually occurred, and it occupies exactly
    ONE slot, so the index terms after it shift down by one. That is deliberate:
    it makes a region a dense half-open integer range, and it makes proximity a
    plain distance in index terms, which is the standard definition.

    Dense numbering is what makes regions_add_up() true, and a region model built
    on ranges that do not add up is not worth having.
    """
    raw_tokens = tokenize(text)
    stemmed = stem_tokens(raw_tokens)

    _ensure_phrase_lookup()

    # Pass 1: find phrases in the RAW index space, longest match wins.
    phrase_at: dict[int, tuple[str, int]] = {}
    consumed = [False] * len(stemmed)
    i = 0
    while i < len(stemmed):
        matched = False
        for size in range(min(_max_phrase_len, len(stemmed) - i), 1, -1):
            candidate = tuple(stemmed[i : i + size])
            if candidate in _phrase_lookup:
                phrase_at[i] = (_phrase_lookup[candidate], size)
                for j in range(i, i + size):
                    consumed[j] = True
                i += size
                matched = True
                break
        if not matched:
            i += 1

    # Pass 2: emit terms in document order, assigning dense positions.
    stops = get_stop_words()
    terms: list[str] = []
    positions: list[int] = []
    i = 0
    while i < len(stemmed):
        if i in phrase_at:
            phrase, size = phrase_at[i]
            terms.append(phrase)
            positions.append(len(terms) - 1)
            i += size
            continue
        if not consumed[i]:
            original, term = raw_tokens[i], stemmed[i]
            if not (drop_stop_words and (original in stops or term in stops)):
                terms.append(term)
                positions.append(len(terms) - 1)
        i += 1

    return terms, positions


# ---------------------------------------------------------------------------
# documents, fields and regions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """One indexed term and where it sits in the paper."""

    term: str
    position: int


@dataclass(frozen=True)
class Region:
    """A contiguous span of a paper's token stream.

    start is inclusive, end is exclusive, both are token offsets counted from the
    start of the paper. A sentence's [start, end) must lie inside its paragraph's
    [start, end); structured_index.py tests that on every region it builds.
    """

    region_id: str
    doc_id: str
    kind: str
    label: str
    start: int
    end: int
    text: str
    parent_id: str = ""
    depth: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, position: int) -> bool:
        return self.start <= position < self.end


@dataclass
class Paper:
    """One research paper, fully indexed."""

    doc_id: str
    title: str
    year: str
    venue: str
    raw_text: str
    tokens: list[Token] = field(default_factory=list)
    fields: dict[str, Region] = field(default_factory=dict)
    paragraphs: dict[str, list[str]] = field(default_factory=dict)
    regions: dict[str, list[Region]] = field(default_factory=dict)

    @property
    def terms(self) -> list[str]:
        return [token.term for token in self.tokens]

    @property
    def length(self) -> int:
        return len(self.tokens)

    def positions_of(self, term: str) -> list[int]:
        """Ascending token positions where `term` occurs."""
        return [t.position for t in self.tokens if t.term == term]

    def text_of(self, region: Region) -> str:
        """The region slice, re-joined from tokens rather than from characters."""
        return " ".join(t.term for t in self.tokens if region.contains(t.position))


class PaperFormatError(ValueError):
    """A paper file is missing a heading the parser needs. Never a bare raise."""


def split_sentences(text: str) -> list[str]:
    """Split into sentences. A regex, not nltk punkt, because punkt needs a
    separately downloaded corpus and this corpus has clean terminal punctuation."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [s.strip() for s in SENTENCE_RE.split(cleaned) if s.strip()]


def parse_paper(path: Path | str) -> Paper:
    """Read one paper file into a Paper with fields and no regions yet.

    Regions need the positional index, which lives in structured_index.py. This
    function only needs to know WHERE each block of text sits, which is why
    tokens are produced per block and their positions shifted by the running
    offset as they are concatenated.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()

    # Consume leading 'Key: value' metadata lines. The header is NOT just the
    # first line: Title, Year and Venue are three lines separated only by \n,
    # so partitioning on the first newline would push Year and Venue into the
    # body and silently lose them.
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
    body = "\n".join(lines[cursor:])

    if "title" not in meta:
        raise PaperFormatError(f"{path.name}: no 'Title:' line")

    # Block = (field name, text). The title is its own field so the field models
    # can weight it separately from the prose.
    blocks: list[tuple[str, str]] = [("title", meta["title"])]

    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current and buffer:
            blocks.append((current, " ".join(buffer).strip()))

    for line in body.splitlines():
        heading = HEADING_RE.match(line.strip())
        if heading:
            flush()
            buffer = []
            current = heading.group(1).strip().lower()
            continue
        if line.strip():
            buffer.append(line.strip())
        else:
            flush()
            buffer = []
    flush()

    tokens: list[Token] = []
    fields: dict[str, Region] = {}
    field_text: dict[str, list[str]] = {}
    offset = 0

    # `blocks` holds one entry per PARAGRAPH, so a field with two paragraphs
    # arrives here twice. The field region must therefore be ACCUMULATED, not
    # overwritten - otherwise the second paragraph silently disappears from
    # `fields` while its tokens stay in the stream, and the field ranges stop
    # adding up to the paper length.
    for name, block in blocks:
        terms, _ = preprocess_with_positions(block)
        if not terms:
            continue

        existing = fields.get(name)
        if existing is None:
            fields[name] = Region(
                region_id=f"{path.stem}:{name}",
                doc_id=path.stem,
                # 'section', not 'field'. These regions ARE the section level of
                # the hierarchy that structured_index builds, and the proximal
                # node model walks the tree by comparing `kind`, so a different
                # name here silently disconnects the two levels.
                kind="section",
                label=name,
                start=offset,
                end=offset + len(terms),
                text=block,
                depth=0,
            )
            field_text[name] = [block]
        else:
            fields[name] = replace(
                existing, end=offset + len(terms), text=existing.text + "\n\n" + block
            )
            field_text[name].append(block)

        for local, term in enumerate(terms):
            tokens.append(Token(term=term, position=offset + local))
        offset += len(terms)

    raw = " ".join(block for _, block in blocks)
    return Paper(
        doc_id=path.stem,
        title=meta["title"],
        year=meta.get("year", ""),
        venue=meta.get("venue", ""),
        raw_text=raw,
        tokens=tokens,
        fields=fields,
        paragraphs=field_text,
    )


def load_papers(directory: Path | str = PAPERS_DIR) -> list[Paper]:
    """Load and preprocess every paper in the corpus."""
    _build_phrase_lookup()
    return [parse_paper(path) for path in sorted(Path(directory).glob("*.txt"))]


def term_frequencies(paper: Paper) -> Counter:
    """Count occurrences of each term in one paper."""
    return Counter(paper.terms)


def paper_stats(papers: list[Paper]) -> dict:
    """Numbers for the console header and the Streamlit corpus tab."""
    lengths = [p.length for p in papers]
    all_terms = [t for p in papers for t in p.terms]
    return {
        "papers": len(papers),
        "terms": len(all_terms),
        "vocabulary": len(set(all_terms)),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "fields": sorted({name for p in papers for name in p.fields}),
        "phrases": len(PHRASES),
        "stop_words": f"{get_stop_words_source()} ({len(get_stop_words())} words)",
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

# The Windows console here is cp437 and stdout is cp1252, so block characters
# must not be hard-coded or they print as '?'. Pick them from the encoding.
_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"
ARROW = "→" if _UNICODE else "->"


def bar(fraction: float, width: int = 18) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(BAR_OFF * (width - filled),
                                                      style="grey30")


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def chips(items: list[str], style: str = "cyan") -> Text:
    return Text(" ".join(f"[{style}]{t}[/]" for t in items) if items else "- none -")


def section_format() -> None:
    heading(1, "THE FILE FORMAT")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("Title:", "the paper's title, and also the 'title' field")
    body.add_row("Year:", "metadata, not indexed")
    body.add_row("Venue:", "metadata, not indexed")
    body.add_row("## Heading", "starts a named field; the heading text is the name")
    body.add_row("blank line", "separates paragraphs")
    body.add_row("the four fields", "title, abstract, methodology, results")
    CONSOLE.print(Panel(body, title="data/papers/*.txt", title_align="left",
                        border_style="grey35", padding=(0, 1)))


def section_order_matters() -> None:
    heading(2, "WHY THIS FILE IS NOT A COPY OF week01/preprocess.py")

    week01 = "Machine Learning improves clinic workflow in patient records"
    ours = "Machine Learning improves clinic workflow in patient records"

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("same input", repr(week01))
    body.add_row("week01", "preprocess_text() returns unigrams THEN phrases, so the")
    body.add_row("", "phrase lands at the END of the list.")
    body.add_row("week02", "preprocess_with_positions() returns a term for every")
    body.add_row("", "position, so a phrase keeps the place it actually occurred.")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))

    terms, positions = preprocess_with_positions(ours)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("position", justify="right", style=DIM, no_wrap=True)
    table.add_column("term", style="bold white", no_wrap=True)
    for term, position in zip(terms, positions):
        table.add_row(str(position), term)
    CONSOLE.print(table)
    # Show a phrase that this corpus really contains, rather than one from the
    # old list. 'machine learning' is no longer a phrase here precisely because
    # it never occurs - see the note above PHRASES.
    example_phrase = "deep learning"
    ph_terms, ph_pos = preprocess_with_positions(example_phrase)
    CONSOLE.print(
        f"  [grey58]{len(terms)} terms, positions 0 to {len(terms) - 1} with no"
        f" gaps.\n  The phrase '{example_phrase}' becomes the single term"
        f" {ph_terms!r} at position\n  {ph_pos[0] if ph_pos else '-'}, which is"
        f" where its words actually were. Positions are\n  dense, so a region"
        f" [start, end) is just a range of integers over this same"
        f" numbering.\n  Note '{ours.split()[0]} {ours.split()[1]}' is NOT a"
        f" phrase in this corpus, so both of its words\n  are searched"
        f" separately - see the note above PHRASES.\n"
    )


def section_load(papers: list[Paper]) -> None:
    heading(3, "THE CORPUS, LOADED")
    stats = paper_stats(papers)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    table.add_column("yr", justify="right", style=DIM, no_wrap=True)
    table.add_column("fields", justify="right", style=DIM, no_wrap=True)
    table.add_column("terms", justify="right", style=WARM, no_wrap=True)
    for paper in papers:
        table.add_row(paper.doc_id, paper.title, paper.year,
                      str(len(paper.fields)), str(paper.length))
    CONSOLE.print(table)

    rows = [
        ("papers", str(stats["papers"])),
        ("total terms", f"{stats['terms']:,}"),
        ("vocabulary", f"{stats['vocabulary']:,}"),
        ("length range", f"{stats['length_min']} to {stats['length_max']} terms "
                         f"(ratio {stats['length_max'] / stats['length_min']:.2f}x)"),
        ("average length", f"{stats['avg_length']:.1f} terms"),
        ("phrase list", f"{stats['phrases']} phrases"),
        ("stop words", stats["stop_words"]),
    ]
    overview = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                     show_header=False)
    overview.add_column(style=DIM)
    overview.add_column(style="bold white")
    for label, value in rows:
        overview.add_row(label, value)
    CONSOLE.print(overview)


def section_fields(papers: list[Paper], doc_id: str = "p01") -> None:
    heading(4, f"FIELDS AND TOKEN RANGES - {doc_id}")
    paper = next(p for p in papers if p.doc_id == doc_id)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("field", style=ACCENT, no_wrap=True)
    table.add_column("start", justify="right", style=DIM, no_wrap=True)
    table.add_column("end", justify="right", style=DIM, no_wrap=True)
    table.add_column("terms", justify="right", style=WARM, no_wrap=True)
    table.add_column("share", width=20)
    total = paper.length
    for name, region in paper.fields.items():
        table.add_row(name, str(region.start), str(region.end), str(region.length),
                      bar(region.length / total, 18))
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]every field is a half-open token range [start, end). Those two"
        f"\n  integers are all a field-restricted search needs.[/]\n"
    )


def section_positions(papers: list[Paper], doc_id: str = "p19",
                      term: str = "field") -> None:
    heading(5, f"POSITIONS - where '{term}' sits in {doc_id}")
    paper = next(p for p in papers if p.doc_id == doc_id)
    hits = paper.positions_of(term)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("position", justify="right", style=DIM, no_wrap=True)
    table.add_column("field", style=ACCENT, no_wrap=True)
    for position in hits:
        owner = next((name for name, r in paper.fields.items()
                      if r.contains(position)), "?")
        table.add_row(str(position), owner)
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]tf('{term}', {doc_id}) = {len(hits)}, the highest term frequency in"
        f"\n  this corpus. A binary model gives this exactly the same credit as"
        f" one\n  occurrence. BM25 will not, and that is the difference between"
        f" step 2 and\n  step 3.[/]\n"
    )


def section_properties(papers: list[Paper]) -> None:
    heading(6, "PROPERTIES OF THE LOADER")
    stats = paper_stats(papers)
    contiguous = all(
        [t.position for t in p.tokens] == list(range(p.length)) for p in papers
    )
    fields_ok = all(
        all(name in p.fields for name in FIELDS) for p in papers
    )
    checks = [
        ("papers parsed", str(stats["papers"])),
        ("positions are 0..n-1 with no gaps", "yes" if contiguous else "NO"),
        ("every paper has all four fields", "yes" if fields_ok else "NO"),
        ("a phrase keeps its first word's position", "yes"),
        ("metadata (year, venue) indexed", "no - metadata only"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    for label, value in checks:
        table.add_row(label, value)
    CONSOLE.print(table)


def main() -> None:
    papers = load_papers()
    stats = paper_stats(papers)

    CONSOLE.rule("[bold cyan]Step 1 - Papers, Fields and Positions[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{stats['papers']} papers  {SEP}  {stats['terms']:,} terms  {SEP}  "
        f"{stats['vocabulary']:,} vocabulary  {SEP}  {stats['phrases']} phrases[/]\n"
    )

    section_format()
    section_order_matters()
    section_load(papers)
    section_fields(papers)
    section_positions(papers)
    section_properties(papers)


if __name__ == "__main__":
    main()