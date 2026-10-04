"""Step 2 - Preprocessing for the university IR mini search engine.

Converts the 20 raw documents in data/documents/ into clean lists of terms.

Pipeline
--------
    raw text
        -> lowercase
        -> tokenization                     (nltk wordpunct_tokenize)
        -> punctuation removal
        -> stemming                         (nltk SnowballStemmer, english)
        -> phrase extraction                (multi-word terms kept as one term)
        -> stop word removal                (nltk english list + academic extras)
        -> final terms

Nothing is written to disk. Later steps (inverted index, TF-IDF, ...) import the
functions in this file instead of re-implementing preprocessing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DOCUMENTS_DIR = Path(__file__).resolve().parent / "data" / "documents"

APOSTROPHE_RE = re.compile(r"(\w)['\u2019](\w)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

EXTRA_STOP_WORDS = frozenset(
    {
        # standard English function words
        "also", "often", "usually", "still", "thus", "however",
        "within", "upon", "whether", "rather", "said", "almost", "whenever", "along",
        # academic filler that survives the nltk list
        "must", "may", "might", "shall", "one", "every", "many", "much",
        "like", "non", "go", "even",
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
        # words the Snowball stemmer would either damage or merge with another
        # word: its -> it, business -> busi (like busy), series -> seri
        "its", "business", "series",
        # the analysis family: Snowball maps these to three different stems, so
        # they are kept as dictionary words to stay predictable
        "analysis", "analyses", "analysing", "analysed",
    }
)

PHRASES = (
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "computer vision",
    "computer science",
    "medical imaging",
    "medical data",
    "patient records",
    "natural language",
    "large language model",
    "language model",
    "tokenization",
    "transformer",
    "anomaly detection",
    "data science",
    "data set",
    "cloud computing",
    "internet of things",
    "training data",
    "real world",
)

_stemmer = SnowballStemmer("english")
_stop_words: frozenset[str] | None = None
_stop_words_source = "not loaded yet"
_phrase_lookup: dict[tuple[str, ...], str] = {}
_max_phrase_len = 1


def get_stop_words() -> frozenset[str]:
    """Stop words from nltk, plus our academic extras. Falls back to a small
    built-in list when the nltk 'stopwords' corpus has not been downloaded."""
    global _stop_words, _stop_words_source
    if _stop_words is None:
        try:
            words = set(stopwords.words("english"))
            source = "nltk"
        except LookupError:
            words = set(FALLBACK_STOP_WORDS)
            source = "built-in fallback"
            print(
                "WARNING: nltk stopwords corpus not found. Run:\n"
                "    python -m nltk.downloader stopwords"
            )
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


def extract_phrases(tokens: list[str]) -> tuple[list[str], list[bool]]:
    """Find multi-word terms in a list of already stemmed tokens.

    Returns the phrase terms plus a mask that is True for tokens that should
    still be kept as single words. Longest match wins, so 'large language
    model' is preferred over 'language model'.
    """
    phrases: list[str] = []
    keep = [True] * len(tokens)
    i = 0
    while i < len(tokens):
        matched = False
        for size in range(min(_max_phrase_len, len(tokens) - i), 1, -1):
            candidate = tuple(tokens[i : i + size])
            if candidate in _phrase_lookup:
                phrases.append(_phrase_lookup[candidate])
                for j in range(i, i + size):
                    keep[j] = False
                i += size
                matched = True
                break
        if not matched:
            i += 1
    return phrases, keep


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Drop stop words from a list of terms.

    preprocess_text() uses a stricter version of this that tests both the
    surface form and the stem. This helper is kept as part of the public API.
    """
    sw = get_stop_words()
    return [t for t in tokens if t not in sw]


def preprocess_text(text: str, drop_stop_words: bool = True) -> list[str]:
    """Run the whole pipeline on a piece of text and return its terms."""
    raw_tokens = tokenize(text)
    stemmed = stem_tokens(raw_tokens)
    phrases, keep = extract_phrases(stemmed)
    stop_words = get_stop_words()

    unigrams = []
    for original, term, is_kept in zip(raw_tokens, stemmed, keep):
        if not is_kept:
            continue
        # both forms are tested: 'likely' is not a stop word but its stem 'like' is
        if drop_stop_words and (original in stop_words or term in stop_words):
            continue
        unigrams.append(term)
    return unigrams + phrases


def preprocess_query(query: str) -> list[str]:
    """Same pipeline for a query. Stop words are NOT dropped so that words such
    as 'not' survive for Boolean expressions later."""
    return preprocess_text(query, drop_stop_words=False)


@dataclass
class Document:
    doc_id: str
    title: str
    raw_text: str
    terms: list[str] = field(default_factory=list)


def load_documents(directory: Path | str = DOCUMENTS_DIR) -> list[Document]:
    """Read every .txt file. Line 1 is the title, the rest is the body."""
    directory = Path(directory)
    documents = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title, _, body = text.partition("\n")
        documents.append(
            Document(
                doc_id=path.stem,
                title=title.strip(),
                raw_text=body.replace("\n", " ").strip(),
            )
        )
    return documents


def preprocess_document(document: Document) -> Document:
    """Fill in document.terms using the title and the body together."""
    document.terms = preprocess_text(document.title) + preprocess_text(document.raw_text)
    return document


def build_documents(directory: Path | str = DOCUMENTS_DIR) -> list[Document]:
    """Convenience helper: load every document and preprocess it."""
    _build_phrase_lookup()
    return [preprocess_document(d) for d in load_documents(directory)]


def _stats(documents: list[Document]) -> tuple[Counter, Counter, list[str]]:
    all_terms = [t for d in documents for t in d.terms]
    collection_freq = Counter(all_terms)
    document_freq: Counter = Counter()
    for document in documents:
        document_freq.update(set(document.terms))
    return collection_freq, document_freq, all_terms


# ---------------------------------------------------------------------------
# console output
# ---------------------------------------------------------------------------

CONSOLE = Console()

ACCENT = "bold cyan"
DIM = "grey58"

# Windows console is cp437 and stdout is cp1252, so block characters must not be
# hard-coded or they print as '?'. Pick them from the detected encoding.
_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"


def bar(fraction: float, width: int = 24) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(
        BAR_OFF * (width - filled), style="grey30"
    )


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def term_list(terms: list[str], style: str = "") -> str:
    return Text(" ".join(terms), style=style) if style else " ".join(terms)


def section_walkthrough(document: Document) -> None:
    """Trace one document through every stage of the real pipeline."""
    heading(1, f"PIPELINE WALKTHROUGH - {document.doc_id}")
    sentences = [s.strip() for s in document.raw_text.split(".") if s.strip()]
    # a sentence containing punctuation, so the tokenizer stages visibly differ
    sentence = next((s for s in sentences if "," in s or ";" in s), sentences[0])
    if len(sentence.split()) > 40:
        sentence = " ".join(sentence.split()[:40]) + " ..."

    lowered = sentence.lower()
    with_punctuation = wordpunct_tokenize(lowered)
    tokens = tokenize(sentence)
    stemmed = stem_tokens(tokens)
    phrases, keep = extract_phrases(stemmed)
    stop_words = get_stop_words()
    survived = [t for t, k in zip(stemmed, keep) if k and t not in stop_words]

    stages = [
        ("1 raw text", sentence),
        ("2 lowercased", lowered),
        ("3 tokenized", term_list(with_punctuation)),
        ("4 punctuation removed", term_list(tokens)),
        ("5 stemmed", term_list(stemmed)),
        ("6 phrases found", term_list(phrases) or "- none -"),
        ("7 stop words removed", term_list(survived)),
    ]

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("stage", style=ACCENT, no_wrap=True, vertical="top")
    table.add_column("output", style="white", overflow="fold")
    for name, value in stages:
        table.add_row(name, value)
    CONSOLE.print(table)

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM)
    body.add_column(style="white", overflow="fold")
    body.add_row("title", document.title)
    body.add_row("terms", f"{len(document.terms)} total")
    body.add_row("", term_list(document.terms))
    CONSOLE.print(Panel(body, title=f"{document.doc_id} final terms",
                        title_align="left", border_style="grey35",
                        padding=(0, 1)))


def section_all_documents(documents: list[Document]) -> None:
    heading(2, "ALL DOCUMENTS")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    table.add_column("terms", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("distinct", justify="right", style=DIM, no_wrap=True)
    table.add_column("repeated", justify="right", style=DIM, no_wrap=True)

    for document in documents:
        counts = Counter(document.terms)
        repeated = sum(1 for tf in counts.values() if tf > 1)
        table.add_row(document.doc_id, document.title, str(len(document.terms)),
                      str(len(counts)), str(repeated))
    CONSOLE.print(table)


def section_statistics(documents: list[Document], collection_freq: Counter,
                       document_freq: Counter, all_terms: list[str]) -> None:
    heading(3, "CORPUS STATISTICS")
    vocabulary = sorted(collection_freq)
    rows = [
        ("total terms (tokens)", f"{len(all_terms):,}"),
        ("unique terms (vocabulary)", f"{len(vocabulary):,}"),
        ("terms in exactly 1 document",
         f"{sum(1 for t in vocabulary if document_freq[t] == 1):,}"),
        ("terms in all 20 documents",
         f"{sum(1 for t in vocabulary if document_freq[t] == len(documents)):,}"),
        ("stop word list", f"{get_stop_words_source()} ({len(get_stop_words())} words)"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    for label, value in rows:
        table.add_row(label, value)
    CONSOLE.print(table)


def section_collection_frequency(collection_freq: Counter, top: int = 20) -> None:
    heading(4, f"TOP {top} TERMS BY COLLECTION FREQUENCY")
    ranked = collection_freq.most_common(top)
    largest = ranked[0][1]

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("count", justify="right", style=ACCENT, no_wrap=True)
    table.add_column("share", width=26)
    for term, count in ranked:
        table.add_row(term, str(count), bar(count / largest, 24))
    CONSOLE.print(table)


def section_stemming_probe(documents: list[Document], collection_freq: Counter,
                           document_freq: Counter) -> None:
    heading(5, "STEMMING PROBE")
    probe = [
        "machine", "learning", "models", "applications", "images",
        "diagnosis", "diagnosed", "networks", "neurons", "analyses",
    ]
    # words that actually occur in the raw text, before any preprocessing
    raw_words = set()
    for document in documents:
        for text in (document.title, document.raw_text):
            raw_words.update(re.findall(r"[A-Za-z]+", text.lower()))

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("surface form", style="white", no_wrap=True)
    table.add_column("became", style="bold cyan", no_wrap=True)
    table.add_column("was in raw text", justify="center", no_wrap=True)
    table.add_column("now in corpus", justify="right", no_wrap=True)
    table.add_column("note", style=DIM, no_wrap=True)

    for word in probe:
        stem = stem_token(word)
        total = collection_freq.get(stem, 0)
        was_present = "yes" if word in raw_words else "no"
        note = "[green]merged[/]" if word in raw_words else "[grey50]never in corpus[/]"
        table.add_row(word, stem, was_present, f"{total:,}", note)
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]'merged' = the word occurs in the raw documents but no longer"
        "\n  exists as its own term; all of its occurrences folded into the stem.[/]\n"
    )


def section_phrases(collection_freq: Counter, document_freq: Counter) -> None:
    heading(6, "PHRASE TERMS (multi-word terms kept as a single index term)")
    phrases = sorted(t for t in collection_freq if " " in t)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("phrase term", style="bold cyan", no_wrap=True)
    table.add_column("occurrences", justify="right", no_wrap=True)
    table.add_column("documents", justify="right", no_wrap=True)
    for phrase in phrases:
        table.add_row(phrase, str(collection_freq[phrase]),
                      str(document_freq[phrase]))
    CONSOLE.print(table)


def section_queries() -> None:
    heading(7, "SAMPLE QUERY PROCESSING")
    queries = [
        "Machine Learning in Healthcare",
        "computer vision NOT finance",
        "anomaly detection",
        "Medical Imaging",
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("query", style="white")
    table.add_column("becomes", style="bold cyan", overflow="fold")
    for query in queries:
        table.add_row(repr(query), term_list(preprocess_query(query)))
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]Queries keep their stop words so 'not' survives for the"
        "\n  Boolean step. Documents drop them.[/]\n"
    )


def main() -> None:
    _build_phrase_lookup()
    documents = [preprocess_document(d) for d in load_documents()]
    collection_freq, document_freq, all_terms = _stats(documents)

    CONSOLE.rule("[bold cyan]Step 2 - Preprocessing[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{len(documents)} documents  {SEP}  "
        f"SnowballStemmer(english)  {SEP}  "
        f"{len(get_stop_words())} stop words  {SEP}  "
        f"{len(PHRASES)} phrases[/]\n"
    )

    section_walkthrough(documents[2])
    section_all_documents(documents)
    section_statistics(documents, collection_freq, document_freq, all_terms)
    section_collection_frequency(collection_freq)
    section_stemming_probe(documents, collection_freq, document_freq)
    section_phrases(collection_freq, document_freq)
    section_queries()


if __name__ == "__main__":
    main()