"""Shared data layer for the mini search engine.

This module holds the retrieval logic that both front-ends use:

    main.py    command line interface
    app.py     Streamlit web interface

Keeping it separate means neither front-end imports the other, and the two can
never disagree about what the engine returns.

It prints nothing and imports no UI framework, so it can be driven from a
terminal, a web server, or a test.

The engine is built once and cached, because building the inverted index,
the IDF table and the TF-IDF weights takes about 0.2 seconds. A web app that
rebuilt it on every click would feel slow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from boolean_search import BooleanSyntaxError
from boolean_search import search as boolean_search
from cosine import cosine_similarity, rank_documents
from idf import IdfModel, build_idf_model
from index import InvertedIndex, build_index
from preprocess import get_stop_words, preprocess_query
from tfidf import TfidfModel, build_tfidf_model

BOOLEAN_WORDS = {"AND", "OR", "NOT"}

EXAMPLE_QUERIES = [
    "machine learning healthcare",
    "artificial intelligence",
    "medical imaging",
    "neural network",
    "security attack anomaly",
]

EXAMPLE_BOOLEAN_QUERIES = [
    "machine learning AND healthcare",
    "artificial intelligence OR robotics",
    "machine learning AND NOT finance",
    "healthcare AND diagnosis",
    "computer vision OR medical imaging",
    "security AND machine learning",
    "NLP AND education",
    "(artificial intelligence OR robotics) AND NOT security",
]


@dataclass(frozen=True)
class Engine:
    """The built corpus. Treat as read-only."""

    index: InvertedIndex
    idf_model: IdfModel
    tfidf_model: TfidfModel

    @property
    def number_of_documents(self) -> int:
        return self.index.number_of_documents

    @property
    def vocabulary(self) -> list[str]:
        return self.index.vocabulary

    @property
    def total_postings(self) -> int:
        return self.index.total_postings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Build the corpus once. Repeated calls return the identical object."""
    index = build_index()
    idf_model = build_idf_model(index)
    tfidf_model = build_tfidf_model(index, idf_model)
    return Engine(index=index, idf_model=idf_model, tfidf_model=tfidf_model)


@dataclass(frozen=True)
class Hit:
    """One retrieved document."""

    rank: int
    doc_id: str
    title: str
    score: float
    length: int
    matched_terms: list[tuple[str, float]] = field(default_factory=list)
    satisfies_and: bool = False


@dataclass(frozen=True)
class SearchResponse:
    """Everything a front-end needs to display one query's results."""

    query: str
    mode: str                       # 'ranked' | 'boolean'
    terms: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    boolean_ids: list[str] = field(default_factory=list)
    unknown_terms: list[str] = field(default_factory=list)
    error: str | None = None
    boolean_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def count(self) -> int:
        return len(self.hits)


def is_boolean(query: str) -> bool:
    """True when the user used Boolean operators or parentheses."""
    tokens = query.replace("(", " ( ").replace(")", " ) ").split()
    return any(token.upper() in BOOLEAN_WORDS or token in {"(", ")"} for token in tokens)


def boolean_form(query: str) -> str:
    """The strict AND reading of a query, built from PREPROCESSED terms.

    Splitting raw words would turn 'machine learning healthcare' into
    'machine AND learning AND healthcare', which matches nothing, because
    'machine learning' is stored as a single phrase term.
    """
    terms = [t for t in preprocess_query(query) if t not in get_stop_words()]
    return " AND ".join(terms)


def search(query: str, top: int | None = None, mode: str = "auto") -> SearchResponse:
    """Run one query. Returns data and prints nothing.

    mode 'auto'     Boolean operators present means boolean, otherwise ranked
    mode 'ranked'   always cosine ranking
    mode 'boolean'  always exact Boolean retrieval
    """
    query = query.strip()
    if mode not in {"auto", "ranked", "boolean"}:
        return SearchResponse(query=query, mode="", error=f"unknown mode {mode!r}")

    if not query:
        return SearchResponse(query=query, mode="", error="Query is empty.")

    chosen = mode
    if mode == "auto":
        chosen = "boolean" if is_boolean(query) else "ranked"

    if chosen == "boolean":
        return _search_boolean(query, top)
    return _search_ranked(query, top)


def _search_boolean(query: str, top: int | None) -> SearchResponse:
    try:
        documents = boolean_search(get_engine().index, query)
    except BooleanSyntaxError as error:
        return SearchResponse(query=query, mode="boolean", error=str(error))

    ids = sorted(d.doc_id for d in documents)
    terms = [t for t in preprocess_query(query) if t not in get_stop_words()]
    hits = [
        Hit(rank=position, doc_id=d.doc_id, title=d.title, score=0.0,
            length=len(d.terms), satisfies_and=True)
        for position, d in enumerate(documents[:top] if top else documents, 1)
    ]
    return SearchResponse(query=query, mode="boolean", terms=terms, hits=hits,
                          boolean_ids=ids)


def _search_ranked(query: str, top: int | None) -> SearchResponse:
    engine = get_engine()
    from vsm import build_query_vector

    vector = build_query_vector(query, engine.idf_model)
    terms = sorted(vector.terms)
    if not terms:
        return SearchResponse(query=query, mode="ranked", terms=[],
                              unknown_terms=vector.unknown_terms,
                              error="No searchable terms in that query.")

    results = rank_documents(vector, engine.index, engine.tfidf_model)
    if top:
        results = results[:top]

    and_form = boolean_form(query)
    and_ids: set[str] = set()
    and_error = ""
    try:
        and_ids = {d.doc_id for d in boolean_search(engine.index, and_form)}
    except BooleanSyntaxError as error:
        and_error = str(error)

    hits = [
        Hit(rank=r.rank, doc_id=r.doc_id, title=r.title, score=r.score,
            length=r.length, matched_terms=r.matched_terms,
            satisfies_and=r.doc_id in and_ids)
        for r in results
    ]
    return SearchResponse(query=query, mode="ranked", terms=terms, hits=hits,
                          boolean_ids=sorted(and_ids),
                          unknown_terms=vector.unknown_terms,
                          boolean_error=and_error or None)


def similarity(query: str, doc_id: str) -> float:
    """Cosine similarity between a query and one document."""
    engine = get_engine()
    from vsm import build_query_vector, document_vector

    vector = build_query_vector(query, engine.idf_model)
    return cosine_similarity(vector.terms,
                             document_vector(doc_id, engine.tfidf_model))


def document_terms(doc_id: str, limit: int = 15) -> list[tuple[str, int, float, float]]:
    """(term, tf, idf, tfidf) for a document's heaviest terms."""
    engine = get_engine()
    document = next((d for d in engine.index.documents if d.doc_id == doc_id), None)
    if document is None:
        return []
    weights = engine.tfidf_model.document_terms(doc_id)
    rows = [
        (term, document.terms.count(term), engine.idf_model.idf_for(term), weight)
        for term, weight in weights.items()
    ]
    rows.sort(key=lambda row: (-row[3], row[0]))
    return rows[:limit]


def corpus_stats() -> dict:
    """Numbers for the corpus tab and the sidebar."""
    engine = get_engine()
    idf_values = list(engine.idf_model.idf.values())
    total_terms = sum(len(d.terms) for d in engine.index.documents)
    top = sorted(engine.idf_model.document_frequency.items(),
                 key=lambda kv: (-kv[1], kv[0]))[:15]
    return {
        "documents": engine.number_of_documents,
        "vocabulary": len(engine.vocabulary),
        "postings": engine.total_postings,
        "total_terms": total_terms,
        "terms_per_document": (
            min(engine.index.doc_lengths.values()),
            max(engine.index.doc_lengths.values()),
        ),
        "idf_min": min(idf_values),
        "idf_max": max(idf_values),
        "hapax": sum(1 for df in engine.idf_model.document_frequency.values()
                     if df == 1),
        "top_by_df": top,
    }
