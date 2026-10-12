"""The shared data layer for the week02 search engine.

Every retrieval model in this week - probability ranking, structured field
search, the two list models and the proximal node model - is reachable through
one function:

    search(query, mode=..., top=...)

The front ends in console_ui.py and app.py call this and nothing else, so the
terminal and the browser can never disagree about what the engine returns. It
prints nothing and imports no UI framework, so it can also be driven from a
test.

Everything expensive is built once and cached. Building the corpus, the
positional index, the IDF tables and the per-field statistics takes about half
a second; a web app that rebuilt them on every keystroke would feel broken.

    mode            what it does
    --------------- ---------------------------------------------------------
    boolean         every query term must appear. No ranking, an exact set.
    tfidf           week01's baseline: TF-IDF vectors and cosine similarity.
    bim             Binary Independence Model, probability ranking
    bm25            BM25, with k1 and b exposed
    fields          field-restricted, with title: and results: prefixes
    bm25f           every field contributes, weighted
    overlapped      every query term inside ONE region
    non-overlapped  query terms in DIFFERENT regions
    proximal        one operator of the paper/section/paragraph/sentence tree

Run:  python engine.py            quick check of every mode
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

from evaluate import Judgements, load_qrels
from fields import DEFAULT_WEIGHTS, FieldQueryError, FieldSearch
from lists import ListModels, ListResult
from proximity import ProximalNodeModel, ProximitySyntaxError
from ranking import (BinaryIndependenceModel, Bm25Model, IdfTable, Query,
                     parse_query)
from structured_index import StructuredIndex, build_index

MODES = ("boolean", "tfidf", "bim", "bm25", "fields", "bm25f",
         "overlapped", "non-overlapped", "proximal")

# Re-exported so the front ends do not have to import structured_index directly
# just to fill a selectbox.
HIERARCHY_NAMES = ("section", "paragraph", "sentence")

MODE_LABELS = {
    "boolean": "exact AND - every query term, no ranking",
    "tfidf": "week01 baseline - TF-IDF vectors, cosine similarity",
    "bim": "Binary Independence Model - probability ranking",
    "bm25": "BM25 - saturated tf, length normalised",
    "fields": "field-restricted - title: and results: prefixes",
    "bm25f": "BM25F - every field weighted and summed",
    "overlapped": "every query term inside ONE region",
    "non-overlapped": "query terms in DIFFERENT regions",
    "proximal": "proximal node operators over the paper tree",
}

EXAMPLE_QUERIES = [
    "proximity window phrase query",
    "structured field search",
    "term weighting saturation",
    "mean average precision",
    "cross encoder reranking passages",
    "document index positions",
]

EXAMPLE_PROXIMAL = [
    "paragraph in methodology",
    "section with(2) retrieval",
    "sentence before(10) index (paragraph)",
    "paragraph child methodology",
    "section parent(2) paragraph",
]

EXAMPLE_FIELD_QUERIES = [
    "title:search",
    "title:search AND results:precision",
    "results:precision",
    "structured field search",
]


@dataclass(frozen=True)
class Hit:
    """One result. Papers and regions both come back as hits.

    A region result fills region_id and snippet; a paper result leaves them
    empty. One type rather than two keeps the UI and the evaluation from
    branching on mode everywhere.
    """

    rank: int
    doc_id: str
    title: str
    score: float
    length: int
    matched: int = 0
    total_terms: int = 0
    field: str = ""
    region_id: str = ""
    snippet: str = ""
    note: str = ""

    @property
    def is_region(self) -> bool:
        return bool(self.region_id)


@dataclass(frozen=True)
class SearchResponse:
    """Everything a front end needs to display one query's results."""

    query: str
    mode: str
    terms: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def count(self) -> int:
        return len(self.hits)


# ---------------------------------------------------------------------------
# the TF-IDF cosine baseline
# ---------------------------------------------------------------------------


class TfidfCosineModel:
    """week01's baseline, rebuilt on this week's index.

    Present so the evaluation has a vector-model column to compare the
    probability models against. Without it, "BM25 beats week01" would be a claim
    about nothing. It uses log(N/df) rather than this week's log-odds variant,
    precisely so it reproduces what week01 actually did.
    """

    def __init__(self, index: StructuredIndex) -> None:
        self.index = index
        self.idf_table = IdfTable(index, "plain")
        self.weights: dict[str, dict[str, float]] = {}
        for paper in index.papers:
            counts: dict[str, int] = {}
            for token in paper.tokens:
                counts[token.term] = counts.get(token.term, 0) + 1
            self.weights[paper.doc_id] = {
                term: tf * self.idf_table.idf_for(term)
                for term, tf in counts.items()
            }

    @property
    def name(self) -> str:
        return "TF-IDF cosine"

    @staticmethod
    def _magnitude(vector: dict[str, float]) -> float:
        return math.sqrt(sum(w * w for w in vector.values()))

    def score_paper(self, doc_id: str, terms: tuple[str, ...]) -> float:
        query = {t: self.idf_table.idf_for(t) for t in terms}
        document = self.weights[doc_id]
        dot = sum(w * document.get(t, 0.0) for t, w in query.items())
        if dot <= 0.0:
            return 0.0
        return dot / (self._magnitude(query) * self._magnitude(document))

    def rank(self, query: str, top: int | None = None) -> list[Hit]:
        parsed = parse_query(query, self.idf_table)
        hits = []
        for paper in self.index.papers:
            score = self.score_paper(paper.doc_id, parsed.terms)
            if score <= 0.0:
                continue
            matched = sum(1 for t in parsed.terms
                          if t in self.weights[paper.doc_id])
            hits.append(Hit(rank=0, doc_id=paper.doc_id, title=paper.title,
                            score=score, length=paper.length, matched=matched,
                            total_terms=len(parsed.terms)))
        return _order(hits, top)


# ---------------------------------------------------------------------------
# the built engine
# ---------------------------------------------------------------------------


@dataclass
class Engine:
    """The built corpus and every model over it. Treat as read-only."""

    index: StructuredIndex
    judgements: Judgements
    idf_table: IdfTable
    bim: BinaryIndependenceModel
    bm25: Bm25Model
    tfidf: TfidfCosineModel
    fields: FieldSearch
    lists: ListModels
    proximal: ProximalNodeModel

    @property
    def number_of_documents(self) -> int:
        return self.index.number_of_documents

    @property
    def vocabulary(self) -> list[str]:
        return self.index.vocabulary

    @property
    def total_postings(self) -> int:
        return self.index.total_postings

    def titles(self) -> dict[str, str]:
        return {p.doc_id: p.title for p in self.index.papers}


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Build everything once. Repeated calls return the identical object."""
    index = build_index()
    variant = "bm25"
    return Engine(
        index=index,
        judgements=load_qrels(),
        idf_table=IdfTable(index, variant),
        bim=BinaryIndependenceModel(index, variant),
        bm25=Bm25Model(index, idf_variant=variant),
        tfidf=TfidfCosineModel(index),
        fields=FieldSearch(index),
        lists=ListModels(index),
        proximal=ProximalNodeModel(index),
    )


def get_index() -> StructuredIndex:
    return get_engine().index


def get_judgements() -> Judgements:
    return get_engine().judgements


def _order(hits: list[Hit], top: int | None) -> list[Hit]:
    """Sort best first and assign 1-based ranks.

    Ties break on doc_id then region_id rather than on whatever order the
    dictionary happened to produce, because evaluate.py's numbers have to be
    reproducible between runs.
    """
    hits.sort(key=lambda h: (-h.score, h.doc_id, h.region_id))
    for position, hit in enumerate(hits, 1):
        object.__setattr__(hit, "rank", position)
    return hits[:top] if top else hits


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search(query: str, mode: str = "bm25", top: int | None = None,
           k1: float = 1.2, b: float = 0.75, hierarchy: str = "paragraph",
           normalise: bool = True) -> SearchResponse:
    """Run one query in one mode. Returns data and prints nothing."""
    query = query.strip()
    if mode not in MODES:
        return SearchResponse(query=query, mode="",
                              error=f"unknown mode {mode!r}; "
                                    f"expected one of {', '.join(MODES)}")
    if not query:
        return SearchResponse(query=query, mode=mode, error="Query is empty.")

    engine = get_engine()
    handler = _HANDLERS[mode]
    try:
        response = handler(engine, query, top, k1, b, hierarchy, normalise)
    except (FieldQueryError, ProximitySyntaxError) as error:
        return SearchResponse(query=query, mode=mode, error=str(error))
    if response.ok and not response.hits and not response.terms:
        note = _nothing_searched_note(query, response)
        if note:
            response.notes.append(note)
    return response


def _nothing_searched_note(query: str, response: SearchResponse) -> str:
    """Explain a zero-hit answer where nothing was searchable at all.

    Without this, "the" returns nothing with no explanation, and the Boolean
    engine - which requires every query term to be present - happily returns
    all 20 papers, because every paper trivially contains zero terms.
    """
    if response.unknown:
        return ""  # coverage_note() already explains a word that is not in the corpus
    return (f"none of the words in {query!r} were searched - they are all stop "
            f"words, so there was nothing left to look for")


def _paper_hits(engine: Engine, query: str, scored, parsed: Query,
                top: int | None, mode: str, parameters: dict) -> SearchResponse:
    hits = [
        Hit(rank=0, doc_id=r.doc_id, title=r.title, score=r.score,
            length=r.length, matched=r.matched,
            total_terms=r.total_query_terms)
        for r in scored
    ]
    ordered = _order(hits, top)
    return SearchResponse(
        query=query, mode=mode, terms=list(parsed.terms), hits=ordered,
        papers=sorted({h.doc_id for h in ordered}),
        unknown=list(parsed.unknown), parameters=parameters,
    )


def _boolean(engine, query, top, k1, b, hierarchy, normalise):
    parsed = parse_query(query, engine.idf_table)
    hits = []
    for paper in engine.index.papers:
        present = {t.term for t in paper.tokens}
        matched = sum(1 for t in parsed.terms if t in present)
        # A query with no terms to require is satisfied by every paper, which is
        # not an answer - 20/20 for "the" is a bug, not a match.
        if not parsed.terms or matched != len(set(parsed.terms)):
            continue
        hits.append(Hit(rank=0, doc_id=paper.doc_id, title=paper.title,
                        score=float(len(parsed.terms)), length=paper.length,
                        matched=matched, total_terms=len(parsed.terms),
                        note="exact AND"))
    ordered = _order(hits, top)
    return SearchResponse(query=query, mode="boolean",
                          terms=list(parsed.terms), hits=ordered,
                          papers=sorted({h.doc_id for h in ordered}),
                          unknown=list(parsed.unknown))


def _tfidf(engine, query, top, k1, b, hierarchy, normalise):
    ranked = engine.tfidf.rank(query, top=top)
    parsed = parse_query(query, engine.tfidf.idf_table)
    return SearchResponse(
        query=query, mode="tfidf", terms=list(parsed.terms),
        hits=ranked, papers=sorted({h.doc_id for h in ranked}),
        unknown=list(parsed.unknown),
        parameters={"idf": "plain log(N/df)", "note": "week01 baseline"},
    )


def _bim(engine, query, top, k1, b, hierarchy, normalise):
    parsed = parse_query(query, engine.bim.idf_table)
    scored = engine.bim.rank(parsed, top=None, idf_table=engine.bim.idf_table)
    return _paper_hits(engine, query, scored, parsed, top, "bim",
                       dict(engine.bim.parameters))


def _bm25(engine, query, top, k1, b, hierarchy, normalise):
    model = engine.bm25.with_parameters(k1=k1, b=b)
    parsed = parse_query(query, model.idf_table)
    scored = model.rank(parsed, top=None, idf_table=model.idf_table)
    return _paper_hits(engine, query, scored, parsed, top, "bm25",
                       dict(model.parameters))


def _fields(engine, query, top, k1, b, hierarchy, normalise):
    result = engine.fields.restrict(query, top=top, k1=k1, b=b,
                                    normalise=normalise)
    return _field_response(result, query, "fields",
                           {"k1": k1, "b": b, "normalise": normalise})


def _bm25f(engine, query, top, k1, b, hierarchy, normalise):
    result = engine.fields.weighted(query, top=top, k1=k1, b=b,
                                    normalise=normalise,
                                    weights=DEFAULT_WEIGHTS)
    return _field_response(result, query, "bm25f",
                           {"k1": k1, "b": b, "normalise": normalise,
                            "weights": dict(DEFAULT_WEIGHTS)})


def _field_response(result, query, mode, parameters) -> SearchResponse:
    hits = [
        Hit(rank=h.rank, doc_id=h.doc_id, title=h.title, score=h.score,
            length=h.field_length, matched=h.matched,
            total_terms=h.total_query_terms, field=h.field,
            region_id=h.region_id)
        for h in result.hits
    ]
    return SearchResponse(query=query, mode=mode, hits=hits,
                          papers=list(result.papers),
                          unknown=list(result.unknown), error=result.error,
                          parameters=parameters)


def _overlapped(engine, query, top, k1, b, hierarchy, normalise):
    result = engine.lists.overlapped(query, hierarchy, top)
    return _list_response(engine, result, query, "overlapped",
                          {"hierarchy": hierarchy})


def _non_overlapped(engine, query, top, k1, b, hierarchy, normalise):
    result = engine.lists.non_overlapped(query, hierarchy, top)
    return _list_response(engine, result, query, "non-overlapped",
                          {"hierarchy": hierarchy})


def _list_response(engine: Engine, result: ListResult, query: str, mode: str,
                   parameters: dict) -> SearchResponse:
    titles = engine.titles()
    # ListRegion carries no rank of its own; the response owns the numbering, so
    # the position is taken from enumerate rather than read off the region.
    hits = [
        Hit(rank=position, doc_id=r.doc_id,
            title=titles.get(r.doc_id, r.doc_id),
            score=r.score, length=r.length, matched=r.matched,
            total_terms=r.total_query_terms, region_id=r.region_id,
            field=r.label, snippet=r.text, note=r.role)
        for position, r in enumerate(result.regions, 1)
    ]
    notes = [
        f"{result.count} regions in {len(result.papers)} papers",
        f"{result.complete_regions} region(s) hold every query term",
    ]
    return SearchResponse(query=query, mode=mode, hits=hits,
                          papers=list(result.papers),
                          unknown=list(result.unknown), error=result.error,
                          notes=notes, parameters=parameters)


def _proximal(engine, query, top, k1, b, hierarchy, normalise):
    titles = engine.titles()
    result = engine.proximal.evaluate(query)
    nodes = result.nodes[:top] if top else result.nodes
    # The proximal model returns a SET of nodes, not a ranking: there is no
    # relevance score attached to "this sentence is a methodology paragraph".
    # score stays 0.0 and the node length rides in the `length` field, so no
    # front end can accidentally sort by it and call the result a ranking.
    hits = [
        Hit(rank=position, doc_id=n.doc_id, title=titles.get(n.doc_id, n.doc_id),
            score=0.0, length=n.length, region_id=n.region_id,
            field=n.kind, snippet=n.text, note=n.label)
        for position, n in enumerate(nodes, 1)
    ]
    return SearchResponse(
        query=query, mode="proximal", hits=hits,
        papers=list(result.papers), error=result.error,
        notes=result.steps + [
            f"{result.count} node(s) - a SET, not a ranking: this model answers"
            f" 'which', not 'in what order'",
        ],
        parameters={"operators": "in, beginin, with(k), parent(k), child, "
                                "before(k), after(k)"},
    )


_HANDLERS = {
    "boolean": _boolean,
    "tfidf": _tfidf,
    "bim": _bim,
    "bm25": _bm25,
    "fields": _fields,
    "bm25f": _bm25f,
    "overlapped": _overlapped,
    "non-overlapped": _non_overlapped,
    "proximal": _proximal,
}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluation_models(index: StructuredIndex | None = None) -> dict:
    """One ranking function per model, for evaluate.py.

    Each takes a QUERY ID, looks the text up in the judgements and returns the
    ranked paper ids. Taking the id rather than the text keeps the ranking
    functions swappable and the query text in exactly one place.
    """
    engine = get_engine()
    judgements = engine.judgements

    def ranking_for(run, description: str = ""):
        def fn(query_id: str) -> list[str]:
            return [h.doc_id for h in run(judgements.queries[query_id])]
        fn.description = description
        return fn

    def boolean(query: str) -> list[Hit]:
        return _boolean(engine, query, None, 1.2, 0.75, "paragraph",
                        True).hits

    def tfidf(query: str) -> list[Hit]:
        return engine.tfidf.rank(query, top=None)

    def bim(query: str) -> list[Hit]:
        return _bim(engine, query, None, 1.2, 0.75, "paragraph", True).hits

    def bm25(query: str) -> list[Hit]:
        return _bm25(engine, query, None, 1.2, 0.75, "paragraph", True).hits

    def bm25f(query: str) -> list[Hit]:
        return _bm25f(engine, query, None, 1.2, 0.75, "paragraph",
                      True).hits

    def overlapped(query: str) -> list[Hit]:
        """Papers that have an overlapped region, best region first.

        The paper is the unit of comparison even though the region is the unit
        of retrieval, because the judgements are about papers.
        """
        result = engine.lists.overlapped(query, "paragraph")
        seen: set[str] = set()
        out: list[Hit] = []
        for region in result.regions:
            if region.doc_id in seen:
                continue
            seen.add(region.doc_id)
            out.append(Hit(rank=0, doc_id=region.doc_id,
                            title=region.doc_id, score=region.score,
                            length=region.length))
        return _order(out, None)

    def non_overlapped(query: str) -> list[Hit]:
        """Papers ranked by how completely their regions cover the query, with
        the best single region breaking ties.

        A paper whose terms all sit in one region outranks one that scattered
        them, which is what makes this different from the overlapped model.
        """
        result = engine.lists.non_overlapped(query, "paragraph")
        coverage: dict[str, set[str]] = {}
        wanted: dict[str, int] = {}
        best: dict[str, float] = {}
        for region in result.regions:
            coverage.setdefault(region.doc_id, set()).update(region.terms)
            wanted[region.doc_id] = region.total_query_terms
            best[region.doc_id] = max(best.get(region.doc_id, 0.0),
                                      region.score)
        out = [
            Hit(rank=0, doc_id=doc_id, title=doc_id,
                score=len(terms) + best[doc_id] / 1000.0,
                length=0, matched=len(terms), total_terms=wanted.get(doc_id, 0))
            for doc_id, terms in coverage.items()
        ]
        return _order(out, None)

    def proximal(query: str) -> list[Hit]:
        """Papers named by `section with(1) <first query term>`.

        Built from the query rather than from the paper's own title, because the
        judgements are paper-level and a node query is the only proximal
        operator that produces a paper ranking at all.
        """
        parsed = parse_query(query, engine.idf_table)
        if not parsed.terms:
            return []
        result = engine.proximal.evaluate(f"section with(1) {parsed.terms[0]}")
        seen: set[str] = set()
        out = []
        for node in result.nodes:
            if node.doc_id in seen:
                continue
            seen.add(node.doc_id)
            out.append(Hit(rank=0, doc_id=node.doc_id, title=node.doc_id,
                            score=float(node.length), length=node.length))
        return _order(out, None)

    return {
        "boolean AND": ranking_for(boolean, "every query term must appear"),
        "tfidf cosine": ranking_for(tfidf, "week01 baseline"),
        "BIM": ranking_for(bim, "probability ranking, binary indicator"),
        "BM25": ranking_for(bm25, "saturated tf, length normalised"),
        "BM25F": ranking_for(bm25f, "per-field BM25, weighted"),
        "overlapped": ranking_for(overlapped, "all terms in one region"),
        "non-overlapped": ranking_for(non_overlapped,
                                      "terms in separate regions"),
        "proximal": ranking_for(proximal, "section with(1) on the first term"),
    }


# ---------------------------------------------------------------------------
# corpus statistics
# ---------------------------------------------------------------------------


def coverage_note(response: SearchResponse) -> str:
    """Say what was NOT searched. Nothing at all when nothing was dropped.

    An earlier version compared search TERMS against the user's WORDS, which is
    wrong in the other direction: a legitimate phrase means two typed words
    become one searched term, and reporting "1 of 2 words" for a perfect match
    on "inverted index" is its own lie. So the note only ever counts DROPPED
    words, which is the one thing the user cannot see for themselves.
    """
    if not response.unknown:
        return ""
    count = len(response.unknown)
    return (f"{count} word{'s' if count != 1 else ''} in your query "
            f"{'were' if count != 1 else 'was'} never searched")


def phrase_hint(response: SearchResponse, engine: "Engine") -> str:
    """Explain each dropped word, when a phrase swallowed it."""
    hints = []
    for term in response.unknown:
        phrases = engine.idf_table.phrase_hints(term)
        if phrases:
            hints.append(f"'{term}' only occurs inside "
                         + " or ".join(f"'{p}'" for p in phrases))
    return "  ".join(hints)


FIELD_MODES = ("fields", "bm25f")


def misrouted_field_prefix(response: SearchResponse) -> str:
    """Warn when a `field:terms` prefix was typed in a mode that ignores it.

    In `bm25`, `bim`, `tfidf` or `boolean`, `title:search` is tokenised into the
    ordinary terms 'titl' and 'search', so the engine searches for the WORD
    "title" and reports six hits with no indication that the field restriction
    was discarded. That is a confident wrong answer to a query the user thought
    was field-restricted.

    `overlapped` and `non-overlapped` behave differently again: they return no
    hits at all, because neither of those terms appears in a single region
    together. Both cases are worth a sentence.
    """
    if response.mode in FIELD_MODES or response.mode == "proximal":
        return ""
    text = response.query
    if not any(f"{name}:" in text.lower() for name in ("title", "abstract",
                                                      "methodology", "results")):
        return ""
    if response.mode in ("overlapped", "non-overlapped"):
        return (f"this query has a field prefix, but {response.mode} searches "
                f"regions, not fields, so 'title' was treated as an ordinary "
                f"word. Use mode 'fields' for a field-restricted search.")
    return (f"this query has a field prefix, but mode '{response.mode}' ranks "
            f"whole papers, so 'title' was searched as an ordinary word. Use "
            f"mode 'fields' or 'bm25f' to restrict to a field.")


def corpus_stats() -> dict:
    """Numbers for the corpus tab, the sidebar and the console header."""
    engine = get_engine()
    index = engine.index
    df = index.document_frequency
    lengths = sorted(index.doc_lengths.values())
    idf_values = list(engine.idf_table.idf.values())
    return {
        "papers": index.number_of_documents,
        "vocabulary": len(index.vocabulary),
        "postings": index.total_postings,
        "positions": index.total_positions,
        "total_terms": sum(lengths),
        "length_min": lengths[0],
        "length_max": lengths[-1],
        "avg_length": sum(lengths) / len(lengths),
        "idf_min": min(idf_values),
        "idf_max": max(idf_values),
        "sections": len(index.regions_of("section")),
        "paragraphs": len(index.regions_of("paragraph")),
        "sentences": len(index.regions_of("sentence")),
        "top_by_df": sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:12],
        "judged_queries": len(engine.judgements.query_ids),
    }


def query_options() -> dict[str, list[str]]:
    """Example queries for the console and the Streamlit sidebar."""
    return {
        "ranked": list(EXAMPLE_QUERIES),
        "field": list(EXAMPLE_FIELD_QUERIES),
        "proximal": list(EXAMPLE_PROXIMAL),
    }


def main() -> None:
    """Run one query in every mode. A smoke test for the whole engine."""
    stats = corpus_stats()

    print(f"{stats['papers']} papers | {stats['vocabulary']} terms | "
          f"{stats['postings']} postings | {stats['positions']} positions")
    print(f"{stats['sections']} sections | {stats['paragraphs']} paragraphs | "
          f"{stats['sentences']} sentences")
    print(f"judged queries: {stats['judged_queries']}\n")

    probes = {"proximal": "paragraph in methodology",
              "fields": "title:search"}
    for mode in MODES:
        probe = probes.get(mode, EXAMPLE_QUERIES[0])
        response = search(probe, mode=mode, top=3)
        state = "ok" if response.ok else f"ERROR {response.error}"
        ids = ", ".join(h.doc_id for h in response.hits) or "-"
        print(f"  {mode:15s} {probe!r:38s} {response.count:3d} hits  "
              f"{ids:18s} {state}")


if __name__ == "__main__":
    main()