"""Step 5 - the search engine both front ends call.

    engine.search("breathing difficulty", Settings(model="lsi"))
    engine.compare(query, settings)           GVSM and LSI on one query
    engine.steps(query, doc_id, settings)     the worked cosine for one document

console_ui.py and app.py only present what this returns, so they cannot disagree. A
question the engine cannot answer comes back as a Response with ok=False and a message,
never as an exception.

Run:  python semantic_search.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from clinical_docs import Article, get_articles, tokenize
from gvsm_model import GvsmModel
from lsi_model import LsiModel, SvdFactors
from term_matrix import TermDocumentMatrix
from vector_cosine import CosineSteps

MODELS = ("gvsm", "lsi")
DEFAULT_K = 15
DEFAULT_TOP = 10


@dataclass(frozen=True)
class Settings:
    model: str = "lsi"
    weighting: str = "tfidf"
    k: int = DEFAULT_K
    top: int = DEFAULT_TOP


@dataclass(frozen=True)
class Hit:
    """One ranked article. `shared_terms` empty means it matched on meaning alone."""

    doc_id: str
    title: str
    specialty: str
    code: str
    score: float
    shared_terms: tuple[str, ...]


@dataclass(frozen=True)
class Response:
    ok: bool
    query: str
    settings: Settings
    terms: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    hits: tuple[Hit, ...] = ()
    count: int = 0
    message: str = ""


class SemanticEngine:
    """The collection, one matrix per weighting, and one SVD per matrix."""

    def __init__(self, articles: tuple[Article, ...]):
        self.articles = articles
        self.by_id = {a.doc_id: a for a in articles}
        self._documents = {a.doc_id: a.terms for a in articles}
        self._term_sets = {a.doc_id: frozenset(a.terms) for a in articles}
        self._matrices: dict[str, TermDocumentMatrix] = {}
        self._factors: dict[str, SvdFactors] = {}

    def matrix(self, weighting: str = "tfidf") -> TermDocumentMatrix:
        if weighting not in self._matrices:
            self._matrices[weighting] = TermDocumentMatrix(self._documents, weighting)
        return self._matrices[weighting]

    def factors(self, weighting: str = "tfidf") -> SvdFactors:
        if weighting not in self._factors:
            self._factors[weighting] = SvdFactors(self.matrix(weighting))
        return self._factors[weighting]

    def lsi(self, weighting: str = "tfidf", k: int = DEFAULT_K) -> LsiModel:
        return LsiModel(self.factors(weighting), k)

    def model(self, settings: Settings) -> GvsmModel | LsiModel:
        if settings.model == "gvsm":
            return GvsmModel(self.matrix(settings.weighting))
        if settings.model == "lsi":
            return self.lsi(settings.weighting, settings.k)
        raise ValueError(f"unknown model {settings.model!r}; use one of {MODELS}")

    def search(self, query: str, settings: Settings = Settings()) -> Response:
        terms = tokenize(query)
        if not terms:
            return Response(False, query, settings,
                            message="This query has no searchable word. Common words such "
                                    "as 'the' and 'of' are ignored.")
        matrix = self.matrix(settings.weighting)
        known = matrix.known_terms(terms)
        unknown = tuple(dict.fromkeys(t for t in terms if t not in matrix.row_of))
        if not known:
            # with no row in A a word has no vector, so LSI cannot help either
            return Response(False, query, settings, terms=tuple(terms), unknown=unknown,
                            message="None of these words appear anywhere in the collection, "
                                    "so the query has no vector to compare.")
        try:
            ranked = self.model(settings).rank(terms)
        except ValueError as error:
            return Response(False, query, settings, terms=tuple(terms), message=str(error))

        hits = tuple(self._hit(doc, score, known) for doc, score in ranked[:settings.top])
        return Response(True, query, settings, terms=tuple(known), unknown=unknown,
                        hits=hits, count=len(ranked))

    def _hit(self, doc_id: str, score: float, known: list[str]) -> Hit:
        article = self.by_id[doc_id]
        return Hit(doc_id, article.title, article.specialty, article.code, score,
                   tuple(t for t in known if t in self._term_sets[doc_id]))

    def compare(self, query: str, settings: Settings = Settings()) -> dict[str, Response]:
        """The same query and weighting under both models."""
        return {m: self.search(query, Settings(m, settings.weighting, settings.k, settings.top))
                for m in MODELS}

    def steps(self, query: str, doc_id: str, settings: Settings = Settings()) -> CosineSteps:
        """The worked cosine between this query and one document, under one model."""
        if doc_id not in self.by_id:
            raise ValueError(f"no article called {doc_id!r}")
        return self.model(settings).steps(tokenize(query), doc_id)


@lru_cache(maxsize=1)
def get_engine() -> SemanticEngine:
    """The one engine for this process."""
    return SemanticEngine(get_articles())


def main() -> None:
    engine = get_engine()
    for model, response in engine.compare("breathing difficulty").items():
        print(model, [(h.doc_id, round(h.score, 2)) for h in response.hits[:5]])


if __name__ == "__main__":
    main()
