"""Step 4 - the search engine both front ends call.

    engine.search("hypertension AND NOT diabetes", Settings(mode="extended"))

This file is the only place where a query becomes a ranked result. The web app
(app.py) and the terminal (console_ui.py) both call it, so they can never disagree.

Three modes, one query language:

    boolean    exact words, binary membership, fuzzy operators. Results are
               unranked: every hit has score 1.
    fuzzy      partial matching, a membership function, fuzzy operators. Every
               article with a score above 0 is returned, ranked by score.
    extended   partial matching, a membership function, p-norm operators. The
               articles returned are those in the crisp alpha-cut set algebra
               (lecture section 3): AND = intersection, OR = union, NOT =
               difference. They are ranked by their p-norm score.

Run:  python fuzzy_search.py "hypertension treatment"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from functools import lru_cache

from articles import Article, tokenize
from fuzzy_logic import Step, evaluate
from membership import DEFAULT_TAU, FUNCTIONS, FuzzyIndex
from query_parser import Node, QueryError, Term, parse_query, terms_in, to_text

MODES = ("boolean", "fuzzy", "extended")
IMPLICIT = ("AND", "OR")


@dataclass(frozen=True)
class Settings:
    """Everything the user can change. Defaults are the recommended ones."""

    mode: str = "fuzzy"
    implicit: str = "AND"
    function: str = "log"
    partial: bool = True
    tau: float = DEFAULT_TAU
    p: float = 2.0
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.implicit not in IMPLICIT:
            raise ValueError("implicit must be AND or OR")
        if self.function not in FUNCTIONS:
            raise ValueError(f"function must be one of {FUNCTIONS}")
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be between 0 and 1")
        if self.p < 1:
            raise ValueError("p must be at least 1 (use math.inf for min/max)")
        if not 0 <= self.alpha < 1:
            raise ValueError("alpha must be at least 0 and below 1")

    def effective(self) -> "Settings":
        """Boolean mode is the classical model: exact words, binary membership."""
        if self.mode == "boolean":
            return replace(self, function="binary", partial=False)
        return self


@dataclass(frozen=True)
class TermMatch:
    """How one search word was matched in one article."""

    word: str
    stem: str
    mu: float
    matched_as: str | None


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    specialty: str
    code: str
    score: float
    matches: tuple[TermMatch, ...]
    working: Step


@dataclass(frozen=True)
class Response:
    query: str
    settings: Settings
    ok: bool
    message: str
    parsed: str
    terms: tuple[str, ...]
    hits: tuple[Hit, ...]
    considered: int

    @property
    def count(self) -> int:
        return len(self.hits)


class FuzzyEngine:
    """Holds the index. Build it once; every query after that is read-only."""

    def __init__(self, articles: list[Article] | tuple[Article, ...] | None = None) -> None:
        self.index = FuzzyIndex(articles)
        self.articles = self.index.articles
        self.by_id = {a.doc_id: a for a in self.articles}

    # -- one article, one query ------------------------------------------------

    def _term_value(self, term: Term, doc_id: str,
                    s: Settings) -> tuple[float, tuple[TermMatch, ...]]:
        matches = []
        for stem in term.stems:
            mu, matched = self.index.best_match(stem, doc_id, s.function, s.partial, s.tau)
            matches.append(TermMatch(term.word, stem, mu, matched))
        # a hyphenated word counts as all of its parts, so the weakest part decides
        return min(m.mu for m in matches), tuple(matches)

    def _evaluate_doc(self, tree: Node, doc_id: str, s: Settings,
                      p: float | None) -> tuple[Step, tuple[TermMatch, ...]]:
        found: dict[str, tuple[TermMatch, ...]] = {}

        def leaf(term: Term) -> tuple[float, str]:
            value, matches = self._term_value(term, doc_id, s)
            found[term.word] = matches
            shown = [m.matched_as for m in matches
                     if m.matched_as and m.matched_as != m.stem]
            detail = ""
            if shown:
                detail = f"found as '{', '.join(shown)}'"
            return value, detail

        step = evaluate(tree, leaf, p)
        matches = tuple(m for group in found.values() for m in group)
        return step, matches

    def _in_crisp_set(self, tree: Node, doc_id: str, s: Settings) -> bool:
        """Is the article in the alpha-cut set the query describes?"""

        def leaf(term: Term) -> tuple[float, str]:
            value, _ = self._term_value(term, doc_id, s)
            return (1.0 if value > s.alpha else 0.0), ""

        return evaluate(tree, leaf, None).value == 1.0

    # -- public API --------------------------------------------------------------

    def search(self, query: str, settings: Settings | None = None) -> Response:
        s = (settings or Settings()).effective()
        try:
            tree = parse_query(query, implicit=s.implicit)
        except QueryError as error:
            return Response(query, s, False, str(error), "", (), (), len(self.articles))

        p = None if s.mode in ("boolean", "fuzzy") else s.p
        terms = tuple(dict.fromkeys(stem for t in terms_in(tree) for stem in t.stems))

        hits: list[Hit] = []
        for article in self.articles:
            step, matches = self._evaluate_doc(tree, article.doc_id, s, p)
            if s.mode == "extended":
                if not self._in_crisp_set(tree, article.doc_id, s):
                    continue
            elif step.value <= 0:
                continue
            hits.append(Hit(article.doc_id, article.title, article.specialty,
                            article.code, step.value, matches, step))

        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return Response(query, s, True, "", to_text(tree), terms, tuple(hits),
                        len(self.articles))

    def compare(self, query: str, settings: Settings | None = None) -> dict[str, Response]:
        """The same query in all three modes, with everything else held equal."""
        base = settings or Settings()
        return {mode: self.search(query, replace(base, mode=mode)) for mode in MODES}

    def crisp_set(self, query: str, settings: Settings | None = None) -> set[str]:
        """The alpha-cut set for a query, as article ids. Raises QueryError if unreadable."""
        s = replace((settings or Settings()), mode="extended").effective()
        tree = parse_query(query, implicit=s.implicit)
        return {a.doc_id for a in self.articles if self._in_crisp_set(tree, a.doc_id, s)}

    def set_view(self, a: str, b: str, c: str,
                 settings: Settings | None = None) -> dict:
        """Lecture section 3 checked on real sets: (A OR B) AND NOT C == (A u B) - C."""
        s = settings or Settings(mode="extended")
        set_a = self.crisp_set(f"({a})", s)
        set_b = self.crisp_set(f"({b})", s)
        set_c = self.crisp_set(f"({c})", s)
        by_sets = (set_a | set_b) - set_c
        by_query = self.crisp_set(f"(({a}) OR ({b})) AND NOT ({c})", s)
        return {
            "A": set_a, "B": set_b, "C": set_c,
            "union": set_a | set_b, "result": by_sets,
            "query": by_query, "equal": by_sets == by_query,
        }

    def membership_table(self, word: str, function: str = "log") -> list[tuple[str, str, float]]:
        """mu of one word in every article, highest first: (doc_id, title, mu)."""
        stems = tokenize(word)
        if not stems:
            raise QueryError(f"'{word}' has no searchable word in it", 0)
        stem = stems[0]
        rows = [(a.doc_id, a.title, self.index.membership(stem, a.doc_id, function))
                for a in self.articles]
        rows.sort(key=lambda row: (-row[2], row[0]))
        return rows

    def partial_matches(self, word: str, tau: float = DEFAULT_TAU) -> list[tuple[str, float]]:
        """The vocabulary stems a word would match, with similarity, best first."""
        stems = tokenize(word)
        if not stems:
            return []
        return self.index.expansions(stems[0], tau)


@lru_cache(maxsize=1)
def get_engine() -> FuzzyEngine:
    return FuzzyEngine()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    query = " ".join(args) or "hypertension treatment"
    response = get_engine().search(query)
    if not response.ok:
        print(f"error: {response.message}")
        return 1
    print(f"query {query!r} parsed as {response.parsed}  [{response.settings.mode}]")
    for rank, hit in enumerate(response.hits, start=1):
        print(f"  {rank:>2}. {hit.score:.2f}  {hit.doc_id}  {hit.title}")
    print(f"{response.count} of {response.considered} articles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
