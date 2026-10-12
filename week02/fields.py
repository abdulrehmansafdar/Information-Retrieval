"""Step 4 - Searching structured fields.

Week 01 searched one flat body of text per document. A research paper is not
flat: the title, abstract, methodology and results are not interchangeable, and
a reader who writes "structured field" in a query usually means the words
structured AND field, which is a statement about the TITLE of this week's p19
rather than about its methodology.

Three designs are compared here, and the difference between the first two is the
single most misunderstood point in structured retrieval:

    restrict      score ONLY the chosen field. The other fields do not exist.
    filter        rank over the whole paper, then throw away everything that
                  did not match the field. Same SET of results, different ORDER.
    weight        BM25F: score every field with its own weight and sum. Nothing
                  is thrown away, a title match counts for more than a
                  methodology match.

Restrict and filter return THE SAME PAPERS in a DIFFERENT ORDER. If you only
check which papers came back, the two designs look identical, and p19 below is
built so that they do not.

BM25F
-----
    score_BM25F(q, d) = SUM over fields f of  weight_f * score_BM25(q, f)

with one important correction, without which field weighting barely works:

    normalize each field's length by ITS OWN average, not the paper's

A title has about 3 terms and an abstract about 45. Without per-field
normalisation the title's raw BM25 is dominated by the abstract's length, and a
weight of 3.0 on the title does almost nothing. p19 in the corpus is about that
failure, which is why the normalisation is exposed as a switch rather than
always on.

Run:  python fields.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from preprocess import FIELDS
from ranking import Bm25Model, IdfTable, Query, ScoredPaper, parse_query
from structured_index import Region, StructuredIndex, build_index

DEFAULT_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "abstract": 2.0,
    "methodology": 1.5,
    "results": 1.0,
}

# Any word followed by a colon looks like a field restriction, so that an
# unknown name raises a readable error instead of being silently treated as a
# search term. Restricting the regex to the four real field names would make
# `keywords:foo` a clean miss, which looks like an answer rather than a typo.
ANY_FIELD_RE = re.compile(r"\b([a-z_][a-z_ ]*)\s*:\s*", re.I)


class FieldQueryError(ValueError):
    """A field restriction names a field that does not exist."""


@dataclass(frozen=True)
class FieldRestriction:
    """A parsed `field:terms` restriction. Empty field means every field.

    `unknown` matters: a restriction whose terms are ALL unknown cannot be
    satisfied, which is very different from a restriction the user left blank.
    Treating the two alike silently deletes a constraint from the query - ask for
    `title:proximity AND results:precision` where 'proximity' is not in the
    vocabulary and you would get back every paper whose RESULTS mention
    precision, with the title constraint quietly ignored.
    """

    field: str = ""
    terms: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    @property
    def unsatisfiable(self) -> bool:
        return not self.terms and bool(self.unknown)

    @property
    def blank(self) -> bool:
        return not self.terms and not self.unknown

    def describe(self) -> str:
        return f"{self.field}:{' '.join(self.terms)}" if self.field else "all fields"


# `AND` binds tighter than `OR`, so `A OR B AND C` means `A OR (B AND C)`.
# That is the SQL convention, and it is the one users already expect.
CONNECTOR_RE = re.compile(r"\s+\b(AND|OR)\b\s+", re.I)


@dataclass
class FieldClause:
    """One AND-group: every restriction here must hold for a paper to match.

    A query is a UNION of clauses. `title:search` is one clause of one
    restriction. `title:search OR results:precision` is two clauses of one
    restriction each. `title:search AND results:precision` is ONE clause of two
    restrictions, which is a genuinely different question.
    """

    restrictions: tuple[FieldRestriction, ...] = ()

    @property
    def wanted(self) -> int:
        return sum(len(r.terms) for r in self.restrictions)

    def describe(self) -> str:
        return " AND ".join(r.describe() for r in self.restrictions)


STRAY_COLON_RE = re.compile(r"^\s*[^\s():]*:")


def _strip_outer_parens(text: str) -> tuple[str, bool]:
    """Peel `( ... )` off, but only if the bracket wraps the WHOLE string."""
    if not (text.startswith("(") and text.endswith(")")):
        return text, False
    depth = 0
    for position, character in enumerate(text):
        depth += (character == "(") - (character == ")")
        if depth == 0 and position < len(text) - 1:
            return text, False
    return text[1:-1].strip(), True


def _space_brackets(text: str) -> str:
    """Pull every bracket out as its own token, so `(a OR b)` and `( a OR b )`
    are the same query and neither hides a connector from the split below."""
    return " ".join(re.findall(r"\(|\)|[^\s()]+", text))


def _split_on_top_level(text: str, keyword: str) -> list[str]:
    """Split on a connector, ignoring any that sit inside brackets.

    Pieces are never broken apart inside a bracket, so a multi-word restriction
    like `title:inverted index` survives as one piece. A connector with nothing
    on one side of it is an error: `title:search AND` used to parse as
    `title:search` and answer confidently, which is worse than refusing.
    """
    pieces: list[str] = []
    current: list[str] = []
    depth = 0
    for token in _space_brackets(text).split():
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise FieldQueryError(f"unmatched ')' in {text!r}")
        current.append(token)
        if depth == 0 and token.upper() == keyword:
            head = " ".join(current[:-1])
            if not head:
                raise FieldQueryError(
                    f"{keyword} at the start of {text!r} has nothing to join "
                    f"to; put a term on both sides of it")
            pieces.append(head)
            current = []
    if depth:
        raise FieldQueryError(f"unmatched '(' in {text!r}")
    if pieces and not current:
        raise FieldQueryError(
            f"{keyword} at the end of {text!r} has nothing to join to; put a "
            f"term on both sides of it")
    pieces.append(" ".join(current))
    return [piece for piece in pieces if piece]


def _clause_of_term(text: str, idf_table: IdfTable) -> FieldClause:
    """Compile a single piece - a `field:terms` or a loose term - to one clause."""
    match = ANY_FIELD_RE.match(text)
    if not match:
        if STRAY_COLON_RE.match(text):
            raise FieldQueryError(
                f"{text!r} has a ':' but no field name in front of it; write "
                f"one of {', '.join(FIELDS)} followed by a colon")
        parsed = parse_query(text, idf_table)
        return FieldClause((FieldRestriction("", parsed.terms, parsed.unknown),))

    name = match.group(1).strip().lower()
    if name not in FIELDS:
        raise FieldQueryError(
            f"{name!r} is not a field of this corpus; "
            f"expected one of {', '.join(FIELDS)}")
    body = ANY_FIELD_RE.sub("", text, count=1).strip()
    if not body:
        raise FieldQueryError(
            f"{name!r} was given nothing to look for; write {name}:word")
    parsed = parse_query(body, idf_table)
    return FieldClause((FieldRestriction(name, parsed.terms, parsed.unknown),))


def _clauses_of(text: str, idf_table: IdfTable) -> list[FieldClause]:
    """Compile a chunk into a list of AND-groups, which is its DNF form.

    OR at the top, AND inside it, brackets override both - the SQL convention,
    and the only one a user can predict without reading the source. AND is
    applied as a cross product of the two sides' clauses, which is exactly
    "every restriction from both sides must hold".
    """
    text, was_grouped = _strip_outer_parens(_space_brackets(text))
    if was_grouped:
        if not text:
            raise FieldQueryError("'()' is not a search term")
        return _clauses_of(text, idf_table)

    groups = _split_on_top_level(text, "OR")
    if len(groups) > 1:
        clauses: list[FieldClause] = []
        for group in groups:
            clauses.extend(_clauses_of(group, idf_table))
        return clauses

    pieces = _split_on_top_level(text, "AND")
    if len(pieces) > 1:
        clauses = [FieldClause(())]
        for piece in pieces:
            right = _clauses_of(piece, idf_table)
            clauses = [FieldClause(left.restrictions + other.restrictions)
                       for left in clauses for other in right]
        return clauses

    if "(" in text or ")" in text:
        raise FieldQueryError(
            f"{text!r} puts a bracket in the middle of a term; write "
            f"'(a OR b) AND c', with a space either side of every bracket")
    return [_clause_of_term(text, idf_table)]


def parse_field_query(text: str, idf_table: IdfTable
                      ) -> tuple[list[FieldClause], Query]:
    """Parse a field query into a union of AND-groups.

    `title:neural AND results:accuracy OR abstract:index`
        -> [ (title:neural AND results:accuracy), (abstract:index) ]

    `(title:neural OR abstract:index) AND results:accuracy`
        -> [ (title:neural AND results:accuracy),
             (abstract:index AND results:accuracy) ]

    Splitting happens on the RAW text, before any preprocessing, for the same
    reason week01's Boolean engine does it: preprocess_query() drops 'OR' as a
    stop word, so a post-preprocessing split would silently turn a user's OR
    into an AND and return a confidently wrong answer.

    NOT is rejected rather than ignored. Negation over fields needs a different
    answer set, and quietly dropping the operator is how OR went wrong.

    A bracket, a colon with no field name, an empty restriction and a dangling
    AND are all refused for the same reason. Every one of them used to parse
    into something plausible and answer a question the user did not ask.
    """
    if re.search(r"\bNOT\b", text, flags=re.I):
        raise FieldQueryError(
            "NOT is not supported in a field query. A negated field is a "
            "different question from a missing one - use plain search, or "
            "invert the result yourself."
        )

    clauses = _clauses_of(text, idf_table)

    unknown: list[str] = []
    for clause in clauses:
        for restriction in clause.restrictions:
            for term in restriction.unknown:
                if term not in unknown:
                    unknown.append(term)

    # Every loose term the query asked for, in the order they were written, so
    # the caller can report a term it dropped whichever branch it sat in.
    loose_terms = tuple(dict.fromkeys(
        term
        for clause in clauses
        for restriction in clause.restrictions
        if not restriction.field
        for term in restriction.terms))

    if not clauses:
        clauses.append(FieldClause((FieldRestriction("", ()),)))
    return clauses, Query(text=text, terms=loose_terms, unknown=tuple(unknown))


@dataclass
class FieldHit:
    """One paper matched through one field. Several FieldHits can share a doc_id."""

    rank: int
    doc_id: str
    title: str
    field: str
    score: float
    field_length: int
    matched: int
    total_query_terms: int
    terms: tuple[str, ...] = ()

    @property
    def region_id(self) -> str:
        return f"{self.doc_id}:{self.field}"


@dataclass
class FieldResult:
    """A whole field-restricted answer."""

    query: str
    mode: str
    hits: list[FieldHit] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)
    unknown: tuple[str, ...] = ()
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.hits)

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# field statistics
# ---------------------------------------------------------------------------


@dataclass
class FieldStats:
    """df and average length PER FIELD.

    A plain corpus idf is wrong for field search: a term in the title of one
    paper is much rarer than df suggests, because df counts it across the whole
    paper. So field-restricted search gets its own df.
    """

    index: StructuredIndex
    idf: dict[str, dict[str, float]] = field(default_factory=dict)
    document_frequency: dict[str, dict[str, int]] = field(default_factory=dict)
    average_length: dict[str, float] = field(default_factory=dict)
    field_length: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in FIELDS:
            counts: dict[str, int] = {}
            lengths: dict[str, int] = {}
            holders = 0
            for paper in self.index.papers:
                region = paper.fields.get(name)
                if region is None:
                    continue
                holders += 1
                lengths[paper.doc_id] = region.length
                present = {t.term for t in paper.tokens
                           if region.contains(t.position)}
                for term in present:
                    counts[term] = counts.get(term, 0) + 1
            self.document_frequency[name] = counts
            self.field_length[name] = lengths
            self.average_length[name] = (
                sum(lengths.values()) / len(lengths) if lengths else 0.0
            )
            # log(1 + (N - df + .5)/(df + .5)) with N = the number of papers
            # that actually HAVE this field, so a title idf is computed over 20
            # titles and not over 20 papers.
            self.idf[name] = {
                term: math.log(1.0 + (holders - df + 0.5) / (df + 0.5))
                for term, df in counts.items()
            }

    def idf_for(self, name: str, term: str) -> float:
        """Field idf, or the weight of a term in every paper's field.

        An unknown term matches nothing, so this value can never reach a score;
        it exists so that a clean miss is arithmetic rather than a KeyError.
        """
        table = self.idf.get(name, {})
        if term in table:
            return table[term]
        return math.log(1.0 + (1.0 + 0.5) / 0.5)

    def length_norm(self, name: str, length: int) -> float:
        """Per-field length normalisation: 1.0 for an average-length field."""
        average = self.average_length.get(name, 0.0)
        return length / average if average else 1.0

    def paper_holders(self, name: str) -> int:
        return len(self.field_length.get(name, {}))


# ---------------------------------------------------------------------------
# the three designs
# ---------------------------------------------------------------------------


class FieldSearch:
    """Field-restricted, filtered and weighted search over one index."""

    def __init__(self, index: StructuredIndex) -> None:
        self.index = index
        self.stats = FieldStats(index)

    # --- building blocks ----------------------------------------------------

    def _terms_in(self, doc_id: str, name: str, terms: tuple[str, ...]
                  ) -> dict[str, int]:
        """tf of each query term inside one field of one paper."""
        paper = next(p for p in self.index.papers if p.doc_id == doc_id)
        region = paper.fields.get(name)
        if region is None:
            return {}
        found: dict[str, int] = {}
        for token in paper.tokens:
            if token.term in terms and region.contains(token.position):
                found[token.term] = found.get(token.term, 0) + 1
        return found

    def _bm25(self, doc_id: str, name: str, terms: tuple[str, ...],
              k1: float, b: float, normalise: bool) -> float:
        """BM25 over one field, with per-field df and per-field length norm."""
        counts = self._terms_in(doc_id, name, terms)
        if not counts:
            return 0.0
        region = next(p for p in self.index.papers
                      if p.doc_id == doc_id).fields[name]
        length = region.length
        norm = (self.stats.length_norm(name, length)
                if normalise else length / self.index.average_document_length)
        kn = (1.0 - b) + b * norm
        total = 0.0
        for term, tf in counts.items():
            sat = (tf * (k1 + 1.0)) / (tf + k1 * kn)
            total += self.stats.idf_for(name, term) * sat
        return total

    # --- the three public designs -------------------------------------------

    def _prepare(self, query: str, mode: str,
                 top: int | None) -> tuple[list[FieldClause], FieldResult] | FieldResult:
        """Shared front half of restrict/filter/weighted.

        On success returns (active clauses, template result). On failure it
        returns the FieldResult itself, which the callers detect with isinstance
        - so both outcomes must be different TYPES, not the same tuple shape,
        or a failure unpacks into (None, ...) and every caller then iterates None.
        """
        try:
            clauses, loose = parse_field_query(query, self._idf_table)
        except FieldQueryError as error:
            return FieldResult(query=query, mode=mode, error=str(error))

        broken = [r for clause in clauses for r in clause.restrictions
                  if r.unsatisfiable]
        if broken:
            names = sorted({t for r in broken for t in r.unknown})
            where = broken[0].field
            return FieldResult(
                query=query, mode=mode, unknown=loose.unknown,
                error=(f"no paper contains {' or '.join(names)}"
                       + (f", so the {where} restriction cannot be satisfied"
                          if where else
                          ", and there is no field to fall back on")),
            )

        active = [c for c in clauses if c.wanted]
        return active, FieldResult(query=query, mode=mode, unknown=loose.unknown)

    def _field_score(self, doc_id: str, restriction: FieldRestriction,
                     k1: float, b: float, normalise: bool) -> tuple[float, str]:
        """Score one restriction against one paper.

        A restriction with NO field name means "anywhere in the paper", which is
        a different thing from a named field and must not be looked up in
        paper.fields - '' is not a field there, so a naive membership test
        rejects every such query and returns nothing.
        """
        if not restriction.field:
            paper = next(p for p in self.index.papers if p.doc_id == doc_id)
            present = {t.term for t in paper.tokens}
            if not set(restriction.terms).issubset(present):
                return 0.0, ""
            model = Bm25Model(self.index, k1=k1, b=b)
            score, _ = model.score_paper(doc_id, restriction.terms)
            return score, "any"
        if restriction.field not in next(p for p in self.index.papers
                                        if p.doc_id == doc_id).fields:
            return 0.0, ""
        return (self._bm25(doc_id, restriction.field, restriction.terms,
                           k1, b, normalise), restriction.field)

    def _matched_in(self, doc_id: str, restriction: FieldRestriction
                    ) -> dict[str, int]:
        if not restriction.field:
            paper = next(p for p in self.index.papers if p.doc_id == doc_id)
            counts: dict[str, int] = {}
            for token in paper.tokens:
                if token.term in restriction.terms:
                    counts[token.term] = counts.get(token.term, 0) + 1
            return counts
        return self._terms_in(doc_id, restriction.field, restriction.terms)

    def restrict(self, query: str, top: int | None = None, k1: float = 1.2,
                 b: float = 0.75, normalise: bool = True) -> FieldResult:
        """RESTRICT: score only the named fields. Nothing outside them counts.

        Restrictions joined by AND form one clause and must ALL hold; clauses
        joined by OR are unioned. A paper satisfying several clauses appears
        once, at its best score.
        """
        prepared = self._prepare(query, "restrict", top)
        if isinstance(prepared, FieldResult):
            return prepared
        active, template = prepared
        if not active:
            return self._finish(query, "restrict", [], top, template.unknown)
        wanted = sum(clause.wanted for clause in active)

        hits: list[FieldHit] = []
        for paper in self.index.papers:
            # A paper may satisfy SEVERAL clauses. Keep the best-scoring one and
            # report the paper once, so an OR does not turn into duplicate rows.
            best: tuple[float, int, list[str], list[str]] | None = None
            for clause in active:
                total = 0.0
                matched = 0
                contributing: list[str] = []
                satisfied = True
                for restriction in clause.restrictions:
                    counts = self._matched_in(paper.doc_id, restriction)
                    # every term of THIS restriction must be present, in its field
                    if len(counts) != len(set(restriction.terms)):
                        satisfied = False
                        break
                    part, name = self._field_score(paper.doc_id, restriction,
                                                   k1, b, normalise)
                    if name and restriction.field:
                        contributing.append(name)
                    elif restriction.field == "":
                        contributing.append("any")
                    matched += len(counts)
                    total += part
                if not satisfied or total <= 0.0:
                    continue
                terms = tuple(sorted({t for r in clause.restrictions
                                      for t in r.terms}))
                if best is None or total > best[0]:
                    best = (total, matched, contributing, list(terms))
            if best is None:
                continue
            total, matched, contributing, terms = best
            hits.append(FieldHit(
                rank=0, doc_id=paper.doc_id, title=paper.title,
                field="+".join(dict.fromkeys(contributing)), score=total,
                field_length=sum(paper.fields[n].length
                                 for n in dict.fromkeys(contributing)
                                 if n in paper.fields) or paper.length,
                matched=matched, total_query_terms=wanted,
                terms=tuple(terms),
            ))
        return self._finish(query, "restrict", hits, top, template.unknown)

    def filter(self, query: str, top: int | None = None, k1: float = 1.2,
               b: float = 0.75) -> FieldResult:
        """Filter after rank. Also an AND over restrictions, for the same reason as
        `restrict`. The ranking uses the WHOLE-PAPER idf and the WHOLE-PAPER
        length before the field test is applied, because that is what this
        design means; using per-field idf here would quietly turn it back into
        `restrict` and the comparison in the console walkthrough would be
        meaningless.
        """
        prepared = self._prepare(query, "filter", top)
        if isinstance(prepared, FieldResult):
            return prepared
        active, template = prepared
        if not active:
            return self._finish(query, "filter", [], top, template.unknown)
        wanted = sum(clause.wanted for clause in active)

        whole = Bm25Model(self.index, k1=k1, b=b)
        hits: list[FieldHit] = []
        for paper in self.index.papers:
            best: tuple[float, int, list[str], list[str]] | None = None
            for clause in active:
                matched = 0
                contributing: list[str] = []
                satisfied = True
                for restriction in clause.restrictions:
                    counts = self._matched_in(paper.doc_id, restriction)
                    if len(counts) != len(set(restriction.terms)):
                        satisfied = False
                        break
                    matched += len(counts)
                    if restriction.field:
                        contributing.append(restriction.field)
                if not satisfied:
                    continue
                all_terms = tuple(sorted({t for r in clause.restrictions
                                          for t in r.terms}))
                score, _ = whole.score_paper(paper.doc_id, all_terms)
                if score <= 0.0:
                    continue
                if best is None or matched > best[1]:
                    best = (score, matched, list(dict.fromkeys(contributing)),
                            list(all_terms))
            if best is None:
                continue
            score, matched, contributing, all_terms = best
            hits.append(FieldHit(
                rank=0, doc_id=paper.doc_id, title=paper.title,
                field="+".join(contributing) if contributing else "any",
                score=score,
                field_length=sum(paper.fields[n].length
                                 for n in contributing) or paper.length,
                matched=matched, total_query_terms=wanted,
                terms=tuple(all_terms),
            ))
        return self._finish(query, "filter", hits, top, template.unknown)

    def weighted(self, query: str, top: int | None = None, k1: float = 1.2,
                 b: float = 0.75, normalise: bool = True,
                 weights: dict[str, float] | None = None) -> FieldResult:
        """BM25F: every field contributes, weighted. Nothing is discarded."""
        prepared = self._prepare(query, "weighted", top)
        if isinstance(prepared, FieldResult):
            return prepared
        active, template = prepared
        wanted = sum(clause.wanted for clause in active)

        table = weights or DEFAULT_WEIGHTS
        hits: list[FieldHit] = []
        for paper in self.index.papers:
            # keep the best-scoring clause, so an OR yields one row per paper
            best: tuple[float, list[str], tuple[str, ...]] | None = None
            for clause in active:
                total = 0.0
                contributing: list[str] = []
                clause_terms: list[str] = []
                for restriction in clause.restrictions:
                    clause_terms.extend(restriction.terms)
                    for name, weight in table.items():
                        if name not in paper.fields:
                            continue
                        part = self._bm25(paper.doc_id, name,
                                          restriction.terms, k1, b, normalise)
                        if part <= 0.0:
                            continue
                        contributing.append(name)
                        total += weight * part
                if total <= 0.0:
                    continue
                if best is None or total > best[0]:
                    best = (total, list(dict.fromkeys(contributing)),
                            tuple(sorted(set(clause_terms))))
            if best is None:
                continue
            total, contributing, clause_terms = best
            hits.append(FieldHit(
                rank=0, doc_id=paper.doc_id, title=paper.title,
                field="+".join(contributing), score=total,
                field_length=sum(paper.fields[n].length for n in contributing),
                matched=self._matched_anywhere(paper.doc_id, clause_terms),
                total_query_terms=wanted,
                terms=clause_terms,
            ))
        return self._finish(query, "weighted", hits, top, template.unknown)

    def _matched_anywhere(self, doc_id: str, terms: tuple[str, ...]) -> int:
        """How many query terms appear ANYWHERE in the paper, ignoring fields."""
        present = {t.term for t in
                   next(p for p in self.index.papers
                        if p.doc_id == doc_id).tokens}
        return sum(1 for term in terms if term in present)

    # --- shared -------------------------------------------------------------

    @property
    def _idf_table(self) -> IdfTable:
        if not hasattr(self, "_idf_cache"):
            self._idf_cache = IdfTable(self.index, "bm25")
        return self._idf_cache

    def _finish(self, query: str, mode: str, hits: list[FieldHit],
                top: int | None,
                unknown: tuple[str, ...] = ()) -> FieldResult:
        """Sort, cut, and package. `unknown` is threaded in from _prepare so the
        result does not have to re-parse the query, which would be the only
        reason to re-parse a query the caller already parsed."""
        hits.sort(key=lambda h: (-h.score, h.doc_id, h.field))
        for position, hit in enumerate(hits, 1):
            hit.rank = position
        kept = hits[:top] if top else hits
        papers = sorted({h.doc_id for h in kept})
        return FieldResult(query=query, mode=mode, hits=kept, papers=papers,
                           unknown=tuple(unknown))


def field_summary(search: FieldSearch, doc_id: str, terms: tuple[str, ...],
                  k1: float = 1.2, b: float = 0.75) -> list[dict]:
    """Per-field rows for the console and the UI's 'why' table."""
    rows = []
    paper = next(p for p in search.index.papers if p.doc_id == doc_id)
    for name in FIELDS:
        if name not in paper.fields:
            continue
        region = paper.fields[name]
        counts = search._terms_in(doc_id, name, terms)
        rows.append({
            "field": name,
            "terms": region.length,
            "matched": ", ".join(f"{t} x{n}" for t, n in sorted(counts.items())) or "-",
            "field df table": len(search.stats.document_frequency.get(name, {})),
            "idf avg": sum(search.stats.idf_for(name, t) for t in counts) / len(counts)
            if counts else 0.0,
            "bm25": search._bm25(doc_id, name, terms, k1, b, True),
            "weight": DEFAULT_WEIGHTS.get(name, 0.0),
        })
    return rows


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

_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"


def bar(fraction: float, width: int = 18) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(BAR_OFF * (width - filled),
                                                      style="grey30")


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def hits_table(result: FieldResult) -> Table:
    """One narrow row per hit.

    No bar column here: with paper, title, field, score and matched already in
    the row, a bar pushes the title past the console width and the table starts
    wrapping every paper name onto four lines.
    """
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style=DIM, no_wrap=True)
    table.add_column("paper", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white", overflow="fold", ratio=2)
    table.add_column("field", style="cyan", overflow="fold", ratio=1)
    table.add_column("score", justify="right", style=WARM, no_wrap=True)
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    for hit in result.hits:
        table.add_row(
            str(hit.rank), hit.doc_id, hit.title, hit.field, f"{hit.score:.4f}",
            f"{hit.matched}/{hit.total_query_terms}",
        )
    return table


def section_designs() -> None:
    heading(1, "THREE DESIGNS, ONE QUERY")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("restrict", "score ONLY the named field. The rest of the paper is"
                             " not visible to the ranking at all.")
    body.add_row("filter", "rank the WHOLE paper, then discard whatever did not"
                           " match the field.")
    body.add_row("weight", "[bold]BM25F[/]  score_BM25F(q,d) = SUM over fields f"
                           " of  weight_f x score_BM25(q,f)")
    body.add_row("same set?", "restrict and filter return the SAME PAPERS. Only the"
                              " ORDER differs, because filter ranked them using"
                              " text outside the field.")
    body.add_row("why it matters", "an interface that says 'searched in the title'"
                                  " is telling the user something filter does not"
                                  " do.")
    CONSOLE.print(Panel(body, title="field-restricted, filtered, weighted",
                        title_align="left", border_style="grey35", padding=(0, 1)))


def section_field_statistics(search: FieldSearch) -> None:
    heading(2, "PER-FIELD STATISTICS")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("field", style=ACCENT, no_wrap=True)
    table.add_column("papers", justify="right", style=DIM, no_wrap=True)
    table.add_column("avg length", justify="right", style=DIM, no_wrap=True)
    table.add_column("distinct terms", justify="right", style=DIM, no_wrap=True)
    table.add_column("default weight", justify="right", style=WARM, no_wrap=True)
    for name in FIELDS:
        table.add_row(name, str(search.stats.paper_holders(name)),
                      f"{search.stats.average_length.get(name, 0):.1f}",
                      str(len(search.stats.document_frequency.get(name, {}))),
                      f"{DEFAULT_WEIGHTS.get(name, 0.0):g}")
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]A title averages"
        f" {search.stats.average_length.get('title', 0):.1f} terms and an abstract"
        f" {search.stats.average_length.get('abstract', 0):.1f}."
        f" That ratio is why\n  per-field length normalisation matters more than"
        f" the weights do.[/]\n"
    )


def section_restrict_vs_filter(search: FieldSearch,
                               query: str = "title:search"):
    heading(3, "RESTRICT versus FILTER, SAME PAPERS DIFFERENT ORDER")
    restricted = search.restrict(query, top=6)
    filtered = search.filter(query, top=6)

    CONSOLE.print(f"  [bold white]{query!r}[/]  [grey58]mode: restrict[/]")
    CONSOLE.print(hits_table(restricted))
    CONSOLE.print(f"  [bold white]{query!r}[/]  [grey58]mode: filter[/]")
    CONSOLE.print(hits_table(filtered))

    same_set = restricted.papers == filtered.papers
    r_order = [h.doc_id for h in restricted.hits]
    f_order = [h.doc_id for h in filtered.hits]
    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("same papers?", f"{len(restricted.papers)} papers, "
                                 f"{'identical set' if same_set else 'DIFFERENT set'}")
    body.add_row("same order?", f"{'yes' if r_order == f_order else 'NO'}")
    body.add_row("restrict order", ", ".join(r_order))
    body.add_row("filter order", ", ".join(f_order))
    body.add_row("why", "filter let the methodology and results text compete in the"
                        " ranking, then removed it afterwards. A paper with a"
                        " strong body but a weak title can outrank a paper that"
                        " is actually about the title.")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))


def section_weighted(search: FieldSearch,
                     query: str = "structured field search") -> None:
    heading(4, "BM25F - EVERY FIELD CONTRIBUTES")
    weighted = search.weighted(query, top=6)
    CONSOLE.print(hits_table(weighted))
    CONSOLE.print(
        f"  [grey58]The 'field' column lists every field that matched, so you can"
        f" see that a\n  title match and a results-only match are both present."
        f" This is the mode that can\n  answer 'find papers about X anywhere, but"
        f" prefer the ones whose TITLE says X'.[/]\n"
    )


def section_normalisation(search: FieldSearch) -> None:
    heading(5, "WHY PER-FIELD LENGTH NORMALISATION IS NOT OPTIONAL")

    # Find a query where one paper matches ONLY in a short field and another
    # ONLY in a long field. That is the only configuration in which
    # normalisation can visibly change the ORDER rather than just rescale every
    # score by the same constant. If the corpus has no such pair, say so
    # instead of printing a table that looks convincing and proves nothing.
    found = None
    for query in ("structured field search", "proximity window search",
                  "phrase query index", "term weighting corpus",
                  "mean average precision", "document retrieval"):
        restrictions, _ = parse_field_query(query, IdfTable(search.index, "bm25"))
        terms = tuple(sorted({t for clause in restrictions
                              for r in clause.restrictions for t in r.terms}))
        if not terms:
            continue
        short_only, long_only = None, None
        for paper in search.index.papers:
            where = [n for n in FIELDS
                     if n in paper.fields and search._terms_in(paper.doc_id, n, terms)]
            if where == ["title"]:
                short_only = paper.doc_id
            elif where == ["results"]:
                long_only = paper.doc_id
            if short_only and long_only:
                break
        if short_only and long_only:
            found = (query, terms, short_only, long_only)
            break

    if not found:
        CONSOLE.print(
            "  [yellow]No query in this corpus pairs a title-only match with a"
            " results-only match,\n  so normalisation cannot be shown changing"
            " an order here. That is a property of\n  the 20-paper corpus, not"
            " of the model. The per-field scores below are the\n  honest"
            " evidence: they show the magnitude of the effect.[/]\n"
        )
    else:
        query, terms, short_id, long_id = found
        on = {h.doc_id: h.rank for h in search.weighted(query, top=20,
                                                        normalise=True).hits
              if h.terms == terms}
        raw = {h.doc_id: h.rank for h in search.weighted(query, top=20,
                                                        normalise=False).hits
               if h.terms == terms}
        CONSOLE.print(f"  [bold white]{query!r}[/]  [grey58]terms:"
                      f" {', '.join(terms)}[/]")
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
        table.add_column("paper", style=ACCENT, no_wrap=True)
        table.add_column("matches only in", style="white", no_wrap=True)
        table.add_column("field terms", justify="right", style=DIM, no_wrap=True)
        table.add_column("rank, raw", justify="right", style=DIM, no_wrap=True)
        table.add_column("rank, normalised", justify="right", style=WARM,
                         no_wrap=True)
        for doc_id, name in ((short_id, "title"), (long_id, "results")):
            paper = next(p for p in search.index.papers if p.doc_id == doc_id)
            table.add_row(doc_id, name, str(paper.fields[name].length),
                          str(raw.get(doc_id, "-")), str(on.get(doc_id, "-")))
        CONSOLE.print(table)
        CONSOLE.print(
            f"  [grey58]{short_id} matches in a"
            f" {next(p for p in search.index.papers if p.doc_id == short_id).fields['title'].length}"
            f"-term title;\n  {long_id} matches in a"
            f" {next(p for p in search.index.papers if p.doc_id == long_id).fields['results'].length}"
            f"-term results section. Without per-field normalisation the long\n"
            f"  field scores higher for no good reason.[/]\n"
        )

    # the magnitude of the effect, whatever the order does
    on = search.weighted("structured field search", top=6, normalise=True)
    off = search.weighted("structured field search", top=6, normalise=False)
    off_scores = {h.doc_id: h.score for h in off.hits}
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("paper", style=ACCENT, no_wrap=True)
    table.add_column("normalised", justify="right", style=WARM, no_wrap=True)
    table.add_column("raw whole-paper length", justify="right", style=DIM,
                     no_wrap=True)
    table.add_column("ratio", justify="right", style=WARM, no_wrap=True)
    for hit in on.hits:
        raw = off_scores.get(hit.doc_id, 0.0)
        table.add_row(hit.doc_id, f"{hit.score:.4f}", f"{raw:.4f}",
                      f"{raw / hit.score:.2f}x" if hit.score else "-")
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]The ratio is the length effect this switch removes. It is"
        " roughly constant\n  across papers, which is why it mostly rescales"
        " rather than reorders - and it would\n  stop being constant on a corpus"
        " with much more varied field lengths.[/]\n"
    )


def section_per_field(search: FieldSearch, query: str = "structured field search",
                      doc_id: str = "p19") -> None:
    heading(6, f"ONE PAPER, FIELD BY FIELD - {doc_id}")
    from ranking import parse_query
    terms = parse_query(query, IdfTable(search.index, "bm25")).terms
    rows = field_summary(search, doc_id, terms)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("field", style=ACCENT, no_wrap=True)
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    table.add_column("matched", style="white", overflow="fold")
    table.add_column("idf", justify="right", style=DIM, no_wrap=True)
    table.add_column("bm25", justify="right", style=WARM, no_wrap=True)
    table.add_column("x weight", justify="right", style=WARM, no_wrap=True)
    for row in rows:
        table.add_row(row["field"], str(row["terms"]), row["matched"],
                      f"{row['idf avg']:.4f}", f"{row['bm25']:.4f}",
                      f"{row['bm25'] * row['weight']:.4f}")
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]The final column is the weighted contribution to BM25F."
        " Nothing is\n  hidden: the score is the sum of those numbers and"
        " nothing else.[/]\n"
    )


def section_cross_field(search: FieldSearch) -> None:
    heading(7, "CROSS-FIELD AND - WHICH WEEK 01 CANNOT EXPRESS")
    query = "title:search AND results:precision"
    result = search.restrict(query, top=6)
    CONSOLE.print(f"  [bold white]{query!r}[/]")
    CONSOLE.print(hits_table(result))
    CONSOLE.print(
        "  [grey58]This query means: the TITLE must contain 'search' AND the"
        " RESULTS must\n  contain 'precision'. week01's Boolean engine sees two"
        " document sets and intersects\n  them, and returns papers that merely"
        " contain both words ANYWHERE. It cannot\n  express the constraint at"
        " all, because week01's documents have no named fields.\n"
    )

    broken = search.restrict("title:proximity AND results:precision")
    CONSOLE.print(f"  [bold white]{broken.query!r}[/]")
    CONSOLE.print(f"  [red]{broken.error}[/]")
    CONSOLE.print(
        "  [grey58]'proximity' is not a term in this corpus - preprocessing"
        " folded it into\n  the phrases 'proximity operator' and 'proximity"
        " window'. Returning the papers whose\n  RESULTS mention precision"
        " anyway would be answering a DIFFERENT query, so\n  this returns"
        " nothing and says why. Silently dropping the impossible constraint is"
        "\n  the bug this check exists to prevent.[/]\n"
    )


def section_errors(search: FieldSearch) -> None:
    heading(8, "ERRORS ARE MESSAGES, NOT STACK TRACES")
    checks = [
        ("title:search", "a field prefix on its own"),
        ("title:search AND results:precision", "two fields, AND-ed (intersected)"),
        ("title:search OR results:precision", "two fields, OR-ed (unioned)"),
        ("title:search NOT results:x", "NOT is rejected, not ignored"),
        ("title:proximity AND results:precision", "unknown term -> 0 hits, explained"),
        ("keywords:search", "unknown field name -> FieldQueryError"),
        ("   ", "empty -> no hits, no crash"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("query", style="bold white", overflow="fold")
    table.add_column("example", style=DIM, overflow="fold", ratio=1)
    table.add_column("result", style="white", overflow="fold", ratio=2)
    for query, note in checks:
        try:
            result = search.restrict(query)
            outcome = (result.error if result.error
                       else f"{result.count} hits in {result.papers}")
        except FieldQueryError as error:
            outcome = f"FieldQueryError: {error}"
        table.add_row(repr(query), note, str(outcome))
    CONSOLE.print(table)
    CONSOLE.print()


def main() -> None:
    index = build_index()
    search = FieldSearch(index)

    CONSOLE.rule("[bold cyan]Step 4 - Searching Structured Fields[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} papers  {SEP}  fields:"
        f" {', '.join(FIELDS)}  {SEP}  weights: "
        f"{', '.join(f'{k} {v:g}' for k, v in DEFAULT_WEIGHTS.items())}[/]\n"
    )

    section_designs()
    section_field_statistics(search)
    section_restrict_vs_filter(search)
    section_weighted(search)
    section_normalisation(search)
    section_per_field(search)
    section_cross_field(search)
    section_errors(search)


if __name__ == "__main__":
    main()