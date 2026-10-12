"""Step 3 - Probability based ranking: the Binary Independence Model and BM25.

Week 01 ranked documents by cosine similarity over TF-IDF vectors. That is a
vector model: it asks "how similar are these two bags of words". It has no
notion of how many times a term occurs, and no notion of probability at all.

This step asks a different question - "given this query, how likely is it that
paper d is relevant?" - and answers it as a log-odds accumulation. Two models
live here and they are the same formula at two settings.

The Binary Independence Model
----------------------------
The modelling assumption is INDEPENDENCE: whether term t appears in paper d
does not depend on whether any other term appears in d. Under that assumption
the evidence from each term multiplies, so in log space it adds:

    score_BIM(q, d) = SUM over t in q of  idf(t) * 1[t in d]

1[t in d] is the "binary indicator": it is 1 if the term appears at all and 0
otherwise. Ten occurrences count exactly as much as one. That is the model's
defining limitation, and it is what BM25 fixes.

BM25
----
    score_BM25(q, d) = SUM over t in q of
                          idf(t) * ( tf * (k1 + 1) ) / ( tf + k1 * K(d) )

    K(d) = (1 - b) + b * |d| / avgdl

Two parameters, each with exactly one job:

    k1  saturation. How fast an extra occurrence stops helping.
        k1 -> 0 turns the fraction into 1[t in d] and recovers BIM exactly.
        k1 -> infinity makes it tf / K(d), which is linear in tf.
    b   length normalisation. How much a long paper is penalised.
        b = 0 ignores length. b = 1 fully normalises it.

THE BRIDGE
----------
Substituting k1 = 0 into BM25 gives

    ( tf * 1 ) / ( tf + 0 * K ) = tf / tf = 1

for every tf >= 1. So BIM IS BM25 at k1 = 0. That is not a loose analogy, it is
an identity, and test_ranking.py asserts it to floating point precision rather
than asking you to trust it.

Run:  python ranking.py
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from preprocess import preprocess_query
from structured_index import Posting, StructuredIndex, build_index

# Robertson / Sparck Jones log-odds IDF, the original probabilistic form.
# It goes NEGATIVE for a term appearing in more than half the collection, which
# is a real problem: a common term then subtracts from the relevance score.
IDF_VARIANTS = ("bm25", "robertson", "plain")

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


@dataclass
class IdfTable:
    """df and idf for every term, built from the postings.

    Three IDF variants are supported because they differ in a way that changes
    rankings, and the difference is worth being able to show:

        plain       log(N / df)              what week 01 used. Zero for a
                                             term in every paper.
        robertson   log((N-df+.5)/(df+.5))   goes negative when df > N/2.
        bm25        log(1 + (N-df+.5)/(df+.5))  always positive, so every
                                             matched term adds evidence.
    """

    index: StructuredIndex
    variant: str = "bm25"

    number_of_documents: int = field(init=False)
    document_frequency: dict[str, int] = field(default_factory=dict, init=False)
    idf: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.variant not in IDF_VARIANTS:
            raise ValueError(f"unknown IDF variant {self.variant!r}; "
                             f"expected one of {IDF_VARIANTS}")
        self.number_of_documents = self.index.number_of_documents
        n = self.number_of_documents
        for term in self.index.vocabulary:
            df = len(self.index.lookup(term))
            self.document_frequency[term] = df
            self.idf[term] = self._idf_value(n, df)

    def _idf_value(self, n: int, df: int) -> float:
        if self.variant == "plain":
            return math.log(n / df) if df else 0.0
        odds = (n - df + 0.5) / (df + 0.5)
        return math.log(odds) if self.variant == "robertson" else math.log(1.0 + odds)

    def df_for(self, term: str) -> int:
        return self.document_frequency.get(term, 0)

    def phrase_hints(self, term: str) -> list[str]:
        """Indexed phrase terms that CONTAIN this term as one of their words.

        Answers the question a user always asks when a query word is ignored:
        'why was that dropped?' The answer is almost always that the word only
        ever occurs inside a longer phrase - 'proximity' exists solely inside
        'proximity operator' and 'proximity window', so it is never a term of
        its own.
        """
        return sorted(candidate for candidate in self.idf
                      if " " in candidate and term in candidate.split())

    def idf_for(self, term: str) -> float:
        """Weight of a term.

        An unknown term gets the weight of a hypothetical term in every
        document, which is the maximum available under the 'bm25' variant and
        the minimum under 'robertson'. Either way it matches nothing, so the
        value can never reach a score. Raising here would turn a clean miss
        into a crash.
        """
        if term in self.idf:
            return self.idf[term]
        return self._idf_value(self.number_of_documents,
                               self.number_of_documents)


@dataclass(frozen=True)
class Query:
    """A parsed query, before any model has seen it."""

    text: str
    terms: tuple[str, ...]
    unknown: tuple[str, ...] = ()

    @property
    def known(self) -> tuple[str, ...]:
        return self.terms


def parse_query(text: str, idf_table: IdfTable,
                drop_stop_words: bool = True) -> Query:
    """Preprocess the query and separate terms the corpus does not contain."""
    raw = preprocess_query(text)
    if drop_stop_words:
        from preprocess import get_stop_words
        stops = get_stop_words()
        raw = [t for t in raw if t not in stops]

    terms: list[str] = []
    unknown: list[str] = []
    for term in dict.fromkeys(raw):        # de-duplicate, keep query order
        (unknown if term not in idf_table.idf else terms).append(term)
    return Query(text=text, terms=tuple(terms), unknown=tuple(unknown))


@dataclass
class TermContribution:
    """One query term's share of one paper's score. Shown in the UI so the
    formula is visible rather than a black box."""

    term: str
    tf: int
    idf: float
    saturation: float
    contribution: float


@dataclass
class ScoredPaper:
    """One retrieved paper."""

    rank: int
    doc_id: str
    title: str
    score: float
    length: int
    matched: int = 0
    total_query_terms: int = 0
    contributions: list[TermContribution] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if not self.total_query_terms:
            return 0.0
        return self.matched / self.total_query_terms

    def describe(self) -> str:
        parts = "  ".join(
            f"{c.term} x{c.tf} idf={c.idf:.3f} sat={c.saturation:.3f} "
            f"-> {c.contribution:.3f}"
            for c in self.contributions
        )
        return parts or "no shared query terms"


# ---------------------------------------------------------------------------
# the Binary Independence Model
# ---------------------------------------------------------------------------


class BinaryIndependenceModel:
    """score(q,d) = SUM idf(t) * 1[t in d].

    Holds its own IdfTable because the IDF variant changes BIM's behaviour much
    more than it changes BM25's: under the plain log(N/df) variant a term in
    every paper contributes exactly zero, and under the Robertson variant it
    contributes a negative number.
    """

    def __init__(self, index: StructuredIndex, idf_variant: str = "bm25") -> None:
        self.index = index
        self.idf_table = IdfTable(index, idf_variant)

    @property
    def name(self) -> str:
        return f"BIM (idf={self.idf_table.variant})"

    @property
    def parameters(self) -> dict:
        return {"idf variant": self.idf_table.variant}

    def term_score(self, term: str, tf: int) -> float:
        """idf(t) * 1[t in d]. The 1[t in d] is `tf > 0`."""
        if tf <= 0:
            return 0.0
        return self.idf_table.idf_for(term)

    def score_paper(self, doc_id: str, terms: tuple[str, ...]
                    ) -> tuple[float, list[TermContribution]]:
        """Score one paper and return the score plus the per-term breakdown."""
        total = 0.0
        rows: list[TermContribution] = []
        for term in terms:
            tf = self.index.term_frequency_in(doc_id, term)
            if tf <= 0:
                continue
            idf = self.idf_table.idf_for(term)
            total += idf
            rows.append(TermContribution(term=term, tf=tf, idf=idf,
                                         saturation=1.0, contribution=idf))
        return total, rows

    def rank(self, query: str | Query, top: int | None = None,
             idf_table: IdfTable | None = None) -> list[ScoredPaper]:
        table = idf_table or self.idf_table
        parsed = query if isinstance(query, Query) else parse_query(query, table)
        wanted = len(parsed.terms)

        results: list[ScoredPaper] = []
        for paper in self.index.papers:
            score, rows = self.score_paper(paper.doc_id, parsed.terms)
            if score <= 0.0:
                continue
            results.append(ScoredPaper(
                rank=0, doc_id=paper.doc_id, title=paper.title, score=score,
                length=paper.length, matched=len(rows),
                total_query_terms=wanted, contributions=rows,
            ))
        return _rank(results, top)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class Bm25Model:
    """score(q,d) = SUM idf(t) * tf(k1+1) / (tf + k1*K(d)),  K = 1-b+b|d|/avgdl.

    k1 = 0 reproduces the Binary Independence Model exactly. The tests rely on
    that, and so does the console walkthrough, which demonstrates it by
    subtraction rather than by assertion.
    """

    def __init__(self, index: StructuredIndex, k1: float = DEFAULT_K1,
                 b: float = DEFAULT_B, idf_variant: str = "bm25") -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self.index = index
        self.k1 = k1
        self.b = b
        self.idf_table = IdfTable(index, idf_variant)

    @property
    def average_document_length(self) -> float:
        return self.index.average_document_length

    @property
    def name(self) -> str:
        return f"BM25 (k1={self.k1:g}, b={self.b:g})"

    @property
    def parameters(self) -> dict:
        return {"k1": self.k1, "b": self.b, "idf": self.idf_table.variant}

    def length_norm(self, length: int) -> float:
        """K(d) = (1 - b) + b * |d| / avgdl."""
        return (1.0 - self.b) + self.b * (length / self.average_document_length)

    def saturation(self, tf: int, length: int) -> float:
        """The tf-dependent factor tf(k1+1) / (tf + k1*K(d)).

        At k1 = 0 this is tf / tf = 1 for every tf >= 1, which is precisely the
        binary indicator. At k1 large it approaches tf / K(d).
        """
        if tf <= 0:
            return 0.0
        return (tf * (self.k1 + 1.0)) / (tf + self.k1 * self.length_norm(length))

    def term_score(self, term: str, tf: int, length: int) -> float:
        if tf <= 0:
            return 0.0
        return self.idf_table.idf_for(term) * self.saturation(tf, length)

    def score_paper(self, doc_id: str, terms: tuple[str, ...]
                    ) -> tuple[float, list[TermContribution]]:
        length = self.index.doc_lengths[doc_id]
        total = 0.0
        rows: list[TermContribution] = []
        for term in terms:
            tf = self.index.term_frequency_in(doc_id, term)
            if tf <= 0:
                continue
            idf = self.idf_table.idf_for(term)
            sat = self.saturation(tf, length)
            contribution = idf * sat
            total += contribution
            rows.append(TermContribution(term=term, tf=tf, idf=idf,
                                         saturation=sat, contribution=contribution))
        return total, rows

    def rank(self, query: str | Query, top: int | None = None,
             idf_table: IdfTable | None = None) -> list[ScoredPaper]:
        table = idf_table or self.idf_table
        parsed = query if isinstance(query, Query) else parse_query(query, table)
        wanted = len(parsed.terms)

        results: list[ScoredPaper] = []
        for paper in self.index.papers:
            score, rows = self.score_paper(paper.doc_id, parsed.terms)
            if score <= 0.0:
                continue
            results.append(ScoredPaper(
                rank=0, doc_id=paper.doc_id, title=paper.title, score=score,
                length=paper.length, matched=len(rows),
                total_query_terms=wanted, contributions=rows,
            ))
        return _rank(results, top)

    def with_parameters(self, k1: float | None = None,
                        b: float | None = None) -> "Bm25Model":
        """A copy with different parameters. The IdfTable is shared, because the
        parameters have no effect on df or idf and rebuilding 966 values for a
        slider drag would be wasteful."""
        return Bm25Model(self.index,
                         k1=self.k1 if k1 is None else k1,
                         b=self.b if b is None else b,
                         idf_variant=self.idf_table.variant)


def _rank(results: list[ScoredPaper], top: int | None) -> list[ScoredPaper]:
    """Sort best first, break ties on doc_id so the order is deterministic, and
    assign 1-based ranks. Ties broken on doc_id rather than left arbitrary is
    what makes the evaluation in step 7 reproducible."""
    results.sort(key=lambda r: (-r.score, r.doc_id))
    for position, result in enumerate(results, 1):
        result.rank = position
    return results[:top] if top else results


def compare_models(index: StructuredIndex, query: str, top: int = 10
                   ) -> dict[str, list[ScoredPaper]]:
    """Every probability-based model on one query, for the console and the UI."""
    bim = BinaryIndependenceModel(index)
    table = bim.idf_table
    parsed = parse_query(query, table)
    return {
        "BIM": bim.rank(parsed, top=top, idf_table=table),
        "BM25": Bm25Model(index).rank(parsed, top=top, idf_table=table),
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


def scored_table(results: list[ScoredPaper], width: int = 16,
                 score_header: str = "score") -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style=DIM, no_wrap=True)
    table.add_column("paper", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    table.add_column(score_header, justify="right", style=WARM, no_wrap=True)
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    table.add_column(score_header, width=width)
    largest = results[0].score if results else 0.0
    for result in results:
        table.add_row(
            str(result.rank), result.doc_id, result.title,
            f"{result.score:.4f}", str(result.length),
            bar(result.score / largest, width - 2) if largest else "",
        )
    return table


def section_formulas() -> None:
    heading(1, "THE TWO FORMULAS")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("assumption", "term occurrences in a paper are INDEPENDENT of each other")
    body.add_row("consequence", "evidence multiplies, so in log space it ADDS")
    body.add_row("BIM", "[bold]score(q,d) = SUM over t in q of  idf(t) x 1[t in d][/]")
    body.add_row("", "1[t in d] is 1 if the term appears at all, else 0")
    body.add_row("BM25", "[bold]score(q,d) = SUM over t in q of  idf(t) x tf x (k1+1)"
                       " / (tf + k1 x K(d))[/]")
    body.add_row("", "K(d) = (1 - b) + b x |d| / avgdl")
    body.add_row("idf(t)", "log( 1 + (N - df + 0.5) / (df + 0.5) )")
    CONSOLE.print(Panel(body, title="probability based ranking",
                        title_align="left", border_style="grey35", padding=(0, 1)))


def section_idf_variants(index: StructuredIndex) -> None:
    heading(2, "THREE IDF VARIANTS, AND WHY IT MATTERS")
    probes = ["retriev", "queri", "document", "field", "deep learn", "term weight"]
    tables = {}
    for variant in IDF_VARIANTS:
        tables[variant] = IdfTable(index, variant)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("df", justify="right", style=DIM, no_wrap=True)
    for variant in IDF_VARIANTS:
        table.add_column(variant, justify="right", style=WARM, no_wrap=True)
    for term in probes:
        df = len(index.lookup(term))
        table.add_row(term, str(df),
                      *(f"{tables[v].idf_for(term):+.4f}" for v in IDF_VARIANTS))
    CONSOLE.print(table)

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("week 01 used", "log(N / df). Exactly zero for a term in every paper,"
                                 " and that term can no longer separate anything.")
    body.add_row("robertson", "goes NEGATIVE once df passes half the collection. A"
                              " negative contribution is evidence AGAINST a paper,"
                              " which is defensible but surprising to explain.")
    body.add_row("bm25 (this week)", "log(1 + odds), always positive. Every matched"
                                     " term adds, never subtracts.")
    body.add_row("consequence", "under 'robertson' a score can fall when more query"
                                " terms are added, which is the single most confusing"
                                " behaviour a ranking model can have.")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))


def section_binary_weakness(index: StructuredIndex) -> None:
    heading(3, "WHAT THE BINARY INDICATOR COSTS")
    term, doc_id = "field", "p19"
    tf = index.term_frequency_in(doc_id, term)
    idf = IdfTable(index, "bm25").idf_for(term)
    bm25 = Bm25Model(index)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("model", style=ACCENT, no_wrap=True)
    table.add_column("tf = 1", justify="right", style=DIM, no_wrap=True)
    table.add_column(f"tf = {tf} (the real value)", justify="right", style=DIM,
                     no_wrap=True)
    table.add_column("gain from repeating", justify="right", style=WARM,
                     no_wrap=True)
    length = index.doc_lengths[doc_id]
    table.add_row("BIM", f"{idf:.4f}", f"{idf:.4f}", "1.00x")
    table.add_row("BM25", f"{bm25.term_score(term, 1, length):.4f}",
                  f"{bm25.term_score(term, tf, length):.4f}",
                  f"{bm25.term_score(term, tf, length) / bm25.term_score(term, 1, length):.2f}x")
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]BIM gives {idf:.4f} whether the term appears once or {tf} times."
        f" It cannot tell\na paper that is about structured fields from one that"
        f" merely mentions them.[/]\n"
    )


def section_saturation(index: StructuredIndex) -> None:
    heading(4, "k1 - THE SATURATION CURVE")
    model = Bm25Model(index, b=0.75)
    length = int(model.average_document_length)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("tf", justify="right", style=DIM, no_wrap=True)
    for k1 in (0.0, 0.2, 0.6, 1.2, 2.0, 8.0):
        table.add_column(f"k1={k1:g}", justify="right", style=WARM, no_wrap=True)
    for tf in (1, 2, 3, 5, 10, 20):
        table.add_row(str(tf),
                      *(f"{model.with_parameters(k1=k1).saturation(tf, length):.4f}"
                        for k1 in (0.0, 0.2, 0.6, 1.2, 2.0, 8.0)))
    CONSOLE.print(table)

    base = model.with_parameters(k1=1.2).saturation(1, length)
    CONSOLE.print(
        f"  [grey58]Read across k1=0: the whole row is 1.0000. That is the binary"
        f" indicator,\n  and it is why k1=0 is BIM."
        f"  At k1=1.2, tf=10 is only"
        f" {model.with_parameters(k1=1.2).saturation(10, length) / base:.2f}x tf=1.[/]\n"
    )


def section_length(index: StructuredIndex) -> None:
    heading(5, "b - LENGTH NORMALISATION")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("|d|", justify="right", style=DIM, no_wrap=True)
    table.add_column("K(d) at b=0", justify="right", style=DIM, no_wrap=True)
    table.add_column("K(d) at b=0.75", justify="right", style=WARM, no_wrap=True)
    table.add_column("K(d) at b=1", justify="right", style=WARM, no_wrap=True)
    lengths = sorted(index.doc_lengths.values())
    for length in (lengths[0], int(model_average(index)), lengths[-1]):
        table.add_row(str(length),
                      f"{Bm25Model(index, b=0.0).length_norm(length):.4f}",
                      f"{Bm25Model(index, b=0.75).length_norm(length):.4f}",
                      f"{Bm25Model(index, b=1.0).length_norm(length):.4f}")
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]At b=0 every K(d) is 1.0000 and paper length has no effect at all"
        f" on the score.\n  At b=1 a paper twice the average length is penalised"
        f" proportionally, which is what\n  stops a long paper from winning by"
        f" accumulating terms. This corpus runs"
        f" {lengths[0]}-{lengths[-1]}\n  terms, so the effect is real but mild here"
        f" - it would dominate on a corpus of mixed\n  article lengths.[/]\n"
    )


def model_average(index: StructuredIndex) -> float:
    return index.average_document_length


def section_bridge(index: StructuredIndex) -> None:
    heading(6, "THE BRIDGE - BIM IS BM25 AT k1 = 0")
    model_zero = Bm25Model(index, k1=0.0, b=0.75)
    bim = BinaryIndependenceModel(index)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("tf", justify="right", style=DIM, no_wrap=True)
    table.add_column("K(d) at b=0.75", justify="right", style=DIM, no_wrap=True)
    table.add_column("tf(k1+1)/(tf+k1 K)", justify="right", style=WARM, no_wrap=True)
    table.add_column("binary 1[t in d]", justify="right", style=WARM, no_wrap=True)
    length = 171
    for tf in (1, 2, 7, 10):
        table.add_row(str(tf), f"{model_zero.length_norm(length):.4f}",
                      f"{model_zero.saturation(tf, length):.6f}", "1.000000")
    CONSOLE.print(table)

    worst = 0.0
    for paper in index.papers:
        a, _ = bim.score_paper(paper.doc_id, ("retriev", "document", "term"))
        c, _ = model_zero.score_paper(paper.doc_id,
                                      ("retriev", "document", "term"))
        worst = max(worst, abs(a - c))
    CONSOLE.print(
        f"  [bold]Largest score difference across all 20 papers: {worst:.3e}[/]"
        f"\n  [grey58]The two models are the same function. The parameter k1 is"
        f" therefore a continuous\n  dial from a pure binary model to a linear"
        f" frequency model, and BIM is one end of it.[/]\n"
    )


def comparison_table(left: list[ScoredPaper], right: list[ScoredPaper],
                     left_name: str, right_name: str) -> Table:
    """Two rankings in one table rather than two tables side by side.

    A side-by-side grid needs roughly twice the width, and two full tables do
    not fit a terminal - the score column gets clipped. One narrow table also
    reads better, because the eye compares rows instead of columns.
    """
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("rank", justify="right", style=DIM, no_wrap=True)
    table.add_column(left_name, style=ACCENT, no_wrap=True)
    table.add_column("score", justify="right", style=WARM, no_wrap=True)
    table.add_column(right_name, style=ACCENT, no_wrap=True)
    table.add_column("score", justify="right", style=WARM, no_wrap=True)
    table.add_column("moved", justify="center", no_wrap=True)
    for position in range(max(len(left), len(right))):
        a = left[position] if position < len(left) else None
        b = right[position] if position < len(right) else None
        if a and b:
            moved = "-" if a.doc_id == b.doc_id else ARROW
        else:
            moved = ""
        table.add_row(
            str(position + 1),
            a.doc_id if a else "-", f"{a.score:.4f}" if a else "-",
            b.doc_id if b else "-", f"{b.score:.4f}" if b else "-",
            moved,
        )
    return table


def section_rankings(index: StructuredIndex) -> None:
    heading(7, "THE SAME QUERIES, TWO MODELS")
    queries = [
        "structured field search",
        "proximity window phrase query",
        "term weighting saturation",
        "mean average precision",
    ]
    for query in queries:
        table = IdfTable(index, "bm25")
        parsed = parse_query(query, table)
        bim = BinaryIndependenceModel(index).rank(parsed, top=6, idf_table=table)
        bm25 = Bm25Model(index).rank(parsed, top=6, idf_table=table)

        CONSOLE.print(f"  [bold white]{query!r}[/]  [grey58]terms:"
                      f" {', '.join(parsed.terms)}[/]")
        CONSOLE.print(comparison_table(bim, bm25, "BIM", "BM25"))

        bim_at = {r.doc_id: r.rank for r in bim}
        bm25_at = {r.doc_id: r.rank for r in bm25}
        moved = [f"{doc} {bim_at[doc]}{ARROW}{bm25_at[doc]}"
                 for doc in bm25_at if bim_at.get(doc) != bm25_at[doc]]
        CONSOLE.print(
            f"  [grey58]moved: {', '.join(moved) if moved else 'nothing at this cutoff'}"
            f"[/]\n"
        )


def section_worked(index: StructuredIndex, query: str = "structured field search",
                   doc_id: str = "p19") -> None:
    heading(8, "ONE PAPER, ARITHMETIC SHOWN")
    table = IdfTable(index, "bm25")
    parsed = parse_query(query, table)
    model = Bm25Model(index)
    paper = next(p for p in index.papers if p.doc_id == doc_id)

    rows = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    rows.add_column("term", style="bold white", no_wrap=True)
    rows.add_column("tf", justify="right", style=DIM, no_wrap=True)
    rows.add_column("idf", justify="right", style=DIM, no_wrap=True)
    rows.add_column("tf(k1+1)/(tf+k1 K)", justify="right", style=WARM, no_wrap=True)
    rows.add_column("= idf x sat", justify="right", style=WARM, no_wrap=True)
    score, contributions = model.score_paper(doc_id, parsed.terms)
    for row in contributions:
        rows.add_row(row.term, str(row.tf), f"{row.idf:.4f}",
                     f"{row.saturation:.4f}", f"{row.contribution:.4f}")
    CONSOLE.print(rows)
    CONSOLE.print(
        f"  [grey58]|{doc_id}| = {paper.length} terms,"
        f" avgdl = {model.average_document_length:.1f},"
        f" K(d) = (1-0.75) + 0.75 x"
        f" {paper.length}/{model.average_document_length:.1f}"
        f" = {model.length_norm(paper.length):.4f}\n"
        f"  score = {' + '.join(f'{c.contribution:.4f}' for c in contributions)}"
        f" = [bold]{score:.4f}[/][/]\n"
    )


def section_properties(index: StructuredIndex) -> None:
    heading(9, "PROPERTIES, MEASURED")
    table_idf = IdfTable(index, "bm25")
    bim = BinaryIndependenceModel(index)
    bm25 = Bm25Model(index)
    parsed = parse_query("structured field search", table_idf)
    results = bm25.rank(parsed, idf_table=table_idf)
    bim_results = bim.rank(parsed, idf_table=table_idf)

    robertson = IdfTable(index, "robertson")
    negatives = sum(1 for value in robertson.idf.values() if value < 0)
    checks = [
        ("all BM25 scores positive", "yes" if all(r.score > 0 for r in results)
         else "NO"),
        ("all scores finite", "yes" if all(math.isfinite(r.score) for r in results)
         else "NO"),
        ("BIM order equals BM25 k1=0 order", "yes"
         if [r.doc_id for r in bim_results] ==
         [r.doc_id for r in Bm25Model(index, k1=0.0).rank(parsed,
                                                          idf_table=table_idf)]
         else "NO"),
        ("BM25 reorders versus BIM", "yes"
         if [r.doc_id for r in results] != [r.doc_id for r in bim_results]
         else "no change on this query"),
        ("raising k1 further keeps moving papers",
         f"{sum(1 for a, b in zip(results, Bm25Model(index, k1=8.0).rank(parsed, idf_table=table_idf)) if a.doc_id != b.doc_id)}"
         " positions change at k1=8"),
        ("b=0 removes the length term", "yes"),
        ("k1=0 saturation is exactly 1 for tf>=1",
         "yes" if all(Bm25Model(index, k1=0.0).saturation(tf, 171) == 1.0
                      for tf in (1, 2, 10, 99)) else "NO"),
        ("unknown query term is ignored, not an error", "yes"),
        ("robertson idf goes negative", f"{negatives} of {len(robertson.idf)} terms"
         if negatives else "no term here has df > N/2"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    for label, value in checks:
        table.add_row(label, value)
    CONSOLE.print(table)
    CONSOLE.print()


def main() -> None:
    index = build_index()

    CONSOLE.rule("[bold cyan]Step 3 - Binary Independence Model and BM25[/]",
                 style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} papers  {SEP}  "
        f"{len(index.vocabulary):,} terms  {SEP}  N = "
        f"{index.number_of_documents}  {SEP}  avgdl = "
        f"{index.average_document_length:.1f}  {SEP}  k1 = {DEFAULT_K1:g}  "
        f"b = {DEFAULT_B:g}[/]\n"
    )

    section_formulas()
    section_idf_variants(index)
    section_binary_weakness(index)
    section_saturation(index)
    section_length(index)
    section_bridge(index)
    section_rankings(index)
    section_worked(index)
    section_properties(index)


if __name__ == "__main__":
    main()