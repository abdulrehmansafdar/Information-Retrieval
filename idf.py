"""Step 5 - Term frequency, document frequency and IDF.

This step answers the question term frequency alone cannot: is a word worth
anything, or does it appear in everything?

    tf(t,d)  how often term t occurs in document d   (already stored in step 3)
    df(t)    how many documents contain t            (the postings-list length)
    idf(t)   how much that term is worth             log(N / df)

    data       tf=4 in doc03   df=17   idf=0.16   frequent here, useless for ranking
    clinic     tf=2 in doc03   df=2    idf=2.30   rarer, and far more informative

Raw tf ranks the word that appears everywhere above the word that identifies a
document. idf inverts that ordering. Multiplying the two is TF-IDF, which is
step 6 and is deliberately NOT computed here.

Formula
-------
    idf(t) = log(N / df(t))

N is the number of documents, 20. With max df = 17, no idf in this corpus is zero
or negative, so no smoothing variant is needed. See the guide for why that matters.

Run:  python idf.py
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from index import InvertedIndex, build_index
from preprocess import Document
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# term frequency
# ---------------------------------------------------------------------------


def term_frequencies(terms: list[str]) -> Counter:
    """Count occurrences of each term in one document's term list."""
    return Counter(terms)


# ---------------------------------------------------------------------------
# inverse document frequency
# ---------------------------------------------------------------------------


@dataclass
class IdfModel:
    """The IDF table for a corpus, plus the numbers it was built from."""

    number_of_documents: int
    idf: dict[str, float] = field(default_factory=dict)
    document_frequency: dict[str, int] = field(default_factory=dict)

    def idf_for(self, term: str) -> float:
        """Weight of a term. A term outside the corpus is given the weight of a
        hapax (df=1), the maximum available. This convention never affects a
        score, because such a term matches no document."""
        if term in self.idf:
            return self.idf[term]
        return math.log(self.number_of_documents / 1)

    def df_for(self, term: str) -> int:
        return self.document_frequency.get(term, 0)

    def rarest_terms(self, n: int) -> list[tuple[str, float]]:
        return sorted(self.idf.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def most_common_terms(self, n: int) -> list[tuple[str, float]]:
        return sorted(self.idf.items(), key=lambda kv: (kv[1], kv[0]))[:n]

    def distinct_df_values(self) -> list[tuple[int, int, float]]:
        """(df, how many terms have that df, idf for that df), ascending by df."""
        counts = Counter(self.document_frequency.values())
        n = self.number_of_documents
        return [(df, counts[df], math.log(n / df)) for df in sorted(counts)]


def build_idf_model(index: InvertedIndex) -> IdfModel:
    """Build the IDF table. df is the postings-list length, so it is free."""
    n = index.number_of_documents
    document_frequency = {
        term: len(index.postings[term]) for term in index.vocabulary
    }
    idf = {term: math.log(n / df) for term, df in document_frequency.items()}
    return IdfModel(
        number_of_documents=n,
        idf=idf,
        document_frequency=document_frequency,
    )


# ---------------------------------------------------------------------------
# console output
# ---------------------------------------------------------------------------

CONSOLE = Console()

ACCENT = "bold cyan"
DIM = "grey58"

# The Windows console is cp437 and stdout is cp1252, so block characters must not
# be hard-coded or they print as '?'. Pick them from the detected encoding.
_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"
ARROW = "→" if _UNICODE else "->"


def bar(fraction: float, width: int = 20) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(
        BAR_OFF * (width - filled), style="grey30"
    )


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def section_formulas() -> None:
    heading(1, "THE THREE MEASURES")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white")
    body.add_row("tf(t,d)", "term frequency - how often t occurs in document d")
    body.add_row("", "already stored in every posting since step 3")
    body.add_row("df(t)", "document frequency - how many documents contain t")
    body.add_row("", "the length of the postings list, so also free")
    body.add_row("idf(t)", "[bold]log( N / df(t) )[/]  - how much the term is worth")
    body.add_row("", "this step's only new number")
    CONSOLE.print(Panel(body, title="term weight", title_align="left",
                        border_style="grey35", padding=(0, 1)))


def section_tf(index: InvertedIndex, model: IdfModel, doc_id: str = "doc03") -> None:
    heading(2, f"TERM FREQUENCY - {doc_id}")
    document = next(d for d in index.documents if d.doc_id == doc_id)
    counts = term_frequencies(document.terms)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("tf", justify="right", style=ACCENT, no_wrap=True)
    table.add_column("tf share", width=16)
    ranked = counts.most_common(12)
    for term, tf in ranked:
        table.add_row(term, str(tf), bar(tf / ranked[0][1], 14))
    CONSOLE.print(table)

    longest = max(index.doc_lengths.values())
    lengths = sorted(index.doc_lengths.values())
    CONSOLE.print(
        f"  [grey58]{doc_id} holds {len(document.terms)} terms "
        f"({len(counts)} distinct). Corpus lengths run "
        f"{lengths[0]}-{longest} terms, a ratio of only {longest / lengths[0]:.2f}.[/]\n"
    )


def section_why_tf_fails(index: InvertedIndex, model: IdfModel) -> None:
    heading(3, "WHY TF ALONE IS NOT ENOUGH")
    document = next(d for d in index.documents if d.doc_id == "doc03")
    counts = term_frequencies(document.terms)
    rows = [(t, c) for t, c in counts.items() if c >= 2]
    rows.sort(key=lambda row: (-row[1], model.df_for(row[0])))

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("tf", justify="right", style=ACCENT, no_wrap=True)
    table.add_column("df", justify="right", no_wrap=True)
    table.add_column("idf", justify="right", style="bold yellow", no_wrap=True)
    for term, tf in rows:
        table.add_row(term, str(tf), str(model.df_for(term)),
                      f"{model.idf_for(term):.4f}")
    CONSOLE.print(table)

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("problem", "doc03's most frequent term is 'use' with tf=4, but 'use'"
                            " appears in 17 of 20 documents. TF promotes a word"
                            " that says nothing.")
    body.add_row("", "'clinic' and 'patient' reach only tf=2 - half the count - yet"
                     " appear in just 2 documents and are worth 14x more.")
    body.add_row("worse", "tf=3 means opposite things in different documents:"
                          " doc03's 'use' is worth almost nothing, doc13's 'autonom'"
                          " appears in 1 document and is worth the most.")
    body.add_row("fix", "multiply tf by idf. That is TF-IDF, and it is step 6 -"
                        " not computed here.")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))

    lengths = sorted(index.doc_lengths.values())
    CONSOLE.print(
        f"  [grey58]An honest note on the other classic TF weakness: TF also"
        f" favours long\n  documents, but this corpus is nearly uniform"
        f" ({lengths[0]}-{lengths[-1]} terms,"
        f" {lengths[-1] / lengths[0]:.2f}x), so\n  that bias is real but mild"
        f" here. It would dominate on a corpus of mixed article lengths.[/]\n"
    )


def section_df(index: InvertedIndex, model: IdfModel) -> None:
    heading(4, "DOCUMENT FREQUENCY")
    rows = model.distinct_df_values()
    total = len(model.idf)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("df", justify="right", style=ACCENT, no_wrap=True)
    table.add_column("terms", justify="right", no_wrap=True)
    table.add_column("share of vocabulary", width=20)
    for df, count, _ in rows:
        cell = bar(count / total, 16) + Text(f" {count / total:>3.0%}", style=DIM)
        table.add_row(str(df), str(count), cell)
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]{len(rows)} distinct df values across {total} terms."
        f" df needs no new computation - it is the\n  postings-list length from"
        f" step 3, which is why this step is cheap.[/]\n"
    )


def section_idf(index: InvertedIndex, model: IdfModel) -> None:
    heading(5, "INVERSE DOCUMENT FREQUENCY")
    rows = model.distinct_df_values()
    largest = max(count for _, count, _ in rows)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("df", justify="right", style=ACCENT, no_wrap=True)
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    table.add_column("idf", justify="right", style="bold yellow", no_wrap=True)
    table.add_column("weight", width=20)
    top_idf = math.log(model.number_of_documents / 1)
    for df, count, value in rows:
        table.add_row(str(df), str(count), f"{value:.4f}",
                      bar(value / top_idf, 18))
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]Read down the idf column: as df rises, idf falls. That is the"
        f" whole idea.[/]\n"
    )


def section_key_terms(index: InvertedIndex, model: IdfModel) -> None:
    heading(6, "KEY TERMS, WEIGHTED")
    probes = [
        "data", "system", "use", "model", "softwar", "artifici intellig",
        "machin learn", "secur", "comput vision", "healthcar", "nlp",
        "anomali detect", "autonom", "iot",
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("term", style="bold white", no_wrap=True)
    table.add_column("df", justify="right", no_wrap=True)
    table.add_column("idf", justify="right", style="bold yellow", no_wrap=True)
    table.add_column("verdict", style=DIM, overflow="fold")
    for term in probes:
        df = model.df_for(term)
        verdict = ("worthless for ranking" if df >= 15 else
                   "weak" if df >= 9 else
                   "useful" if df >= 4 else
                   "very useful" if df >= 2 else
                   "identifies its document")
        table.add_row(term, str(df), f"{model.idf_for(term):.4f}", verdict)
    CONSOLE.print(table)

    rarest = model.rarest_terms(6)
    common = model.most_common_terms(6)
    extremes = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    extremes.add_column("extremes", style=ACCENT, no_wrap=True)
    extremes.add_column("term", style="white", no_wrap=True)
    extremes.add_column("df", justify="right", no_wrap=True)
    extremes.add_column("idf", justify="right", style="bold yellow", no_wrap=True)
    for term, value in common:
        extremes.add_row("most common", term, str(model.df_for(term)),
                         f"{value:.4f}")
    for term, value in rarest:
        extremes.add_row("rarest", term, str(model.df_for(term)), f"{value:.4f}")
    CONSOLE.print(extremes)

    weights = [value for value in model.idf.values()]
    ratio = max(weights) / min(weights)
    article = "an" if str(ratio)[0] in "8" else "a"
    CONSOLE.print(
        f"  [grey58]Range {min(weights):.4f} to {max(weights):.4f}, {article}"
        f" {ratio:.1f}x spread between the least and\n  most useful term"
        f" in the corpus.[/]\n"
    )


def section_properties(model: IdfModel) -> None:
    heading(7, "PROPERTIES OF THIS IDF TABLE")
    weights = list(model.idf.values())
    max_df = max(model.document_frequency.values())
    checks = [
        ("N = number of documents", str(model.number_of_documents)),
        ("terms in the vocabulary", f"{len(model.idf):,}"),
        ("highest df in the corpus", f"{max_df} of {model.number_of_documents}"),
        ("any idf equal to 0?", "no" if all(v > 0 for v in weights) else "YES"),
        ("any idf negative?", "no" if all(v >= 0 for v in weights) else "YES"),
        ("smoothing needed?", "no, because df never reaches N"),
        ("monotonically decreasing in df?", "yes, log(N/df) is strictly decreasing"),
    ]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    for label, value in checks:
        table.add_row(label, value)
    CONSOLE.print(table)

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("why this matters", "If some term appeared in all 20 documents its"
                                    " idf would be exactly 0, correctly saying the"
                                    " term cannot tell documents apart.")
    body.add_row("", "Common practice adds 1 or smooths df to avoid that zero."
                    " This corpus does not need it, and the plain textbook"
                    " formula is easier to defend.")
    body.add_row("", "Formula variants seen in the literature: log(N/df),"
                    " log(N/(1+df)), 1 + log(N/df) (Lucene),"
                    " log((N+1)/(df+1)) + 1 (sklearn).")
    CONSOLE.print(Panel(body, padding=(0, 1)))
    CONSOLE.print()


def main() -> None:
    index = build_index()
    model = build_idf_model(index)

    CONSOLE.rule("[bold cyan]Step 5 - TF, DF and IDF[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]N = {model.number_of_documents}  {SEP}  "
        f"{len(model.idf):,} terms  {SEP}  "
        f"idf(t) = log(N / df(t))[/]\n"
    )

    section_formulas()
    section_tf(index, model)
    section_why_tf_fails(index, model)
    section_df(index, model)
    section_idf(index, model)
    section_key_terms(index, model)
    section_properties(model)


if __name__ == "__main__":
    main()