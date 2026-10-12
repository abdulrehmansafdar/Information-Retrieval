"""Step 7 - Does any of this actually work?

Every number printed by steps 3 to 6 is a property of the MODEL, not of its
quality. A system that ranks p01 first for "proximity window phrase query" is
not more right than one that ranks p19 first unless someone has said which
paper answers the question. That is what this file is for.

The judgements live in data/relevance/qrels.txt, in TREC format:

    <query_id> <paper_id> <0 or 1>

A line of 0 means an assessor looked at that paper for that query and decided
it does NOT answer it. Those lines matter as much as the 1s: without them,
recall and MAP are undefined, and a system that returns nothing looks perfect.

Twelve queries, judged by hand against the 20 papers in this corpus. They were
chosen to cover the cases the earlier steps claim to differ on:

    q01 single clear topic            BM25 and BIM both find it
    q02 two-word topic               df matters
    q03 title-only topic             field restriction must matter
    q04 results-only topic           the title must NOT win
    q05 terms spread over the paper  non-overlapped must beat overlapped
    q06 terms adjacent in one place  overlapped must beat non-overlapped
    q07 an evaluation topic          several papers are genuinely relevant
    q08 a method topic               several papers, different vocabulary
    q09 a term in only one paper     highest idf in the corpus
    q10 a term in most papers        lowest idf, should barely matter
    q11 proximity wording            the window should be what matters
    q12 metadata wording             field weighting should be what matters

The metrics
-----------
    Precision@k   of the first k results, how many were judged relevant
    Recall@k      of everything judged relevant, how many are in the first k
    MRR           1 / rank of the first relevant result, averaged
    MAP           mean over queries of the average of Precision@rank over
                  every relevant result found

MRR only looks at the first hit and MAP only counts hits, so a model that puts
one good answer first and then fills with noise looks good on MRR and bad on
MAP. Reporting both is the minimum honest thing to do.

Run:  python evaluate.py
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

QRELS_PATH = Path(__file__).resolve().parent / "data" / "relevance" / "qrels.txt"

DEFAULT_CUTOFFS = (1, 3, 5, 10)


@dataclass
class Judgements:
    """query id -> paper id -> 0 or 1."""

    relevant: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    judged: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    queries: dict[str, str] = field(default_factory=dict)

    def is_relevant(self, query_id: str, doc_id: str) -> bool:
        return doc_id in self.relevant[query_id]

    @property
    def query_ids(self) -> list[str]:
        return sorted(self.queries)


def load_qrels(path: Path | str = QRELS_PATH) -> Judgements:
    """Read a TREC qrels file.

    Judgement lines are `<query_id> <paper_id> <0 or 1>`. Blank lines and lines
    starting with # are ignored, and a comment of the form

        # q01 | the query text

    supplies the human-readable query. Storing the text in the file rather than
    in the Python means the evaluation can be re-run, or read by someone else,
    without opening this module.

    Every pair is recorded in `judged` as well as `relevant`, because a 0 is a
    decision - it is the only thing that makes precision meaningful.
    """
    out = Judgements()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if "|" in body:
                query_id, _, query_text = body.partition("|")
                out.queries[query_id.strip()] = query_text.strip()
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        query_id, doc_id, grade = parts[0], parts[1], parts[2]
        out.queries.setdefault(query_id, query_id)
        out.judged[query_id].add(doc_id)
        try:
            value = int(grade)
        except ValueError:
            continue
        if value > 0:
            out.relevant[query_id].add(doc_id)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """How many of the first k are relevant. 0.0 when k is 0."""
    if k <= 0:
        return 0.0
    top = ranking[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / len(top)


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """How many of everything relevant are in the first k.

    Undefined when nothing is judged relevant, so it is defined to be 0.0 and
    the query is excluded from the average by the caller instead.
    """
    if not relevant:
        return 0.0
    return sum(1 for doc in ranking[:k] if doc in relevant) / len(relevant)


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant hit. 0.0 when there is none."""
    for position, doc in enumerate(ranking, 1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def average_precision(ranking: list[str], relevant: set[str]) -> float:
    """Sum of Precision@rank over every relevant hit, divided by the number of
    relevant hits.

    A relevant hit at rank 1 out of 20 scores 1.0/20 = 0.05; one at rank 3 out
    of 20 scores (1/3 + 2/20) = 0.433. The sum is NOT truncated at any cutoff,
    which is why MAP is called the untruncated average.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for position, doc in enumerate(ranking, 1):
        if doc in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant)


@dataclass
class QueryScore:
    """One query under one model."""

    query_id: str
    text: str
    model: str
    ranking: list[str]
    relevant_total: int
    found: int = 0
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    rr: float = 0.0
    ap: float = 0.0

    @property
    def complete(self) -> bool:
        return self.relevant_total > 0 and self.found == self.relevant_total


@dataclass
class ModelScore:
    """Every query under one model, plus the averages."""

    model: str
    queries: list[QueryScore] = field(default_factory=list)
    description: str = ""

    def mean(self, attribute: str, cutoff: int | None = None) -> float:
        values = []
        for query in self.queries:
            if query.relevant_total == 0:
                continue
            if cutoff is not None:
                values.append(getattr(query, attribute)[cutoff])
            else:
                values.append(getattr(query, attribute))
        return statistics.fmean(values) if values else 0.0

    @property
    def mrr(self) -> float:
        return self.mean("rr")

    @property
    def map(self) -> float:
        return self.mean("ap")

    @property
    def queries_with_no_hit(self) -> list[str]:
        return [q.query_id for q in self.queries if q.rr == 0.0]


def score_model(model_name: str, ranking_fn, judgements: Judgements,
                cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
                description: str = "") -> ModelScore:
    """Run one model over every judged query and score it."""
    out = ModelScore(model=model_name, description=description)
    for query_id in judgements.query_ids:
        ranking = list(ranking_fn(query_id))
        relevant = judgements.relevant[query_id]
        score = QueryScore(
            query_id=query_id, text=judgements.queries[query_id],
            model=model_name, ranking=ranking, relevant_total=len(relevant),
            found=sum(1 for doc in ranking if doc in relevant),
        )
        for cutoff in cutoffs:
            score.precision[cutoff] = precision_at_k(ranking, relevant, cutoff)
            score.recall[cutoff] = recall_at_k(ranking, relevant, cutoff)
        score.rr = reciprocal_rank(ranking, relevant)
        score.ap = average_precision(ranking, relevant)
        out.queries.append(score)
    return out


def evaluate_all(judgements: Judgements, ranking_fns: dict,
                 cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS
                 ) -> list[ModelScore]:
    return [score_model(name, fn, judgements, cutoffs,
                        description=fn.description if hasattr(fn, "description") else "")
            for name, fn in ranking_fns.items()]


def query_text(judgements: Judgements) -> dict[str, str]:
    return dict(judgements.queries)


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


def bar(fraction: float, width: int = 14) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(BAR_OFF * (width - filled),
                                                      style="grey30")


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def section_judgements(judgements: Judgements) -> None:
    heading(1, "THE JUDGEMENTS")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("id", style=ACCENT, no_wrap=True)
    table.add_column("query", style="white", overflow="fold")
    table.add_column("relevant", justify="right", style=WARM, no_wrap=True)
    table.add_column("judged", justify="right", style=DIM, no_wrap=True)
    table.add_column("papers judged relevant", style=DIM, overflow="fold")
    for query_id in judgements.query_ids:
        relevant = judgements.relevant[query_id]
        judged = judgements.judged[query_id]
        table.add_row(query_id, judgements.queries[query_id], str(len(relevant)),
                      str(len(judged)), ", ".join(sorted(relevant)) or "-")
    CONSOLE.print(table)

    positives = sum(len(v) for v in judgements.relevant.values())
    negatives = sum(len(v) - len(judgements.relevant[v])
                    for v in judgements.judged)
    CONSOLE.print(
        f"  [grey58]{positives} positive and {negatives} negative judgements"
        f" across {len(judgements.query_ids)} queries.\n  The negatives are the"
        f" ones that make precision meaningful - without them a system that"
        f" returns one paper\n  and is right by luck scores 1.0.[/]\n"
    )


SHORT_NAMES = {
    "boolean AND": "bool",
    "tfidf cosine": "tfidf",
    "BIM": "bim",
    "BM25": "bm25",
    "BM25F": "bm25f",
    "overlapped": "overlap",
    "non-overlapped": "non-ovl",
    "proximal": "proxim",
}


def short(name: str) -> str:
    return SHORT_NAMES.get(name, name[:9])


def section_overall(scores: list[ModelScore]) -> None:
    heading(2, "OVERALL, EVERY MODEL ON EVERY QUERY")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("model", style=ACCENT, no_wrap=True)
    table.add_column("P@1", justify="right", style=WARM, no_wrap=True)
    table.add_column("P@5", justify="right", style=WARM, no_wrap=True)
    table.add_column("R@10", justify="right", style=WARM, no_wrap=True)
    table.add_column("MRR", justify="right", style=WARM, no_wrap=True)
    table.add_column("MAP", justify="right", style=WARM, no_wrap=True)
    table.add_column("nohit", justify="right", style=DIM, no_wrap=True)
    for score in scores:
        table.add_row(
            short(score.model),
            f"{score.mean('precision', 1):.3f}",
            f"{score.mean('precision', 5):.3f}",
            f"{score.mean('recall', 10):.3f}",
            f"{score.mrr:.3f}", f"{score.map:.3f}",
            str(len(score.queries_with_no_hit)),
        )
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]'nohit' counts queries where the model never returned a"
        " judged-relevant paper\n  at all. It is the number that exposes a"
        " model which is precise on the easy queries\n  and useless on the"
        " hard ones, and it is invisible in MRR.[/]\n"
    )


def section_per_query(scores: list[ModelScore], judgements: Judgements,
                      cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS) -> None:
    heading(3, "QUERY BY QUERY, MRR PER MODEL")
    names = [short(s.model) for s in scores]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("q", style=ACCENT, no_wrap=True)
    for name in names:
        table.add_column(name, justify="right", style=WARM, no_wrap=True)
    table.add_column("rel", justify="right", style=DIM, no_wrap=True)
    for query_id in judgements.query_ids:
        row = [query_id]
        for score in scores:
            match = next(q for q in score.queries if q.query_id == query_id)
            row.append(f"{match.rr:.2f}")
        row.append(str(len(judgements.relevant[query_id])))
        table.add_row(*row)
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]queries, in the order given in the qrels file. 1.00 means"
        f" the correct paper came\n  first, 0.00 means it never appeared at"
        f" all. Reading down a column shows which\n  query type each model is"
        f" good at.[/]\n"
    )


def section_precision_recall(scores: list[ModelScore],
                             cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS) -> None:
    heading(4, "PRECISION AND RECALL AT EVERY CUTOFF")
    for cutoff in cutoffs:
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
        table.add_column("model", style=ACCENT, no_wrap=True)
        table.add_column(f"P@{cutoff}", justify="right", style=WARM, no_wrap=True)
        table.add_column(f"R@{cutoff}", justify="right", style=WARM, no_wrap=True)
        table.add_column(f"P@{cutoff}", width=18)
        best = max((s.mean("precision", cutoff) for s in scores), default=0.0)
        for score in scores:
            precision = score.mean("precision", cutoff)
            table.add_row(score.model, f"{precision:.3f}",
                          f"{score.mean('recall', cutoff):.3f}",
                          bar(precision / best if best else 0.0, 16))
        CONSOLE.print(table)
        CONSOLE.print()


def section_diagnose(scores: list[ModelScore]) -> None:
    heading(5, "WHAT THE NUMBERS SAY")
    if not scores:
        return
    ranked = sorted(scores, key=lambda s: (-s.mrr, -s.map))
    best, worst = ranked[0], ranked[-1]
    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("best MRR", f"{best.model} at {best.mrr:.3f}")
    body.add_row("best MAP", max(scores, key=lambda s: s.map).model
                 + f" at {max(s.map for s in scores):.3f}")
    body.add_row("worst", f"{worst.model} at {worst.mrr:.3f}")
    spread = best.mrr - worst.mrr
    body.add_row("spread", f"{spread:.3f} across {len(scores)} models")
    body.add_row("read honestly", f"{spread:.3f} is the only thing these numbers"
                                  " support. Twelve hand-made queries over 20"
                                  " papers is a demo, not a benchmark, and no"
                                  " claim about 'which model is better in"
                                  " general' follows from it.")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))

    collisions = [s for s in scores if s.queries_with_no_hit]
    if collisions:
        names = ", ".join(f"{s.model} ({len(s.queries_with_no_hit)})"
                          for s in collisions)
        CONSOLE.print(f"  [grey58]Models that returned nothing for at least one"
                      f" query: {names}[/]\n")


def main(judgements: Judgements | None = None,
         ranking_fns: dict | None = None) -> None:
    from engine import evaluation_models, get_index, get_judgements

    judgements = judgements or get_judgements()
    ranking_fns = ranking_fns or evaluation_models(get_index())

    CONSOLE.rule("[bold cyan]Step 7 - Effectiveness Evaluation[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{len(judgements.query_ids)} judged queries  {SEP}  "
        f"{sum(len(v) for v in judgements.relevant.values())} relevant pairs"
        f"  {SEP}  cutoffs {', '.join(str(c) for c in DEFAULT_CUTOFFS)}[/]\n"
    )

    scores = evaluate_all(judgements, ranking_fns)
    section_judgements(judgements)
    section_overall(scores)
    section_per_query(scores, judgements)
    section_precision_recall(scores)
    section_diagnose(scores)


if __name__ == "__main__":
    main()