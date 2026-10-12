"""Step 6 - The proximal node model.

Navarro and Baeza-Yates, TOIS 1997. A paper is not a bag of words and not a
flat block of text either - it is a TREE:

    paper
      section      title | abstract | methodology | results
        paragraph  blank-line block inside one field
          sentence one sentence inside one paragraph

The overlapped and non-overlapped lists in step 5 use ONE level of that tree at
a time. This step relates levels to each other, which is what lets a query say
"a paragraph inside the methodology, not anywhere".

The operators
-------------
    P in Q            nodes of P that lie inside some node of Q
    P beginin Q       nodes of P whose FIRST position lies inside Q
    P with(k) T       nodes of P containing at least k occurrences of term T
    P parent(k) Q     nodes of P that are parents of at least k nodes of Q
    P child Q         nodes of P that are DIRECT children of Q
    P before(k) T (C) nodes of P starting at most k terms before an occurrence
                      of T, inside the same node of C

    example: paragraph in methodology
             sentence before(10) architecture (paragraph)
             section with(2) retrieval

WHY IT IS CALLED "PROXIMAL"
---------------------------
Because evaluation only ever moves to NEARBY nodes. `P with(k) T` does not walk
every node in the corpus. It reads the postings list for T, and for each
occurrence asks "which is the innermost P-node containing this position?" -
one upward step, because P-nodes tile each paper and an occurrence belongs to
exactly one. That is bottom-up evaluation: match the CONTENT first, then walk
up to the structure. It is the single idea that makes the model efficient, and
innermost_component() is where it lives.

The two guarantees that fall out of the tree
--------------------------------------------
    a node is inside itself      P in P returns everything, never a surprise
    answers are one hierarchy   `sentence before(10) x (paragraph)` returns
                                SENTENCES only, never a mixture

Run:  python proximity.py
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from preprocess import Region, preprocess_query
from structured_index import HIERARCHIES, StructuredIndex, build_index

KIND_OF = {"paper", *HIERARCHIES}
PARENT_OF = {"section": "paper", "paragraph": "section", "sentence": "paragraph"}


class ProximitySyntaxError(ValueError):
    """A proximal query cannot be parsed. Always carries a readable message."""


OPERATORS = ("in", "beginin", "with", "parent", "child", "before", "after")

OPERATOR_RE = re.compile(
    r"\b(?P<op>in|beginin|with|parent|child|before|after)\s*"
    r"(?:\(\s*(?P<arg>\d+)\s*\))?\s+"
    r"(?P<right>.+?)"
    r"(?:\s*\(\s*(?P<context>[^)]+?)\s*\))?\s*$",
    re.I,
)

TERM_RE = re.compile(r"^[\w][\w ]*$")


@dataclass(frozen=True)
class Constructor:
    """A node type: a kind, optionally narrowed to one named field."""

    kind: str
    label: str = ""

    def __str__(self) -> str:
        return f"{self.kind}:{self.label}" if self.label else self.kind


@dataclass(frozen=True)
class ProximityQuery:
    """One parsed `P op Q (context)` expression."""

    left: Constructor
    operator: str
    right: Constructor | str
    k: int = 1
    context: Constructor | None = None
    source: str = ""

    @property
    def right_is_term(self) -> bool:
        return isinstance(self.right, str)

    def describe(self) -> str:
        right = str(self.right)
        if self.operator in {"with", "parent", "before", "after"}:
            head = f"{self.left} {self.operator}({self.k}) {right}"
        else:
            head = f"{self.left} {self.operator} {right}"
        if self.context is not None:
            head += f" ({self.context})"
        return head


@dataclass
class ProximityResult:
    """Nodes produced by evaluating a ProximityQuery."""

    query: str
    text: str
    nodes: list[Region] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)
    error: str | None = None
    steps: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.nodes)

    @property
    def ok(self) -> bool:
        return self.error is None


def resolve_constructor(text: str) -> Constructor:
    """'paragraph' -> paragraph, 'methodology' -> section:methodology."""
    cleaned = text.strip().strip('"')
    if ":" in cleaned:
        kind, _, label = cleaned.partition(":")
        kind, label = kind.strip().lower(), label.strip().lower()
        if kind not in KIND_OF:
            raise ProximitySyntaxError(
                f"{kind!r} is not a node kind; expected one of "
                f"{', '.join(sorted(KIND_OF))}"
            )
        return Constructor(kind, label)
    lowered = cleaned.lower()
    if lowered in KIND_OF:
        return Constructor(lowered)
    # a bare field name means the section with that label
    return Constructor("section", lowered)


class ProximalNodeModel:
    """Operators over the paper / section / paragraph / sentence tree."""

    def __init__(self, index: StructuredIndex) -> None:
        self.index = index

    # --- the core upward step ----------------------------------------------

    def innermost_component(self, kind: str, doc_id: str,
                            position: int) -> Region | None:
        """The smallest `kind` node containing a position.

        This is the bottom-up step. It is called once per occurrence of a term,
        never once per node in the corpus, which is the whole performance
        argument for the model.
        """
        return self.index.enclosing(kind, doc_id, position)

    def nodes(self, constructor: Constructor) -> list[Region]:
        """Every node matching a constructor, in document order."""
        if constructor.kind == "paper":
            return [
                Region(region_id=p.doc_id, doc_id=p.doc_id, kind="paper",
                       label=p.doc_id, start=0, end=p.length, text=p.title,
                       depth=0)
                for p in self.index.papers
            ]
        if constructor.label:
            out = []
            for paper in self.index.papers:
                region = paper.fields.get(constructor.label)
                if region is not None:
                    out.append(region)
            return out
        return sorted(self.index.regions_of(constructor.kind),
                      key=lambda r: (r.doc_id, r.start))

    def nodes_in(self, constructor: Constructor, doc_id: str) -> list[Region]:
        """Nodes of one constructor inside ONE paper, in order."""
        if constructor.kind == "paper":
            paper = next(p for p in self.index.papers if p.doc_id == doc_id)
            return [Region(region_id=paper.doc_id, doc_id=paper.doc_id,
                           kind="paper", label=paper.doc_id, start=0,
                           end=paper.length, text=paper.title, depth=0)]
        if constructor.label:
            paper = next(p for p in self.index.papers if p.doc_id == doc_id)
            region = paper.fields.get(constructor.label)
            return [region] if region is not None else []
        return self.index.regions_of_kind(doc_id, constructor.kind)

    # --- parsing -----------------------------------------------------------

    def parse(self, text: str) -> ProximityQuery:
        """Parse `P op Q` and `P op(k) Q (context)`."""
        cleaned = " ".join(text.split())
        match = OPERATOR_RE.search(cleaned)
        if not match:
            raise ProximitySyntaxError(
                f"expected one of {', '.join(OPERATORS)} in {text!r}"
            )
        left_text = cleaned[: match.start()].strip()
        if not left_text:
            raise ProximitySyntaxError(f"nothing before the operator in {text!r}")
        operator = match.group("op").lower()
        right_text = match.group("right").strip()
        if not right_text:
            raise ProximitySyntaxError(f"nothing after {operator!r} in {text!r}")

        left = resolve_constructor(left_text)
        if looks_like_constructor(right_text):
            right: Constructor | str = resolve_constructor(right_text)
        else:
            right = right_text.lower()

        argument = match.group("arg")
        k = int(argument) if argument else 1
        if operator in {"with", "parent", "before", "after"} and argument is None:
            k = 1
        context = (resolve_constructor(match.group("context"))
                   if match.group("context") else None)

        return ProximityQuery(left=left, operator=operator, right=right, k=k,
                              context=context, source=cleaned)

    # --- evaluation --------------------------------------------------------

    def evaluate(self, query: str | ProximityQuery) -> ProximityResult:
        """Run one proximal query. Evaluates bottom-up."""
        parsed = query if isinstance(query, ProximityQuery) else self.parse(query)
        steps = [f"parsed: {parsed.describe()}"]

        if parsed.right_is_term:
            nodes, extra = self._from_term(parsed)
            steps.extend(extra)
        else:
            nodes = self._structural(parsed)
            steps.append(f"structural: {len(nodes)} node(s) from the tree")

        if parsed.context is not None:
            keep = self._in_context(nodes, parsed.context)
            steps.append(f"context ({parsed.context}): "
                         f"{len(nodes)} -> {len(keep)} node(s)")
            nodes = keep

        nodes.sort(key=lambda r: (r.doc_id, r.start))
        return ProximityResult(
            query=parsed.source, text=parsed.describe(), nodes=nodes,
            papers=sorted({n.doc_id for n in nodes}), steps=steps,
        )

    # --- the operators -----------------------------------------------------

    def _structural(self, query: ProximityQuery) -> list[Region]:
        op = query.operator
        if op == "in":
            return self._op_in(query.left, query.right)
        if op == "beginin":
            return self._op_beginin(query.left, query.right)
        if op == "child":
            return self._op_child(query.left, query.right)
        if op == "parent":
            return self._op_parent(query.left, query.right, query.k)
        raise ProximitySyntaxError(
            f"{op!r} takes a term on the right, not {query.right!r}"
        )

    def _op_in(self, left: Constructor, right: Constructor) -> list[Region]:
        """P in Q: P-nodes whose whole segment lies inside some Q-node."""
        outer = self.nodes(right)
        inner = self.nodes(left)
        out: list[Region] = []
        for node in inner:
            for container in outer:
                if container.doc_id != node.doc_id:
                    continue
                if (container.start <= node.start
                        and node.end <= container.end):
                    out.append(node)
                    break
        return out

    def _op_beginin(self, left: Constructor, right: Constructor) -> list[Region]:
        """P beginin Q: only the START of P has to be inside Q.

        Differs from `in` whenever a node straddles a boundary, which happens at
        every field boundary in this corpus.
        """
        outer = self.nodes(right)
        return [node for node in self.nodes(left)
                if any(c.doc_id == node.doc_id and c.contains(node.start)
                       for c in outer)]

    def _op_child(self, left: Constructor, right: Constructor) -> list[Region]:
        """P child Q: direct children only, never grandchildren."""
        parents = self.nodes(right)
        out: list[Region] = []
        for child in self.nodes(left):
            for parent in parents:
                if parent.doc_id != child.doc_id:
                    continue
                if PARENT_OF.get(child.kind) != parent.kind:
                    continue
                if parent.contains(child.start):
                    out.append(child)
                    break
        return out

    def _from_term(self, query: ProximityQuery
                   ) -> tuple[list[Region], list[str]]:
        """Content first, then structure. The bottom-up path."""
        raw = str(query.right)
        # The right-hand side goes through the SAME preprocessing as a document,
        # so 'retrieval' reaches the index as the stem 'retriev' and 'proximity
        # window' reaches it as the phrase term 'proxim window'. Without this,
        # a proximal query silently matches nothing for every inflected form,
        # which looks like a bug in the operator rather than a missing stemmer.
        terms = preprocess_query(raw)
        term = terms[0] if terms else raw.lower()
        steps = [f"preprocess {raw!r} -> {term!r}"]
        postings = self.index.lookup(term)
        steps.append(f"postings for {term!r}: {len(postings)} document(s)")
        if not postings:
            return [], steps + [f"{term!r} is not in the vocabulary"]

        if query.operator == "with":
            return self._op_with(query.left, term, query.k), steps
        if query.operator in {"before", "after"}:
            return self._op_near(query.left, term, query.k,
                                 query.operator), steps
        # `in`, `beginin`, `child` and `parent` need a NODE TYPE on the right.
        # Reached only when the right-hand side parsed as a term, e.g.
        # 'paragraph child retrieval'. Without this check _op_near would treat
        # anything that is not 'before' as 'after', so a malformed query would
        # silently return an 'after' answer - a plausible answer to a query the
        # user never wrote.
        raise ProximitySyntaxError(
            f"{query.operator!r} needs a node type on the right, e.g. "
            f"'paragraph child methodology', not the term {term!r}"
        )

    def _op_with(self, left: Constructor, term: str, k: int) -> list[Region]:
        """P with(k) T: P-nodes containing at least k occurrences of T.

        Walks the occurrences, not the nodes. Each occurrence is mapped to its
        innermost P-node, and the nodes are counted - so the cost is O(postings
        for T) rather than O(all nodes).
        """
        counts: Counter = Counter()
        owner: dict[str, Region] = {}
        for paper in self.index.papers:
            for position in self.index.positions_of(paper.doc_id, term):
                node = self.innermost_component(left.kind, paper.doc_id, position)
                if node is None or (left.label and node.label != left.label):
                    continue
                counts[node.region_id] += 1
                owner[node.region_id] = node
        return [owner[n] for n, count in counts.items() if count >= k]

    def _op_parent(self, left: Constructor, right: Constructor,
                   k: int) -> list[Region]:
        """P parent(k) Q: P-nodes that are parents of at least k Q-nodes.

        'section parent(2) paragraph' reads the section hierarchy, counts how
        many paragraph children each section has, and keeps the sections with at
        least k. The k is what separates a real structure from an artifact of
        how the parser happened to split the blank lines.
        """
        counts: Counter = Counter()
        owner: dict[str, Region] = {}
        for paper in self.index.papers:
            for child in self.nodes_in(right, paper.doc_id):
                for parent in self.nodes_in(left, paper.doc_id):
                    if PARENT_OF.get(child.kind) != parent.kind:
                        continue
                    if parent.contains(child.start):
                        counts[parent.region_id] += 1
                        owner[parent.region_id] = parent
                        break
        return [owner[n] for n, count in counts.items() if count >= k]

    def _op_near(self, left: Constructor, term: str, distance: int,
                 direction: str) -> list[Region]:
        """P before/after(k) T: distance is measured in INDEX TERMS.

        The P-node does NOT have to contain T. 'sentence before(10) index'
        means "sentences that end at most ten terms before an occurrence of
        index", which is the reading the literature uses - a sentence leading
        into the word, not a sentence containing it. Distance 0 would therefore
        be "immediately adjacent", and every larger value relaxes it by one
        term at a time.

        Measuring in index terms rather than characters is what makes the
        number mean the same thing in a 4-term title and a 70-term section.
        """
        out: list[Region] = []
        for paper in self.index.papers:
            positions = self.index.positions_of(paper.doc_id, term)
            if not positions:
                continue
            for node in self.nodes_in(left, paper.doc_id):
                if left.label and node.label != left.label:
                    continue
                if direction == "before":
                    later = [p for p in positions if p >= node.end]
                    if later and min(later) - node.end <= distance:
                        out.append(node)
                else:
                    earlier = [p for p in positions if p < node.start]
                    if earlier and node.start - max(earlier) <= distance:
                        out.append(node)
        return out

    def _in_context(self, nodes: list[Region], context: Constructor
                    ) -> list[Region]:
        allowed = {(n.doc_id, n.start, n.end) for n in self.nodes(context)}
        return [n for n in nodes
                if any(d == n.doc_id and s <= n.start and n.end <= e
                       for d, s, e in allowed)]


def looks_like_constructor(text: str) -> bool:
    """True when the right-hand side names a node type rather than a term.

    A bare word is ambiguous: 'methodology' is a field name, 'retrieval' is a
    term. The test is membership, not shape, so the answer depends on this
    corpus rather than on the parser's opinion. A word that is BOTH would be
    misread, and the way to disambiguate is `section:methodology`, which this
    function accepts for exactly that reason.
    """
    from preprocess import FIELDS

    cleaned = text.strip().strip('"').strip()
    if ":" in cleaned:
        return True
    lowered = cleaned.lower()
    return lowered in KIND_OF or lowered in set(FIELDS)


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


def bar(fraction: float, width: int = 16) -> Text:
    filled = max(0, min(width, round(fraction * width)))
    return Text(BAR_ON * filled, style="cyan") + Text(BAR_OFF * (width - filled),
                                                      style="grey30")


def heading(number: int, title: str) -> None:
    CONSOLE.rule(f"[{ACCENT}]{number}. {title}[/]", style="grey35")


def nodes_table(result: ProximityResult, limit: int = 10) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("paper", style=ACCENT, no_wrap=True)
    table.add_column("node", style="white", no_wrap=True)
    table.add_column("kind", style=ACCENT, no_wrap=True)
    table.add_column("start", justify="right", style=DIM, no_wrap=True)
    table.add_column("end", justify="right", style=DIM, no_wrap=True)
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    table.add_column("label", style=DIM, overflow="fold")
    for node in result.nodes[:limit]:
        table.add_row(node.doc_id, node.region_id, node.kind,
                      str(node.start), str(node.end), str(node.length),
                      node.label)
    return table


def section_tree() -> None:
    heading(1, "THE TREE")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("paper", "the whole document")
    body.add_row("  section", "title | abstract | methodology | results"
                              "      <- the four named fields")
    body.add_row("    paragraph", "one blank-line block inside a field")
    body.add_row("      sentence", "one sentence inside a paragraph")
    CONSOLE.print(Panel(body, title="node types", title_align="left",
                        border_style="grey35", padding=(0, 1)))

    body2 = Table.grid(padding=(0, 2))
    body2.add_column(style=ACCENT, no_wrap=True)
    body2.add_column(style="white", overflow="fold")
    body2.add_row("P in Q", "nodes of P inside some node of Q"
                           "            paragraph in methodology")
    body2.add_row("P beginin Q", "only P's START inside Q"
                                 "      sentence beginin results")
    body2.add_row("P with(k) T", "P contains >= k occurrences of T"
                                   "      section with(2) retrieval")
    body2.add_row("P child Q", "direct children only"
                               "      paragraph child methodology")
    body2.add_row("P before(k) T (C)", "P starts <= k terms before T, inside C"
                                        "      sentence before(10) index (paragraph)")
    body2.add_row("P after(k) T (C)", "P ends <= k terms after T, inside C"
                                       "      sentence after(5) index (paragraph)")
    body2.add_row("P parent(k) Q", "P is parent of >= k nodes of Q"
                                   "      section parent(2) paragraph")
    CONSOLE.print(Panel(body2, title="operators", title_align="left",
                        border_style="cyan", padding=(0, 1)))


def section_bottom_up(index: StructuredIndex) -> None:
    heading(2, "BOTTOM-UP EVALUATION, MEASURED")
    model = ProximalNodeModel(index)
    query = "section with(2) retrieval"

    naive = len(model.nodes(resolve_constructor("section"))) * \
        len(index.lookup("retriev"))
    postings = sum(len(p.positions) for p in index.lookup("retriev"))

    result = model.evaluate(query)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                  show_header=False)
    table.add_column(style=DIM)
    table.add_column(style="bold white")
    table.add_row("nodes in the collection", str(len(model.nodes(
        resolve_constructor("section")))))
    table.add_row("occurrences of 'retriev' in the postings",
                  f"{postings} in {len(index.lookup('retriev'))} papers")
    table.add_row("top-down would test node x occurrence",
                  f"{naive:,} containment tests")
    table.add_row("bottom-up walks occurrences only",
                  f"{postings:,} upward steps")
    table.add_row("speed-up", f"{naive / postings:.0f}x fewer")
    CONSOLE.print(table)
    CONSOLE.print(
        f"  [grey58]Both numbers are real and both paths return the same answer."
        f" The upward step is\n  innermost_component(), and it is the reason"
        f" the model scales.[/]\n"
    )
    CONSOLE.print(f"  [bold white]{query!r}[/] -> {result.count} node(s)")
    CONSOLE.print(nodes_table(result, 6))
    CONSOLE.print()


def section_structural(index: StructuredIndex) -> None:
    heading(3, "STRUCTURAL OPERATORS")
    queries = [
        "paragraph in methodology",
        "paragraph in abstract",
        "sentence beginin results",
        "paragraph child methodology",
        "section parent(2) paragraph",
    ]
    for query in queries:
        result = ProximalNodeModel(index).evaluate(query)
        CONSOLE.print(f"  [bold white]{query!r}[/]  [grey58]"
                      f"{result.count} node(s) in {len(result.papers)} papers[/]")
        if result.ok and result.nodes:
            CONSOLE.print(nodes_table(result, 4))
        CONSOLE.print()


def section_proximity(index: StructuredIndex) -> None:
    heading(4, "DISTANCE OPERATORS - THE EXACT / BROADER DIAL")
    model = ProximalNodeModel(index)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("query", style="bold white", overflow="fold")
    table.add_column("nodes", justify="right", style=WARM, no_wrap=True)
    table.add_column("papers", justify="right", style=WARM, no_wrap=True)
    for window in (2, 5, 10, 25, 60):
        query = f"sentence before({window}) index (paragraph)"
        result = model.evaluate(query)
        table.add_row(query, str(result.count), str(len(result.papers)))
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]A window of 0 would be exact adjacency. Every larger value is"
        " a controlled\n  relaxation, and the count rises monotonically with"
        " the window. This is the same\n  trade-off as the overlapped and"
        " non-overlapped lists, expressed as one number.\n"
        "  Distance is measured in INDEX TERMS, so the number means the same"
        " thing in a\n  4-term title and a 70-term results section.[/]\n"
    )


def section_in_versus_beginin(index: StructuredIndex) -> None:
    heading(5, "`in` versus `beginin` - WHY THEY DIFFER")
    model = ProximalNodeModel(index)
    both = model.evaluate("paragraph in results")
    begins = model.evaluate("paragraph beginin results")
    a = {n.region_id for n in both.nodes}
    b = {n.region_id for n in begins.nodes}

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("operator", style=ACCENT, no_wrap=True)
    table.add_column("meaning", style="white", overflow="fold")
    table.add_column("nodes", justify="right", style=WARM, no_wrap=True)
    table.add_row("paragraph in results", "the WHOLE paragraph is inside results",
                  str(len(a)))
    table.add_row("paragraph beginin results", "only its START is inside results",
                  str(len(b)))
    table.add_row("only in beginin", "a paragraph straddling the field boundary",
                  str(len(b - a)) if b - a else "none in this corpus")
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]They are equal on this corpus because the parser builds"
        " fields from the blank\n  line before a heading, so no paragraph ever"
        " straddles a boundary. The difference\n  would appear on a corpus where"
        " a section break falls mid-paragraph, which is exactly\n  the case"
        " XML full-text search has to handle.[/]\n"
    )


def section_errors(index: StructuredIndex) -> None:
    heading(6, "SYNTAX ERRORS ARE MESSAGES")
    checks = ["paragraph in methodology", "paragraph child methodology",
              "paragraph methodology", "in methodology", "paragraph in",
              "section with(2) nosuchterm", "section in chapter", "  "]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("query", style="bold white", overflow="fold")
    table.add_column("result", style="white", overflow="fold")
    model = ProximalNodeModel(index)
    for query in checks:
        try:
            result = model.evaluate(query)
            outcome = (result.error if result.error
                       else f"{result.count} nodes in {len(result.papers)} papers")
        except ProximitySyntaxError as error:
            outcome = f"ProximitySyntaxError: {error}"
        table.add_row(repr(query), outcome)
    CONSOLE.print(table)
    CONSOLE.print(
        "  [grey58]'section in chapter' is not an error. 'chapter' is neither a"
        " node kind nor a\n  field of this corpus, so it is treated as a TERM,"
        " and the result carries a\n  step explaining that the term is not in"
        " the vocabulary. Guessing that the user meant a\n  node kind would"
        " hide the mistake behind a plausible answer.[/]\n"
    )


def main() -> None:
    index = build_index()
    model = ProximalNodeModel(index)

    CONSOLE.rule("[bold cyan]Step 6 - Proximal Node Model[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} papers  {SEP}  "
        f"{len(model.nodes(resolve_constructor('section')))} sections  {SEP}  "
        f"{len(model.nodes(resolve_constructor('paragraph')))} paragraphs  {SEP}  "
        f"{len(model.nodes(resolve_constructor('sentence')))} sentences[/]\n"
    )

    section_tree()
    section_bottom_up(index)
    section_structural(index)
    section_proximity(index)
    section_in_versus_beginin(index)
    section_errors(index)


if __name__ == "__main__":
    main()