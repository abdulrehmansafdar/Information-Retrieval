"""Step 4 - Boolean retrieval (AND, OR, NOT) for the university IR mini search engine.

Boolean retrieval is the simplest retrieval model: a query is an expression over
sets of documents, and the answer is a set. No scoring, no ranking.

    machine learning AND healthcare      intersection   A & B
    artificial intelligence OR robotics  union          A | B
    machine learning AND NOT finance     difference      A - B

Because the inverted index already stores each term as a set of document ids
(documents_containing), the three operators are Python's own set operations.

Why the query is parsed BEFORE preprocessing
--------------------------------------------
preprocess_query() returns unigrams first and phrases last, so word order is lost:

    'Machine Learning OR Robotics' -> ['or', 'robot', 'machin learn']

Parsing that stream would put the operator before its operands. Parentheses are
also destroyed, because tokenize() deletes every non-alphanumeric character:

    '(a OR b) AND c' -> ['a', 'or', 'b', 'and', 'c']

So this module works on the raw query text and only preprocesses the term chunks,
one chunk at a time, when it reaches them.

    raw text -> tokens -> AST -> evaluate (chunks preprocessed here)

Operator precedence: NOT binds tightest, then AND, then OR. Parentheses override.

Run:  python boolean_search.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from index import InvertedIndex, build_index, documents_containing
from preprocess import Document, preprocess_query
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# query syntax
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
OPERATORS = {"AND", "OR", "NOT"}


class BooleanSyntaxError(ValueError):
    """Raised when a query cannot be parsed. Always carries a readable message."""


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


class Node:
    """Base class for every node in a parsed query."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self})"


@dataclass(frozen=True)
class Term(Node):
    """A chunk of user words, preprocessed only when it is evaluated.

    'artificial intelligence' is one Term, not two.
    """

    text: str

    def terms(self) -> list[str]:
        found = preprocess_query(self.text)
        if len(found) > 1:
            raise BooleanSyntaxError(
                f"{self.text!r} became {len(found)} separate terms {found}. "
                f"Separate them with AND or OR."
            )
        return found


@dataclass(frozen=True)
class Not(Node):
    operand: Node


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node


# ---------------------------------------------------------------------------
# tokenising the raw query
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawToken:
    kind: str  # 'op' | 'word' | '(' | ')'
    text: str


def scan(query: str) -> list[RawToken]:
    """Split the raw query into parens, operators and word chunks.

    Consecutive words are merged into one chunk so a multi-word phrase such as
    'artificial intelligence' stays a single operand.
    """
    tokens: list[RawToken] = []
    for piece in TOKEN_RE.findall(query):
        if piece == "(":
            tokens.append(RawToken("(", "("))
        elif piece == ")":
            tokens.append(RawToken(")", ")"))
        elif piece.upper() in OPERATORS:
            tokens.append(RawToken("op", piece.upper()))
        elif tokens and tokens[-1].kind == "word":
            tokens[-1] = RawToken("word", tokens[-1].text + " " + piece)
        else:
            tokens.append(RawToken("word", piece))
    return tokens


# ---------------------------------------------------------------------------
# parser:  expr := factor (OR factor)*
#           factor := NOT factor | '(' expr ')' | TERM
# ---------------------------------------------------------------------------


class Parser:
    def __init__(self, tokens: list[RawToken]) -> None:
        self.tokens = tokens
        self.position = 0

    def peek(self) -> RawToken | None:
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def next(self) -> RawToken:
        token = self.peek()
        if token is None:
            raise BooleanSyntaxError("Query ended unexpectedly.")
        self.position += 1
        return token

    def parse(self) -> Node:
        if not self.tokens:
            raise BooleanSyntaxError("Query is empty.")
        node = self.parse_or()
        leftover = self.peek()
        if leftover is not None:
            if leftover.kind == "op":
                raise BooleanSyntaxError(
                    f"Unexpected {leftover.text!r} between two terms. Write "
                    f"'A {leftover.text} B' only after AND or OR, or use one of "
                    f"'A AND B', 'A OR B', 'A AND NOT B'."
                )
            raise BooleanSyntaxError(f"Unexpected {leftover.text!r}.")
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while True:
            token = self.peek()
            if token is not None and token.kind == "op" and token.text == "OR":
                self.next()
                node = Or(node, self.parse_and())
            else:
                return node

    def parse_and(self) -> Node:
        node = self.parse_factor()
        while True:
            token = self.peek()
            if token is not None and token.kind == "op" and token.text == "AND":
                self.next()
                node = And(node, self.parse_factor())
            else:
                return node

    def parse_factor(self) -> Node:
        token = self.next()
        if token.kind == "op" and token.text == "NOT":
            return Not(self.parse_factor())
        if token.kind == "(":
            node = self.parse_or()
            closing = self.peek()
            if closing is None or closing.text != ")":
                raise BooleanSyntaxError("Unbalanced parenthesis: missing ')'.")
            self.next()
            return node
        if token.kind == ")":
            raise BooleanSyntaxError("Unbalanced parenthesis: unexpected ')'.")
        if token.kind == "op":
            raise BooleanSyntaxError(
                f"{token.text!r} is missing a right-hand operand."
            )
        return Term(token.text)


def parse_query(query: str) -> Node:
    """Turn a raw query string into an AST. Raises BooleanSyntaxError."""
    return Parser(scan(query)).parse()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate(node: Node, index: InvertedIndex) -> set[str]:
    """Evaluate an AST into a set of document ids."""
    if isinstance(node, Term):
        found = node.terms()
        if not found:
            return set()
        term = found[0]
        if term not in index.postings:
            return set()
        return documents_containing(index, term)
    if isinstance(node, Not):
        everything = {d.doc_id for d in index.documents}
        return everything - evaluate(node.operand, index)
    if isinstance(node, And):
        return evaluate(node.left, index) & evaluate(node.right, index)
    if isinstance(node, Or):
        return evaluate(node.left, index) | evaluate(node.right, index)
    raise TypeError(f"Unknown node {node!r}")


def search(index: InvertedIndex, query: str) -> list[Document]:
    """Parse, evaluate and return the matching documents sorted by id."""
    matches = evaluate(parse_query(query), index)
    by_id = {d.doc_id: d for d in index.documents}
    return [by_id[doc_id] for doc_id in sorted(matches)]


def format_ast(node: Node) -> str:
    """Readable one-line form of a parsed query."""
    if isinstance(node, Term):
        return node.text
    if isinstance(node, Not):
        return f"NOT {format_ast(node.operand)}"
    if isinstance(node, And):
        return f"({format_ast(node.left)} AND {format_ast(node.right)})"
    if isinstance(node, Or):
        return f"({format_ast(node.left)} OR {format_ast(node.right)})"
    return "?"


def explain(node: Node, index: InvertedIndex) -> str:
    """Describe what the expression evaluates to, with set sizes."""
    if isinstance(node, Term):
        found = node.terms()
        term = found[0] if found else ""
        size = len(documents_containing(index, term))
        return f"{size} docs contain {term!r}"
    if isinstance(node, Not):
        inner = evaluate(node.operand, index)
        return f"{len(inner)} docs excluded, {index.number_of_documents - len(inner)} kept"
    if isinstance(node, And):
        left, right = evaluate(node.left, index), evaluate(node.right, index)
        return f"{len(left)} & {len(right)} = {len(left & right)} docs"
    if isinstance(node, Or):
        left, right = evaluate(node.left, index), evaluate(node.right, index)
        return f"{len(left)} | {len(right)} = {len(left | right)} docs"
    return "?"


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


def result_table(documents: list[Document]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("doc", style=ACCENT, no_wrap=True)
    table.add_column("title", style="white")
    table.add_column("terms", justify="right", style=DIM, no_wrap=True)
    for document in documents:
        table.add_row(document.doc_id, document.title, str(len(document.terms)))
    return table


def show_query(index: InvertedIndex, query: str, note: str = "") -> list[Document]:
    tree = parse_query(query)
    documents = search(index, query)
    matches = {d.doc_id for d in documents}

    head = Table.grid(padding=(0, 2))
    head.add_column(style=DIM, no_wrap=True)
    head.add_column(style="white", overflow="fold")
    head.add_row("query", Text(query, style="bold white"))
    head.add_row("parsed", Text(format_ast(tree), style=ACCENT))
    head.add_row("steps", explain(tree, index))
    head.add_row("result", f"[bold]{len(documents)}[/] of {index.number_of_documents}"
                          f" documents")
    if note:
        head.add_row("note", f"[grey58]{note}[/]")
    CONSOLE.print(Panel(head, border_style="grey35", padding=(0, 1)))

    if documents:
        CONSOLE.print(result_table(documents))
    else:
        CONSOLE.print("  [grey58]no documents matched[/]")
    CONSOLE.print(f"  [grey58]ids: {', '.join(sorted(matches)) or '-'}[/]\n")
    return documents


DEMO_QUERIES = [
    "machine learning AND healthcare",
    "artificial intelligence OR robotics",
    "machine learning AND NOT finance",
    "healthcare AND diagnosis",
    "computer vision OR medical imaging",
    "security AND machine learning",
    "NLP AND education",
]


def section_overview(index: InvertedIndex) -> None:
    heading(1, "THE MODEL")
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ACCENT, no_wrap=True)
    body.add_column(style="white")
    body.add_row("A AND B", "intersection   A & B   documents matching every term")
    body.add_row("A OR B", "union          A | B   documents matching any term")
    body.add_row("A NOT B", "difference     A - B   A without B")
    body.add_row("precedence", "NOT binds tightest, then AND, then OR")
    body.add_row("unknown term", "empty set, never an error")
    CONSOLE.print(Panel(body, title="Boolean retrieval", title_align="left",
                        border_style="grey35", padding=(0, 1)))


def section_worked_trace(index: InvertedIndex) -> None:
    heading(2, "WORKED TRACE - security AND (healthcare OR finance)")
    query = "security AND (healthcare OR finance)"
    tree = parse_query(query)

    tree_table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    tree_table.add_column("term", style="white", no_wrap=True)
    tree_table.add_column("documents", style=DIM, overflow="fold")
    for node in (tree.right.left, tree.right.right):
        term = node.terms()[0]
        ids = sorted(documents_containing(index, term))
        tree_table.add_row(term, f"{len(ids):>2}  " + ", ".join(ids))
    CONSOLE.print(tree_table)

    healthcare = documents_containing(index, "healthcar")
    finance = documents_containing(index, "financ")
    security = documents_containing(index, "secur")
    grouped = healthcare | finance
    final = security & grouped

    steps = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    steps.add_column("operation", style=ACCENT, no_wrap=True)
    steps.add_column("size", justify="right", no_wrap=True)
    steps.add_column("documents", style=DIM, overflow="fold")
    steps.add_row("healthcare", str(len(healthcare)),
                  ", ".join(sorted(healthcare)))
    steps.add_row("finance", str(len(finance)), ", ".join(sorted(finance)))
    steps.add_row("healthcare OR finance", str(len(grouped)),
                  ", ".join(sorted(grouped)))
    steps.add_row("security", str(len(security)), ", ".join(sorted(security)))
    steps.add_row("security AND (...)", str(len(final)), ", ".join(sorted(final)))
    CONSOLE.print(steps)
    CONSOLE.print(f"  [grey58]The parentheses matter: without them this would be"
                  f" (security AND healthcare) OR finance.[/]\n")


def section_demo_queries(index: InvertedIndex) -> None:
    heading(3, "YOUR SEVEN DEMO QUERIES")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("query", style="bold white", overflow="fold")
    table.add_column("hits", justify="right", style=ACCENT, no_wrap=True)
    table.add_column("share", width=14)
    table.add_column("documents", style=DIM, overflow="fold")
    for query in DEMO_QUERIES:
        ids = sorted({d.doc_id for d in search(index, query)})
        share = bar(len(ids) / index.number_of_documents, 12)
        table.add_row(query, str(len(ids)), share,
                      ", ".join(ids) if ids else "[grey50]none[/]")
    CONSOLE.print(table)


def section_parentheses(index: InvertedIndex) -> None:
    heading(4, "PARENTHESES CHANGE THE ANSWER")
    for query in [
        "artificial intelligence OR robotics AND security",
        "(artificial intelligence OR robotics) AND security",
        "artificial intelligence OR robotics AND NOT security",
        "(artificial intelligence OR robotics) AND NOT security",
    ]:
        tree = parse_query(query)
        ids = sorted({d.doc_id for d in search(index, query)})
        console_line = Table.grid(padding=(0, 2))
        console_line.add_column(style="white", overflow="fold")
        console_line.add_column(justify="right", style=ACCENT, no_wrap=True)
        console_line.add_row(Text(query, style="bold white"), str(len(ids)))
        CONSOLE.print(console_line)
        CONSOLE.print(f"    [grey50]{format_ast(tree)}[/]")
        CONSOLE.print(f"    [grey58]{', '.join(ids) or 'none'}[/]\n")


def section_teaching_case(index: InvertedIndex) -> None:
    heading(5, "TEACHING CASE - WHY 'machine AND healthcare' RETURNS NOTHING")
    show_query(index, "machine AND healthcare")

    body = Table.grid(padding=(0, 2))
    body.add_column(style=DIM, no_wrap=True)
    body.add_column(style="white", overflow="fold")
    body.add_row("why", "Step 2 turned 'machine learning' into the single phrase"
                        " term 'machin learn' and consumed both words.")
    body.add_row("", "The free-standing 'machin' therefore survives only where"
                     " 'machine' meant a physical device.")
    body.add_row("machin", "doc12, doc13, doc20  (robots, vehicles)")
    body.add_row("machin learn", "9 documents, including doc03 and doc04")
    body.add_row("healthcar", "doc03, doc04, doc09")
    body.add_row("result", "[bold]no overlap[/] - the correct answer, but a "
                           "surprise worth explaining")
    body.add_row("fix", "write 'machine learning AND healthcare'")
    CONSOLE.print(Panel(body, border_style="yellow", padding=(0, 1)))
    CONSOLE.print(f"  [grey58]{ARROW} write[/] [bold]machine learning AND healthcare[/]\n")
    show_query(index, "machine learning AND healthcare")


def section_edge_cases(index: InvertedIndex) -> None:
    heading(6, "EDGE CASES")

    CONSOLE.print("  [bold]NOT on a term that is not in the corpus[/]")
    show_query(index, "machine learning AND NOT blockchain",
               note="'blockchain' is unknown, so its document set is empty and "
                    "A - empty = A. NOT can never remove anything.")

    CONSOLE.print("  [bold]An unknown term on the left of AND[/]")
    show_query(index, "blockchain AND machine learning",
               note="empty AND anything = empty. No error is raised.")

    CONSOLE.print("  [bold]Double negation[/]")
    ids = sorted({d.doc_id for d in search(index, "NOT NOT machine learning")})
    plain = sorted({d.doc_id for d in search(index, "machine learning")})
    CONSOLE.print(f"    NOT NOT machine learning {ARROW} {len(ids)} documents")
    CONSOLE.print(f"    machine learning          {ARROW} {len(plain)} documents")
    CONSOLE.print(f"    [grey58]identical: {ids == plain}[/]\n")

    CONSOLE.print("  [bold]Syntax errors[/]")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("bad query", style="white", no_wrap=True)
    table.add_column("message", style=DIM, overflow="fold")
    for bad in [
        "machine healthcare",
        "machine AND",
        "AND machine",
        "machine AND (healthcare",
        "machine healthcare)",
        "machine learning OR OR robotics",
        "NOT",
    ]:
        try:
            search(index, bad)
            message = "[red]NO ERROR RAISED - this is a bug[/]"
        except BooleanSyntaxError as error:
            message = str(error)
        table.add_row(bad, message)
    CONSOLE.print(table)
    CONSOLE.print()


def main() -> None:
    index = build_index()

    CONSOLE.rule("[bold cyan]Step 4 - Boolean Retrieval[/]", style="cyan")
    CONSOLE.print(
        f"[grey58]{index.number_of_documents} documents  {SEP}  "
        f"{len(index.vocabulary):,} terms  {SEP}  "
        f"AND = &   OR = |   NOT = -[/]\n"
    )

    section_overview(index)
    section_worked_trace(index)
    section_demo_queries(index)
    section_parentheses(index)
    section_teaching_case(index)
    section_edge_cases(index)


if __name__ == "__main__":
    main()