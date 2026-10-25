"""Step 2 - the query language: words, AND, OR, NOT, BUT NOT, brackets.

Turns a query string into a small tree that the fuzzy and extended evaluators can
walk. The tree is the same whichever model later scores it, so the parser is the
one place where the query language is defined.

    hypertension AND NOT diabetes          And(hypertension, Not(diabetes))
    (asthma OR COPD) BUT NOT smoking        And(Or(asthma, COPD), Not(smoking))
    -fever                                  Not(fever)
    heart failure                           implicit AND (or OR, see `implicit`)

Precedence, tightest first: NOT, AND, OR. Brackets override it.

Operators are upper case only. "and", "or" and "not" in lower case are ordinary
words, so a clinical search for "not" does not turn into an operator by accident.

Errors are QueryError with a character position, never a crash and never a guess.

Run:  python query_parser.py "hypertension AND NOT diabetes"
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from articles import tokenize

TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
OPERATORS = ("AND", "OR", "NOT", "BUT")
IMPLICIT = ("AND", "OR")


class QueryError(ValueError):
    """The query cannot be read. `position` is the character where it went wrong."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(f"{message} (at character {position + 1})")
        self.message = message
        self.position = position


# ---------------------------------------------------------------------------
# the tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """One search word. `stems` has more than one entry for a hyphenated word."""

    word: str
    stems: tuple[str, ...]


@dataclass(frozen=True)
class And:
    children: tuple["Node", ...]


@dataclass(frozen=True)
class Or:
    children: tuple["Node", ...]


@dataclass(frozen=True)
class Not:
    child: "Node"


Node = Term | And | Or | Not


def to_text(node: Node, top: bool = True) -> str:
    """The query as it was understood, with the brackets the parser added."""
    if isinstance(node, Term):
        return node.word
    if isinstance(node, Not):
        return f"NOT {to_text(node.child, top=False)}"
    joiner = " AND " if isinstance(node, And) else " OR "
    text = joiner.join(to_text(child, top=False) for child in node.children)
    return text if top else f"({text})"


def terms_in(node: Node) -> list[Term]:
    """Every Term in the tree, in the order they were typed, repeats included."""
    if isinstance(node, Term):
        return [node]
    if isinstance(node, Not):
        return terms_in(node.child)
    return [t for child in node.children for t in terms_in(child)]


# ---------------------------------------------------------------------------
# lexing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    kind: str    # "WORD", "AND", "OR", "NOT", "BUT", "(" or ")"
    text: str
    position: int


def lex(query: str) -> list[Token]:
    """Split the query into tokens. A leading '-' on a word means NOT."""
    tokens: list[Token] = []
    for match in TOKEN_RE.finditer(query):
        text, position = match.group(), match.start()
        if text in ("(", ")"):
            tokens.append(Token(text, text, position))
        elif text in OPERATORS:
            tokens.append(Token(text, text, position))
        elif text.startswith("-") and len(text) > 1:
            tokens.append(Token("NOT", "-", position))
            tokens.append(Token("WORD", text[1:], position + 1))
        else:
            tokens.append(Token("WORD", text, position))
    return tokens


# ---------------------------------------------------------------------------
# parsing: recursive descent, one function per precedence level
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, query: str, implicit: str) -> None:
        self.query = query
        self.implicit = implicit
        self.tokens = lex(query)
        self.at = 0

    def peek(self) -> Token | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self) -> Token:
        token = self.tokens[self.at]
        self.at += 1
        return token

    def end_position(self) -> int:
        return len(self.query)

    def starts_operand(self, token: Token | None) -> bool:
        return token is not None and token.kind in ("WORD", "NOT", "(")

    def parse(self) -> Node:
        if not self.tokens:
            raise QueryError("the query is empty", 0)
        node = self.expr()
        leftover = self.peek()
        if leftover is not None:
            raise QueryError(f"'{leftover.text}' is not expected here", leftover.position)
        return node

    def expr(self) -> Node:
        """OR level. Implicit OR joins neighbouring words when implicit == 'OR'."""
        items = [self.and_expr()]
        while True:
            token = self.peek()
            if token is not None and token.kind == "OR":
                self.take()
                items.append(self.and_expr())
            elif self.implicit == "OR" and self.starts_operand(token):
                items.append(self.and_expr())
            else:
                break
        return items[0] if len(items) == 1 else Or(tuple(items))

    def and_expr(self) -> Node:
        """AND level, including BUT NOT. Implicit AND joins neighbours by default."""
        items = [self.unary()]
        while True:
            token = self.peek()
            if token is not None and token.kind == "AND":
                self.take()
                items.append(self.unary())
            elif token is not None and token.kind == "BUT":
                self.take()
                following = self.peek()
                if following is None or following.kind != "NOT":
                    position = following.position if following else self.end_position()
                    raise QueryError("BUT must be followed by NOT", position)
                self.take()
                items.append(Not(self.unary()))
            elif self.implicit == "AND" and self.starts_operand(token):
                items.append(self.unary())
            else:
                break
        return items[0] if len(items) == 1 else And(tuple(items))

    def unary(self) -> Node:
        token = self.peek()
        if token is None:
            raise QueryError("the query ends with an operator", self.end_position())
        if token.kind == "NOT":
            self.take()
            return Not(self.unary())
        return self.primary()

    def primary(self) -> Node:
        token = self.take()
        if token.kind == "WORD":
            return self.term(token)
        if token.kind == "(":
            node = self.expr()
            closing = self.peek()
            if closing is None or closing.kind != ")":
                raise QueryError("this '(' is never closed", token.position)
            self.take()
            return node
        raise QueryError(f"'{token.text}' needs a word or a bracket next to it",
                         token.position)

    def term(self, token: Token) -> Term:
        stems = tuple(tokenize(token.text))
        if not stems:
            raise QueryError(f"'{token.text}' has no searchable word in it "
                             f"(it may be a stop word)", token.position)
        return Term(word=token.text, stems=stems)


def parse_query(query: str, implicit: str = "AND") -> Node:
    """The query as a tree. `implicit` is what two neighbouring words mean: AND or OR."""
    if implicit not in IMPLICIT:
        raise ValueError(f"implicit must be AND or OR, not {implicit!r}")
    return _Parser(query, implicit).parse()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    query = " ".join(args) or "hypertension AND NOT diabetes"
    try:
        tree = parse_query(query)
    except QueryError as error:
        print(f"error: {error}")
        return 1
    print(f"query:     {query}")
    print(f"parsed as: {to_text(tree)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
