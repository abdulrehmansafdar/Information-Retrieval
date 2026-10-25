"""Step 3 - combining membership values with AND, OR and NOT.

One evaluator walks the query tree and works with two sets of operators:

    fuzzy (p = None)     AND = min,  OR = max,  NOT = 1 - x        (Zadeh)
    p-norm (p = 2 ...)   AND = 1 - ( mean (1 - x)^p )^(1/p)
                         OR  = ( mean x^p )^(1/p)                   (Salton)
                         NOT = 1 - x

The p-norm reduces to the fuzzy operators at the extremes: p = inf gives exactly
min and max, and p = 1 gives the plain mean. Lowering p makes the operators behave
more like averages, so a document strong on some terms and weak on others scores
better; raising p makes AND stricter and OR more generous.

Classical Boolean retrieval is the same evaluator run with binary membership and
the fuzzy operators. A value of 1 means "in the result", and nothing else is used.

Every evaluation returns a Step: the value and the working that produced it, so the
interface can show how a score was reached rather than only the number.

Run:  python fuzzy_logic.py
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from query_parser import And, Node, Not, Or, Term

Leaf = Callable[[Term], tuple[float, str]]


@dataclass(frozen=True)
class Step:
    """One node of the working: the operator or word, its value and how it was got."""

    label: str
    value: float
    detail: str = ""
    children: tuple["Step", ...] = ()


def conjunction(values: list[float], p: float | None = None) -> float:
    """Fuzzy AND (min), or the p-norm AND when p is given."""
    if p is None or math.isinf(p):
        return min(values)
    n = len(values)
    return 1 - (sum((1 - x) ** p for x in values) / n) ** (1 / p)


def disjunction(values: list[float], p: float | None = None) -> float:
    """Fuzzy OR (max), or the p-norm OR when p is given."""
    if p is None or math.isinf(p):
        return max(values)
    n = len(values)
    return (sum(x ** p for x in values) / n) ** (1 / p)


def operator_name(p: float | None) -> str:
    if p is None:
        return "min / max"
    if math.isinf(p):
        return "min / max (p = inf)"
    return f"p-norm, p = {p:g}"


def evaluate(node: Node, leaf: Leaf, p: float | None = None) -> Step:
    """Score one document. `leaf` gives the membership of each Term."""
    if isinstance(node, Term):
        value, detail = leaf(node)
        return Step(f'"{node.word}"', value, detail)

    if isinstance(node, Not):
        child = evaluate(node.child, leaf, p)
        return Step("NOT", 1 - child.value, f"1 - {child.value:.2f}", (child,))

    children = tuple(evaluate(child, leaf, p) for child in node.children)
    values = [c.value for c in children]
    shown = ", ".join(f"{v:.2f}" for v in values)
    if isinstance(node, And):
        value = conjunction(values, p)
        label = "AND"
    else:
        value = disjunction(values, p)
        label = "OR"
    return Step(label, value, f"over {shown}  [{operator_name(p)}]", children)


def explain(step: Step, indent: int = 0) -> list[str]:
    """The working as indented lines, for the console and the 'why' panel."""
    head = f"{'  ' * indent}{step.label}  {step.value:.2f}"
    if step.detail:
        head += f"   {step.detail}"
    lines = [head]
    for child in step.children:
        lines.extend(explain(child, indent + 1))
    return lines


def describe(value: float) -> str:
    """How well an article matches one word, in words a reader can follow."""
    if value >= 0.8:
        return "strong match"
    if value >= 0.4:
        return "medium match"
    if value > 0:
        return "weak match"
    return "no match"


def plain_english(step: Step, level: int = 0) -> list[str]:
    """The working as plain sentences, one bullet per step, nested by operator."""
    indent = "  " * level
    if step.label == "AND":
        head = (f"ALL of these parts must hold. The weakest one sets the score: "
                f"{step.value:.2f}")
    elif step.label == "OR":
        head = (f"ANY one of these parts is enough. The strongest one sets the score: "
                f"{step.value:.2f}")
    elif step.label == "NOT":
        child = step.children[0].value
        head = (f"This part is reversed (NOT). A match of {child:.2f} becomes "
                f"{step.value:.2f}")
    else:
        head = f"{step.label}: {describe(step.value)} ({step.value:.2f})"
        if step.detail:
            head += f", {step.detail}"
    lines = [f"{indent}- {head}"]
    for child in step.children:
        lines.extend(plain_english(child, level + 1))
    return lines


def main() -> int:
    from query_parser import parse_query

    tree = parse_query("hypertension AND (treatment OR diabetes)")
    values = {"hypertension": 0.92, "treatment": 0.61, "diabetes": 0.40}

    def leaf(term: Term) -> tuple[float, str]:
        return values[term.word], f"membership {values[term.word]:.2f}"

    for p in (None, 1.0, 2.0, math.inf):
        print("\n".join(explain(evaluate(tree, leaf, p))))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
