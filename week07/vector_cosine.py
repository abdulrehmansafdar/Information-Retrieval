"""Step 2 - cosine similarity, written out from the formula on slide 7.

    Cosine Similarity(Q, D) = (Q . D) / (||Q|| x ||D||)

    Q . D  = dot product       = sum of q_i x d_i
    ||Q||  = length of Q       = square root of the sum of q_i squared
    ||D||  = length of D       = square root of the sum of d_i squared

No library computes this: dot_product and vector_length are plain loops, so each line
of the formula is one function here. cosine_steps returns every intermediate number,
which is what the console and the web page print as the worked steps of slide 8.

Run:  python vector_cosine.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Sequence

Vector = Sequence[float]


def dot_product(q: Vector, d: Vector) -> float:
    """Q . D: multiply matching positions, then add them up."""
    if len(q) != len(d):
        raise ValueError(f"vectors differ in length: {len(q)} and {len(d)}")
    total = 0.0
    for q_i, d_i in zip(q, d):
        total += q_i * d_i
    return total


def vector_length(v: Vector) -> float:
    """||V||: the square root of the sum of the squares."""
    total = 0.0
    for v_i in v:
        total += v_i * v_i
    return math.sqrt(total)


def cosine_similarity(q: Vector, d: Vector) -> float:
    """The formula. An empty (all-zero) vector has no direction, so it scores 0."""
    denominator = vector_length(q) * vector_length(d)
    if denominator == 0:
        return 0.0
    # rounding can push a true 1.0 to 1.0000000000000002
    return max(-1.0, min(1.0, dot_product(q, d) / denominator))


def _number(value: float) -> str:
    """5.0 prints as 5 and 0.1234 prints as 0.123, so the steps read like the slide."""
    return f"{value:.0f}" if value == int(value) else f"{value:.3f}"


def _squared(value: float) -> str:
    """-0.5 prints as (-0.5)², because -0.5² would mean minus the square."""
    return f"({_number(value)})²" if value < 0 else f"{_number(value)}²"


def _joined(parts: list[str], limit: int = 6) -> str:
    if len(parts) <= limit:
        return " + ".join(parts) or "0"
    return " + ".join(parts[:limit]) + f" + ... ({len(parts) - limit} more)"


@dataclass(frozen=True)
class CosineSteps:
    """Every number on the way to one cosine, in the order of slide 8."""

    q: tuple[float, ...]
    d: tuple[float, ...]
    pairs: tuple[tuple[float, float], ...]  # the (q_i, d_i) whose product is not zero
    dot: float
    q_length: float
    d_length: float
    denominator: float
    score: float

    def lines(self) -> list[tuple[str, str]]:
        """(step name, worked arithmetic) for the boxes of slide 8."""
        pairs = _joined([f"({_number(a)}×{_number(b)})" for a, b in self.pairs])
        q_squares = _joined([_squared(v) for v in self.q if v])
        d_squares = _joined([_squared(v) for v in self.d if v])
        return [
            ("Dot product", f"Q · D = {pairs} = {_number(self.dot)}"),
            ("Length of Q", f"||Q|| = √({q_squares}) = {self.q_length:.3f}"),
            ("Length of D", f"||D|| = √({d_squares}) = {self.d_length:.3f}"),
            ("Cosine similarity",
             f"(Q · D) / (||Q|| × ||D||) = {_number(self.dot)} / "
             f"({self.q_length:.3f} × {self.d_length:.3f}) = {_number(self.dot)} / "
             f"{self.denominator:.3f} = {self.score:.3f}"),
        ]


def cosine_steps(q: Vector, d: Vector) -> CosineSteps:
    """The same calculation as cosine_similarity, keeping each intermediate number."""
    q_length = vector_length(q)
    d_length = vector_length(d)
    return CosineSteps(
        q=tuple(q), d=tuple(d),
        pairs=tuple((a, b) for a, b in zip(q, d) if a * b != 0),
        dot=dot_product(q, d), q_length=q_length, d_length=d_length,
        denominator=q_length * d_length,
        score=cosine_similarity(q, d),
    )


def angle_degrees(cosine: float) -> float:
    """The angle a cosine stands for: arccos, in degrees. 1 is 0 and 0 is 90."""
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def main() -> None:
    # the root sign is not in the old Windows console code page
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for name, working in cosine_steps((1, 1), (5, 3)).lines():
        print(f"{name:18s} {working}")


if __name__ == "__main__":
    main()
