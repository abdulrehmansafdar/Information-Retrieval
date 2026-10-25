"""Step 1 - how strongly a word belongs to an article: the membership value.

The Fuzzy Model needs one number, between 0 and 1, for every pair of (word, article).
This file builds that number in two parts.

1. Membership functions, mu_t(d) in [0, 1], which turn a word's weighted count in
   an article into a degree of membership. Three are offered side by side:

       binary   1 if the word occurs at all              (classical Boolean)
       linear   weight / the largest weight of the word   (proportional)
       log      log(1 + weight) / log(1 + largest)        (default: damps repeats)

   A word's weight counts a title occurrence as TITLE_WEIGHT body occurrences,
   because a title is a stronger statement of what an article is about.

2. Partial term matching. A query word is compared with every stem in the
   vocabulary using character trigrams and the Dice coefficient,

       s(q, v) = 2 |tri(q) & tri(v)| / (|tri(q)| + |tri(v)|)

   so "hypertenion" (a typo) still reaches "hypertens". The membership of the query
   word in an article is then the best of the matches, each discounted by how close
   it is:

       mu_q(d) = max over v of  s(q, v) * mu_v(d)

   A match below the threshold tau is not used. Boolean mode turns partial matching
   off, which is the contrast the Compare page shows.

Run:  python membership.py hypertenion
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from collections.abc import Iterable

from articles import Article, get_articles

TITLE_WEIGHT = 3
FUNCTIONS = ("binary", "linear", "log")
DEFAULT_TAU = 0.6
DEFAULT_K = 3


def trigrams(word: str) -> frozenset[str]:
    """Character trigrams of the word, with a space at each end."""
    padded = f" {word} "
    return frozenset(padded[i:i + 3] for i in range(len(padded) - 2))


def dice(a: str, b: str) -> float:
    """Dice coefficient of the two words' trigram sets, in [0, 1]."""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


class FuzzyIndex:
    """Weighted counts for every stem, in every article, plus the membership maths."""

    def __init__(self, articles: Iterable[Article] | None = None) -> None:
        self.articles: tuple[Article, ...] = tuple(articles or get_articles())
        self.weights: dict[str, dict[str, int]] = {}
        for article in self.articles:
            body = Counter(article.body_terms)
            title = Counter(article.title_terms)
            for stem in set(body) | set(title):
                weight = body[stem] + TITLE_WEIGHT * title[stem]
                self.weights.setdefault(stem, {})[article.doc_id] = weight
        self.largest: dict[str, int] = {
            stem: max(per_doc.values()) for stem, per_doc in self.weights.items()
        }
        self.vocabulary: tuple[str, ...] = tuple(sorted(self.weights))
        self._trigrams = {stem: trigrams(stem) for stem in self.vocabulary}

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(a.doc_id for a in self.articles)

    def weight(self, stem: str, doc_id: str) -> int:
        return self.weights.get(stem, {}).get(doc_id, 0)

    def membership(self, stem: str, doc_id: str, function: str = "log") -> float:
        """mu_stem(doc): a value in [0, 1] for one stem in one article."""
        if function not in FUNCTIONS:
            raise ValueError(f"unknown membership function {function!r}")
        weight = self.weight(stem, doc_id)
        if weight == 0:
            return 0.0
        if function == "binary":
            return 1.0
        largest = self.largest[stem]
        if function == "linear":
            return weight / largest
        return math.log(1 + weight) / math.log(1 + largest)

    def expansions(self, stem: str, tau: float = DEFAULT_TAU,
                   k: int = DEFAULT_K) -> list[tuple[str, float]]:
        """The stems a query word may mean, best first: (stem, similarity)."""
        query = trigrams(stem)
        scored: list[tuple[str, float]] = []
        for vocab in self.vocabulary:
            if vocab == stem:
                scored.append((vocab, 1.0))
                continue
            other = self._trigrams[vocab]
            if not query or not other:
                continue
            similarity = 2 * len(query & other) / (len(query) + len(other))
            if similarity >= tau:
                scored.append((vocab, similarity))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def best_match(self, stem: str, doc_id: str, function: str = "log",
                   partial: bool = True, tau: float = DEFAULT_TAU) -> tuple[float, str | None]:
        """mu_q(doc) for one query stem, and the vocabulary stem that gave it."""
        if partial:
            candidates = self.expansions(stem, tau)
        else:
            candidates = [(stem, 1.0)] if stem in self.weights else []
        best, matched = 0.0, None
        for vocab, similarity in candidates:
            value = similarity * self.membership(vocab, doc_id, function)
            if value > best:
                best, matched = value, vocab
        return best, matched


def main(argv: list[str] | None = None) -> int:
    from articles import tokenize

    args = sys.argv[1:] if argv is None else argv
    word = " ".join(args) or "hypertension"
    index = FuzzyIndex()
    stems = tokenize(word)
    if not stems:
        print(f"{word!r} has no searchable stem")
        return 1
    stem = stems[0]
    print(f"query stem: {stem}")
    print("partial matches (similarity >= tau):")
    for vocab, similarity in index.expansions(stem):
        print(f"  {vocab:<16} {similarity:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
