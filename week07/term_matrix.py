"""Step 1 - the term-document matrix A (lecture 3.1, slide 11).

    rows      = terms
    columns   = documents
    values    = term weights, or presence/absence

The GVSM row of slide 9 says the weighting is flexible, so three are offered. The
choice changes the numbers in A; nothing downstream needs to know which was picked.

    count     how many times the term occurs. Slide 6: D1 = (5, 3)
    binary    1 if present, 0 if absent. Slide 11
    tfidf     count x log10(N / df). Slide 15: "e.g. TF-IDF vectors"

A query becomes a vector in the same space, one number per row of A. Under count and
binary, "fruits vegetables" becomes Q = (1, 1), exactly as on slide 6. Under tfidf,
each query count is multiplied by that term's idf, so query and documents are weighted
alike.

Run:  python term_matrix.py
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

WEIGHTINGS = ("tfidf", "count", "binary")


class TermDocumentMatrix:
    """The matrix A. `values[i, j]` is the weight of terms[i] in doc_ids[j]."""

    def __init__(self, documents: Mapping[str, Sequence[str]], weighting: str = "tfidf"):
        if weighting not in WEIGHTINGS:
            raise ValueError(f"unknown weighting {weighting!r}; use one of {WEIGHTINGS}")
        if not documents:
            raise ValueError("a term-document matrix needs at least one document")
        self.weighting = weighting
        self.doc_ids = tuple(documents)
        self.terms = tuple(sorted({t for terms in documents.values() for t in terms}))
        self.row_of = {term: i for i, term in enumerate(self.terms)}
        self.column_of = {doc: j for j, doc in enumerate(self.doc_ids)}

        counts = np.zeros((len(self.terms), len(self.doc_ids)))
        for j, terms in enumerate(documents.values()):
            for term in terms:
                counts[self.row_of[term], j] += 1
        self.counts = counts
        self.df = (counts > 0).sum(axis=1)
        self.idf = np.log10(len(self.doc_ids) / self.df)
        self.values = self._weigh(counts)

    def _weigh(self, counts: np.ndarray) -> np.ndarray:
        if self.weighting == "count":
            return counts.copy()
        if self.weighting == "binary":
            return (counts > 0).astype(float)
        return counts * self.idf[:, None]

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def document_vector(self, doc_id: str) -> list[float]:
        """One column of A: the document as a list of weights, one per term."""
        return self.values[:, self.column_of[doc_id]].tolist()

    def query_vector(self, query_terms: Iterable[str]) -> list[float]:
        """The query in the same space as the documents. Unknown terms have no row."""
        counts = np.zeros(len(self.terms))
        for term in query_terms:
            if term in self.row_of:
                counts[self.row_of[term]] += 1
        return self._weigh(counts[:, None])[:, 0].tolist()

    def known_terms(self, query_terms: Iterable[str]) -> list[str]:
        """The query terms that have a row in A, in query order, without repeats."""
        return list(dict.fromkeys(t for t in query_terms if t in self.row_of))


def main() -> None:
    documents = {"D1": ["fruit"] * 5 + ["veget"] * 3, "D2": ["fruit", "market"]}
    matrix = TermDocumentMatrix(documents, "count")
    print("terms:", matrix.terms, "| documents:", matrix.doc_ids)
    print(matrix.values)
    print("query 'fruit veget':", matrix.query_vector(["fruit", "veget"]))


if __name__ == "__main__":
    main()
