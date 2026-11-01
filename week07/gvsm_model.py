"""Step 3 - the Generalized Vector Space Model, as lecture section 2 defines it.

Documents and queries are vectors in a multi-dimensional space where each dimension is
a term or feature. Documents are ranked by how similar their vector is to the query
vector, using cosine similarity (slides 6 to 8). The weighting is flexible (slide 9),
so it is a choice made when the matrix is built, not something fixed here.

Only a document that shares at least one term with the query can score above 0. That
is the limit LSI, in lsi_model.py, exists to remove.

Run:  python gvsm_model.py
"""

from __future__ import annotations

from typing import Iterable

from term_matrix import TermDocumentMatrix
from vector_cosine import CosineSteps, cosine_similarity, cosine_steps


class GvsmModel:
    """Rank documents by the cosine between their column of A and the query vector."""

    def __init__(self, matrix: TermDocumentMatrix):
        self.matrix = matrix

    def document_vector(self, doc_id: str) -> list[float]:
        return self.matrix.document_vector(doc_id)

    def query_vector(self, query_terms: Iterable[str]) -> list[float]:
        return self.matrix.query_vector(query_terms)

    def rank(self, query_terms: Iterable[str]) -> list[tuple[str, float]]:
        """(doc_id, cosine), best first, for every document scoring above 0."""
        q = self.query_vector(query_terms)
        scored = [(doc, cosine_similarity(q, self.document_vector(doc)))
                  for doc in self.matrix.doc_ids]
        return sorted(((d, s) for d, s in scored if s > 0), key=lambda r: (-r[1], r[0]))

    def steps(self, query_terms: Iterable[str], doc_id: str) -> CosineSteps:
        return cosine_steps(self.query_vector(query_terms), self.document_vector(doc_id))


def main() -> None:
    matrix = TermDocumentMatrix({"D1": ["fruit"] * 5 + ["veget"] * 3}, "count")
    model = GvsmModel(matrix)
    print("Q =", model.query_vector(["fruit", "veget"]))
    print("D1 =", model.document_vector("D1"))
    print("rank:", model.rank(["fruit", "veget"]))


if __name__ == "__main__":
    main()
