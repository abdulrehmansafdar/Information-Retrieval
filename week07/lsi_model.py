"""Step 4 - Latent Semantic Indexing: SVD of A, cut down to k concepts (slide 12).

    A  =  U  Sigma  V^T          the full decomposition
    A  ~  U_k Sigma_k V_k^T      keep only the top k singular values

    U_k        term-concept matrix: how much each term belongs to each concept
    Sigma_k    singular values: the strength of each concept
    V_k^T      document-concept matrix: how much each document belongs to each concept

Comparing in concept space. A document is its column of Sigma_k V_k^T, and a query q is
placed with q_k = U_k^T q. Both are the same operation, the projection of a term vector
onto the k concept axes, because U_k^T A = Sigma_k V_k^T. Ranking then uses the same
cosine as GVSM. The slides do not give this query step, so it is this module's choice;
docs/understanding/lsi-and-svd.md gives the reason.

Why the truncation finds meaning: terms that keep appearing together load on the same
concept, so a query can land near a document that uses other words for the same theme.

Run:  python lsi_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from term_matrix import TermDocumentMatrix
from vector_cosine import CosineSteps, cosine_similarity, cosine_steps

# singular values below this fraction of the largest are rounding noise, not concepts
RANK_TOLERANCE = 1e-10
SCORE_FLOOR = 1e-9


@dataclass(frozen=True)
class Concept:
    """One hidden theme: its strength and the terms and documents that load on it most."""

    number: int
    strength: float
    terms: tuple[tuple[str, float], ...]
    documents: tuple[tuple[str, float], ...]


class SvdFactors:
    """The full decomposition of one matrix, computed once and shared by every k."""

    def __init__(self, matrix: TermDocumentMatrix):
        self.matrix = matrix
        u, sigma, vt = np.linalg.svd(matrix.values, full_matrices=False)
        # An SVD may flip the sign of any concept (u and v together) and stay correct.
        # Fixing the sign keeps the printed matrices the same from run to run.
        for c in range(len(sigma)):
            if u[np.argmax(np.abs(u[:, c])), c] < 0:
                u[:, c] *= -1
                vt[c, :] *= -1
        self.u, self.sigma, self.vt = u, sigma, vt
        self.rank = int((sigma > sigma[0] * RANK_TOLERANCE).sum()) if sigma[0] > 0 else 0


class LsiModel:
    """Rank documents by cosine in the space of the top k concepts."""

    def __init__(self, factors: SvdFactors, k: int):
        if not 1 <= k <= factors.rank:
            raise ValueError(f"k must be between 1 and {factors.rank}, got {k}")
        self.matrix = factors.matrix
        self.k = k
        self.u_k = factors.u[:, :k]
        self.sigma_k = factors.sigma[:k]
        self.vt_k = factors.vt[:k, :]
        self.total_energy = float((factors.sigma ** 2).sum())
        # the columns of Sigma_k V_k^T: each document in concept coordinates
        self._documents = (self.vt_k.T * self.sigma_k).tolist()

    def reconstruct(self) -> np.ndarray:
        """A_k = U_k Sigma_k V_k^T: the matrix rebuilt from k concepts."""
        return (self.u_k * self.sigma_k) @ self.vt_k

    def energy_kept(self) -> float:
        """The share of A's total squared weight that the top k concepts hold."""
        return float((self.sigma_k ** 2).sum()) / self.total_energy

    def document_vector(self, doc_id: str) -> list[float]:
        return self._documents[self.matrix.column_of[doc_id]]

    def query_vector(self, query_terms: Iterable[str]) -> list[float]:
        """q_k = U_k^T q: the query's term vector projected onto the concepts."""
        return (self.u_k.T @ np.array(self.matrix.query_vector(query_terms))).tolist()

    def rank(self, query_terms: Iterable[str]) -> list[tuple[str, float]]:
        """(doc_id, cosine), best first, for every document scoring above 0.

        A cosine here can be negative, because concept coordinates can be. A negative
        score means the document points away from the query, so it is not returned.
        Scores under SCORE_FLOOR are rounding noise around a true 0, not a weak match.
        """
        q = self.query_vector(query_terms)
        scored = [(doc, cosine_similarity(q, self.document_vector(doc)))
                  for doc in self.matrix.doc_ids]
        return sorted(((d, s) for d, s in scored if s > SCORE_FLOOR),
                      key=lambda r: (-r[1], r[0]))

    def steps(self, query_terms: Iterable[str], doc_id: str) -> CosineSteps:
        return cosine_steps(self.query_vector(query_terms), self.document_vector(doc_id))

    def concepts(self, top: int = 8) -> list[Concept]:
        """The hidden themes (slide 13): the strongest terms and documents of each."""
        result = []
        for c in range(self.k):
            term_order = np.argsort(-np.abs(self.u_k[:, c]), kind="stable")[:top]
            doc_order = np.argsort(-np.abs(self.vt_k[c, :]), kind="stable")[:top]
            result.append(Concept(
                number=c + 1,
                strength=float(self.sigma_k[c]),
                terms=tuple((self.matrix.terms[i], float(self.u_k[i, c])) for i in term_order),
                documents=tuple((self.matrix.doc_ids[j], float(self.vt_k[c, j]))
                                for j in doc_order),
            ))
        return result


def main() -> None:
    matrix = TermDocumentMatrix({"D1": ["nasa", "rocket"], "D2": ["rocket", "moon"],
                                 "D3": ["fruit", "veget"]}, "count")
    model = LsiModel(SvdFactors(matrix), 2)
    print("singular values:", np.round(model.sigma_k, 3))
    print("query 'moon' finds:", model.rank(["moon"]))


if __name__ == "__main__":
    main()
