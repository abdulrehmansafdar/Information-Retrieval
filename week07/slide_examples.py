"""The lecture's own worked examples, run through the real code.

Nothing here computes anything. Each example is only its input, passed through the same
TermDocumentMatrix, GvsmModel and LsiModel that rank the clinical articles, so if the
code disagreed with the slides these examples would show it.

    fruits_and_vegetables()   slides 6 to 8: Q = (1, 1), D = (5, 3), cosine 0.97
    slide_matrix()            slide 11: the 3 x 3 term-document matrix A
    space_exploration()       slides 4, 10, 13: the query that shares no word with a
                              relevant document. The six short documents are written
                              for this demo, using the slide's words.

Run:  python slide_examples.py
"""

from __future__ import annotations

from clinical_docs import tokenize
from gvsm_model import GvsmModel
from lsi_model import LsiModel, SvdFactors
from term_matrix import TermDocumentMatrix

FRUIT_QUERY = "fruits vegetables"
SPACE_QUERY = "space exploration"
SPACE_K = 2

SPACE_DOCUMENTS = {
    "D1": "space exploration mission",
    "D2": "NASA space mission rocket",
    "D3": "NASA rocket moon landing",
    "D4": "astronaut programme moon rocket",
    "D5": "fruits vegetables market",
    "D6": "fresh fruits vegetables diet",
}


def fruits_and_vegetables() -> GvsmModel:
    """D1 mentions fruits 5 times and vegetables 3 times, weighted by plain count."""
    return GvsmModel(TermDocumentMatrix(
        {"D1": tokenize("fruits " * 5 + "vegetables " * 3)}, "count"))


def slide_matrix() -> TermDocumentMatrix:
    """t1 is in D1 and D2, t2 in D1 and D3, t3 in D2 and D3: slide 11's A."""
    return TermDocumentMatrix({"D1": ("t1", "t2"), "D2": ("t1", "t3"), "D3": ("t2", "t3")},
                              "binary")


def space_matrix() -> TermDocumentMatrix:
    documents = {d: tokenize(text) for d, text in SPACE_DOCUMENTS.items()}
    return TermDocumentMatrix(documents, "count")


def space_exploration() -> dict[str, dict[str, float]]:
    """Each document's score for 'space exploration' under GVSM and under LSI."""
    matrix = space_matrix()
    terms = tokenize(SPACE_QUERY)
    gvsm = dict(GvsmModel(matrix).rank(terms))
    lsi = dict(LsiModel(SvdFactors(matrix), SPACE_K).rank(terms))
    return {doc: {"gvsm": gvsm.get(doc, 0.0), "lsi": lsi.get(doc, 0.0)}
            for doc in matrix.doc_ids}


def main() -> None:
    model = fruits_and_vegetables()
    terms = tokenize(FRUIT_QUERY)
    print("Q =", model.query_vector(terms), " D =", model.document_vector("D1"))
    print(f"cosine = {model.steps(terms, 'D1').score:.2f}")
    print("A =", slide_matrix().values.tolist())
    for doc, scores in space_exploration().items():
        print(f"{doc}  {SPACE_DOCUMENTS[doc]:32s} GVSM {scores['gvsm']:.2f}  LSI {scores['lsi']:.2f}")


if __name__ == "__main__":
    main()
