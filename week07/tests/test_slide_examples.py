"""The lecture's worked examples, run through the real models."""

from __future__ import annotations

from clinical_docs import tokenize
from slide_examples import (FRUIT_QUERY, SPACE_DOCUMENTS, fruits_and_vegetables,
                            slide_matrix, space_exploration, space_matrix)


class TestFruitsAndVegetables:
    def test_the_document_vector_is_5_3(self):
        assert fruits_and_vegetables().document_vector("D1") == [5, 3]

    def test_the_query_vector_is_1_1(self):
        assert fruits_and_vegetables().query_vector(tokenize(FRUIT_QUERY)) == [1, 1]

    def test_the_cosine_is_0_97(self):
        steps = fruits_and_vegetables().steps(tokenize(FRUIT_QUERY), "D1")
        assert steps.dot == 8
        assert round(steps.score, 2) == 0.97


class TestSlideMatrix:
    def test_it_is_the_matrix_on_slide_11(self):
        assert slide_matrix().values.tolist() == [[1, 1, 0], [1, 0, 1], [0, 1, 1]]

    def test_rows_are_t1_to_t3_and_columns_d1_to_d3(self):
        matrix = slide_matrix()
        assert matrix.terms == ("t1", "t2", "t3")
        assert matrix.doc_ids == ("D1", "D2", "D3")


class TestSpaceExploration:
    def test_d3_and_d4_contain_neither_query_word(self):
        query = set(tokenize("space exploration"))
        for doc in ("D3", "D4"):
            assert not query & set(tokenize(SPACE_DOCUMENTS[doc]))

    def test_gvsm_cannot_see_them(self):
        scores = space_exploration()
        assert scores["D3"]["gvsm"] == 0 and scores["D4"]["gvsm"] == 0

    def test_lsi_reaches_them(self):
        scores = space_exploration()
        assert scores["D3"]["lsi"] > 0.9 and scores["D4"]["lsi"] > 0.9

    def test_the_food_documents_stay_at_0_under_both(self):
        scores = space_exploration()
        for doc in ("D5", "D6"):
            assert scores[doc] == {"gvsm": 0.0, "lsi": 0.0}

    def test_the_matrix_uses_plain_counts(self):
        assert space_matrix().weighting == "count"
