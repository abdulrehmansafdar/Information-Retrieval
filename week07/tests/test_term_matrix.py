"""The term-document matrix: rows are terms, columns are documents, three weightings."""

from __future__ import annotations

import math

import pytest

from term_matrix import WEIGHTINGS, TermDocumentMatrix

DOCS = {"D1": ("fruit", "fruit", "veget"), "D2": ("fruit", "market"), "D3": ("veget",)}


class TestTheShape:
    def test_rows_are_terms_and_columns_are_documents(self):
        matrix = TermDocumentMatrix(DOCS, "count")
        assert matrix.terms == ("fruit", "market", "veget")
        assert matrix.doc_ids == ("D1", "D2", "D3")
        assert matrix.shape == (3, 3)

    def test_a_cell_is_the_term_in_the_document(self):
        matrix = TermDocumentMatrix(DOCS, "count")
        assert matrix.values[matrix.row_of["fruit"], matrix.column_of["D1"]] == 2
        assert matrix.values[matrix.row_of["market"], matrix.column_of["D1"]] == 0

    def test_no_documents_is_refused(self):
        with pytest.raises(ValueError):
            TermDocumentMatrix({}, "count")

    def test_an_unknown_weighting_is_refused(self):
        with pytest.raises(ValueError):
            TermDocumentMatrix(DOCS, "bm25")


class TestTheWeightings:
    def test_count_is_the_raw_count(self):
        assert TermDocumentMatrix(DOCS, "count").document_vector("D1") == [2, 0, 1]

    def test_binary_is_presence_or_absence(self):
        assert TermDocumentMatrix(DOCS, "binary").document_vector("D1") == [1, 0, 1]

    def test_tfidf_is_count_times_log_n_over_df(self):
        matrix = TermDocumentMatrix(DOCS, "tfidf")
        fruit = matrix.values[matrix.row_of["fruit"], matrix.column_of["D1"]]
        assert fruit == pytest.approx(2 * math.log10(3 / 2))

    def test_a_term_in_every_document_has_tfidf_0(self):
        matrix = TermDocumentMatrix({"D1": ("a", "b"), "D2": ("a", "c")}, "tfidf")
        assert matrix.values[matrix.row_of["a"]].tolist() == [0, 0]

    def test_the_three_weightings_are_offered(self):
        assert set(WEIGHTINGS) == {"tfidf", "count", "binary"}


class TestTheQueryVector:
    def test_it_has_one_number_per_term(self):
        matrix = TermDocumentMatrix(DOCS, "count")
        assert len(matrix.query_vector(["fruit"])) == len(matrix.terms)

    def test_count_gives_the_slide_query_vector(self):
        matrix = TermDocumentMatrix({"D1": ("fruit", "veget")}, "count")
        assert matrix.query_vector(["fruit", "veget"]) == [1, 1]

    def test_a_repeated_word_counts_twice_under_count_only(self):
        assert TermDocumentMatrix(DOCS, "count").query_vector(["fruit", "fruit"])[0] == 2
        assert TermDocumentMatrix(DOCS, "binary").query_vector(["fruit", "fruit"])[0] == 1

    def test_tfidf_weights_the_query_like_the_documents(self):
        matrix = TermDocumentMatrix(DOCS, "tfidf")
        assert matrix.query_vector(["market"])[matrix.row_of["market"]] == pytest.approx(
            math.log10(3))

    def test_an_unknown_word_adds_nothing(self):
        assert sum(TermDocumentMatrix(DOCS, "count").query_vector(["zebra"])) == 0

    def test_known_terms_drops_unknown_words_and_repeats(self):
        assert TermDocumentMatrix(DOCS, "count").known_terms(
            ["veget", "zebra", "veget", "fruit"]) == ["veget", "fruit"]


class TestTheCorpus:
    def test_it_is_30_articles_by_1128_terms(self, semantic_engine):
        assert semantic_engine.matrix().shape == (1128, 30)

    def test_the_specialty_is_not_indexed(self, semantic_engine):
        # 'nephrology' labels three articles, and appears in none of their text
        assert "nephrolog" not in semantic_engine.matrix().row_of
