"""The measured comparison: the metric itself, then what it reports."""

from __future__ import annotations

from collections import Counter

import pytest

from semantic_eval import JUDGED_QUERIES, average_precision, evaluate
from semantic_search import Settings


class TestAveragePrecision:
    def test_perfect_order_is_1(self):
        assert average_precision(["a", "b", "c"], {"a", "b"}) == 1.0

    def test_a_relevant_document_lower_down_counts_for_less(self):
        assert average_precision(["x", "a"], {"a"}) == 0.5

    def test_a_relevant_document_never_found_counts_as_0(self):
        assert average_precision(["a"], {"a", "b"}) == 0.5

    def test_nothing_relevant_is_0(self):
        assert average_precision(["a"], set()) == 0.0


class TestTheQueries:
    def test_there_are_two_per_specialty(self):
        assert set(Counter(s for _, s in JUDGED_QUERIES).values()) == {2}
        assert len(JUDGED_QUERIES) == 20

    def test_every_specialty_exists_in_the_corpus(self, semantic_engine):
        known = {a.specialty for a in semantic_engine.articles}
        assert {s for _, s in JUDGED_QUERIES} == known


class TestTheResults:
    def test_lsi_at_the_default_k_beats_gvsm(self):
        assert (evaluate(Settings("lsi")).mean_average_precision
                > evaluate(Settings("gvsm")).mean_average_precision)

    def test_gvsm_can_never_have_a_no_word_hit(self):
        assert evaluate(Settings("gvsm")).no_word_hits == 0

    def test_too_few_concepts_is_worse_than_gvsm(self):
        # the number the docs warn about: LSI is not better at every k
        assert (evaluate(Settings("lsi", k=2)).mean_average_precision
                < evaluate(Settings("gvsm")).mean_average_precision)

    def test_full_rank_lsi_scores_the_same_as_gvsm(self):
        assert evaluate(Settings("lsi", k=30)).mean_average_precision == pytest.approx(
            evaluate(Settings("gvsm")).mean_average_precision)
