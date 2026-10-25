"""Membership functions, weights and partial matching.

The claims tested here are the ones the docs make: every value is in [0, 1], the
best-matching article scores 1, binary membership is the Boolean model, and a typo
reaches its word through trigram similarity and not through a hand-written list.
"""

from __future__ import annotations

import pytest

from membership import FUNCTIONS, TITLE_WEIGHT, dice, trigrams


class TestTrigramSimilarity:
    def test_a_word_is_identical_to_itself(self):
        assert dice("hypertens", "hypertens") == 1.0

    def test_unrelated_words_share_nothing(self):
        assert dice("asthma", "malaria") < 0.2

    def test_a_typo_stays_close(self):
        assert dice("hypertenion", "hypertens") >= 0.6

    def test_padding_marks_word_edges(self):
        assert trigrams("h") == frozenset({" h "})
        assert " ht" in trigrams("ht") and "ht " in trigrams("ht")


class TestMembershipValues:
    def test_every_value_is_between_zero_and_one(self, index):
        for stem in index.vocabulary:
            for doc_id in index.doc_ids:
                for function in FUNCTIONS:
                    value = index.membership(stem, doc_id, function)
                    assert 0.0 <= value <= 1.0, (stem, doc_id, function)

    def test_a_word_absent_from_an_article_has_no_membership(self, index):
        assert index.membership("hypertens", "m04", "log") == 0.0
        assert index.membership("hypertens", "m04", "binary") == 0.0

    def test_binary_is_one_wherever_the_word_occurs(self, index):
        for stem in index.vocabulary[:200]:
            for doc_id in index.doc_ids:
                expected = 1.0 if index.weight(stem, doc_id) else 0.0
                assert index.membership(stem, doc_id, "binary") == expected

    def test_the_strongest_article_for_a_word_scores_one(self, index):
        for function in ("linear", "log"):
            best = max(index.membership("hypertens", d, function) for d in index.doc_ids)
            assert best == pytest.approx(1.0), function

    def test_log_is_never_below_linear(self, index):
        for stem in index.vocabulary[:300]:
            for doc_id in index.doc_ids:
                linear = index.membership(stem, doc_id, "linear")
                log = index.membership(stem, doc_id, "log")
                assert log >= linear - 1e-12, (stem, doc_id)

    def test_a_title_word_counts_as_several_body_words(self, index):
        weight = index.weight("hypertens", "m01")
        assert weight >= TITLE_WEIGHT

    def test_an_unknown_function_is_refused(self, index):
        with pytest.raises(ValueError):
            index.membership("hypertens", "m01", "cosine")


class TestPartialMatching:
    def test_an_exact_word_comes_first_with_similarity_one(self, index):
        found = index.expansions("hypertens")
        assert found[0] == ("hypertens", 1.0)

    def test_a_typo_expands_to_the_right_stem(self, index):
        found = index.expansions("hypertenion")
        assert found[0][0] == "hypertens"
        assert found[0][1] >= 0.6

    def test_the_threshold_removes_weak_matches(self, index):
        assert index.expansions("hypertenion", tau=0.95) == []

    def test_at_most_k_expansions_are_kept(self, index):
        assert len(index.expansions("diabet", tau=0.1, k=3)) <= 3

    def test_a_typo_is_found_when_partial_matching_is_on(self, index):
        value, matched = index.best_match("hypertenion", "m01", "log", partial=True)
        assert matched == "hypertens"
        assert value > 0.5

    def test_a_typo_is_not_found_when_partial_matching_is_off(self, index):
        assert index.best_match("hypertenion", "m01", "log", partial=False) == (0.0, None)

    def test_the_exact_word_is_found_when_partial_matching_is_off(self, index):
        value, matched = index.best_match("hypertens", "m01", "log", partial=False)
        assert matched == "hypertens"
        assert value == pytest.approx(1.0)

    def test_a_discount_applies_to_a_partial_match(self, index):
        exact, _ = index.best_match("hypertens", "m02", "log", partial=True)
        typo, _ = index.best_match("hypertenion", "m02", "log", partial=True)
        assert typo < exact
        assert typo == pytest.approx(exact * index.expansions("hypertenion")[0][1])

    def test_a_missing_word_scores_zero(self, index):
        assert index.best_match("zzzzzz", "m01", "log") == (0.0, None)
