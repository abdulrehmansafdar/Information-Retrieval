"""The operators: fuzzy min/max/complement, the p-norm, and the set identity.

The numbers used here are the lecture's own (0.8 and 0.6 give 0.6; 0.7 and 0.4 give
0.4), so the code is checked against the notes and not against itself.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from fuzzy_logic import conjunction, disjunction, evaluate, explain, plain_english
from query_parser import Term, parse_query


class TestFuzzyOperators:
    def test_the_lecture_and_example(self):
        assert conjunction([0.8, 0.6]) == pytest.approx(0.6)

    def test_the_self_assessment_and_example(self):
        assert conjunction([0.7, 0.4]) == pytest.approx(0.4)

    def test_fuzzy_or_is_the_larger_value(self):
        assert disjunction([0.8, 0.6]) == pytest.approx(0.8)

    def test_fuzzy_not_is_one_minus_the_value(self):
        leaf = {"asthma": 0.3}
        step = evaluate(parse_query("NOT asthma"), lambda t: (leaf[t.word], ""))
        assert step.value == pytest.approx(0.7)


class TestPNorm:
    def test_p_one_and_is_one_minus_the_mean_complement(self):
        assert conjunction([0.8, 0.6], 1.0) == pytest.approx(0.7)

    def test_p_one_or_is_the_plain_mean(self):
        assert disjunction([0.8, 0.6], 1.0) == pytest.approx(0.7)

    def test_p_two_matches_the_hand_calculation(self):
        assert disjunction([0.8, 0.6], 2.0) == pytest.approx(math.sqrt(0.5))
        assert conjunction([0.8, 0.6], 2.0) == pytest.approx(1 - math.sqrt(0.1))

    def test_infinite_p_is_exactly_min_and_max(self):
        assert conjunction([0.8, 0.6], math.inf) == pytest.approx(0.6)
        assert disjunction([0.8, 0.6], math.inf) == pytest.approx(0.8)

    def test_every_p_norm_result_stays_between_zero_and_one(self):
        rng = random.Random(6)
        for _ in range(500):
            values = [rng.random() for _ in range(rng.randint(1, 5))]
            p = rng.choice([1.0, 1.5, 2.0, 5.0, 10.0])
            assert 0.0 <= conjunction(values, p) <= 1.0 + 1e-12
            assert 0.0 <= disjunction(values, p) <= 1.0 + 1e-12

    def test_and_is_never_below_the_weakest_term_and_or_never_above_the_strongest(self):
        rng = random.Random(7)
        for _ in range(500):
            values = [rng.random() for _ in range(rng.randint(2, 4))]
            p = rng.choice([1.0, 2.0, 5.0])
            assert conjunction(values, p) >= min(values) - 1e-12
            assert disjunction(values, p) <= max(values) + 1e-12


class TestTreeEvaluation:
    LEAVES = {"hypertension": 0.92, "treatment": 0.61, "diabetes": 0.40}

    def leaf(self, term: Term):
        return self.LEAVES[term.word], "membership"

    def test_a_nested_query_follows_the_precedence(self):
        tree = parse_query("hypertension AND (treatment OR diabetes)")
        assert evaluate(tree, self.leaf).value == pytest.approx(0.61)

    def test_the_working_names_every_operator(self):
        tree = parse_query("hypertension AND (treatment OR diabetes)")
        lines = explain(evaluate(tree, self.leaf))
        assert lines[0].startswith("AND")
        assert any(line.strip().startswith("OR") for line in lines)
        assert len(lines) == 5

    def test_the_plain_working_says_what_each_operator_does(self):
        tree = parse_query("hypertension AND (treatment OR diabetes)")
        lines = plain_english(evaluate(tree, self.leaf))
        assert lines[0].startswith("- ALL of these parts must hold")
        assert any("ANY one of these parts is enough" in line for line in lines)
        assert any('"hypertension": strong match (0.92)' in line for line in lines)

    def test_a_partial_match_says_what_it_was_found_as(self):
        tree = parse_query("fever")
        step = evaluate(tree, lambda t: (0.23, "found as 'sever'"))
        assert plain_english(step) == ["- \"fever\": weak match (0.23), found as 'sever'"]

    def test_the_same_query_under_a_p_norm_gives_a_different_number(self):
        tree = parse_query("hypertension AND (treatment OR diabetes)")
        fuzzy = evaluate(tree, self.leaf).value
        pnorm = evaluate(tree, self.leaf, 1.0).value
        assert pnorm != pytest.approx(fuzzy)


class TestBooleanIsFuzzyWithBinaryValues:
    """Lecture section 3: with 0/1 membership, the fuzzy operators are set algebra."""

    @pytest.mark.parametrize("a, b, c", list(itertools.product((0.0, 1.0), repeat=3)))
    def test_or_and_not_match_union_and_difference(self, a, b, c):
        values = {"science": a, "technology": b, "art": c}
        tree = parse_query("(science OR technology) AND NOT art")
        result = evaluate(tree, lambda t: (values[t.word], "")).value
        in_union = a == 1.0 or b == 1.0
        in_difference = in_union and c == 0.0
        assert result == (1.0 if in_difference else 0.0)
