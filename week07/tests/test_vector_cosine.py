"""Cosine similarity against the formula and the numbers on slides 7 and 8."""

from __future__ import annotations

import math

import pytest

from vector_cosine import (angle_degrees, cosine_similarity, cosine_steps, dot_product,
                           vector_length)


class TestTheSlideExample:
    def test_dot_product_is_8(self):
        assert dot_product((1, 1), (5, 3)) == 8

    def test_length_of_q_is_root_2(self):
        assert vector_length((1, 1)) == pytest.approx(math.sqrt(2))

    def test_length_of_d_is_root_34(self):
        assert vector_length((5, 3)) == pytest.approx(math.sqrt(34))

    def test_denominator_is_8_point_24(self):
        assert cosine_steps((1, 1), (5, 3)).denominator == pytest.approx(8.246, abs=0.001)

    def test_the_cosine_is_0_97(self):
        assert round(cosine_similarity((1, 1), (5, 3)), 2) == 0.97


class TestTheFormula:
    def test_same_direction_is_1(self):
        assert cosine_similarity((1, 2, 3), (2, 4, 6)) == pytest.approx(1.0)

    def test_orthogonal_is_0(self):
        assert cosine_similarity((1, 0), (0, 5)) == 0.0

    def test_opposite_direction_is_minus_1(self):
        assert cosine_similarity((1, 2), (-1, -2)) == pytest.approx(-1.0)

    def test_length_does_not_matter_only_direction(self):
        assert cosine_similarity((1, 1), (5, 3)) == pytest.approx(
            cosine_similarity((1, 1), (500, 300)))

    def test_it_is_symmetric(self):
        assert cosine_similarity((1, 4, 2), (3, 0, 1)) == pytest.approx(
            cosine_similarity((3, 0, 1), (1, 4, 2)))

    def test_it_never_leaves_the_range(self):
        assert cosine_similarity((0.1,) * 7, (0.1,) * 7) <= 1.0

    @pytest.mark.parametrize("q, d", [((0, 0), (1, 2)), ((1, 2), (0, 0)), ((0, 0), (0, 0))])
    def test_a_zero_vector_scores_0_rather_than_dividing_by_0(self, q, d):
        assert cosine_similarity(q, d) == 0.0

    def test_vectors_of_different_lengths_are_refused(self):
        with pytest.raises(ValueError):
            dot_product((1, 2), (1, 2, 3))


class TestTheAngle:
    @pytest.mark.parametrize("cosine, degrees", [(1.0, 0.0), (0.0, 90.0), (-1.0, 180.0),
                                                 (0.5, 60.0)])
    def test_a_cosine_is_an_angle(self, cosine, degrees):
        assert angle_degrees(cosine) == pytest.approx(degrees)

    def test_a_rounding_overshoot_does_not_crash(self):
        assert angle_degrees(1.0000000000000002) == 0.0


class TestTheWrittenSteps:
    def test_the_steps_read_like_slide_8(self):
        lines = dict(cosine_steps((1, 1), (5, 3)).lines())
        assert lines["Dot product"] == "Q · D = (1×5) + (1×3) = 8"
        assert lines["Length of Q"] == "||Q|| = √(1² + 1²) = 1.414"
        assert lines["Length of D"] == "||D|| = √(5² + 3²) = 5.831"
        assert lines["Cosine similarity"].endswith("8 / 8.246 = 0.970")

    def test_a_negative_is_bracketed_before_it_is_squared(self):
        assert "(-2)²" in dict(cosine_steps((1, -2), (1, 1)).lines())["Length of Q"]

    def test_zero_entries_are_left_out_of_the_working(self):
        steps = cosine_steps((1, 0, 2), (3, 9, 0))
        assert steps.pairs == ((1, 3),)
        assert "9" not in dict(steps.lines())["Dot product"]

    def test_a_long_vector_is_shortened_but_the_answer_uses_all_of_it(self):
        steps = cosine_steps((1,) * 20, (1,) * 20)
        assert "(14 more)" in dict(steps.lines())["Dot product"]
        assert steps.dot == 20

    def test_no_shared_terms_prints_a_zero_dot_product(self):
        assert dict(cosine_steps((1, 0), (0, 1)).lines())["Dot product"] == "Q · D = 0 = 0"

    def test_the_steps_agree_with_the_plain_function(self):
        q, d = (0.3, 1.2, 0.0, 4.0), (1.0, 0.5, 2.0, 0.1)
        assert cosine_steps(q, d).score == cosine_similarity(q, d)
