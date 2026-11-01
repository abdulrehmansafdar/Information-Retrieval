"""LSI: the SVD is a real decomposition, truncation behaves, and ranking is correct."""

from __future__ import annotations

import numpy as np
import pytest

from gvsm_model import GvsmModel
from lsi_model import LsiModel, SvdFactors
from slide_examples import slide_matrix, space_matrix
from term_matrix import TermDocumentMatrix


@pytest.fixture(scope="module")
def factors():
    return SvdFactors(space_matrix())


class TestTheDecomposition:
    def test_full_rank_rebuilds_a_exactly(self, factors):
        model = LsiModel(factors, factors.rank)
        assert np.allclose(model.reconstruct(), factors.matrix.values)

    def test_u_has_orthonormal_columns(self, factors):
        assert np.allclose(factors.u.T @ factors.u, np.eye(factors.u.shape[1]))

    def test_v_has_orthonormal_rows(self, factors):
        assert np.allclose(factors.vt @ factors.vt.T, np.eye(factors.vt.shape[0]))

    def test_singular_values_are_sorted_and_not_negative(self, factors):
        assert (np.diff(factors.sigma) <= 0).all()
        assert (factors.sigma >= 0).all()

    def test_the_slide_matrix_has_singular_values_2_1_1(self):
        assert np.allclose(SvdFactors(slide_matrix()).sigma, [2, 1, 1])

    def test_u_transpose_a_equals_sigma_v_transpose(self, factors):
        # the identity that lets a query and a document share one projection
        model = LsiModel(factors, 3)
        assert np.allclose(model.u_k.T @ factors.matrix.values,
                           np.diag(model.sigma_k) @ model.vt_k)

    def test_the_sign_is_fixed_so_output_is_repeatable(self, factors):
        for c in range(factors.u.shape[1]):
            assert factors.u[np.argmax(np.abs(factors.u[:, c])), c] > 0

    def test_a_rank_deficient_matrix_reports_its_true_rank(self):
        duplicate = TermDocumentMatrix({"D1": ("a", "b"), "D2": ("a", "b")}, "count")
        assert SvdFactors(duplicate).rank == 1


class TestTruncation:
    def test_the_shapes_are_m_by_k_k_and_k_by_n(self, factors):
        model = LsiModel(factors, 2)
        m, n = factors.matrix.shape
        assert model.u_k.shape == (m, 2)
        assert model.sigma_k.shape == (2,)
        assert model.vt_k.shape == (2, n)

    @pytest.mark.parametrize("k", [0, -1, 99])
    def test_a_k_outside_the_rank_is_refused(self, factors, k):
        with pytest.raises(ValueError):
            LsiModel(factors, k)

    def test_more_concepts_never_rebuild_a_worse(self, factors):
        errors = [np.linalg.norm(factors.matrix.values - LsiModel(factors, k).reconstruct())
                  for k in range(1, factors.rank + 1)]
        assert errors == sorted(errors, reverse=True)

    def test_the_rebuild_error_is_exactly_the_dropped_singular_values(self, factors):
        error = np.linalg.norm(factors.matrix.values - LsiModel(factors, 2).reconstruct())
        assert error == pytest.approx(np.sqrt((factors.sigma[2:] ** 2).sum()))

    def test_energy_kept_grows_to_1(self, factors):
        energy = [LsiModel(factors, k).energy_kept() for k in range(1, factors.rank + 1)]
        assert energy == sorted(energy)
        assert energy[-1] == pytest.approx(1.0)

    def test_truncation_fills_in_zeros(self, factors):
        # the hidden relationship: moon is absent from D2, but not in the rebuilt matrix
        matrix = factors.matrix
        i, j = matrix.row_of["moon"], matrix.column_of["D2"]
        assert matrix.values[i, j] == 0
        assert LsiModel(factors, 2).reconstruct()[i, j] > 0.1


class TestRanking:
    def test_it_reaches_a_document_with_no_shared_word(self, factors):
        ranked = dict(LsiModel(factors, 2).rank(["space", "explor"]))
        assert "D4" in ranked
        assert not {"space", "explor"} & {t for t in factors.matrix.terms
                                          if factors.matrix.counts[factors.matrix.row_of[t],
                                                                   factors.matrix.column_of["D4"]]}

    def test_an_unrelated_topic_stays_out(self, factors):
        ranked = dict(LsiModel(factors, 2).rank(["space", "explor"]))
        assert "D5" not in ranked and "D6" not in ranked

    def test_full_rank_orders_documents_exactly_as_gvsm(self, semantic_engine):
        # the dot products survive and each ||d|| survives; only ||q|| changes, by a
        # constant, so at full rank LSI must give GVSM's order and the same set
        matrix = semantic_engine.matrix()
        full = semantic_engine.lsi(k=semantic_engine.factors().rank)
        gvsm = GvsmModel(matrix)
        for terms in (["hypertens"], ["wheez"], ["chest", "pain"], ["blood", "sugar"]):
            assert [d for d, _ in full.rank(terms)] == [d for d, _ in gvsm.rank(terms)]

    def test_rounding_noise_is_not_a_match_at_full_rank(self, semantic_engine):
        # wheez is in one article; 1e-17 scores on the others must not count as hits
        full = semantic_engine.lsi(k=30)
        assert [d for d, _ in full.rank(["wheez"])] == ["m09"]

    def test_scores_are_sorted_and_positive(self, semantic_engine):
        scores = [s for _, s in semantic_engine.lsi().rank(["chest", "pain"])]
        assert scores == sorted(scores, reverse=True)
        assert all(0 < s <= 1 for s in scores)

    def test_the_steps_agree_with_the_rank(self, semantic_engine):
        model = semantic_engine.lsi()
        ranked = dict(model.rank(["wheez"]))
        assert model.steps(["wheez"], "m10").score == pytest.approx(ranked["m10"])


class TestConcepts:
    def test_there_are_k_concepts_strongest_first(self, semantic_engine):
        concepts = semantic_engine.lsi(k=5).concepts()
        strengths = [c.strength for c in concepts]
        assert len(concepts) == 5
        assert strengths == sorted(strengths, reverse=True)

    def test_each_lists_its_top_terms_by_absolute_weight(self, semantic_engine):
        weights = [abs(w) for _, w in semantic_engine.lsi(k=3).concepts(top=8)[1].terms]
        assert len(weights) == 8
        assert weights == sorted(weights, reverse=True)
