"""Step 3: the Binary Independence Model and BM25.

The central test here is `bim_is_bm25_with_k1_zero`. It is an identity rather
than an approximation:

    ( tf * (k1 + 1) ) / ( tf + k1 * K )   at k1 = 0   becomes   tf / tf = 1

which is exactly the binary indicator BIM multiplies by. If that drifts, every
statement made about the two models being one function is wrong.
"""

from __future__ import annotations

import math

import pytest

from ranking import (BinaryIndependenceModel, Bm25Model, IdfTable, IDF_VARIANTS,
                     parse_query)


class TestTheBridge:
    def test_bim_is_bm25_with_k1_zero(self, index):
        """Every paper, every query term, identical to floating point."""
        bim = BinaryIndependenceModel(index)
        zero = Bm25Model(index, k1=0.0)
        terms = ("retriev", "document", "term", "index", "field")
        for paper in index.papers:
            bim_score, _ = bim.score_paper(paper.doc_id, terms)
            bm25_score, _ = zero.score_paper(paper.doc_id, terms)
            assert bim_score == pytest.approx(bm25_score, abs=0.0, rel=1e-12), (
                paper.doc_id
            )

    def test_k1_zero_saturation_is_exactly_one(self, bm25):
        zero = bm25.with_parameters(k1=0.0)
        for tf in (1, 2, 7, 10, 99, 1000):
            assert zero.saturation(tf, length=200) == 1.0

    def test_the_same_order_comes_out_at_k1_zero(self, index):
        bim = BinaryIndependenceModel(index)
        zero = Bm25Model(index, k1=0.0)
        parsed = parse_query("structured field search", IdfTable(index, "bm25"))
        assert [r.doc_id for r in bim.rank(parsed)] == \
               [r.doc_id for r in zero.rank(parsed)]

    def test_b1_is_the_only_setting_that_matters_at_k1_zero(self, index):
        """With k1 = 0 the term tf + k1*K is just tf, so b cannot matter."""
        for b in (0.0, 0.5, 1.0):
            model = Bm25Model(index, k1=0.0, b=b)
            assert model.saturation(3, length=999) == 1.0


class TestSaturation:
    def test_the_tf_factor_is_always_at_least_one(self, bm25):
        for tf in range(1, 30):
            assert bm25.saturation(tf, 170) >= 1.0

    def test_the_gain_from_repeating_diminishes(self, bm25):
        """tf = 2 must buy more than tf = 10 buys over tf = 9."""
        first = bm25.saturation(2, 170) - bm25.saturation(1, 170)
        later = bm25.saturation(10, 170) - bm25.saturation(9, 170)
        assert later < first

    def test_bm25_credits_repetition_and_bim_does_not(self, index, bm25, bim):
        doc, term, length = "p19", "field", 150
        tf = index.term_frequency_in(doc, term)
        assert tf >= 5, "this test needs a high tf term in the corpus"

        # BIM's term score takes no length: the binary indicator cannot depend
        # on anything but presence, which is the whole point of the comparison.
        assert bim.term_score(term, 1) == bim.term_score(term, tf)
        assert bm25.term_score(term, tf, length) > \
            bm25.term_score(term, 1, length)

    def test_larger_k1_rewards_repetition_more(self, index):
        lenient = Bm25Model(index, k1=0.2)
        strict = Bm25Model(index, k1=3.0)
        ratio_lenient = (lenient.saturation(10, 170)
                         / lenient.saturation(1, 170))
        ratio_strict = (strict.saturation(10, 170) / strict.saturation(1, 170))
        assert ratio_strict > ratio_lenient


class TestLengthNormalisation:
    def test_b_zero_ignores_length_entirely(self, index):
        model = Bm25Model(index, b=0.0)
        for length in (10, 100, 1000):
            assert model.length_norm(length) == 1.0

    def test_length_norm_rises_with_length(self, bm25):
        assert bm25.length_norm(50) < bm25.length_norm(170) < \
            bm25.length_norm(400)

    def test_the_average_length_gets_a_norm_of_about_one(self, bm25):
        assert bm25.length_norm(int(bm25.average_document_length)) == \
            pytest.approx(1.0, abs=0.05)

    def test_b_is_rejected_outside_zero_to_one(self, index):
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError):
                Bm25Model(index, b=bad)

    def test_negative_k1_is_rejected(self, index):
        with pytest.raises(ValueError):
            Bm25Model(index, k1=-0.5)


class TestIdf:
    def test_the_bm25_variant_is_always_positive(self, index):
        for variant in IDF_VARIANTS:
            table = IdfTable(index, variant)
            if variant == "robertson":
                continue
            assert all(v >= 0.0 for v in table.idf.values()), variant

    def test_idf_decreases_as_df_increases(self, index):
        table = IdfTable(index, "bm25")
        by_df = sorted(table.document_frequency.items(), key=lambda kv: kv[1])
        values = [table.idf[term] for term, _ in by_df]
        assert values == sorted(values, reverse=True)

    def test_a_rarer_term_is_worth_more_than_a_common_one(self, index, idf_table):
        rare = max(idf_table.document_frequency.items(),
                   key=lambda kv: (kv[1] == 1, kv[1]))
        common = max(idf_table.document_frequency.items(), key=lambda kv: kv[1])
        assert idf_table.idf_for(rare[0]) > idf_table.idf_for(common[0])

    def test_an_unknown_idf_variant_raises(self, index):
        with pytest.raises(ValueError) as caught:
            IdfTable(index, "magic")
        assert "bm25" in str(caught.value)

    def test_an_unknown_term_gets_an_extreme_weight_without_raising(
            self, idf_table):
        value = idf_table.idf_for("nosuchterm")
        assert math.isfinite(value)


class TestRanking:
    def test_scores_are_finite_and_positive(self, bm25, idf_table):
        for result in bm25.rank("document index positions", idf_table=idf_table):
            assert math.isfinite(result.score)
            assert result.score > 0.0

    def test_ranks_are_dense_and_ordered(self, bm25, idf_table):
        results = bm25.rank("proximity window phrase query",
                            idf_table=idf_table)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ties_break_on_doc_id_so_results_are_reproducible(self, bm25,
                                                            idf_table):
        first = bm25.rank("field search", idf_table=idf_table)
        second = bm25.rank("field search", idf_table=idf_table)
        assert [r.doc_id for r in first] == [r.doc_id for r in second]

    def test_top_limits_the_list(self, bm25, idf_table):
        everything = bm25.rank("document retrieval search", idf_table=idf_table)
        limited = bm25.rank("document retrieval search", top=2,
                            idf_table=idf_table)
        assert len(limited) == 2
        assert [r.doc_id for r in limited] == [r.doc_id for r in everything[:2]]

    def test_matching_more_terms_never_lowers_the_bim_score(self, index):
        bim = BinaryIndependenceModel(index)
        table = bim.idf_table
        few = bim.score_paper("p04", ("retriev",))[0]
        more = bim.score_paper("p04", ("retriev", "document", "term"))[0]
        assert more > few

    def test_an_empty_query_produces_no_results_not_an_error(self, bm25,
                                                             idf_table):
        assert bm25.rank("the of and", idf_table=idf_table) == []

    def test_an_unknown_term_is_reported_not_crashed(self, bm25, idf_table):
        parsed = parse_query("zzzqqq document", idf_table)
        assert "zzzqqq" in parsed.unknown
        assert parsed.terms == ("document",)

    def test_contributions_sum_to_the_score(self, bm25, idf_table):
        parsed = parse_query("structured field search", idf_table)
        total, rows = bm25.score_paper("p19", parsed.terms)
        assert sum(r.contribution for r in rows) == pytest.approx(total)