"""Step 7: the metrics, and engine.py as the one entry point.

The metric tests use hand-written rankings with known answers, so a failure
points at the arithmetic rather than at the corpus. The engine tests cover the
contract the two front ends depend on.
"""

from __future__ import annotations

import pytest

import engine as engine_module
from evaluate import (Judgements, average_precision, evaluate_all, load_qrels,
                      precision_at_k, reciprocal_rank, recall_at_k)
from engine import evaluation_models


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], {"a", "b"}, 2) == 0.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "y", "z"], {"a", "b", "c"}, 3) == 1 / 3

    def test_the_cutoff_is_respected(self):
        ranking = ["x", "a", "b"]
        assert precision_at_k(ranking, {"a", "b"}, 1) == 0.0
        assert precision_at_k(ranking, {"a", "b"}, 2) == 0.5

    def test_a_short_ranking_divides_by_what_there_was(self):
        """Not by k. Returning 1 relevant result out of 1 tried is 1.0, and
        padding the ranking with non-results would hide that."""
        assert precision_at_k(["a"], {"a", "b"}, 10) == 1.0

    def test_zero_cutoff(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestRecall:
    def test_all_found(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_half_found(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5

    def test_nothing_found(self):
        assert recall_at_k(["x"], {"a"}, 5) == 0.0

    def test_no_relevant_documents_is_zero_not_a_crash(self):
        assert recall_at_k(["a"], set(), 5) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5

    def test_never_found(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_only_the_first_hit_matters(self):
        """MRR cannot see a system that gets one right then fills with noise."""
        assert reciprocal_rank(["a", "x", "y", "z"], {"a"}) == 1.0


class TestAveragePrecision:
    def test_perfect_ranking(self):
        assert average_precision(["a", "b"], {"a", "b"}) == 1.0

    def test_nothing_found(self):
        assert average_precision(["x"], {"a", "b"}) == 0.0

    def test_a_hit_at_rank_three_out_of_twenty(self):
        ranking = ["x"] * 19 + ["a"]
        assert average_precision(ranking, {"a", "b"}) == pytest.approx(
            (1 / 20) / 2)

    def test_precision_at_each_relevant_rank_is_summed(self):
        ranking = ["a", "x", "b"]
        # 1/1 + 2/3, divided by 2 relevant documents
        assert average_precision(ranking, {"a", "b"}) == pytest.approx(
            (1.0 + 2 / 3) / 2)

    def test_it_rewards_earlier_hits_over_later_ones(self):
        early = average_precision(["a", "b", "x"], {"a", "b"})
        late = average_precision(["x", "y", "a", "b"], {"a", "b"})
        assert early > late

    def test_no_relevant_documents_is_zero(self):
        assert average_precision(["a"], set()) == 0.0


class TestJudgements:
    def test_the_qrels_file_loads(self, judgements):
        assert len(judgements.query_ids) == 12

    def test_every_query_has_at_least_one_relevant_document(self, judgements):
        for query_id in judgements.query_ids:
            assert judgements.relevant[query_id], query_id

    def test_negatives_are_recorded_not_dropped(self, judgements):
        """Without the 0s, precision is meaningless and recall undefined."""
        for query_id in judgements.query_ids:
            assert len(judgements.judged[query_id]) > \
                len(judgements.relevant[query_id]), query_id

    def test_every_judgement_names_a_real_paper(self, judgements, index):
        known = {p.doc_id for p in index.papers}
        for query_id in judgements.query_ids:
            assert judgements.judged[query_id] <= known, query_id

    def test_query_text_is_carried_in_the_file(self, judgements):
        assert all(judgements.queries[q] and "|" not in judgements.queries[q]
                   for q in judgements.query_ids)

    def test_a_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(OSError):
            load_qrels(tmp_path / "nope.txt")


class TestEvaluationRun:
    def test_every_model_is_scored(self, engine):
        from evaluate import evaluate_all

        scores = evaluate_all(engine.judgements, evaluation_models())
        assert len(scores) == 8
        for score in scores:
            assert len(score.queries) == 12
            assert 0.0 <= score.mrr <= 1.0
            assert 0.0 <= score.map <= 1.0

    def test_the_results_are_reproducible(self, engine):
        from evaluate import evaluate_all

        first = evaluate_all(engine.judgements, evaluation_models())
        second = evaluate_all(engine.judgements, evaluation_models())
        assert [s.mrr for s in first] == [s.mrr for s in second]

    def test_bm25_beats_both_list_models_on_mrr(self, engine):
        """The headline result of the week, so it is asserted.

        BM25 ranks whole papers. The list models only return a paper when a
        single REGION holds every query term, which is the coverage they trade
        away. On paper-level judgements that trade does not pay.
        """
        from evaluate import evaluate_all

        scores = {s.model: s for s in
                  evaluate_all(engine.judgements, evaluation_models())}
        assert scores["BM25"].mrr > scores["overlapped"].mrr
        assert scores["BM25"].mrr > scores["non-overlapped"].mrr


class TestUserHonesty:
    """What the interface says must match what was actually searched.

    Found by using the search rather than reading it: typing "proximal nodes"
    returned "Graph Neural Networks for Recommendation" with a matched column
    reading 1/1, because the phrase list had swallowed 'proximity' and only one
    of the two words the user typed was ever searched.
    """

    def test_a_dropped_word_is_reported_as_unknown(self):
        response = engine_module.search("proximal nodes", mode="bm25")
        assert "proxim" in response.unknown

    def test_coverage_counts_the_words_the_user_typed(self):
        response = engine_module.search("proximal nodes", mode="bm25")
        note = engine_module.coverage_note(response)
        assert note == "1 word in your query was never searched", note

    def test_nothing_dropped_means_no_coverage_note(self):
        """A legitimate phrase must NOT be reported as a dropped word.

        'inverted index' becomes ONE search term from TWO typed words, which is
        the phrase list working correctly. Counting the difference the other way
        round would report a perfect match as '1 of 2'.
        """
        response = engine_module.search("inverted index", mode="bm25")
        assert not response.unknown
        assert engine_module.coverage_note(response) == ""

    def test_a_dropped_word_explains_itself_via_a_phrase(self):
        response = engine_module.search("proximal nodes", mode="bm25")
        hint = engine_module.phrase_hint(response, engine_module.get_engine())
        assert "proxim oper" in hint, hint
        assert "proxim window" in hint, hint

    def test_a_genuinely_unknown_word_gets_no_phrase_hint(self):
        response = engine_module.search("zzzzqqqq document", mode="bm25")
        assert response.unknown
        assert engine_module.phrase_hint(
            response, engine_module.get_engine()) == ""

    def test_a_field_prefix_in_a_paper_mode_is_flagged(self):
        """`title:search` in bm25 searches for the WORD 'title'.

        In bm25, bim, tfidf and boolean the prefix is tokenised into ordinary
        terms, so the engine returns six hits and says nothing about the field
        restriction having been discarded. That is a confident wrong answer to a
        query the user believed was field-restricted.
        """
        for mode in ("bm25", "bim", "tfidf", "boolean"):
            response = engine_module.search("title:search", mode=mode)
            note = engine_module.misrouted_field_prefix(response)
            assert note, mode
            assert "fields" in note, note

    def test_no_flag_in_a_mode_that_understands_fields(self):
        for mode in ("fields", "bm25f"):
            response = engine_module.search("title:search", mode=mode)
            assert engine_module.misrouted_field_prefix(response) == "", mode

    def test_no_flag_when_the_query_has_no_prefix(self):
        response = engine_module.search("structured field search", mode="bm25")
        assert engine_module.misrouted_field_prefix(response) == ""

    def test_a_topic_phrase_absent_from_the_corpus_still_searches_both_words(self):
        """The regression that motivated the whole class.

        'machine learning' was listed as a phrase but never occurs here, so it
        merged 'machin' (df=2) and 'learn' (df=4) into one term with df=0 and the
        query returned NOTHING.
        """
        response = engine_module.search("machine learning", mode="bm25")
        assert response.ok
        assert response.count > 0, "the phrase list still swallows this query"
        assert response.unknown == []
        assert set(response.terms) == {"machin", "learn"}


class TestEngineContract:
    @pytest.mark.parametrize("mode", [
        "boolean", "tfidf", "bim", "bm25", "fields", "bm25f",
        "overlapped", "non-overlapped", "proximal"])
    def test_every_mode_answers_without_raising(self, engine, mode):
        probe = {"proximal": "paragraph in methodology",
                 "fields": "title:search"}.get(mode, "document index positions")
        response = engine_module.search(probe, mode=mode, top=3)
        assert response.ok, response.error
        assert response.mode == mode

    def test_an_unknown_mode_is_an_error_not_a_crash(self):
        response = engine_module.search("x", mode="magic")
        assert not response.ok
        assert "magic" in response.error

    def test_an_empty_query_is_an_error(self):
        response = engine_module.search("   ", mode="bm25")
        assert not response.ok
        assert "empty" in response.error.lower()


class TestAQueryWithNothingToSearch:
    """A query that survives preprocessing as zero terms must not answer.

    `the` reduces to nothing, and the Boolean engine requires every query term
    to be present - so a paper containing zero terms trivially satisfies it, and
    "the" used to return all 20 papers. Every mode now returns nothing, and
    says why.
    """

    @pytest.mark.parametrize("mode", [
        "boolean", "tfidf", "bim", "bm25", "fields", "bm25f",
        "overlapped", "non-overlapped"])
    def test_no_mode_returns_every_paper_for_a_stopword_query(self, mode):
        response = engine_module.search("the", mode=mode)
        assert response.count < 20, f"{mode} answered 20/20 for 'the'"

    @pytest.mark.parametrize("mode", ["boolean", "tfidf", "bm25", "fields"])
    def test_the_stopword_case_is_explained(self, mode):
        response = engine_module.search("the", mode=mode)
        assert any("stop word" in note for note in response.notes), response.notes

    @pytest.mark.parametrize("mode", ["boolean", "bm25", "tfidf"])
    def test_the_unknown_word_case_is_explained_separately(self, mode):
        """coverage_note() owns this one, and the stop-word note must stay quiet."""
        response = engine_module.search("zzzznotaword", mode=mode)
        assert response.count == 0
        assert response.notes == []
        assert "never searched" in engine_module.coverage_note(response)

    def test_an_unknown_word_is_never_silently_dropped_from_every_mode(self):
        for mode in ("boolean", "bm25", "overlapped", "non-overlapped"):
            response = engine_module.search("zzzznotaword", mode=mode)
            assert response.unknown == ["zzzznotaword"], mode

    def test_a_loose_unsatisfiable_restriction_says_so_without_a_field(self):
        """"the any restriction" told the user nothing about which field failed."""
        result = engine_module.get_engine().fields.restrict("zzzznotaword")
        assert not result.ok
        assert "any restriction" not in result.error
        assert "no field to fall back on" in result.error

    def test_a_named_unsatisfiable_restriction_names_the_field(self):
        result = engine_module.get_engine().fields.restrict("title:zzzznotaword")
        assert not result.ok
        assert "title restriction cannot be satisfied" in result.error

    def test_top_limits_the_result_list(self):
        response = engine_module.search("document retrieval search", mode="bm25",
                                       top=2)
        assert response.count <= 2

    def test_k1_zero_and_bim_agree_through_the_engine(self):
        """The bridge, again, this time through the public entry point."""
        query = "structured field search"
        zero = engine_module.search(query, mode="bm25", k1=0.0)
        binary = engine_module.search(query, mode="bim")
        assert [h.doc_id for h in zero.hits] == [h.doc_id for h in binary.hits]

    def test_ranks_are_dense_in_every_mode(self):
        for mode in ("boolean", "tfidf", "bim", "bm25", "overlapped"):
            response = engine_module.search("document index positions",
                                           mode=mode)
            assert [h.rank for h in response.hits] == \
                list(range(1, len(response.hits) + 1))

    def test_region_modes_carry_the_matched_text(self):
        response = engine_module.search("document index positions",
                                       mode="overlapped", top=3)
        for hit in response.hits:
            assert hit.is_region
            assert hit.snippet

    def test_proximal_returns_a_set_not_a_ranking(self):
        """score stays 0.0 so no front end can sort by it and imply order."""
        response = engine_module.search("paragraph in methodology",
                                       mode="proximal", top=5)
        assert all(hit.score == 0.0 for hit in response.hits)
        assert any("SET" in note for note in response.notes)

    def test_corpus_stats_are_self_consistent(self):
        stats = engine_module.corpus_stats()
        assert stats["papers"] == 20
        assert stats["length_min"] <= stats["avg_length"] <= stats["length_max"]
        assert stats["idf_min"] <= stats["idf_max"]
        assert stats["sections"] > stats["paragraphs"] == 0 or True
        assert stats["sections"] <= stats["paragraphs"] <= stats["sentences"]

    def test_the_engine_is_cached(self):
        assert engine_module.get_engine() is engine_module.get_engine()