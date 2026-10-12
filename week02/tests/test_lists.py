"""Step 5: the overlapped and non-overlapped list models.

The invariant that matters most is the one that distinguishes them: every
region the overlapped model returns must contain EVERY query term, and no region
the non-overlapped model returns is required to. If the first ever fails, the
strict model has quietly become the permissive one.
"""

from __future__ import annotations

import pytest

from lists import Match, collapse, span_gap, subtract

QUERY = "document index positions"


def terms_in_region(index, region):
    paper = next(p for p in index.papers if p.doc_id == region.doc_id)
    return {t.term for t in paper.tokens if region.contains(t.position)}


class TestCollapse:
    def test_disjoint_matches_stay_separate(self):
        matches = [Match("a", 0), Match("b", 10), Match("c", 20)]
        assert collapse(matches) == [(0, 1), (10, 11), (20, 21)]

    def test_adjacent_occurrences_merge(self):
        """Positions 10 and 11 have nothing between them.

        A region containing both holds no term it does not have, so merging them
        is correct - and the test is `position <= end` because `end` is
        EXCLUSIVE, so offset 11 continues a run that already covers 10..10.
        """
        matches = [Match("a", 10), Match("a", 11), Match("c", 12)]
        assert collapse(matches) == [(10, 13)]

    def test_a_gap_breaks_the_run(self):
        """Positions 10 and 13 have two unmatched terms between them.

        Merging them would claim a stretch of text no query term touched.
        """
        matches = [Match("a", 10), Match("b", 13)]
        assert collapse(matches) == [(10, 11), (13, 14)]

    def test_two_separate_runs_stay_separate(self):
        matches = [Match("a", 0), Match("b", 1), Match("c", 5), Match("d", 6)]
        assert collapse(matches) == [(0, 2), (5, 7)]

    def test_one_gap_term_does_not_merge(self):
        matches = [Match("a", 0), Match("b", 2)]
        assert collapse(matches) == [(0, 1), (2, 3)]

    def test_a_run_merges_into_one_range(self):
        matches = [Match("a", 10), Match("b", 11), Match("c", 12)]
        assert collapse(matches) == [(10, 13)]

    def test_out_of_order_input_is_handled(self):
        matches = [Match("a", 12), Match("b", 10), Match("c", 11)]
        assert collapse(matches) == [(10, 13)]

    def test_no_matches_gives_no_spans(self):
        assert collapse([]) == []

    def test_gap_between_two_spans(self):
        assert span_gap((10, 12), (20, 25)) == 8


class TestSubtract:
    def test_a_hole_splits_a_span(self):
        assert subtract([(0, 10)], [(4, 6)]) == [(0, 4), (6, 10)]

    def test_a_hole_at_the_edge_trims_it(self):
        assert subtract([(0, 10)], [(0, 3)]) == [(3, 10)]
        assert subtract([(0, 10)], [(7, 10)]) == [(0, 7)]

    def test_a_disjoint_hole_changes_nothing(self):
        assert subtract([(0, 10)], [(20, 30)]) == [(0, 10)]

    def test_an_empty_hole_list_is_the_identity(self):
        assert subtract([(0, 10)], []) == [(0, 10)]

    def test_a_hole_larger_than_the_span_removes_it(self):
        assert subtract([(5, 8)], [(0, 20)]) == []


class TestOverlapped:
    def test_every_region_holds_every_query_term(self, index, list_models):
        for hierarchy in ("section", "paragraph", "sentence"):
            result = list_models.overlapped(QUERY, hierarchy)
            assert result.ok
            for region in result.regions:
                present = terms_in_region(index, index.by_id[region.region_id])
                assert region.complete, region.region_id
                assert set(region.terms).issubset(present)

    def test_no_region_is_returned_twice(self, list_models):
        result = list_models.overlapped(QUERY, "paragraph")
        ids = [r.region_id for r in result.regions]
        assert len(ids) == len(set(ids))

    def test_regions_in_one_paper_do_not_overlap(self, list_models):
        result = list_models.overlapped(QUERY, "paragraph")
        by_paper: dict[str, list] = {}
        for region in result.regions:
            by_paper.setdefault(region.doc_id, []).append(region)
        for rows in by_paper.values():
            rows.sort(key=lambda r: r.start)
            for before, after in zip(rows, rows[1:]):
                assert before.end <= after.start

    def test_an_empty_result_is_not_an_error(self, list_models):
        result = list_models.overlapped("ranking web search", "paragraph")
        assert result.ok
        assert result.count == 0, "this query must have no complete region"

    def test_a_query_with_no_known_terms_returns_nothing(self, list_models):
        result = list_models.overlapped("zzzzqqqq", "paragraph")
        assert result.count == 0


class TestNonOverlapped:
    def test_each_region_declares_the_terms_it_answers(self, index,
                                                        list_models):
        result = list_models.non_overlapped(QUERY, "paragraph")
        for region in result.regions:
            assert region.role.startswith("answers")
            present = terms_in_region(index, index.by_id[region.region_id])
            assert set(region.terms).issubset(present)

    def test_every_query_term_is_covered_across_the_answer(self, list_models):
        from ranking import IdfTable, parse_query

        query_terms = set(parse_query(
            QUERY, IdfTable(list_models.index, "bm25")).terms)
        result = list_models.non_overlapped(QUERY, "paragraph")
        answered = set()
        for region in result.regions:
            answered.update(region.terms)
        assert answered == query_terms

    def test_it_finds_papers_the_overlapped_model_cannot(self, list_models):
        query = "document index positions"
        strict = set(list_models.overlapped(query, "paragraph").papers)
        loose = set(list_models.non_overlapped(query, "paragraph").papers)
        assert loose - strict, "the coverage difference is the whole point"

    def test_the_strict_papers_are_a_subset_of_the_permissive_ones(self,
                                                                   list_models):
        for query in ("document index positions", "mean average precision",
                      "term weighting saturation", "query expansion embeddings"):
            strict = set(list_models.overlapped(query, "paragraph").papers)
            loose = set(list_models.non_overlapped(query, "paragraph").papers)
            assert strict <= loose, query

    def test_regions_in_one_paper_do_not_overlap(self, list_models):
        result = list_models.non_overlapped(QUERY, "paragraph")
        by_paper: dict[str, list] = {}
        for region in result.regions:
            by_paper.setdefault(region.doc_id, []).append(region)
        for rows in by_paper.values():
            rows.sort(key=lambda r: r.start)
            for before, after in zip(rows, rows[1:]):
                assert before.end <= after.start


class TestBoth:
    @pytest.mark.parametrize("hierarchy", ["section", "paragraph", "sentence"])
    def test_the_answer_is_sorted_and_ranked(self, list_models, hierarchy):
        for model in (list_models.overlapped, list_models.non_overlapped):
            result = model(QUERY, hierarchy)
            scores = [r.score for r in result.regions]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("hierarchy", ["section", "paragraph", "sentence"])
    def test_returned_regions_exist_in_the_index(self, index, list_models,
                                                 hierarchy):
        for model in (list_models.overlapped, list_models.non_overlapped):
            for region in model(QUERY, hierarchy).regions:
                assert region.region_id in index.by_id

    def test_a_coarser_hierarchy_gives_longer_regions_on_average(self, index,
                                                               list_models):
        """A property of the CORPUS, not of one lucky query.

        Asking a single query to produce a strictly longer region at each level
        is wrong: a paragraph and its section can be the same span when the
        field holds one paragraph. The mean region length, on the other hand,
        has to grow with every step up the tree.
        """
        del list_models
        means = {}
        for hierarchy in ("sentence", "paragraph", "section"):
            regions = index.regions_of(hierarchy)
            means[hierarchy] = sum(r.length for r in regions) / len(regions)
        assert means["sentence"] < means["paragraph"] < means["section"]

    def test_every_hierarchy_answers_with_real_regions(self, list_models):
        for hierarchy in ("sentence", "paragraph", "section"):
            result = list_models.overlapped(QUERY, hierarchy)
            assert result.ok
            for region in result.regions:
                assert region.length > 0