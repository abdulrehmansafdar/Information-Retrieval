"""The engine end to end: the three modes, ranking, the set identity, and errors."""

from __future__ import annotations

import math

import pytest

from fuzzy_search import MODES, FuzzyEngine, Settings
from query_parser import QueryError

QUERIES = (
    "hypertension treatment",
    "hypertension AND NOT diabetes",
    "heart failure BUT NOT (asthma OR -fever)",
    "asthma OR COPD",
    "covid-19",
    "cardiology",
)


class TestSettings:
    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            Settings(mode="vector")

    def test_p_below_one_is_refused(self):
        with pytest.raises(ValueError):
            Settings(p=0.5)

    def test_tau_must_be_between_zero_and_one(self):
        with pytest.raises(ValueError):
            Settings(tau=1.5)

    def test_boolean_mode_is_exact_and_binary(self):
        effective = Settings(mode="boolean").effective()
        assert effective.partial is False and effective.function == "binary"

    def test_fuzzy_mode_keeps_its_settings(self):
        assert Settings(mode="fuzzy", function="linear").effective().function == "linear"


class TestTheThreeModes:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("query", QUERIES)
    def test_every_score_is_between_zero_and_one(self, engine, mode, query):
        response = engine.search(query, Settings(mode=mode))
        assert response.ok
        assert all(0.0 <= hit.score <= 1.0 for hit in response.hits)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("query", QUERIES)
    def test_results_are_sorted_best_first(self, engine, mode, query):
        hits = engine.search(query, Settings(mode=mode)).hits
        keys = [(-h.score, h.doc_id) for h in hits]
        assert keys == sorted(keys)

    def test_fuzzy_returns_only_articles_with_a_positive_score(self, engine):
        hits = engine.search("fever", Settings(mode="fuzzy")).hits
        assert hits and all(h.score > 0 for h in hits)

    def test_boolean_hits_are_all_one_and_unranked_by_score(self, engine):
        hits = engine.search("asthma OR COPD", Settings(mode="boolean")).hits
        assert {h.score for h in hits} == {1.0}

    def test_boolean_and_is_exact_word_intersection(self, engine, index):
        response = engine.search("hypertension AND treatment",
                                 Settings(mode="boolean"))
        expected = [d for d in index.doc_ids
                    if index.weight("hypertens", d) and index.weight("treatment", d)]
        assert [h.doc_id for h in response.hits] == expected

    def test_a_typo_finds_nothing_in_boolean_mode(self, engine):
        assert engine.search("hypertenion", Settings(mode="boolean")).count == 0

    def test_a_typo_finds_hypertension_in_fuzzy_mode(self, engine):
        response = engine.search("hypertenion", Settings(mode="fuzzy"))
        assert response.hits[0].doc_id == "m01"
        assert response.hits[0].matches[0].matched_as == "hypertens"

    def test_a_typo_is_found_in_extended_mode_too(self, engine):
        response = engine.search("hypertenion", Settings(mode="extended"))
        assert "m01" in [h.doc_id for h in response.hits]


class TestExtendedBooleanSets:
    @pytest.mark.parametrize("query", QUERIES)
    def test_the_extended_hits_are_exactly_the_alpha_cut_set(self, engine, query):
        settings = Settings(mode="extended")
        hits = {h.doc_id for h in engine.search(query, settings).hits}
        assert hits == engine.crisp_set(query, settings)

    def test_and_not_removes_the_excluded_article(self, engine):
        hits = [h.doc_id for h in engine.search("hypertension AND NOT diabetes",
                                                Settings(mode="extended")).hits]
        assert hits == ["m01"]

    def test_implicit_or_returns_a_wider_set_than_implicit_and(self, engine):
        narrow = engine.crisp_set("asthma COPD", Settings(mode="extended", implicit="AND"))
        wide = engine.crisp_set("asthma COPD", Settings(mode="extended", implicit="OR"))
        assert narrow <= wide and narrow != wide

    @pytest.mark.parametrize("a, b, c", [
        ("diabetes", "hypertension", "kidney"),
        ("asthma", "COPD", "smoking"),
        ("fever", "malaria", "liver"),
        ("pain", "stroke", "epilepsy"),
    ])
    def test_the_lecture_identity_holds_on_the_corpus(self, engine, a, b, c):
        view = engine.set_view(a, b, c)
        assert view["equal"]
        assert view["result"] == (view["union"] - view["C"])

    def test_an_alpha_cut_above_zero_keeps_only_strong_articles(self, engine):
        loose = engine.crisp_set("hypertension", Settings(mode="extended", alpha=0.0))
        strict = engine.crisp_set("hypertension", Settings(mode="extended", alpha=0.5))
        assert strict <= loose and len(strict) < len(loose)


class TestCompareAndErrors:
    def test_compare_returns_one_response_per_mode(self, engine):
        results = engine.compare("asthma OR COPD")
        assert set(results) == set(MODES)

    def test_a_bad_query_is_an_error_response_not_an_exception(self, engine):
        response = engine.search("the")
        assert not response.ok
        assert "stop word" in response.message

    def test_an_empty_query_is_refused_with_a_message(self, engine):
        response = engine.search("")
        assert not response.ok and "empty" in response.message

    def test_crisp_set_raises_on_a_bad_query(self, engine):
        with pytest.raises(QueryError):
            engine.crisp_set("hypertension AND")


class TestSupportingViews:
    def test_the_membership_table_is_sorted_and_complete(self, engine):
        rows = engine.membership_table("hypertension", "log")
        assert len(rows) == 30
        values = [mu for _, _, mu in rows]
        assert values == sorted(values, reverse=True)

    def test_partial_matches_include_the_exact_stem_and_a_variant(self, engine):
        stems = dict(engine.partial_matches("anemia"))
        assert stems["anemia"] == 1.0
        assert "anaemia" in stems

    def test_a_hyphenated_query_finds_the_covid_article(self, engine):
        response = engine.search("covid-19", Settings(mode="fuzzy"))
        assert response.hits[0].doc_id == "m17"

    def test_the_specialty_is_searchable(self, engine):
        hits = [h.doc_id for h in engine.search("cardiology", Settings(mode="fuzzy")).hits]
        assert {"m01", "m02", "m03", "m04"} <= set(hits)

    def test_the_working_behind_a_score_is_kept(self, engine):
        hit = engine.search("hypertension AND treatment").hits[0]
        assert hit.working.label == "AND"
        assert math.isclose(hit.score, min(c.value for c in hit.working.children))

    def test_engine_is_built_from_any_article_list(self, articles):
        small = FuzzyEngine(articles[:5])
        assert len(small.articles) == 5
        assert small.search("hypertension").considered == 5
