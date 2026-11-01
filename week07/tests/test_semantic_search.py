"""The engine: the contract both front ends rely on, and the corpus it reads."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from clinical_docs import CORPUS_DIR, CorpusFormatError, get_articles, parse_article, tokenize
from semantic_search import DEFAULT_K, MODELS, Settings

MAIN = Path(__file__).resolve().parent.parent / "main.py"


class TestTheCorpus:
    def test_it_reads_the_shared_corpus_at_the_root(self):
        assert CORPUS_DIR.parts[-2:] == ("corpus", "articles")
        assert len(get_articles()) == 30

    def test_week07_keeps_no_copy_of_the_data(self):
        assert not (MAIN.parent / "data").exists()

    def test_stemming_joins_word_forms(self):
        assert tokenize("wheezing") == tokenize("wheeze")

    def test_stop_words_are_dropped(self):
        assert tokenize("the and of") == []

    def test_each_article_has_its_four_sections(self):
        for article in get_articles():
            assert [name for name, _ in article.sections] == [
                "Overview", "Symptoms", "Diagnosis", "Treatment"]

    @pytest.mark.parametrize("text", ["nonsense line\n\n## A\nb", "Title: x\n\n## A\nb",
                                      "Title: a\nSpecialty: b\nCode: c\nUpdated: 1\n\nbody"])
    def test_a_broken_file_is_named_in_the_error(self, tmp_path, text):
        path = tmp_path / "bad.txt"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(CorpusFormatError, match="bad.txt"):
            parse_article(path)


class TestSearch:
    @pytest.mark.parametrize("model", MODELS)
    def test_hits_are_ranked_best_first_within_range(self, semantic_engine, model):
        hits = semantic_engine.search("chest pain", Settings(model=model)).hits
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(0 < s <= 1 for s in scores)

    def test_top_limits_the_list_but_not_the_count(self, semantic_engine):
        response = semantic_engine.search("chest pain", Settings(top=3))
        assert len(response.hits) == 3
        assert response.count > 3

    def test_gvsm_only_returns_articles_with_a_shared_word(self, semantic_engine):
        hits = semantic_engine.search("chest pain", Settings("gvsm", top=30)).hits
        assert hits and all(h.shared_terms for h in hits)

    def test_lsi_reaches_copd_for_wheezing_without_the_word(self, semantic_engine):
        hits = semantic_engine.search("wheezing").hits
        copd = next(h for h in hits if h.doc_id == "m10")
        assert copd.shared_terms == ()
        assert copd.score > 0.8

    def test_gvsm_finds_only_asthma_for_wheezing(self, semantic_engine):
        response = semantic_engine.search("wheezing", Settings("gvsm"))
        assert [h.doc_id for h in response.hits] == ["m09"]

    def test_the_default_model_is_lsi_at_k_15(self):
        assert Settings().model == "lsi"
        assert Settings().k == DEFAULT_K == 15

    def test_an_unknown_word_is_reported_and_the_rest_still_run(self, semantic_engine):
        response = semantic_engine.search("wheezing zzzqqq")
        assert response.ok
        assert response.unknown == ("zzzqqq",)


class TestRefusals:
    def test_only_stop_words_is_a_message(self, semantic_engine):
        response = semantic_engine.search("the of and")
        assert not response.ok and "searchable" in response.message

    def test_an_empty_query_is_a_message(self, semantic_engine):
        assert not semantic_engine.search("   ").ok

    @pytest.mark.parametrize("model", MODELS)
    def test_words_outside_the_collection_are_a_message(self, semantic_engine, model):
        response = semantic_engine.search("zzzqqq", Settings(model=model))
        assert not response.ok and "collection" in response.message

    def test_a_bad_k_is_a_message_not_a_crash(self, semantic_engine):
        response = semantic_engine.search("wheezing", Settings(k=99))
        assert not response.ok and "between" in response.message

    def test_steps_for_an_unknown_article_is_refused(self, semantic_engine):
        with pytest.raises(ValueError):
            semantic_engine.steps("wheezing", "m99")


class TestCompareAndSteps:
    def test_compare_returns_both_models(self, semantic_engine):
        assert set(semantic_engine.compare("wheezing")) == set(MODELS)

    def test_a_query_of_stop_words_fails_in_both(self, semantic_engine):
        assert not any(r.ok for r in semantic_engine.compare("the").values())

    @pytest.mark.parametrize("model", MODELS)
    def test_steps_match_the_score_shown(self, semantic_engine, model):
        settings = Settings(model=model)
        for hit in semantic_engine.search("chest pain", settings).hits[:5]:
            assert semantic_engine.steps("chest pain", hit.doc_id, settings).score == (
                pytest.approx(hit.score))

    def test_gvsm_vectors_have_one_dimension_per_term(self, semantic_engine):
        steps = semantic_engine.steps("wheezing", "m09", Settings("gvsm"))
        assert len(steps.q) == semantic_engine.matrix().shape[0]

    def test_lsi_vectors_have_k_dimensions(self, semantic_engine):
        assert len(semantic_engine.steps("wheezing", "m09", Settings(k=7)).q) == 7


class TestTheConsole:
    @pytest.mark.parametrize("flags, expected", [
        (["--query", "wheezing", "--compare"], "m10"),
        (["--query", "wheezing", "--steps", "m10"], "Cosine similarity"),
        (["--slide"], "0.97"),
        (["--svd", "slide"], "Sigma_k"),
        (["--matrix", "slide"], "t1"),
        (["--concepts"], "strength"),
    ])
    def test_each_command_prints_its_answer(self, flags, expected):
        done = subprocess.run([sys.executable, str(MAIN), "--console", *flags],
                              capture_output=True, encoding="utf-8", timeout=120)
        assert done.returncode == 0, done.stderr
        assert expected in done.stdout

    @pytest.mark.parametrize("flags", [["--query", "the"], ["--query", "wheezing", "--k", "99"],
                                       ["--query", "wheezing", "--steps", "m99"]])
    def test_a_bad_request_exits_1_without_a_traceback(self, flags):
        done = subprocess.run([sys.executable, str(MAIN), "--console", *flags],
                              capture_output=True, encoding="utf-8", timeout=120)
        assert done.returncode == 1
        assert "Traceback" not in done.stderr
