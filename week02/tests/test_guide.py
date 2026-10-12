"""The GUIDE.md at the repository root must stay true.

This exists for the same reason check_doc_lines.py and check_doc_claims.py do:
the documentation in this repository was wrong twice for different reasons, and
both times nothing caught it. A guide that tells a reader the wrong expected
output is worse than no guide, because it is trusted.

So the guide's promises are asserted here. If the code changes and the guide
does not, this fails instead of quietly misleading somebody.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import engine as ir
from evaluate import evaluate_all
from fields import DEFAULT_WEIGHTS
from lists import ListModels
from proximity import ProximalNodeModel
from ranking import Bm25Model, parse_query

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "GUIDE.md"


@pytest.fixture(scope="module")
def guide() -> str:
    assert GUIDE.exists(), "GUIDE.md is missing from the repository root"
    return GUIDE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def engine():
    return ir.get_engine()


@pytest.fixture(scope="module")
def scores(engine):
    return {s.model: s
            for s in evaluate_all(engine.judgements, ir.evaluation_models())}


def resolves(name: str) -> bool:
    """True if the guide's name is real, from the root or from week02/.

    The guide runs `cd week02` before listing the standalone modules, so it
    names `ranking.py` where a root-level reader would expect `week02/ranking.py`.
    Both spellings are legitimate; neither should rot.
    """
    path = name.replace("\\", "/")
    return any((REPO / p).exists()
               for p in (path, f"week02/{path}", f"week01/{path}"))


class TestTheGuideExists:
    def test_it_is_at_the_repository_root(self):
        assert GUIDE.name == "GUIDE.md"
        assert GUIDE.parent == REPO

    def test_it_covers_both_modules(self, guide):
        assert "week01" in guide
        assert "week02" in guide


class TestEveryCommandItDocumentsResolves:
    """A guide with a dead command in it is worse than one without a command."""

    @staticmethod
    def _is_a_placeholder(name: str) -> bool:
        return "XX" in name or name == "file.py"

    def test_every_python_file_it_names_exists(self, guide):
        named = set(re.findall(r"[\w/\\]+\.py", guide))
        dead = [n for n in named
                if not self._is_a_placeholder(n) and not resolves(n)]
        assert dead == [], f"GUIDE.md names files that do not exist: {dead}"

    def test_every_data_path_it_names_exists(self, guide):
        named = set(re.findall(r"[\w/\\]*data/[\w/\\.]+", guide))
        dead = [n for n in named
                if not self._is_a_placeholder(n) and not resolves(n)]
        assert dead == [], f"GUIDE.md names data paths that do not exist: {dead}"

    def test_it_lists_all_nine_modes(self, guide, engine):
        for mode in ir.MODES:
            assert mode in guide, f"GUIDE.md never mentions the {mode} mode"

    def test_it_lists_all_six_tabs(self, guide):
        for tab in ("Search", "Fields & Lists", "Proximity", "Evaluation",
                    "Corpus", "Pipeline"):
            assert tab in guide

    def test_its_stated_test_count_is_the_real_one(self, guide,
                                                   collected_count):
        """Adding a test makes this fail until the guide is updated.

        Deliberately self-referential: the number a reader is told to expect is
        the one most likely to rot, because adding a test does not announce that
        it changed a document.
        """
        assert collected_count > 0
        assert re.search(rf"\|\s*`?-m pytest`?\s*\|[^|]*\|[^|]*"
                         rf"{collected_count} passed", guide), (
            f"GUIDE.md should say pytest reports {collected_count} passed")


class TestTheNumbersTheGuideQuotes:
    """Section 4 is a promise: 'you must see X'. X has to be X."""

    def test_corpus_figures(self, guide, engine):
        stats = ir.corpus_stats()
        for value in (stats["papers"], stats["vocabulary"], stats["postings"],
                      stats["sections"], stats["paragraphs"],
                      stats["sentences"], stats["positions"]):
            assert str(value) in guide or f"{value:,}" in guide, \
                f"GUIDE.md never states {value}"
        assert "144 to 197" in guide
        assert "171.7" in guide

    def test_the_most_frequent_term_claim(self, guide, engine):
        """"`two` in 16 of 20 papers" - the guide tells you to go and check it."""
        assert engine.index.document_frequency["two"] == 16
        assert "two" in guide
        assert "16 of 20" in guide

    def test_the_qrels_figures(self, guide, engine):
        judgements = engine.judgements
        positives = sum(len(judgements.relevant[q]) for q in judgements.query_ids)
        negatives = sum(len(judgements.judged[q]) - len(judgements.relevant[q])
                        for q in judgements.query_ids)
        assert len(judgements.query_ids) == 12
        assert positives == 18
        assert negatives == 61
        assert "18 positives" in guide
        assert "61" in guide

    def test_the_field_weights(self, guide):
        assert list(DEFAULT_WEIGHTS.values()) == [3.0, 2.0, 1.5, 1.0]
        assert "title" in guide

    def test_the_length_normalisation_range(self, guide, engine):
        """The guide claims 1.35-1.66x. This was 1.5-1.7x once, wrongly."""
        raw = engine.fields.weighted("structured field search", normalise=False)
        normalised = engine.fields.weighted("structured field search",
                                            normalise=True)
        ratios = [round(a.score / b.score, 2)
                  for a, b in zip(raw.hits, normalised.hits) if b.score]
        assert (min(ratios), max(ratios)) == (1.35, 1.66)
        assert "1.35" in guide and "1.66" in guide


class TestSection4Promises:
    """Every 'you must see' in section 4, checked against the engine."""

    def test_4_2_bm25_beats_bim_on_mrr(self, guide, scores):
        assert scores["BM25"].mrr == 1.0
        assert round(scores["BIM"].mrr, 3) == 0.917
        assert "1.000" in guide and "0.917" in guide

    def test_4_3_overlapped_is_a_strict_subset(self, guide, engine):
        lists = ListModels(engine.index)
        query = "document index positions"
        strict = lists.overlapped(query, "paragraph")
        loose = lists.non_overlapped(query, "paragraph")
        assert strict.count == 1
        assert strict.regions[0].region_id == "p06:results:p2"
        assert loose.count == 4
        assert sorted(set(loose.papers) - set(strict.papers)) == ["p11"]
        assert "p06:results:p2" in guide
        assert "3/3" in guide

    def test_4_4_a_loose_phrase_is_found_but_not_a_strict_one(self, engine):
        lists = ListModels(engine.index)
        assert lists.overlapped("ranking web search", "paragraph").count == 0
        assert len(lists.non_overlapped("ranking web search",
                                        "paragraph").papers) == 2

    def test_4_5_cross_field_and_returns_one_paper(self, engine):
        result = engine.fields.restrict("title:search AND results:precision")
        assert result.ok
        assert result.papers == ["p19"]

    def test_4_6_brackets_and_the_flat_form_agree(self, guide, engine):
        grouped = engine.fields.restrict(
            "(title:search OR abstract:rank) AND abstract:result")
        flat = engine.fields.restrict(
            "title:search AND abstract:result OR abstract:rank AND abstract:result")
        assert grouped.ok and flat.ok
        assert grouped.papers == flat.papers
        assert grouped.papers == ["p03", "p15", "p18", "p19", "p20"]
        for paper in grouped.papers:
            assert paper in guide, f"GUIDE.md omits {paper} from the 4.6 result set"

    def test_4_7_a_stopword_query_never_returns_everything(self, engine):
        for mode in ir.MODES:
            if mode == "proximal":
                continue  # needs its own constructor syntax
            response = ir.search("the", mode=mode)
            assert response.count < len(engine.index.papers), mode

    def test_4_7_machine_learning_is_a_partial_match(self, guide, engine):
        response = ir.search("machine learning")
        assert response.terms == ["machin", "learn"]
        assert response.count == 6
        assert all(h.matched < h.total_terms for h in response.hits)
        assert "1/2" in guide

    def test_4_8_malformed_queries_are_refused(self, engine):
        expected = {
            "title:": "nothing to look for",
            ":search": "no field name",
            "title:search AND": "nothing to join to",
            "(title:a OR title:b": "unmatched",
            "keywords:search": "not a field of this corpus",
            "title:search NOT results:x": "NOT is not supported",
        }
        for text, fragment in expected.items():
            result = engine.fields.restrict(text)
            assert not result.ok, text
            assert fragment in result.error, (text, result.error)

    def test_4_9_the_proximal_model_is_a_set(self, guide, engine):
        result = ProximalNodeModel(engine.index).evaluate(
            "section with(2) retrieval")
        assert result.count == 5
        response = ir.search("section with(2) retrieval", mode="proximal")
        assert all(hit.score == 0.0 for hit in response.hits)
        assert "set, not a ranking" in guide.lower()

    def test_the_bm25_bim_bridge_the_guide_points_at(self, engine):
        """"set k1 to 0 and the list matches bim" - section 4.2."""
        parsed = parse_query("structured field search", engine.idf_table)
        zero = Bm25Model(engine.index).with_parameters(k1=0.0)
        worst = max(abs(zero.score_paper(p.doc_id, parsed.terms)[0]
                        - engine.bim.score_paper(p.doc_id, parsed.terms)[0])
                    for p in engine.index.papers)
        assert worst == 0.0


class TestTheGuideTellsTheTruthAboutLimits:
    """A guide that oversells is worse than no guide."""

    def test_it_warns_that_20_papers_is_not_a_benchmark(self, guide, engine):
        assert "demonstration, not a benchmark" in guide
        assert len(engine.index.papers) == 20

    def test_it_names_nohit_as_the_column_that_exposes_failure(self, guide, scores):
        assert len(scores["overlapped"].queries_with_no_hit) == 3
        assert "nohit" in guide
        assert "three" in guide

    def test_it_does_not_claim_to_fix_the_expected_value_on_failure(self, guide):
        assert 'Never "fix" a failing check' in guide