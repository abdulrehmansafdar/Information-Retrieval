"""The shared corpus and the stemming that every later step depends on."""

from __future__ import annotations

from pathlib import Path

import pytest

from articles import CORPUS_DIR, CorpusFormatError, parse_article, tokenize

REPO = Path(__file__).resolve().parents[2]
WEEK06 = REPO / "week06"
SECTIONS = ("Overview", "Symptoms", "Diagnosis", "Treatment")


class TestTheSharedCorpus:
    def test_there_are_thirty_articles(self, articles):
        assert len(articles) == 30

    def test_ids_run_from_m01_to_m30(self, articles):
        assert [a.doc_id for a in articles] == [f"m{i:02d}" for i in range(1, 31)]

    def test_every_article_has_the_four_sections_in_order(self, articles):
        assert all(a.section_names == SECTIONS for a in articles)

    def test_no_section_is_empty(self, articles):
        assert all(s.text for a in articles for s in a.sections)

    def test_every_article_has_its_metadata(self, articles):
        for a in articles:
            assert a.title and a.specialty and a.code, a.doc_id
            assert 2020 <= a.updated <= 2026, a.doc_id

    def test_there_are_ten_specialties(self, articles):
        assert len({a.specialty for a in articles}) == 10

    def test_the_corpus_lives_once_at_the_repository_root(self):
        assert CORPUS_DIR == REPO / "corpus" / "articles"
        assert CORPUS_DIR.is_dir()

    def test_week06_keeps_no_copy_of_the_corpus(self):
        assert not (WEEK06 / "data").exists()
        assert not list(WEEK06.glob("**/m*.txt"))


class TestStemming:
    def test_a_word_is_lower_cased_and_stemmed(self):
        assert tokenize("Hypertension") == ["hypertens"]

    def test_stop_words_are_dropped(self):
        assert tokenize("the of and") == []

    def test_inflections_share_one_stem(self):
        assert tokenize("treatments") == tokenize("treatment")

    def test_a_hyphen_splits_a_word_into_parts(self):
        assert tokenize("iron-deficiency") == ["iron", "defici"]


class TestReadingAnArticle:
    def write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "m99.txt"
        path.write_text(body, encoding="utf-8")
        return path

    def test_a_missing_field_is_named_in_the_error(self, tmp_path):
        path = self.write(tmp_path, "Title: X\nSpecialty: Y\nUpdated: 2025\n\n## Overview\nText\n")
        with pytest.raises(CorpusFormatError, match="missing code"):
            parse_article(path)

    def test_the_year_must_be_a_number(self, tmp_path):
        path = self.write(tmp_path, "Title: X\nSpecialty: Y\nCode: Z\nUpdated: soon\n\n"
                                    "## Overview\nText\n")
        with pytest.raises(CorpusFormatError, match="must be a year"):
            parse_article(path)

    def test_an_article_with_no_sections_is_refused(self, tmp_path):
        path = self.write(tmp_path, "Title: X\nSpecialty: Y\nCode: Z\nUpdated: 2025\n")
        with pytest.raises(CorpusFormatError, match="no '## Heading' sections"):
            parse_article(path)

    def test_a_stray_line_in_the_metadata_is_refused(self, tmp_path):
        path = self.write(tmp_path, "Title: X\nwhat is this\n\n## Overview\nText\n")
        with pytest.raises(CorpusFormatError, match="unexpected line"):
            parse_article(path)
