"""Step 0: the collection loads, and every document is complete.

A browser built on half-loaded documents produces plausible pages from missing
data, so the loader's contract is asserted rather than assumed: every file has
its three metadata lines, its named sections, and a term list to search.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from collection import (CollectionFormatError, corpus_stats, parse_document,
                        tokenize)

WEEK03 = Path(__file__).resolve().parent.parent
WEEK02_PAPERS = WEEK03.parent / "week02" / "data" / "papers"


class TestLoading:
    def test_twenty_documents(self, documents):
        assert len(documents) == 20

    def test_they_are_sorted_by_id(self, documents):
        assert [d.doc_id for d in documents] == \
            [f"p{i:02d}" for i in range(1, 21)]

    def test_every_document_has_metadata(self, documents):
        for document in documents:
            assert document.title, document.doc_id
            assert 2018 <= document.year <= 2026, document.doc_id
            assert document.venue, document.doc_id

    def test_every_document_has_the_three_sections(self, documents):
        for document in documents:
            assert document.section_names == \
                ("Abstract", "Methodology", "Results"), document.doc_id

    def test_every_section_has_text(self, documents):
        for document in documents:
            for section in document.sections:
                assert section.text.strip(), (document.doc_id, section.name)
                assert section.words > 0

    def test_every_document_has_searchable_terms(self, documents):
        for document in documents:
            assert document.terms, document.doc_id
            assert document.topic_terms, document.doc_id

    def test_the_abstract_helper_finds_the_abstract(self, documents):
        for document in documents:
            assert document.section("abstract") is not None
            assert document.abstract == document.section("abstract").text


class TestParsing:
    def test_a_missing_title_is_a_named_error(self, tmp_path):
        broken = tmp_path / "broken.txt"
        broken.write_text("Year: 2024\n\n## Abstract\nx\n", encoding="utf-8")
        with pytest.raises(CollectionFormatError) as caught:
            parse_document(broken)
        assert "Title" in str(caught.value)

    def test_a_document_with_no_sections_is_a_named_error(self, tmp_path):
        broken = tmp_path / "broken.txt"
        broken.write_text("Title: x\nYear: 2024\nVenue: y\n", encoding="utf-8")
        with pytest.raises(CollectionFormatError):
            parse_document(broken)

    def test_two_metadata_lines_are_not_pushed_into_the_body(self, tmp_path):
        path = tmp_path / "ok.txt"
        path.write_text("Title: A paper\nYear: 2021\nVenue: Somewhere\n\n"
                        "## Abstract\nthe body\n", encoding="utf-8")
        document = parse_document(path)
        assert document.year == 2021
        assert document.venue == "Somewhere"
        assert "Year" not in document.abstract
        assert "2021" not in " ".join(document.terms)


class TestTokenizer:
    def test_it_lowercases_and_drops_punctuation(self):
        assert tokenize("Browsing, Navigation!") == ["browsing", "navigation"]

    def test_stop_words_and_single_characters_go(self):
        assert tokenize("a the of to x") == []
        assert tokenize("the model and the data") == ["model", "data"]


class TestStatistics:
    def test_the_stats_agree_with_the_documents(self, documents):
        stats = corpus_stats()
        assert stats["documents"] == len(documents)
        assert stats["sections"] == sum(len(d.sections) for d in documents)
        assert stats["words"] == sum(d.words for d in documents)
        assert stats["section_names"] == ["Abstract", "Methodology", "Results"]


class TestStandaloneCopy:
    """week03 ships its own corpus; it must not drift from week02's copy.

    week02's collection is the curated source. week03 duplicates it so the
    module runs standalone, and this test is what stops the duplicate going
    stale. Skipped when week02 is not checked out alongside, so the module still
    tests on its own.
    """

    def test_the_copy_matches_week02(self):
        if not WEEK02_PAPERS.is_dir():
            pytest.skip("week02 corpus not present")
        ours = {p.name: p.read_text(encoding="utf-8").strip()
                for p in (WEEK03 / "data" / "papers").glob("*.txt")}
        theirs = {p.name: p.read_text(encoding="utf-8").strip()
                  for p in WEEK02_PAPERS.glob("*.txt")}
        assert ours == theirs, ("week03/data/papers has drifted from "
                                "week02/data/papers")
