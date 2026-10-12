"""Step 1-2: preprocessing, positional index and the region hierarchies.

These are the properties every model in the week leans on. A model built on
regions that do not tile their document still runs and still produces numbers,
so the only protection is asserting them.
"""

from __future__ import annotations

import pytest

from preprocess import FIELDS, PHRASES, preprocess_with_positions
from structured_index import HIERARCHIES, check_invariants


class TestPhraseList:
    """A phrase entry can only help if the text it names actually occurs.

    A DEAD entry is worse than no entry: listing "machine learning" when the
    corpus never uses it merges two perfectly good words ('machin', df=2 and
    'learn', df=4) into one term with df=0, so a user typing "machine learning"
    gets zero results. That bug shipped in the first draft of this file and was
    found by running the search, not by reading it.
    """

    def test_every_phrase_occurs_in_the_corpus(self, index):
        dead = []
        for phrase in PHRASES:
            term, _ = preprocess_with_positions(phrase, drop_stop_words=False)
            joined = " ".join(term)
            if index.document_frequency.get(joined, 0) == 0:
                dead.append(f"{phrase!r} -> {joined!r}")
        assert dead == [], ("phrases that never occur, and so only break "
                           "queries:\n  " + "\n  ".join(dead))

    def test_a_topic_phrase_the_corpus_lacks_still_searches_both_words(self):
        """'machine learning' is not a phrase here, so both words survive.

        This is the regression test for the bug above, stated as behaviour
        rather than as a property of the list.
        """
        terms, positions = preprocess_with_positions("machine learning")
        assert terms == ["machin", "learn"]
        assert positions == [0, 1]

    def test_a_real_phrase_is_still_one_term(self):
        terms, positions = preprocess_with_positions("deep learning")
        assert terms == ["deep learn"]
        assert positions == [0]


class TestPositions:
    def test_positions_are_dense_and_in_order(self):
        terms, positions = preprocess_with_positions(
            "Hospitals and clinics collect large amounts of medical data"
        )
        assert positions == list(range(len(terms)))

    def test_phrase_keeps_its_place_in_the_stream(self):
        """The one behaviour week01's preprocess_text() cannot produce.

        week01 returns `unigrams + phrases`, so the phrase term lands at the END
        of the list and document order is destroyed. Here a phrase occupies the
        position of its first word, which is what makes regions possible.
        """
        terms, _ = preprocess_with_positions(
            "Deep learning improves clinic workflow in patient records"
        )
        assert terms[0] == "deep learn"
        # 'improv' immediately follows it - order was preserved
        assert terms[1] == "improv"
        assert "deep learn" in terms

    def test_a_phrase_occupies_exactly_one_slot(self):
        terms, positions = preprocess_with_positions("deep learning model")
        assert len(terms) == len(positions) == 2
        assert positions == [0, 1]
        assert terms == ["deep learn", "model"]

    def test_stop_words_are_dropped_for_documents(self):
        terms, _ = preprocess_with_positions("the model and the data")
        assert terms == ["model", "data"]

    def test_stop_words_survive_for_queries(self):
        terms, _ = preprocess_with_positions("the model", drop_stop_words=False)
        assert "the" in terms

    def test_every_paper_has_dense_positions(self, index):
        for paper in index.papers:
            positions = [t.position for t in paper.tokens]
            assert positions == list(range(paper.length)), paper.doc_id


class TestFields:
    def test_all_twenty_papers_parsed(self, index):
        assert index.number_of_documents == 20

    def test_every_paper_has_all_four_named_fields(self, index):
        for paper in index.papers:
            for name in FIELDS:
                assert name in paper.fields, f"{paper.doc_id} missing {name}"
                assert paper.fields[name].length > 0

    def test_year_and_venue_are_read_as_metadata(self, index):
        for paper in index.papers:
            assert paper.year.isdigit(), f"{paper.doc_id} year {paper.year!r}"
            assert paper.venue, f"{paper.doc_id} has no venue"

    def test_fields_tile_the_paper_exactly(self, index):
        for paper in index.papers:
            total = sum(r.length for r in paper.fields.values())
            assert total == paper.length, paper.doc_id

    def test_a_paper_missing_its_title_line_raises(self, tmp_path):
        from preprocess import PaperFormatError, parse_paper

        broken = tmp_path / "broken.txt"
        broken.write_text("Year: 2024\n\n## Abstract\ntext\n", encoding="utf-8")
        with pytest.raises(PaperFormatError) as caught:
            parse_paper(broken)
        assert "Title" in str(caught.value)


class TestHierarchies:
    def test_invariants_hold_for_the_whole_corpus(self, index):
        problems = check_invariants(index)
        assert problems == [], "\n".join(problems)

    @pytest.mark.parametrize("kind", HIERARCHIES)
    def test_each_hierarchy_tiles_every_paper(self, index, kind):
        for paper in index.papers:
            regions = sorted(index.regions_of_kind(paper.doc_id, kind),
                             key=lambda r: r.start)
            assert regions, f"{paper.doc_id} has no {kind}"
            assert regions[0].start == 0
            assert regions[-1].end == paper.length
            for before, after in zip(regions, regions[1:]):
                assert before.end == after.start, (
                    f"{paper.doc_id}/{kind}: gap or overlap"
                )

    def test_children_are_inside_their_parents(self, index):
        for kind in ("paragraph", "sentence"):
            for region in index.regions_of(kind):
                parent = index.by_id.get(region.parent_id)
                assert parent is not None, region.region_id
                assert parent.start <= region.start
                assert region.end <= parent.end

    def test_regions_of_one_paper_never_overlap(self, index):
        for kind in HIERARCHIES:
            for paper in index.papers:
                rows = sorted(index.regions_of_kind(paper.doc_id, kind),
                              key=lambda r: r.start)
                for before, after in zip(rows, rows[1:]):
                    assert before.end <= after.start


class TestPostings:
    def test_posting_positions_match_the_document(self, index):
        for term in index.vocabulary:
            for post in index.lookup(term):
                paper = next(p for p in index.papers if p.doc_id == post.doc_id)
                actual = [t.position for t in paper.tokens if t.term == term]
                assert list(post.positions) == actual

    def test_document_frequency_equals_the_posting_count(self, index):
        for term in index.vocabulary:
            assert index.document_frequency[term] == len(index.lookup(term))

    def test_term_frequency_is_not_document_frequency(self, index):
        """tf is a count inside one document; df is a count of documents.

        Mixing the two up is the most common IR bug there is, and the console
        walkthrough spends a page on it, so it is worth an assertion.
        """
        assert index.term_frequency("field") > index.document_frequency["field"]

    def test_an_unknown_term_is_a_clean_miss_not_an_error(self, index):
        assert index.lookup("nosuchterm") == []
        assert index.positions_of("p01", "nosuchterm") == []

    def test_every_position_is_stored_exactly_once(self, index):
        total = sum(len(post.positions)
                    for plist in index.postings.values() for post in plist)
        assert total == index.total_positions
        assert total == sum(p.length for p in index.papers)


class TestLookups:
    def test_enclosing_finds_the_smallest_region(self, index):
        paper = next(p for p in index.papers if p.doc_id == "p01")
        middle = paper.fields["methodology"].start + 2
        section = index.enclosing("section", "p01", middle)
        paragraph = index.enclosing("paragraph", "p01", middle)
        sentence = index.enclosing("sentence", "p01", middle)
        assert section.label == "methodology"
        assert paragraph.length <= section.length
        assert sentence.length <= paragraph.length

    def test_field_at_reports_the_named_field(self, index):
        paper = next(p for p in index.papers if p.doc_id == "p01")
        for name in FIELDS:
            region = paper.fields[name]
            probe = region.start + min(1, region.length - 1)
            assert index.field_at("p01", probe) == name

    def test_an_unknown_hierarchy_raises_with_the_valid_names(self, index):
        with pytest.raises(KeyError) as caught:
            index.regions_of("chapter")
        assert "paragraph" in str(caught.value)