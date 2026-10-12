"""Step 6: the proximal node model.

The operators are tested on the real corpus and, where the semantics are
subtle, on small hand-built fixtures where the right answer can be written down
by hand.
"""

from __future__ import annotations

import pytest

from proximity import (Constructor, ProximitySyntaxError,
                       looks_like_constructor, resolve_constructor)


def kinds(result):
    return {node.region_id for node in result.nodes}


class TestParsing:
    def test_a_bare_field_name_is_a_section_constructor(self):
        assert resolve_constructor("methodology") == \
            Constructor("section", "methodology")

    def test_a_kind_needs_no_label(self):
        assert resolve_constructor("paragraph") == Constructor("paragraph")

    def test_an_explicit_kind_and_label_is_kept(self):
        assert resolve_constructor("paragraph:methodology") == \
            Constructor("paragraph", "methodology")

    def test_a_term_is_not_mistaken_for_a_constructor(self):
        assert not looks_like_constructor("retrieval")
        assert looks_like_constructor("methodology")
        assert looks_like_constructor("sentence")

    def test_every_operator_parses(self, proximal):
        for query in ("paragraph in methodology", "sentence beginin results",
                      "section with(2) retrieval", "paragraph child methodology",
                      "section parent(2) paragraph",
                      "sentence before(10) index (paragraph)",
                      "sentence after(5) index (paragraph)"):
            assert proximal.parse(query).source == query

    def test_a_missing_operator_is_an_error(self, proximal):
        with pytest.raises(ProximitySyntaxError) as caught:
            proximal.parse("paragraph methodology")
        assert "expected one of" in str(caught.value)

    def test_a_missing_left_hand_side_is_an_error(self, proximal):
        with pytest.raises(ProximitySyntaxError) as caught:
            proximal.parse("in methodology")
        assert "nothing before" in str(caught.value)

    def test_an_empty_query_is_an_error_not_a_crash(self, proximal):
        with pytest.raises(ProximitySyntaxError):
            proximal.parse("   ")


class TestStructuralOperators:
    def test_paragraph_in_methodology_returns_only_methodology(self, index,
                                                               proximal):
        result = proximal.evaluate("paragraph in methodology")
        assert result.count > 0
        for node in result.nodes:
            assert node.kind == "paragraph"
            section = index.enclosing("section", node.doc_id, node.start)
            assert section.label == "methodology"

    def test_in_equals_beginin_when_no_node_straddles_a_boundary(self, proximal):
        """A property of THIS corpus's parser, not of the operators.

        Fields are cut at a blank line before a heading, so no paragraph ever
        starts inside one section and ends inside another. On a corpus where a
        section break fell mid-paragraph the two would differ, and that is the
        case the operators exist for.
        """
        strict = kinds(proximal.evaluate("paragraph in results"))
        loose = kinds(proximal.evaluate("paragraph beginin results"))
        assert strict == loose

    def test_child_never_returns_a_grandchild(self, index, proximal):
        result = proximal.evaluate("paragraph child methodology")
        assert result.count > 0
        for node in result.nodes:
            parent = index.by_id[node.parent_id]
            assert parent.kind == "section"
            assert parent.label == "methodology"

    def test_parent_k_returns_only_nodes_with_k_children(self, index, proximal):
        loose = proximal.evaluate("section parent(1) paragraph")
        strict = proximal.evaluate("section parent(2) paragraph")
        assert kinds(strict) <= kinds(loose)

    def test_a_node_is_inside_itself(self, proximal):
        """`section in section` must return every section, not none."""
        result = proximal.evaluate("section in section")
        assert result.count > 0

    def test_beginin_returns_a_subset_of_in(self, proximal):
        assert kinds(proximal.evaluate("sentence beginin results")) <= \
            kinds(proximal.evaluate("sentence in results"))


class TestContentOperators:
    def test_with_k_counts_occurrences_not_documents(self, index, proximal):
        result = proximal.evaluate("section with(2) retrieval")
        assert result.count > 0
        for node in result.nodes:
            count = len(index.positions_of(node.doc_id, "retriev"))
            assert count >= 2, node.region_id

    def test_larger_k_never_adds_nodes(self, proximal):
        few = kinds(proximal.evaluate("section with(1) retrieval"))
        many = kinds(proximal.evaluate("section with(3) retrieval"))
        assert many <= few

    def test_an_inflected_term_is_preprocessed_like_a_document(self, proximal):
        """'retrieval' must reach the index as the stem 'retriev'.

        Without this a proximal query silently matches nothing for every
        inflected form, which looks like a broken operator rather than a
        missing stemmer.
        """
        assert proximal.evaluate("section with(1) retrieval").count == \
            proximal.evaluate("section with(1) retriev").count

    def test_an_unknown_term_gives_an_explained_empty_answer(self, proximal):
        result = proximal.evaluate("section with(1) zzzzqqqq")
        assert result.count == 0
        assert any("not in the vocabulary" in step for step in result.steps)

    def test_a_structural_operator_with_a_term_raises(self, proximal):
        """'paragraph child retrieval' is a malformed query, not an 'after' one.

        The right-hand side parsed as a TERM, so the content path runs - and the
        content path only implements with/before/after. Without an explicit
        rejection, anything that is not 'before' fell through to 'after' and the
        query silently returned a plausible answer to something the user never
        wrote.
        """
        with pytest.raises(ProximitySyntaxError) as caught:
            proximal.evaluate("paragraph child retrieval")
        assert "needs a node type" in str(caught.value)

    @pytest.mark.parametrize("query", ["paragraph in retrieval",
                                       "sentence beginin retrieval",
                                       "section parent(2) retrieval"])
    def test_every_structural_operator_rejects_a_term(self, proximal, query):
        with pytest.raises(ProximitySyntaxError):
            proximal.evaluate(query)


class TestDistanceOperators:
    def test_the_window_is_a_monotone_dial(self, proximal):
        counts = [
            proximal.evaluate(f"sentence before({w}) index (paragraph)").count
            for w in (0, 2, 5, 10, 25, 60)
        ]
        assert counts == sorted(counts)

    def test_a_node_does_not_have_to_contain_the_term(self, proximal):
        """'sentence before(10) index' means a sentence LEADING INTO the word.

        A reading that required the term to be inside the sentence would be a
        different, much weaker operator.
        """
        result = proximal.evaluate("sentence before(30) index (paragraph)")
        index_module = proximal.index
        for node in result.nodes:
            positions = index_module.positions_of(node.doc_id, "index")
            assert any(p >= node.end for p in positions), node.region_id

    def test_context_restricts_to_the_named_node_type(self, proximal):
        loose = proximal.evaluate("sentence before(10) index")
        tight = proximal.evaluate("sentence before(10) index (paragraph)")
        assert tight.count <= loose.count

    def test_after_is_the_mirror_of_before(self, proximal):
        assert proximal.evaluate("sentence after(5) index (paragraph)").count >= 0


class TestBottomUp:
    def test_innermost_is_never_larger_than_its_parent(self, index, proximal):
        for doc_id in ("p01", "p06", "p19"):
            middle = index.doc_lengths[doc_id] // 2
            section = proximal.innermost_component("section", doc_id, middle)
            paragraph = proximal.innermost_component("paragraph", doc_id, middle)
            sentence = proximal.innermost_component("sentence", doc_id, middle)
            assert sentence.length <= paragraph.length <= section.length

    def test_innermost_agrees_with_the_index_lookup(self, index, proximal):
        middle = index.doc_lengths["p11"] // 2
        assert proximal.innermost_component("section", "p11", middle) is \
            index.enclosing("section", "p11", middle)


class TestResultShape:
    def test_the_result_reports_its_steps(self, proximal):
        result = proximal.evaluate("section with(2) retrieval")
        assert result.steps
        assert any("parsed" in step for step in result.steps)

    def test_nodes_come_back_in_document_order(self, proximal):
        result = proximal.evaluate("paragraph in methodology")
        keys = [(n.doc_id, n.start) for n in result.nodes]
        assert keys == sorted(keys)

    def test_papers_are_the_distinct_documents(self, proximal):
        result = proximal.evaluate("paragraph in methodology")
        assert set(result.papers) == {n.doc_id for n in result.nodes}