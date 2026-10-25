"""Step 1: the category hierarchy is derived, complete and walkable.

The point of deriving categories from the papers is that the derivation can be
checked. These tests re-derive the assignment, prove no document falls through to
the fallback, and confirm the tree the browser walks is the one the rules built.
"""

from __future__ import annotations

import pytest

from taxonomy import FALLBACK, RULES, build_tree, classify


class TestClassification:
    def test_every_document_is_assigned(self, documents):
        for document in documents:
            domain, subtopic = classify(document)
            assert domain and subtopic, document.doc_id

    def test_no_document_falls_through_to_the_fallback(self, documents):
        falls = [d.doc_id for d in documents if classify(d) == FALLBACK]
        assert falls == [], f"no rule matched: {falls}"

    def test_the_title_decides_before_the_abstract(self, documents):
        """p01 is 'Deep Learning ... Medical Image Segmentation'.

        'learning' is an ML word and 'medical' is an Applied word; the rule order
        puts medical first, so the title must yield Applied AI, not ML.
        """
        p01 = next(d for d in documents if d.doc_id == "p01")
        assert classify(p01) == ("applied", "healthcare")

    def test_a_systems_paper_is_not_swallowed_by_ir(self, documents):
        p18 = next(d for d in documents if d.doc_id == "p18")
        assert classify(p18) == ("systems", "distributed")

    def test_the_rules_have_no_duplicate_domain_subtopic_pairs(self):
        pairs = [(r.domain, r.subtopic) for r in RULES]
        assert len(pairs) == len(set(pairs))

    def test_classification_is_deterministic(self, documents):
        first = [classify(d) for d in documents]
        second = [classify(d) for d in documents]
        assert first == second


class TestTheTree:
    def test_the_root_holds_every_document(self, taxonomy, documents):
        assert taxonomy.root.doc_ids == tuple(sorted(d.doc_id for d in documents))

    def test_there_are_six_domains(self, taxonomy):
        assert len(taxonomy.domains()) == 6

    def test_no_category_is_empty(self, taxonomy):
        for node in taxonomy.walk():
            assert node.count > 0, node.node_id

    def test_every_document_hangs_off_exactly_one_subtopic(self, taxonomy,
                                                           documents):
        seen = []
        for leaf in taxonomy.leaves():
            assert leaf.kind == "subtopic"
            seen.extend(leaf.doc_ids)
        assert sorted(seen) == sorted(d.doc_id for d in documents)
        assert len(seen) == len(set(seen))

    def test_a_domain_count_is_the_sum_of_its_subtopics(self, taxonomy):
        for domain in taxonomy.domains():
            subtotals = sum(n.count for n in taxonomy.subtopics(domain.node_id))
            assert domain.count == subtotals, domain.node_id

    def test_the_depths_are_zero_one_two(self, taxonomy):
        for node in taxonomy.walk():
            assert node.depth == node.node_id.count("/")

    def test_parents_point_at_children_and_back(self, taxonomy):
        for node in taxonomy.walk():
            for child in node.children:
                assert taxonomy.find(child).parent_id == node.node_id


class TestLookups:
    def test_category_of_returns_the_subtopic(self, taxonomy, documents):
        for document in documents:
            leaf = taxonomy.category_of(document.doc_id)
            assert leaf.kind == "subtopic"
            assert document.doc_id in leaf.doc_ids

    def test_category_of_an_unknown_document_raises(self, taxonomy):
        with pytest.raises(KeyError):
            taxonomy.category_of("p99")

    def test_documents_in_a_domain_is_their_union(self, taxonomy):
        for domain in taxonomy.domains():
            expected = sorted(d for leaf in taxonomy.subtopics(domain.node_id)
                              for d in leaf.doc_ids)
            assert taxonomy.documents_in(domain.node_id) == expected

    def test_an_unknown_category_raises_with_a_hint(self, taxonomy):
        with pytest.raises(KeyError) as caught:
            taxonomy.find("root/nope")
        assert "nope" in str(caught.value)


class TestBreadcrumbs:
    def test_a_document_trail_is_repository_domain_subtopic(self, taxonomy,
                                                            documents):
        for document in documents:
            leaf = taxonomy.category_of(document.doc_id)
            trail = taxonomy.breadcrumb(leaf.node_id)
            assert [n.kind for n in trail] == ["root", "domain", "subtopic"]
            assert trail[0].name == "Repository"
            assert trail[-1] is leaf

    def test_the_root_trail_is_itself(self, taxonomy):
        assert [n.node_id for n in taxonomy.breadcrumb("root")] == ["root"]

    def test_every_node_has_a_trail_from_the_root(self, taxonomy):
        for node in taxonomy.walk():
            trail = taxonomy.breadcrumb(node.node_id)
            assert trail[0].node_id == "root"
            assert trail[-1].node_id == node.node_id


class TestFacets:
    def test_the_year_facet_covers_every_document(self, taxonomy, documents):
        assert sum(taxonomy.facets()["year"].values()) == len(documents)

    def test_the_domain_facet_matches_the_domains(self, taxonomy):
        facet = taxonomy.facets()["domain"]
        for domain in taxonomy.domains():
            assert facet[domain.name] == domain.count

    def test_the_venue_facet_counts_are_positive(self, taxonomy):
        assert all(v > 0 for v in taxonomy.facets()["venue"].values())


class TestRebuild:
    def test_building_twice_gives_the_same_shape(self, documents):
        first = build_tree(list(documents))
        second = build_tree(list(documents))
        assert [n.node_id for n in first.walk()] == \
            [n.node_id for n in second.walk()]
        assert first.root.doc_ids == second.root.doc_ids
