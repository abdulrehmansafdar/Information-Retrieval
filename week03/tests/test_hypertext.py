"""Step 2: H = (N, L).

The graph is the whole point of the hypertext model, so what is asserted here is
its structure: the counts add up, every node is reachable from the repository,
the hierarchy links ARE the category tree, and the related links are the answer
the browser will show.
"""

from __future__ import annotations

import pytest

from hypertext import (LINK_KINDS, RELATED_TOP_K, build_graph, document_node,
                       related_scores, section_node)


class TestShape:
    def test_the_node_counts_add_up(self, graph):
        stats = graph.graph_stats()
        assert stats["nodes"] == (stats["repository"] + stats["categories"]
                                  + stats["documents"] + stats["sections"])

    def test_one_node_per_document_and_three_per_document_for_sections(self, graph):
        stats = graph.graph_stats()
        assert stats["documents"] == 20
        assert stats["sections"] == 60

    def test_the_repository_and_categories_are_the_taxonomy(self, graph,
                                                            taxonomy):
        assert graph.nodes["root"].kind == "repository"
        categories = {n for n in graph.nodes if graph.nodes[n].kind == "category"}
        assert categories == {n.node_id for n in taxonomy.walk()
                              if n.kind != "root"}

    def test_every_link_kind_is_used(self, graph):
        used = set(graph.graph_stats()["by_kind"]) 
        assert used == set(LINK_KINDS)

    def test_links_are_sorted_and_stable(self, documents):
        first = build_graph(list(documents))
        second = build_graph(list(documents))
        assert first.links == second.links


class TestHierarchy:
    def test_the_hierarchy_links_are_the_category_tree(self, graph, taxonomy):
        for node in taxonomy.walk():
            for child in node.children:
                assert any(l.kind == "hierarchy" and l.source == node.node_id
                           and l.target == child
                           for l in graph.out_links(node.node_id))

    def test_every_document_hangs_off_a_subtopic(self, graph, taxonomy,
                                                 documents):
        for document in documents:
            parents = [l.source for l in graph.in_links(document_node(document.doc_id))
                       if l.kind == "hierarchy"]
            assert parents == [taxonomy.leaf_of(document.doc_id)]

    def test_every_node_is_reachable_from_the_root(self, graph):
        seen = {graph.shortest_path("root", node)[-1] for node in graph.nodes}
        assert seen == set(graph.nodes)


class TestInternalLinks:
    def test_each_document_links_to_its_sections(self, graph, documents):
        for document in documents:
            targets = {l.target for l in graph.out_links(document_node(document.doc_id),
                                                         "internal")}
            expected = {section_node(document.doc_id, name)
                        for name in document.section_names}
            assert targets == expected

    def test_sections_link_to_the_next_one_in_reading_order(self, graph,
                                                             documents):
        document = documents[0]
        names = document.section_names
        assert any(l.target == section_node(document.doc_id, names[1])
                   and l.kind == "internal"
                   for l in graph.out_links(section_node(document.doc_id,
                                                         names[0])))


class TestRelatedLinks:
    def test_every_document_has_related_documents(self, graph, documents):
        for document in documents:
            assert graph.related(document.doc_id), document.doc_id

    def test_at_most_top_k_each(self, graph, documents):
        for document in documents:
            assert len(graph.related(document.doc_id)) <= RELATED_TOP_K

    def test_scores_are_positive_and_descending(self, graph, documents):
        for document in documents:
            scores = [l.weight for l in graph.related(document.doc_id)]
            assert all(s > 0.0 for s in scores)
            assert scores == sorted(scores, reverse=True)

    def test_the_two_faceted_search_papers_relate_to_each_other(self, graph):
        """p15 (faceted search interfaces) and p20 (user interfaces for ranked
        lists) are the two HCI papers; a content model that did not link them
        would be a content model not reading the abstracts.
        """
        neighbours = {d for d, _ in graph.related_docs("p15")}
        assert "p20" in neighbours

    def test_ties_break_on_doc_id(self, documents):
        scores = related_scores(list(documents))
        for doc_id, pairs in scores.items():
            ordered = sorted(pairs, key=lambda p: (-p[1], p[0]))
            assert pairs == ordered, doc_id

    def test_a_document_is_never_related_to_itself(self, graph, documents):
        for document in documents:
            assert document.doc_id not in {d for d, _ in
                                           graph.related_docs(document.doc_id)}


class TestSharedAttributes:
    def test_shared_year_links_the_year_group(self, graph, documents):
        by_year: dict[int, list[str]] = {}
        for document in documents:
            by_year.setdefault(document.year, []).append(document.doc_id)
        for year, members in by_year.items():
            if len(members) < 2:
                continue
            a = document_node(sorted(members)[0])
            targets = {l.target for l in graph.out_links(a, "shared-year")}
            assert {document_node(m) for m in members[1:]} <= targets

    def test_shared_venue_links_are_symmetric(self, graph):
        for link in graph.out_links(document_node("p03")):  # any doc
            if link.kind == "shared-venue":
                assert any(l.target == link.source
                           for l in graph.out_links(link.target, "shared-venue"))


class TestPaths:
    def test_a_node_is_its_own_path(self, graph):
        assert graph.shortest_path("root", "root") == ["root"]

    def test_a_path_is_a_chain_of_real_neighbours(self, graph):
        path = graph.shortest_path(document_node("p01"), document_node("p18"))
        assert path[0] == document_node("p01")
        assert path[-1] == document_node("p18")
        for source, target in zip(path, path[1:]):
            assert target in graph.neighbors(source)

    def test_the_path_is_the_shortest_one(self, graph):
        path = graph.shortest_path("root", document_node("p04"))
        # root -> domain -> subtopic -> document
        assert path == ["root", "root/ir", "root/ir/structured", "doc/p04"]

    def test_an_unknown_node_in_a_path_raises(self, graph):
        with pytest.raises(KeyError):
            graph.shortest_path("root", "nope")

    def test_degree_counts_both_directions(self, graph, documents):
        node = document_node(documents[0].doc_id)
        outgoing = len(graph.out_links(node))
        incoming = len(graph.in_links(node))
        assert graph.degree(node) > 0
        assert graph.degree(node) >= outgoing
        assert incoming >= 1  # at least the hierarchy link from its subtopic


class TestOutInLinks:
    def test_in_links_are_the_reverse_of_out_links(self, graph):
        for link in graph.out_links("root"):
            assert link in graph.in_links(link.target)

    def test_filtering_by_kind(self, graph):
        document = document_node("p04")
        assert all(l.kind == "related"
                   for l in graph.out_links(document, "related"))
