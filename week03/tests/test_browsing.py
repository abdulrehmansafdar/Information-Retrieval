"""Step 3: the browsing engine and the search box.

The engine is the one entry point both front ends call, so these tests cover the
contract they depend on: categories list, breadcrumbs read, filters narrow, and
search answers a query with either results or a sentence explaining why not.
"""

from __future__ import annotations

import pytest

from browsing import Filters, get_engine


class TestTreeAccess:
    def test_documents_in_the_root_are_all_twenty(self, engine):
        assert len(engine.documents_in("root")) == 20

    def test_documents_in_are_sorted(self, engine):
        ids = [s.doc_id for s in engine.documents_in("root")]
        assert ids == sorted(ids)

    def test_a_summary_carries_its_path(self, engine):
        summary = engine.documents_in("root")[0]
        assert " / " in summary.path
        assert summary.category
        assert summary.title

    def test_children_of_the_root_are_the_domains(self, engine):
        assert [c.node_id for c in engine.children("root")] == \
            [d.node_id for d in engine.taxonomy.domains()]


class TestFilters:
    def test_a_year_filter_narrows_the_set(self, engine):
        everything = engine.documents_in("root")
        filtered = engine.documents_in("root", Filters(year=2021))
        assert 0 < len(filtered) < len(everything)
        assert all(s.year == 2021 for s in filtered)

    def test_a_domain_filter_narrows_the_set(self, engine):
        filtered = engine.documents_in("root", Filters(domain="root/ml"))
        assert filtered
        assert all(s.category in {"Neural Retrieval", "Representation & Graphs"}
                   for s in filtered)

    def test_a_subtopic_filter_is_the_tightest(self, engine):
        filtered = engine.documents_in("root", Filters(subtopic="root/ir/ranking"))
        assert [s.doc_id for s in filtered] == ["p03", "p16"]

    def test_filters_combine_as_an_intersection(self, engine):
        both = engine.documents_in("root", Filters(domain="root/ir", year=2021))
        assert all(s.year == 2021 for s in both)
        assert set(s.doc_id for s in both) <= \
            set(s.doc_id for s in engine.documents_in("root", Filters(domain="root/ir")))

    def test_an_empty_filter_does_not_change_anything(self, engine):
        assert engine.documents_in("root", Filters()) == \
            engine.documents_in("root")

    def test_facets_expose_nodes_for_the_ui(self, engine):
        facets = engine.filters_available()
        assert "root/ir" in facets["domain_nodes"].values()
        assert facets["subtopic_nodes"]["Neural Retrieval"] == "root/ml/neural"


class TestSearch:
    def test_a_title_word_finds_its_paper(self, engine):
        response = engine.search("segmentation")
        assert response.ok
        assert response.hits[0].doc_id == "p01"

    def test_a_proximity_query_ranks_p04_first(self, engine):
        response = engine.search("proximity window")
        assert response.hits[0].doc_id == "p04"

    def test_an_empty_query_explains_itself(self, engine):
        response = engine.search("   ")
        assert not response.ok
        assert "Type something" in response.error

    def test_a_stop_word_query_explains_itself(self, engine):
        response = engine.search("the of and")
        assert not response.ok
        assert "stop word" in response.error

    def test_an_unknown_word_is_reported(self, engine):
        response = engine.search("zzzznotaword")
        assert response.unknown == ["zzzznotaword"]

    def test_search_respects_the_filters(self, engine):
        loose = engine.search("search")
        tight = engine.search("search", Filters(year=2021))
        assert tight.considered < loose.considered
        assert all(h.doc_id in {s.doc_id for s in
                                engine.documents_in("root", Filters(year=2021))}
                   for h in tight.hits)

    def test_ranks_are_dense(self, engine):
        response = engine.search("retrieval")
        assert [h.rank for h in response.hits] == \
            list(range(1, response.count + 1))

    def test_top_limits_the_list(self, engine):
        response = engine.search("retrieval", top=2)
        assert response.count <= 2

    def test_the_hit_carries_its_category(self, engine):
        for hit in engine.search("proximity").hits:
            assert hit.category
            assert " / " in hit.path


class TestPreview:
    def test_the_preview_has_the_sections(self, engine):
        preview = engine.preview("p17")
        assert [name for name, _ in preview.sections] == \
            ["Abstract", "Methodology", "Results"]

    def test_the_preview_has_a_trail(self, engine):
        preview = engine.preview("p17")
        assert [n.kind for n in preview.trails] == ["root", "domain", "subtopic"]
        assert preview.path.startswith("Repository / ")

    def test_the_preview_has_related_links(self, engine):
        preview = engine.preview("p17")
        assert preview.related
        assert all(other != "p17" for other, _, _ in preview.related)

    def test_the_preview_lists_internal_links(self, engine):
        preview = engine.preview("p17")
        assert preview.internal == ["Abstract", "Methodology", "Results"]

    def test_an_unknown_document_raises(self, engine):
        with pytest.raises(KeyError):
            engine.preview("p99")


class TestPaths:
    def test_a_path_between_two_documents(self, engine):
        path = engine.path_between("p01", "p18")
        assert path[0] == "doc/p01" and path[-1] == "doc/p18"

    def test_describe_path_labels_every_hop(self, engine):
        hops = engine.describe_path(engine.path_between("p01", "p18"))
        assert hops
        for label, kind, target in hops:
            assert label and kind and target

    def test_a_category_to_a_document_path(self, engine):
        path = engine.path_between("root", "p04")
        assert path == ["root", "root/ir", "root/ir/structured", "doc/p04"]


class TestCaching:
    def test_the_engine_is_built_once(self):
        assert get_engine() is get_engine()
