"""Final verification for week03. Run from the repository root:

    .venv\\Scripts\\python.exe week03\\verify.py

or from inside the folder:

    python verify.py

Covers the corpus, the derived taxonomy, the hypergraph, the search contract, the
navigation state, and both documentation checkers - the same shape as week02's
verify.py.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import browsing as browse
import check_doc_claims
import check_doc_lines
from collection import corpus_stats
from hypertext import document_node
from taxonomy import FALLBACK, classify

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark} {name}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    engine = browse.get_engine()
    taxonomy = engine.taxonomy
    graph = engine.graph
    stats = corpus_stats()

    print("== collection ==")
    check("20 documents", stats["documents"] == 20, str(stats["documents"]))
    check("60 sections", stats["sections"] == 60, str(stats["sections"]))
    check("every document has three sections",
          all(d.section_names == ("Abstract", "Methodology", "Results")
              for d in engine.documents))
    check("every document has metadata",
          all(d.title and d.year and d.venue for d in engine.documents))

    print("\n== taxonomy ==")
    check("6 domains", len(taxonomy.domains()) == 6)
    check("11 subtopics", len(taxonomy.leaves()) == 11)
    check("no document falls through to the fallback",
          [d.doc_id for d in engine.documents if classify(d) == FALLBACK] == [])
    check("no category is empty",
          all(node.count > 0 for node in taxonomy.walk()))
    check("subtopics partition the collection",
          sorted(d for leaf in taxonomy.leaves() for d in leaf.doc_ids)
          == [f"p{i:02d}" for i in range(1, 21)])
    check("every breadcrumb is repository -> domain -> subtopic",
          all([n.kind for n in taxonomy.breadcrumb(leaf.node_id)]
              == ["root", "domain", "subtopic"] for leaf in taxonomy.leaves()))

    print("\n== the hypertext graph ==")
    graph_stats = graph.graph_stats()
    check("|N| = 98", graph_stats["nodes"] == 98, str(graph_stats["nodes"]))
    check("|L| = 261", graph_stats["links"] == 261, str(graph_stats["links"]))
    check("every node is reachable from the root",
          all(graph.shortest_path("root", node) for node in graph.nodes))
    check("every document has a related link",
          all(graph.related_docs(d.doc_id) for d in engine.documents))
    check("every document links to its sections",
          all({l.target for l in graph.out_links(document_node(d.doc_id),
                                                 "internal")}
              == {f"doc/{d.doc_id}/{s.lower()}" for s in d.section_names}
              for d in engine.documents))
    check("the hierarchy links are the category tree",
          all(any(l.kind == "hierarchy" and l.source == node.node_id
                  and l.target == child
                  for l in graph.out_links(node.node_id))
              for node in taxonomy.walk() for child in node.children))

    print("\n== search ==")
    response = engine.search("proximity window")
    check("a proximity query finds p04 first",
          response.ok and response.hits and response.hits[0].doc_id == "p04")
    check("a filter narrows the candidates",
          engine.search("search").considered
          > engine.search("search", browse.Filters(year=2021)).considered)
    check("an empty query is a message, not a crash", not engine.search("").ok)
    check("a stop-word query explains itself", not engine.search("the of").ok)

    print("\n== navigation ==")
    state = browse.NavigationState()
    state.visit("root")
    state.visit("root/ir")
    state.visit("doc/p04")
    check("history records the visits",
          state.history == ["root", "root/ir", "doc/p04"])
    check("back walks the stack", state.back() == "root/ir")
    check("forward walks it again", state.forward() == "doc/p04")
    state.back()
    state.visit("root/ml")
    check("a new visit truncates the future",
          state.history == ["root", "root/ir", "root/ml"])
    check("recent is most-recent-first and unique",
          state.recent() == ["root/ml", "root/ir", "root"])

    print("\n== documentation ==")
    check("doc line references", check_doc_lines.main() == 0)
    check("doc numbers", check_doc_claims.main() == 0)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
