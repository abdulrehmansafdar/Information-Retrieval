"""Verify the NUMBERS quoted in the week03 docs.

check_doc_lines.py proves every `file.py:NN` reference points at the right code.
This proves two more things: that every figure the docs quote is still what the
code computes, and that the docs actually contain it. The second matters because
a page can keep the right function and quietly lose the number.

    python check_doc_claims.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import browsing as browse
from collection import corpus_stats
from taxonomy import get_taxonomy

DOCS = ROOT / "docs" / "understanding"
DOC_TEXT = "\n".join(p.read_text(encoding="utf-8")
                     for p in sorted(DOCS.glob("*.md")))

FAILURES: list[str] = []


def is_(label: str, actual, expected, needle: str | None = None) -> None:
    """`needle` is the exact text the docs must contain, when they quote it."""
    ok = actual == expected
    detail = ""
    if ok and needle is not None and needle not in DOC_TEXT:
        ok = False
        detail = f"  (the docs no longer state {needle!r})"
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {actual}"
          + ("" if ok else f"  (expected {expected}){detail}"))
    if not ok:
        FAILURES.append(label)


def main() -> int:
    stats = corpus_stats()
    taxonomy = get_taxonomy()
    engine = browse.get_engine()
    graph = engine.graph
    gstats = graph.graph_stats()

    print("== collection (browsing-models.md) ==")
    is_("documents", stats["documents"], 20, "twenty documents")
    is_("sections", stats["sections"], 60)
    is_("section names", ",".join(stats["section_names"]),
        "Abstract,Methodology,Results")
    is_("words", stats["words"], 5911)
    is_("distinct terms", stats["vocabulary"], 1283)

    print("\n== taxonomy (taxonomy.md) ==")
    is_("domains", len(taxonomy.domains()), 6, "**6** domains")
    is_("subtopics", len(taxonomy.leaves()), 11, "**11** subtopics")
    totals = {d.name: d.count for d in taxonomy.domains()}
    is_("Information Retrieval", totals["Information Retrieval"], 10, "**10**")
    is_("Machine Learning & NLP", totals["Machine Learning & NLP"], 4, "**4**")
    is_("Applied AI", totals["Applied AI"], 2)
    is_("Human-Computer Interaction", totals["Human-Computer Interaction"], 2)
    is_("Systems & Scalability", totals["Systems & Scalability"], 1)
    is_("Evaluation & Measurement", totals["Evaluation & Measurement"], 1)
    is_("subtopics partition the collection",
        sorted(d for leaf in taxonomy.leaves() for d in leaf.doc_ids),
        [f"p{i:02d}" for i in range(1, 21)])
    is_("p04 breadcrumb", " ".join(
        n.name for n in taxonomy.breadcrumb(taxonomy.leaf_of("p04"))),
        "Repository Information Retrieval Structured & Proximity Search",
        "Repository / Information Retrieval / Structured & Proximity Search")
    is_("the fallback is empty",
        [d.doc_id for d in engine.documents
         if taxonomy.category_of(d.doc_id).name == "Uncategorised"], [])

    print("\n== the hypertext graph (navigation.md) ==")
    is_("|N| nodes", gstats["nodes"], 98, "**98 nodes**")
    is_("|L| links", gstats["links"], 261, "**261 links**")
    is_("categories", gstats["categories"], 17, "17")
    is_("documents", gstats["documents"], 20)
    is_("sections", gstats["sections"], 60)
    is_("hierarchy links", gstats["by_kind"]["hierarchy"], 37, "| 37 |")
    is_("internal links", gstats["by_kind"]["internal"], 100, "| 100 |")
    is_("related links", gstats["by_kind"]["related"], 60, "| 60 |")
    is_("shared-venue links", gstats["by_kind"]["shared-venue"], 6, "| 6 |")
    is_("shared-year links", gstats["by_kind"]["shared-year"], 58, "| 58 |")
    is_("every document has a related link",
        all(graph.related_docs(d.doc_id) for d in engine.documents), True)
    is_("related is three per document",
        {len(graph.related_docs(d.doc_id)) for d in engine.documents}, {3},
        "20 × 3 = 60")

    print("\n== the worked path (navigation.md) ==")
    path = engine.path_between("p01", "p18")
    is_("p01 -> p18 hops", len(path) - 1, 2, "Two moves")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print("every number in the docs is still true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
