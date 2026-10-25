"""Verify the NUMBERS quoted in the week06 docs.

check_doc_lines.py proves every `file.py:NN` reference points at the right code.
This proves two more things: that every figure the docs quote is still what the
code computes, and that the docs actually contain it. The second matters because
a page can keep the right function and quietly lose the number.

    python check_doc_claims.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from articles import corpus_stats, tokenize
from fuzzy_logic import conjunction, disjunction
from fuzzy_search import Settings, get_engine

DOCS = ROOT / "docs" / "understanding"
DOC_TEXT = "\n".join(p.read_text(encoding="utf-8") for p in sorted(DOCS.glob("*.md")))

FAILURES: list[str] = []


def is_(label: str, actual, expected, needle: str | None = None) -> None:
    """`needle` is the exact text the docs must contain, when they quote it."""
    if isinstance(expected, float):
        ok = math.isclose(actual, expected, abs_tol=0.005)
    else:
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
    engine = get_engine()
    index = engine.index

    print("== corpus (fuzzy-model.md, membership-functions.md) ==")
    is_("articles", stats["articles"], 30, "30 articles")
    is_("sections", stats["sections"], 120, "120 sections")
    is_("specialties", stats["specialties"], 10, "10 specialties")

    print("\n== membership (membership-functions.md) ==")
    is_("m02 hypertension, log", index.membership("hypertens", "m02", "log"), 0.61,
        "0.61")
    is_("m02 hypertension, linear", index.membership("hypertens", "m02", "linear"), 0.40,
        "0.40")
    is_("m03 hypertension, log", index.membership("hypertens", "m03", "log"), 0.39,
        "0.39")
    is_("m01 title weight of hypertens", index.weight("hypertens", "m01") >= 3, True)
    is_("strongest treatment weight", max(index.weights["treatment"].values()), 3)

    print("\n== the worked fuzzy example (fuzzy-model.md) ==")
    worked = engine.search("hypertension treatment", Settings(mode="fuzzy"))
    is_("hypertension treatment, count", worked.count, 4, "returns **4 articles of 30**")
    is_("hypertension treatment, top score", worked.hits[0].score, 0.50, "0.50")
    is_("hypertension treatment, top article", worked.hits[0].doc_id, "m01")
    is_("min(0.8, 0.6)", conjunction([0.8, 0.6]), 0.6, "min(0.8, 0.6) = 0.6")
    is_("min(0.7, 0.4)", conjunction([0.7, 0.4]), 0.4)

    print("\n== partial matching (partial-matching.md) ==")
    typo = dict(engine.partial_matches("hypertenion"))
    is_("hypertenion -> hypertens", typo["hypertens"], 0.70, "0.70")
    is_("hypertenion at tau 0.95", engine.partial_matches("hypertenion", 0.95), [],
        "no match at all")
    is_("hypertenion fuzzy hits", engine.search("hypertenion", Settings(mode="fuzzy")).count,
        7)
    is_("hypertenion boolean hits",
        engine.search("hypertenion", Settings(mode="boolean")).count, 0)
    anemia = dict(engine.partial_matches("anemia"))
    is_("anemia -> anaemia", anemia["anaemia"], 0.62, "0.62")
    diabetis = dict(engine.partial_matches("diabetis"))
    is_("diabetis -> diabet", diabetis["diabet"], 0.77, "0.77")
    is_("failure is stemmed to failur", tokenize("failure"), ["failur"], "failur")
    is_("fails is stemmed to fail", tokenize("fails"), ["fail"], "fails")

    print("\n== p-norm (extended-boolean.md) ==")
    is_("p=2 AND of 1.00 and 0.50", conjunction([1.0, 0.5], 2.0), 0.65, "0.65")
    is_("p=2 OR of 0.8 and 0.6", disjunction([0.8, 0.6], 2.0), 0.71, "0.71")
    is_("p=2 AND of 0.8 and 0.6", conjunction([0.8, 0.6], 2.0), 0.68, "0.68")
    extended = engine.search("hypertension treatment", Settings(mode="extended"))
    is_("extended hypertension treatment, top", round(extended.hits[0].score, 2), 0.65,
        "0.65")
    asthma = engine.search("asthma OR COPD", Settings(mode="extended"))
    is_("extended asthma OR COPD, top", round(asthma.hits[0].score, 2), 0.71)
    exclude = engine.search("hypertension AND NOT diabetes", Settings(mode="extended"))
    is_("extended hypertension AND NOT diabetes", [h.doc_id for h in exclude.hits], ["m01"],
        "m01 only")

    print("\n== the set identity (extended-boolean.md) ==")
    view = engine.set_view("diabetes", "hypertension", "kidney")
    is_("|diabetes|", len(view["A"]), 11, "diabetes: 11 articles")
    is_("|hypertension|", len(view["B"]), 7, "hypertension: 7 articles")
    is_("|A u B|", len(view["union"]), 12)
    is_("(A u B) - C", sorted(view["result"]), ["m02", "m08", "m17", "m18", "m22"],
        "m02, m08, m17, m18, m22")
    is_("the query gives the same set", view["equal"], True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} number(s) out of date: {', '.join(FAILURES)}")
        return 1
    print("every quoted number matches the code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
