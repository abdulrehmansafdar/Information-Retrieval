"""Verify the NUMBERS quoted in the week07 docs.

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

from clinical_docs import tokenize
from lsi_model import SvdFactors
from semantic_eval import evaluate
from semantic_search import Settings, get_engine
from slide_examples import fruits_and_vegetables, slide_matrix, space_exploration
from vector_cosine import cosine_similarity

DOCS = ROOT / "docs" / "understanding"
DOC_TEXT = "\n".join(p.read_text(encoding="utf-8") for p in sorted(DOCS.glob("*.md")))

FAILURES: list[str] = []


def is_(label: str, actual, expected, needle: str | None = None) -> None:
    """`needle` is the exact text the docs must contain, when they quote it."""
    if isinstance(expected, float):
        ok = math.isclose(actual, expected, abs_tol=0.0005)
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


def map_at(k: int) -> float:
    return evaluate(Settings("lsi", k=k)).mean_average_precision


def main() -> int:
    engine = get_engine()

    print("== the slide's cosine (gvsm-and-cosine.md) ==")
    is_("cosine of (1,1) and (5,3)", round(cosine_similarity((1, 1), (5, 3)), 2), 0.97, "0.97")
    steps = fruits_and_vegetables().steps(tokenize("fruits vegetables"), "D1")
    is_("dot product", steps.dot, 8.0, "(1×5) + (1×3) = **8**")
    is_("denominator", steps.denominator, 8.246, "8.246")

    print("\n== the matrix and the SVD (lsi-and-svd.md) ==")
    is_("slide A", slide_matrix().values.tolist(), [[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    is_("slide A singular values",
        [round(float(s), 6) for s in SvdFactors(slide_matrix()).sigma], [2.0, 1.0, 1.0],
        "**2, 1, 1**")
    is_("corpus A", engine.matrix().shape, (1128, 30), "1,128 terms by 30 articles")
    is_("energy kept at k = 15", engine.lsi().energy_kept(), 0.673, "67.3%")

    print("\n== choosing k (lsi-and-svd.md) ==")
    gvsm = evaluate(Settings("gvsm"))
    lsi = evaluate(Settings("lsi"))
    is_("GVSM MAP", gvsm.mean_average_precision, 0.679, "0.679")
    is_("k = 2", map_at(2), 0.369, "0.369")
    is_("k = 5", map_at(5), 0.646, "0.646")
    is_("k = 12", map_at(12), 0.781, "0.781")
    is_("k = 15", lsi.mean_average_precision, 0.758, "0.758")
    is_("k = 25", map_at(25), 0.740, "0.740")
    is_("k = 30 equals GVSM", map_at(30), gvsm.mean_average_precision)
    is_("every k from 14 to 24 beats GVSM",
        all(map_at(k) > gvsm.mean_average_precision for k in range(14, 25)), True,
        "k = 14 to 24 all beat GVSM")

    print("\n== the slide's space example (semantic-matching.md) ==")
    space = space_exploration()
    is_("D1 GVSM", space["D1"]["gvsm"], 0.816, "0.82")
    is_("D2 GVSM", space["D2"]["gvsm"], 0.354, "0.35")
    for doc in ("D1", "D2", "D3", "D4"):
        is_(f"{doc} LSI", space[doc]["lsi"], 1.0)
    for doc in ("D3", "D4", "D5", "D6"):
        is_(f"{doc} GVSM", space[doc]["gvsm"], 0.0)
    for doc in ("D5", "D6"):
        is_(f"{doc} LSI", space[doc]["lsi"], 0.0)

    print("\n== wheezing (semantic-matching.md) ==")
    results = engine.compare("wheezing", Settings(top=30))
    plain, deep = results["gvsm"], results["lsi"]
    scores = {h.doc_id: h for h in deep.hits}
    is_("GVSM finds only asthma", [h.doc_id for h in plain.hits], ["m09"])
    is_("GVSM asthma", plain.hits[0].score, 0.212, "0.212")
    is_("LSI asthma", scores["m09"].score, 0.986, "0.986")
    is_("LSI COPD", scores["m10"].score, 0.874, "0.874")
    is_("COPD shares no word", scores["m10"].shared_terms, ())
    is_("LSI migraine", scores["m19"].score, 0.369, "0.369")
    is_("LSI coronary artery disease", scores["m02"].score, 0.339, "0.339")
    is_("LSI reaches 13 articles", deep.count, 13, "LSI 13")

    print("\n== the measurement (semantic-matching.md) ==")
    is_("GVSM recall", gvsm.mean_recall_at_r, 0.688, "0.688")
    is_("LSI recall", lsi.mean_recall_at_r, 0.724, "0.724")
    is_("no-word hits, GVSM", gvsm.no_word_hits, 0)
    is_("no-word hits, LSI", lsi.no_word_hits, 3)
    pairs = list(zip(gvsm.per_query, lsi.per_query))
    better = sum(b.average_precision > a.average_precision + 0.005 for a, b in pairs)
    worse = sum(b.average_precision < a.average_precision - 0.005 for a, b in pairs)
    is_("queries improved", better, 10, "10 improve")
    is_("queries worse", worse, 0, "0 get worse")
    mosquito = next((a, b) for a, b in pairs if a.query == "mosquito bite")
    is_("mosquito bite, GVSM", round(mosquito[0].average_precision, 2), 0.40, "0.40")
    is_("mosquito bite, LSI", round(mosquito[1].average_precision, 2), 0.93, "0.93")
    is_("mosquito bite, no-word hits", mosquito[1].no_word_hits, ("m15", "m17"))
    is_("sadness has no row in A", "sad" in engine.matrix().row_of, False)
    is_("sadness stems to sad", tokenize("sadness"), ["sad"], "`sad`")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} number(s) out of date: {', '.join(FAILURES)}")
        return 1
    print("every quoted number matches the code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
