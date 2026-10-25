"""Final verification for week06. Run from the repository root:

    .venv\\Scripts\\python.exe week06\\verify.py

or from inside the folder:

    python verify.py

Covers the shared corpus, the membership functions and partial matching, the query
language, the operators, the engine's three modes and the lecture's set identity,
and both documentation checkers. The same shape as week03's verify.py.
"""

from __future__ import annotations

import itertools
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import check_doc_claims
import check_doc_lines
from articles import CORPUS_DIR, corpus_stats, get_articles
from fuzzy_logic import conjunction, disjunction, evaluate
from fuzzy_search import MODES, Settings, get_engine
from membership import FUNCTIONS
from query_parser import QueryError, Term, parse_query, to_text

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark} {name}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    engine = get_engine()
    index = engine.index
    stats = corpus_stats()

    print("== shared corpus ==")
    check("30 articles", stats["articles"] == 30, str(stats["articles"]))
    check("120 sections (four per article)", stats["sections"] == 120,
          str(stats["sections"]))
    check("10 specialties", stats["specialties"] == 10, str(stats["specialties"]))
    check("corpus is at the repository root", CORPUS_DIR.is_dir())
    check("ids run m01 to m30",
          [a.doc_id for a in get_articles()] == [f"m{i:02d}" for i in range(1, 31)])

    print("\n== membership functions ==")
    values = [index.membership(s, d, f) for s in index.vocabulary[:300]
              for d in index.doc_ids for f in FUNCTIONS]
    check("every membership is between 0 and 1", all(0 <= v <= 1 for v in values))
    check("binary is 1 exactly where the word occurs",
          all(index.membership(s, d, "binary") == (1.0 if index.weight(s, d) else 0.0)
              for s in index.vocabulary[:300] for d in index.doc_ids))
    check("the strongest article scores 1 under log",
          max(index.membership("hypertens", d, "log") for d in index.doc_ids) == 1.0)
    check("log is never below linear",
          all(index.membership(s, d, "log") >= index.membership(s, d, "linear") - 1e-12
              for s in index.vocabulary[:300] for d in index.doc_ids))

    print("\n== partial matching ==")
    typo = engine.search("hypertenion", Settings(mode="fuzzy"))
    check("a typo finds hypertension in fuzzy mode",
          typo.hits and typo.hits[0].doc_id == "m01", str(typo.count))
    check("the same typo finds nothing in boolean mode",
          engine.search("hypertenion", Settings(mode="boolean")).count == 0)

    print("\n== query language ==")
    check("AND binds tighter than OR",
          to_text(parse_query("asthma OR COPD AND smoking")) == "asthma OR (COPD AND smoking)")
    check("BUT NOT means AND NOT",
          to_text(parse_query("asthma BUT NOT smoking")) == "asthma AND NOT smoking")
    try:
        parse_query("hypertension AND")
        check("a dangling operator is refused", False)
    except QueryError:
        check("a dangling operator is refused", True)

    print("\n== operators (lecture numbers) ==")
    check("fuzzy AND of 0.8 and 0.6 is 0.6", math.isclose(conjunction([0.8, 0.6]), 0.6))
    check("fuzzy AND of 0.7 and 0.4 is 0.4", math.isclose(conjunction([0.7, 0.4]), 0.4))
    check("p = inf gives exactly min and max",
          math.isclose(conjunction([0.8, 0.6], math.inf), 0.6)
          and math.isclose(disjunction([0.8, 0.6], math.inf), 0.8))
    check("p = 1 gives the plain mean", math.isclose(disjunction([0.8, 0.6], 1.0), 0.7))

    binary_ok = True
    tree = parse_query("(science OR technology) AND NOT art")
    for a, b, c in itertools.product((0.0, 1.0), repeat=3):
        values_ = {"science": a, "technology": b, "art": c}
        got = evaluate(tree, lambda t: (values_[t.word], "")).value
        binary_ok &= got == (1.0 if (a or b) and not c else 0.0)
    check("with 0/1 values, fuzzy operators are set algebra", binary_ok)

    print("\n== engine ==")
    scores_ok = True
    modes_ok = True
    for mode in MODES:
        for query in ("hypertension treatment", "hypertension AND NOT diabetes",
                      "heart failure BUT NOT (asthma OR -fever)", "covid-19"):
            response = engine.search(query, Settings(mode=mode))
            scores_ok &= all(0.0 <= h.score <= 1.0 for h in response.hits)
            keys = [(-h.score, h.doc_id) for h in response.hits]
            modes_ok &= keys == sorted(keys)
    check("every score is between 0 and 1 in every mode", scores_ok)
    check("results are ranked best first in every mode", modes_ok)

    identity = all(engine.set_view(a, b, c)["equal"] for a, b, c in
                   (("diabetes", "hypertension", "kidney"),
                    ("asthma", "COPD", "smoking"),
                    ("fever", "malaria", "liver")))
    check("(A OR B) AND NOT C equals (A u B) - C on the corpus", identity)
    extended = {h.doc_id for h in engine.search("hypertension AND NOT diabetes",
                                                Settings(mode="extended")).hits}
    check("extended AND NOT keeps hypertension and drops diabetes", extended == {"m01"})
    check("compare returns all three modes",
          set(engine.compare("asthma OR COPD")) == set(MODES))

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
