"""Final verification for week02. Run from the repository root:

    .venv\\Scripts\\python.exe -m pytest
    cd week02 ; python check_doc_lines.py ; python check_doc_claims.py ; python apptest_smoke.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import engine as ir
import check_doc_claims
import check_doc_lines

FAILURES = []


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark} {name}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def main():
    root = pathlib.Path(__file__).resolve().parent
    e = ir.get_engine()
    s = ir.corpus_stats()
    j = e.judgements

    print("== corpus ==")
    check("20 papers", s["papers"] == 20, str(s["papers"]))
    check("4 named fields everywhere",
          all(len(p.fields) == 4 for p in e.index.papers))
    check("3 region hierarchies",
          s["sections"] == 80 and s["paragraphs"] == 120 and s["sentences"] == 299,
          f"{s['sections']}/{s['paragraphs']}/{s['sentences']}")
    check("positional index built", s["positions"] == s["total_terms"])

    print("\n== invariants ==")
    from structured_index import check_invariants
    problems = check_invariants(e.index)
    check("check_invariants clean", problems == [], "; ".join(problems[:3]))

    print("\n== every mode answers ==")
    probes = {"proximal": "paragraph in methodology", "fields": "title:search",
              "overlapped": "document index positions",
              "non-overlapped": "document index positions"}
    for mode in ir.MODES:
        probe = probes.get(mode, "document index positions")
        r = ir.search(probe, mode=mode, top=3)
        check(f"mode {mode}", r.ok and r.count > 0, r.error or "0 hits")

    print("\n== error paths are messages, not crashes ==")
    check("empty query", not ir.search("").ok)
    check("unknown mode", not ir.search("x", mode="magic").ok)
    check("unknown field", not ir.search("keywords:x", mode="fields").ok)
    check("unknown term in a restriction",
          not ir.search("title:proximity AND results:precision",
                        mode="fields").ok)
    check("proximal syntax error is caught by search()",
          not ir.search("paragraph methodology", mode="proximal").ok)

    print("\n== the BIM = BM25(k1=0) identity ==")
    from ranking import BinaryIndependenceModel, Bm25Model
    bim = BinaryIndependenceModel(e.index)
    zero = Bm25Model(e.index, k1=0.0)
    worst = 0.0
    terms = ("retriev", "document", "term", "index", "field")
    for p in e.index.papers:
        worst = max(worst,
                    abs(bim.score_paper(p.doc_id, terms)[0]
                        - zero.score_paper(p.doc_id, terms)[0]))
    check("identical to floating point", worst == 0.0, str(worst))

    print("\n== evaluation is reproducible ==")
    from evaluate import evaluate_all
    first = evaluate_all(j, ir.evaluation_models())
    second = evaluate_all(j, ir.evaluation_models())
    check("two runs agree", [x.mrr for x in first] == [x.mrr for x in second])
    by_name = {x.model: x for x in first}
    check("BM25 beats both list models on MRR",
          by_name["BM25"].mrr > by_name["overlapped"].mrr
          and by_name["BM25"].mrr > by_name["non-overlapped"].mrr)
    check("qrels has 12 queries with positives and negatives",
          len(j.query_ids) == 12
          and all(j.relevant[q] for q in j.query_ids)
          and all(len(j.judged[q]) > len(j.relevant[q]) for q in j.query_ids))

    print("\n== documentation ==")
    code = check_doc_lines.main()
    check("doc line references", code == 0)
    code = check_doc_claims.main()
    check("doc numbers", code == 0)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())