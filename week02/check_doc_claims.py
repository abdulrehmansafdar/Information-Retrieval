"""Verify the NUMBERS quoted in the week02 docs.

check_doc_lines.py proves every `file.py:NN` reference still points at the right
code. This proves the arithmetic the docs quote is still true.

Both were needed: the docs were wrong twice for different reasons - once because
a line moved, and once because a measurement in them was never taken. A page of
confident numbers that nobody re-derives is the worst thing in a graded report,
so these claims are executable.

    python check_doc_claims.py
"""
from __future__ import annotations

import sys

import engine as ir
from evaluate import evaluate_all
from fields import DEFAULT_WEIGHTS, FieldSearch
from lists import Match, ListModels, collapse
from proximity import ProximalNodeModel, resolve_constructor
from ranking import BinaryIndependenceModel, Bm25Model, IdfTable, parse_query

FAILURES: list[str] = []


def is_(label, actual, expected, tolerance=1e-6):
    ok = (abs(actual - expected) <= tolerance
          if isinstance(expected, float)
          else actual == expected)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {actual}"
          + ("" if ok else f"  (docs say {expected})"))
    if not ok:
        FAILURES.append(label)


def main() -> int:
    engine = ir.get_engine()
    index = engine.index
    table = engine.idf_table
    stats = ir.corpus_stats()

    print("== corpus (formulas.md, structured-index.py) ==")
    is_("papers", stats["papers"], 20)
    is_("vocabulary", stats["vocabulary"], 966)
    is_("postings", stats["postings"], 2134)
    is_("positions", stats["positions"], 3434)
    is_("length min", stats["length_min"], 144)
    is_("length max", stats["length_max"], 197)
    is_("avg length", stats["avg_length"], 171.7, 0.05)
    is_("sections", stats["sections"], 80)
    is_("paragraphs", stats["paragraphs"], 120)
    is_("sentences", stats["sentences"], 299)

    print("\n== IDF table (binary-independence-model.md) ==")
    is_("idf(retriev)", round(table.idf_for("retriev"), 4), 0.7932)
    is_("idf(field)", round(table.idf_for("field"), 4), 2.1282)
    is_("idf(search)", round(table.idf_for("search"), 4), 1.1727)
    is_("idf(document)", round(table.idf_for("document"), 4), 0.5188)
    robertson = IdfTable(index, "robertson")
    negatives = sum(1 for v in robertson.idf.values() if v < 0)
    is_("negative robertson idfs", negatives, 17)
    is_("idf(quer) robertson", round(robertson.idf_for("queri"), 4), -0.5878)
    is_("idf(document) robertson", round(robertson.idf_for("document"), 4), -0.3857)

    print("\n== BIM vs BM25 on one term (binary-independence-model.md) ==")
    bim = BinaryIndependenceModel(index)
    bm25 = Bm25Model(index)
    is_("tf(field, p19)", index.term_frequency_in("p19", "field"), 15)
    is_("BIM term score", round(bim.term_score("field", 15), 4), 2.1282)
    is_("BIM ignores repetition",
        bim.term_score("field", 1) == bim.term_score("field", 15), True)
    is_("BM25 sat(tf=1)", round(bm25.saturation(1, 161), 4), 1.0262)
    is_("BM25 sat(tf=15)", round(bm25.saturation(15, 161), 4), 2.0441)
    is_("BM25 repetition gain",
        round(bm25.term_score("field", 15, 161) / bm25.term_score("field", 1, 161), 2), 1.99)

    print("\n== worked example, q12 p19 (formulas.md, bm25.md) ==")
    parsed = parse_query("structured field search", table)
    is_("terms", list(parsed.terms), ["structur field", "search"])
    score, rows = bm25.score_paper("p19", parsed.terms)
    is_("K(p19)", round(bm25.length_norm(161), 4), 0.9533)
    is_("structur field contribution", round(rows[0].contribution, 4), 2.7081)
    is_("search contribution", round(rows[1].contribution, 4), 1.6413)
    is_("score", round(score, 4), 4.3494)

    print("\n== the k1 = 0 identity (formulas.md, bm25.md) ==")
    zero = bm25.with_parameters(k1=0.0)
    worst = max(abs(zero.saturation(tf, p.length) - 1.0)
                for p in index.papers for tf in range(1, 40))
    is_("max |sat - 1| at k1=0", worst, 0.0)
    diff = max(abs(bim.score_paper(p.doc_id, ("retriev", "document", "term"))[0]
                   - zero.score_paper(p.doc_id, ("retriev", "document", "term"))[0])
               for p in index.papers)
    is_("max |BIM - BM25(0)|", diff, 0.0)

    print("\n== field statistics (structured-fields.md) ==")
    fields = engine.fields
    is_("avg title length", round(fields.stats.average_length["title"], 2), 3.45)
    is_("avg abstract length", round(fields.stats.average_length["abstract"], 2), 34.85)
    is_("avg methodology length", round(fields.stats.average_length["methodology"], 2), 66.20)
    is_("avg results length", round(fields.stats.average_length["results"], 2), 67.20)
    is_("title/results ratio", round(fields.stats.average_length["results"]
                                     / fields.stats.average_length["title"], 0), 19)
    is_("weights", list(DEFAULT_WEIGHTS.values()), [3.0, 2.0, 1.5, 1.0])
    is_("df_title(search)", fields.stats.document_frequency["title"].get("search"), 5)
    is_("idf_title(search)", round(fields.stats.idf_for("title", "search"), 4), 1.3398)
    is_("tf(search) in p19 title",
        fields._terms_in("p19", "title", ("search",)), {"search": 1})

    print("\n== collapse (lists.py, overlapped-...-lists.md) ==")
    is_("collapse demo", collapse([Match("index", 10), Match("index", 22),
                                   Match("phrase", 12), Match("phrase", 23),
                                   Match("query", 14)]),
        [(10, 11), (12, 13), (14, 15), (22, 24)])

    print("\n== the two list models (overlapped-...-lists.md) ==")
    lists = ListModels(index)
    over = lists.overlapped("document index positions", "paragraph")
    non = lists.non_overlapped("document index positions", "paragraph")
    is_("overlapped regions", over.count, 1)
    is_("overlapped region id", over.regions[0].region_id, "p06:results:p2")
    is_("non-overlapped regions", non.count, 4)
    is_("non-overlapped adds p11", sorted(set(non.papers) - set(over.papers)),
        ["p11"])
    is_("strict papers are a subset", set(over.papers) <= set(non.papers), True)
    strict_none = lists.overlapped("ranking web search", "paragraph")
    is_("'ranking web search' strict is empty", strict_none.count, 0)
    is_("'ranking web search' loose papers",
        len(lists.non_overlapped("ranking web search", "paragraph").papers), 2)

    print("\n== bottom-up cost (proximal-nodes.md) ==")
    proximal = ProximalNodeModel(index)
    occurrences = sum(len(p.positions) for p in index.lookup("retriev"))
    sections = len(proximal.nodes(resolve_constructor("section")))
    is_("occurrences of 'retriev'", occurrences, 28)
    is_("sections", sections, 80)
    is_("top-down tests", occurrences * sections, 2240)
    is_("speed-up", sections, 80)
    result = proximal.evaluate("section with(2) retrieval")
    is_("nodes found", result.count, 5)
    is_("papers found", len(result.papers), 3)

    print("\n== distance dial (proximal-nodes.md) ==")
    for window, expected in ((0, 2), (2, 3), (5, 6), (10, 9), (25, 17), (60, 33)):
        got = proximal.evaluate(
            f"sentence before({window}) index (paragraph)").count
        is_(f"before({window})", got, expected)

    print("\n== evaluation (evaluation.md, formulas.md) ==")
    scores = {s.model: s for s in evaluate_all(engine.judgements,
                                                ir.evaluation_models())}
    expected = {
        "boolean AND": (0.833, 0.833, 0.667, 0.833, 0.667, 2),
        "tfidf cosine": (0.917, 0.617, 0.958, 0.958, 0.883, 0),
        "BIM": (0.833, 0.633, 0.958, 0.917, 0.875, 0),
        "BM25": (1.000, 0.617, 0.958, 1.000, 0.925, 0),
        "BM25F": (0.917, 0.617, 0.958, 0.958, 0.904, 0),
        "overlapped": (0.750, 0.750, 0.542, 0.750, 0.542, 3),
        "non-overlapped": (0.833, 0.833, 0.667, 0.833, 0.667, 2),
        "proximal": (0.750, 0.683, 0.708, 0.764, 0.609, 2),
    }
    for model, want in expected.items():
        s = scores[model]
        got = (round(s.mean("precision", 1), 3), round(s.mean("precision", 5), 3),
               round(s.mean("recall", 10), 3), round(s.mrr, 3), round(s.map, 3),
               len(s.queries_with_no_hit))
        is_(model, got, want)

    print("\n== the two decisive queries (evaluation.md) ==")
    is_("reciproc df", index.document_frequency.get("reciproc"), 1)
    is_("reciproc only in", sorted(index.documents_containing("reciproc")), ["p12"])
    is_("mean reciproc rank df",
        index.document_frequency["mean reciproc rank"], 5)
    bim_score = bim.score_paper("p15", ("metadata", "structur field", "search"))[0]
    is_("BIM ties p15/p19 exactly",
        bim_score == bim.score_paper("p19",
                                     ("metadata", "structur field", "search"))[0],
        True)
    is_("the tie's value", round(bim_score, 4), 3.8118)
    is_("K(p15)", round(bm25.length_norm(167), 4), 0.9795)
    is_("K(p19)", round(bm25.length_norm(161), 4), 0.9533)
    is_("BM25 p15", round(bm25.score_paper(
        "p15", ("metadata", "structur field", "search"))[0], 4), 4.2939)
    is_("BM25 p19", round(bm25.score_paper(
        "p19", ("metadata", "structur field", "search"))[0], 4), 4.3494)

    print("\n== phrase list (preprocess.py note above PHRASES) ==")
    from preprocess import PHRASES, preprocess_with_positions
    dead = [p for p in PHRASES
            if index.document_frequency.get(
                " ".join(preprocess_with_positions(p, drop_stop_words=False)[0]), 0) == 0]
    is_("dead phrases", dead, [])
    is_("'machine learning' is two terms now",
        preprocess_with_positions("machine learning")[0], ["machin", "learn"])

    print("\n== field query syntax (structured-fields.md) ==")
    fq = engine.fields
    grouped = fq.restrict("(title:search OR abstract:rank) AND abstract:result")
    flat = fq.restrict("title:search AND abstract:result "
                       "OR abstract:rank AND abstract:result")
    is_("(A OR B) AND C answers like (A AND C) OR (B AND C)",
        sorted(grouped.papers) == sorted(flat.papers), True)
    is_("...and both find these papers", sorted(grouped.papers),
        ["p03", "p15", "p18", "p19", "p20"])
    explicit = fq.restrict("title:search OR (abstract:rank AND abstract:result)")
    implicit = fq.restrict("title:search OR abstract:rank AND abstract:result")
    is_("brackets do not change SQL precedence",
        sorted(explicit.papers) == sorted(implicit.papers), True)
    is_("AND is an intersection", fq.restrict(
        "title:search AND results:precision").papers, ["p19"])
    is_("bm25f keeps all 10 papers", len(fq.weighted(
        "title:search AND results:precision").papers), 10)

    print("\n== length normalisation is nearly a rescale (structured-fields.md) ==")
    raw = fq.weighted("structured field search", normalise=False)
    normalised = fq.weighted("structured field search", normalise=True)
    ratios = [round(a.score / b.score, 2) for a, b in zip(raw.hits, normalised.hits)
              if b.score]
    is_("lowest raw/norm ratio", min(ratios), 1.35)
    is_("highest raw/norm ratio", max(ratios), 1.66)

    print("\n== malformed field queries are refused (structured-fields.md) ==")
    for text, fragment in [("keywords:search", "not a field"),
                           ("title:search NOT results:x", "NOT is not supported"),
                           ("(title:a OR title:b", "unmatched"),
                           (":search", "no field name"),
                           ("title:", "nothing to look for"),
                           ("title:search AND", "nothing to join to")]:
        result = fq.restrict(text)
        is_(repr(text), (result.ok, result.error is not None
                         and fragment in result.error), (False, True))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} doc claim(s) no longer true: "
              + ", ".join(FAILURES))
        return 1
    print("every number in the docs is still true")
    return 0


if __name__ == "__main__":
    sys.exit(main())