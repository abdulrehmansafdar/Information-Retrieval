"""Week 02 - structured search engine, web interface.

    python main.py --web
    streamlit run app.py

Every number and every result here is produced by engine.py, which console_ui.py
also uses, so the browser and the terminal can never disagree about an answer.

Tabs
    Search        the probability models side by side, with the BM25 parameters
                  exposed so k1 and b can be watched rather than believed
    Fields & Lists field-restricted search and the two list models
    Proximity     the proximal node operators, with the matched text shown
    Evaluation    Precision@k, MRR and MAP for every model on 12 judged queries
    Corpus        what was indexed, and how

Streamlit re-runs the whole script on every interaction, so the engine is built
once with @st.cache_resource. Without it each click would rebuild the corpus,
the positional index and every IDF table.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import engine as ir
from evaluate import DEFAULT_CUTOFFS, SHORT_NAMES, evaluate_all
from proximity import ProximitySyntaxError

st.set_page_config(page_title="Week 02 Structured Search", layout="wide")

REGION_MODES = ("overlapped", "non-overlapped", "proximal")
PAPER_MODES = ("bm25", "bim", "tfidf", "boolean", "fields", "bm25f")

MODE_ORDER = ("bm25", "bim", "tfidf", "boolean", "fields", "bm25f",
              "overlapped", "non-overlapped", "proximal")


@st.cache_resource(show_spinner="Building the corpus...")
def cached_engine() -> ir.Engine:
    """Built once per server process, not once per interaction."""
    return ir.get_engine()


@st.cache_data(show_spinner="Running the evaluation...")
def cached_scores() -> list:
    return evaluate_all(ir.get_judgements(), ir.evaluation_models())


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def hits_frame(response: ir.SearchResponse) -> pd.DataFrame:
    """Results as a DataFrame, which st.dataframe can sort."""
    rows = []
    # A two-word query that loses one word to the phrase list would otherwise
    # read "1/1" - a complete match. Put the dropped words in the denominator.
    dropped = len(response.unknown)
    for hit in response.hits:
        row = {
            "rank": hit.rank,
            "paper": hit.doc_id,
            "score": round(hit.score, 4),
        }
        if hit.is_region:
            row["region"] = hit.region_id
            row["label"] = hit.field
            row["terms"] = hit.length
        else:
            row["title"] = hit.title
            row["field"] = hit.field
            row["terms"] = hit.length
        if hit.total_terms + dropped:
            row["matched"] = f"{hit.matched}/{hit.total_terms + dropped}"
        if hit.snippet:
            row["excerpt"] = " ".join(hit.snippet.split())[:160]
        rows.append(row)
    return pd.DataFrame(rows)


def show_response(response: ir.SearchResponse, key: str) -> None:
    """Render one SearchResponse. Returns nothing."""
    if response.unknown:
        st.warning(f"not in the vocabulary, ignored: "
                   f"{', '.join(response.unknown)}")
        hint = ir.phrase_hint(response, cached_engine())
        if hint:
            st.info(
                f"{hint} — so {len(response.unknown)} of the words you typed were "
                f"**never searched**. The results below come from the rest of "
                f"the query only. Search the phrase itself, e.g. "
                f"`proximity operator`, to reach the papers that use it."
            )
    if response.error:
        st.error(response.error)
        return

    misrouted = ir.misrouted_field_prefix(response)
    if misrouted:
        st.warning(misrouted)

    if response.terms:
        st.caption(f"query terms: `{'`, `'.join(response.terms)}`")
    for note in response.notes:
        st.caption(note)

    if not response.hits:
        st.info("nothing matched")
        return

    papers = len({h.doc_id for h in response.hits})
    st.success(f"{response.count} results in {papers} of "
               f"{cached_engine().number_of_documents} papers")
    coverage = ir.coverage_note(response)
    if coverage:
        st.caption(coverage)
    frame = hits_frame(response)
    st.dataframe(frame, width="stretch", hide_index=True)

    if any(h.snippet for h in response.hits):
        with st.expander("the matched text"):
            for hit in response.hits:
                if not hit.snippet:
                    continue
                snippet = " ".join(hit.snippet.split())
                st.markdown(
                    f"**{hit.rank}. {hit.region_id}**  \n"
                    f"{hit.doc_id} — {hit.title}  \n"
                    f"`{hit.note}`\n\n"
                    f"> {snippet}\n"
                )

    st.download_button("download as CSV", frame.to_csv(index=False),
                       file_name=f"{response.query.replace(' ', '_')[:40]}"
                                 f"-{response.mode}.csv",
                       mime="text/csv", key=key)


def show_comparison(query: str, top: int, hierarchy: str) -> None:
    """Overlapped and non-overlapped next to each other, with the difference."""
    left, right = st.columns(2)
    strict = ir.search(query, mode="overlapped", top=top, hierarchy=hierarchy)
    loose = ir.search(query, mode="non-overlapped", top=top, hierarchy=hierarchy)

    with left:
        st.markdown("#### Overlapped")
        st.caption("every query term must occur inside **one** region. "
                   "High precision, low coverage.")
        show_response(strict, "dl_overlapped")
    with right:
        st.markdown("#### Non-overlapped")
        st.caption("query terms may occur in **different** regions. "
                   "High coverage, lower precision.")
        show_response(loose, "dl_non_overlapped")

    only_loose = sorted(set(loose.papers) - set(strict.papers))
    only_strict = sorted(set(strict.papers) - set(loose.papers))
    shared = sorted(set(strict.papers) & set(loose.papers))

    st.markdown("#### The difference")
    rows = [
        {"group": "both models", "papers": ", ".join(shared) or "-",
         "count": len(shared)},
        {"group": "non-overlapped only", "papers": ", ".join(only_loose) or "-",
         "count": len(only_loose)},
        {"group": "overlapped only", "papers": ", ".join(only_strict) or "-",
         "count": len(only_strict)},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if only_loose:
        st.info(
            f"`{'`, `'.join(only_loose)}` contain every query term but never two "
            f"of them in the same region. The overlapped model cannot return "
            f"them at all — which is the coverage you give up for precision."
        )


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------


def sidebar() -> dict:
    st.sidebar.title("Controls")
    stats = ir.corpus_stats()

    columns = st.sidebar.columns(2)
    columns[0].metric("papers", stats["papers"])
    columns[1].metric("vocabulary", f"{stats['vocabulary']:,}")
    columns = st.sidebar.columns(2)
    columns[0].metric("postings", f"{stats['postings']:,}")
    columns[1].metric("positions", f"{stats['positions']:,}")
    st.sidebar.caption(
        f"{stats['sections']} sections · {stats['paragraphs']} paragraphs · "
        f"{stats['sentences']} sentences · "
        f"{stats['judged_queries']} judged queries"
    )

    st.sidebar.divider()
    mode = st.sidebar.radio("retrieval model", list(MODE_ORDER),
                            index=0, format_func=lambda m: f"{m} — "
                            f"{ir.MODE_LABELS[m].split(' — ')[0]}")
    top_k = st.sidebar.slider("results to show", 1, 20, 10)

    k1 = 1.2
    b = 0.75
    if mode in {"bm25", "bm25f", "fields"}:
        st.sidebar.divider()
        st.sidebar.markdown("**BM25 parameters**")
        k1 = st.sidebar.slider("k1 — tf saturation", 0.0, 3.0, 1.2, 0.1,
                               help="k1 = 0 makes this the Binary "
                                    "Independence Model exactly. Larger values "
                                    "let extra occurrences count for more.")
        b = st.sidebar.slider("b — length normalisation", 0.0, 1.0, 0.75, 0.05,
                              help="b = 0 ignores paper length entirely. "
                                   "b = 1 normalises it fully.")
        st.sidebar.caption(f"`score = Σ idf(t) · tf(k₁+1) / "
                           f"(tf + k₁((1−b) + b·|d|/avgdl))` with "
                           f"k₁ = {k1:g}, b = {b:g}")
        # Surfaced next to the results as well as in the sidebar: the claim that
        # k1 = 0 reproduces BIM is the single most useful thing on this screen,
        # and a note the user has to scroll up to find is not worth having.
        if k1 == 0.0:
            st.info(
                "**k₁ = 0**, so the saturation factor is exactly `tf/tf = 1` "
                "and this mode **is** the Binary Independence Model. "
                "test_ranking.py asserts the two produce identical scores."
            )

    hierarchy = "paragraph"
    if mode in REGION_MODES:
        st.sidebar.divider()
        hierarchy = st.sidebar.selectbox(
            "region hierarchy", list(ir.HIERARCHY_NAMES),
            index=1, help="which flat view of the text the list models use")

    st.sidebar.divider()
    options = ir.query_options()
    if mode in REGION_MODES:
        pool = options["proximal"] if mode == "proximal" else options["ranked"]
    elif mode in {"fields", "bm25f"}:
        pool = options["field"]
    else:
        pool = options["ranked"]

    example = st.sidebar.selectbox("example query", ["choose one", *pool],
                                   key=f"ex_{mode}")
    if example != "choose one":
        st.session_state[f"q_{mode}"] = example

    return {"mode": mode, "top": top_k, "k1": k1, "b": b,
            "hierarchy": hierarchy, "pool": pool}


def current_query(key: str, placeholder: str) -> str:
    return st.text_input("query", key=key, placeholder=placeholder)


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------


def tab_search(options: dict) -> None:
    mode, top = options["mode"], options["top"]
    st.subheader(f"{mode} — {ir.MODE_LABELS[mode]}")
    st.caption(ir.MODE_LABELS[mode])

    if mode in {"fields", "bm25f"}:
        st.markdown(
            "Write `title:segmentation` to search one field, or "
            "`title:search AND results:precision` to require a term in each of "
            "two fields. A field name that does not exist is an error, not a "
            "quiet miss."
        )
    elif mode == "proximal":
        st.markdown(
            "Operators: `paragraph in methodology`, "
            "`section with(2) retrieval`, `sentence before(10) index (paragraph)`, "
            "`paragraph child methodology`, `section parent(2) paragraph`."
        )

    query = current_query(f"q_{mode}", options["pool"][0])
    if not query:
        st.info("type a query, or pick one of the examples in the sidebar")
        return

    response = ir.search(query, mode=mode, top=top, k1=options["k1"],
                         b=options["b"], hierarchy=options["hierarchy"])
    show_response(response, f"dl_{mode}")

    if "history" not in st.session_state:
        st.session_state["history"] = []
    if query not in st.session_state["history"]:
        st.session_state["history"].insert(0, f"{mode}: {query}")
        st.session_state["history"] = st.session_state["history"][:8]
    if st.session_state["history"]:
        st.caption("recent: " + "  |  ".join(st.session_state["history"][:5]))


def tab_compare(options: dict) -> None:
    st.subheader("Fields and lists, side by side")
    st.caption("Two questions the same engine answers differently: which field "
               "does this term live in, and do the terms ever meet?")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Structured field search")
        st.caption("`title:...` and `results:...` restrict the ranking to one "
                   "field. `bm25f` weights every field instead.")
        field_query = st.text_input("field query", key="q_fields_compare",
                                    placeholder="title:search")
        design = st.radio("design", ["restrict", "filter", "bm25f"],
                          horizontal=True, key="design_compare")
        if field_query:
            search = cached_engine().fields
            if design == "restrict":
                result = search.restrict(field_query, top=options["top"])
            elif design == "filter":
                result = search.filter(field_query, top=options["top"])
            else:
                result = search.weighted(field_query, top=options["top"])
            if result.error:
                st.error(result.error)
            elif not result.hits:
                st.info("nothing matched")
            else:
                st.dataframe(pd.DataFrame([
                    {"rank": h.rank, "paper": h.doc_id, "field": h.field,
                     "score": round(h.score, 4),
                     "matched": f"{h.matched}/{h.total_query_terms}"}
                    for h in result.hits
                ]), width="stretch", hide_index=True)
                st.caption(
                    f"`restrict` and `filter` return the same papers in a "
                    f"different order — filter ranks the whole paper and then "
                    f"discards, so the order reflects text outside the field."
                )

    with right:
        st.markdown("#### Overlapped vs non-overlapped")
        st.caption("precision against coverage, over the chosen hierarchy.")
        list_query = st.text_input("list query", key="q_list_compare",
                                   placeholder="document index positions")
        if list_query:
            show_comparison(list_query, options["top"], options["hierarchy"])


def tab_proximity(options: dict) -> None:
    st.subheader("Proximal node model")
    st.caption("The paper is a tree: paper → section → paragraph → sentence. "
               "These operators answer questions about that tree, not about "
               "whole documents.")

    query = current_query("q_prox_tab", "paragraph in methodology")
    if not query:
        return

    engine = cached_engine()
    try:
        result = engine.proximal.evaluate(query)
    except ProximitySyntaxError as error:
        # A bad operator is ordinary input, not a crash. engine.search() already
        # catches this; the tab calls the model directly, so it has to catch it
        # too or one malformed query takes the whole page down.
        st.error(f"{error}")
        return
    if result.error:
        st.error(result.error)
        return

    for step in result.steps:
        st.caption(step)

    if not result.nodes:
        st.info("no node satisfies that query")
        return

    st.success(f"{result.count} node(s) in {len(result.papers)} papers")
    st.info("This model returns a **set** of nodes, not a ranking. There is no "
            "relevance score, so the order shown is document order.")

    top = options["top"]
    rows = []
    for position, node in enumerate(result.nodes[:top], 1):
        rows.append({
            "#": position,
            "paper": node.doc_id,
            "node": node.region_id,
            "kind": node.kind,
            "label": node.label,
            "start": node.start,
            "end": node.end,
            "terms": node.length,
            "text": " ".join(node.text.split())[:220],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if len(result.nodes) > top:
        st.caption(f"showing {top} of {len(result.nodes)} nodes")

    with st.expander("the operators this query language accepts"):
        st.markdown(
            "| operator | meaning | example |\n"
            "|---|---|---|\n"
            "| `P in Q` | P nodes inside some Q node | `paragraph in methodology` |\n"
            "| `P beginin Q` | only P's start inside Q | `sentence beginin results` |\n"
            "| `P with(k) T` | P contains ≥ k occurrences of T | "
            "`section with(2) retrieval` |\n"
            "| `P child Q` | direct children only | `paragraph child methodology` |\n"
            "| `P parent(k) Q` | P is parent of ≥ k Q nodes | "
            "`section parent(2) paragraph` |\n"
            "| `P before(k) T (C)` | P ends ≤ k terms before T, inside C | "
            "`sentence before(10) index (paragraph)` |\n"
            "| `P after(k) T (C)` | P starts ≤ k terms after T, inside C | "
            "`sentence after(5) index (paragraph)` |\n"
        )


def tab_evaluation() -> None:
    st.subheader("Effectiveness")
    st.caption("12 hand-made queries over 20 papers, judged by hand. This is a "
               "demonstration of how to measure, not a benchmark.")

    scores = cached_scores()
    overall = pd.DataFrame([{
        "model": SHORT_NAMES.get(s.model, s.model),
        "P@1": round(s.mean("precision", 1), 3),
        "P@5": round(s.mean("precision", 5), 3),
        "R@10": round(s.mean("recall", 10), 3),
        "MRR": round(s.mrr, 3),
        "MAP": round(s.map, 3),
        "queries with no relevant hit": len(s.queries_with_no_hit),
    } for s in scores])
    st.dataframe(overall, width="stretch", hide_index=True)

    st.markdown("#### Query by query, MRR per model")
    judgements = ir.get_judgements()
    rows = []
    for query_id in judgements.query_ids:
        row = {"query": query_id, "text": judgements.queries[query_id],
               "relevant": len(judgements.relevant[query_id])}
        for score in scores:
            match = next(q for q in score.queries if q.query_id == query_id)
            row[SHORT_NAMES.get(score.model, score.model)] = round(match.rr, 2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("1.00 means the correct paper came first. 0.00 means it never "
               "appeared at all. Read down a column to see which query type a "
               "model handles.")

    best = max(scores, key=lambda s: s.mrr)
    worst = min(scores, key=lambda s: s.mrr)
    st.markdown("#### What this does and does not support")
    st.markdown(
        f"- **{best.model}** has the best MRR at **{best.mrr:.3f}**; "
        f"**{worst.model}** the worst at **{worst.mrr:.3f}**. The spread is "
        f"**{best.mrr - worst.mrr:.3f}**.\n"
        "- The list models score lower because they answer a *different* "
        "question: a region must hold every term. That is a real cost, not a "
        "bug, and it is the price of the precision they buy.\n"
        "- Twelve hand-made queries over twenty papers is a demonstration. No "
        "claim about which model is better *in general* follows from it, and "
        "the qrels file says so in its header."
    )
    st.download_button("download the table", overall.to_csv(index=False),
                       file_name="week02-evaluation.csv", mime="text/csv",
                       key="dl_eval")


def tab_corpus() -> None:
    st.subheader("Corpus and structure")
    engine = cached_engine()
    stats = ir.corpus_stats()

    columns = st.columns(4)
    columns[0].metric("papers", stats["papers"])
    columns[1].metric("terms", f"{stats['total_terms']:,}")
    columns[2].metric("distinct terms", f"{stats['vocabulary']:,}")
    columns[3].metric("idf max", f"{stats['idf_max']:.3f}")
    st.caption(
        f"paper length {stats['length_min']}–{stats['length_max']} terms "
        f"(average {stats['avg_length']:.1f})  |  "
        f"{stats['sections']} sections  |  {stats['paragraphs']} paragraphs  |  "
        f"{stats['sentences']} sentences"
    )

    st.markdown("#### Papers")
    st.dataframe(pd.DataFrame([{
        "paper": p.doc_id, "title": p.title, "year": p.year, "venue": p.venue,
        "terms": p.length,
        "fields": ", ".join(sorted(p.fields)),
        "paragraphs": sum(len(v) for v in p.paragraphs.values()),
    } for p in engine.index.papers]), width="stretch", hide_index=True)

    st.markdown("#### Most frequent terms carry the least information")
    st.dataframe(pd.DataFrame([{
        "term": term, "df": df,
        "idf": round(engine.idf_table.idf_for(term), 4),
    } for term, df in stats["top_by_df"]]), width="stretch", hide_index=True)

    st.markdown("#### Regions of one paper")
    doc_id = st.selectbox("paper", [p.doc_id for p in engine.index.papers],
                          key="corpus_paper")
    rows = []
    for kind in ("section", "paragraph", "sentence"):
        for region in sorted(engine.index.regions_of_kind(doc_id, kind),
                             key=lambda r: r.start):
            rows.append({
                "kind": kind, "region": region.region_id,
                "start": region.start, "end": region.end,
                "terms": region.length, "label": region.label,
                "excerpt": " ".join(region.text.split())[:120],
            })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def tab_pipeline() -> None:
    st.subheader("Pipeline walkthrough")
    st.caption("One sentence through the whole engine: preprocessing, the "
               "positional index, both probability models, and the region "
               "models. Every number below is computed, not typed.")

    sentence = st.text_input("sentence", key="pipe_sentence",
                             value="Machine Learning improves clinic workflow "
                                   "in patient records")
    if not sentence.strip():
        return

    from preprocess import preprocess_with_positions

    terms, positions = preprocess_with_positions(sentence)
    st.markdown("#### 1  Preprocessing — positions, not just counts")
    st.markdown("`" + "`, `".join(f"{t}" for t in terms) + "`")
    st.caption(f"{len(terms)} terms at dense positions "
               f"0–{max(len(terms) - 1, 0)}. A phrase term keeps the position "
               f"of its first word, which is what makes regions work. "
               f"week01's `preprocess_text()` cannot do this — it returns "
               f"unigrams then phrases, and loses the order.")

    query = st.text_input("run it as a query", key="pipe_query",
                          value=sentence)
    if query.strip():
        from ranking import parse_query as parse_ranked_query

        parsed = parse_ranked_query(query, cached_engine().idf_table)
        columns = st.columns(2)
        for column, mode in zip(columns, ("bim", "bm25")):
            with column:
                response = ir.search(query, mode=mode, top=5)
                st.markdown(f"#### 2  {mode.upper()}")
                if not response.hits:
                    st.info("no paper shares a term with that sentence")
                    continue
                st.dataframe(hits_frame(response), width="stretch",
                             hide_index=True)
                top = response.hits[0]
                # Show the formula's own terms rather than just the total, so
                # the score can be checked by hand against step 3 of the notes.
                model = (cached_engine().bim if mode == "bim"
                         else cached_engine().bm25)
                score, rows = model.score_paper(top.doc_id, parsed.terms)
                st.caption(" + ".join(
                    f"idf({r.term})×sat({r.saturation:.3f}) = {r.contribution:.4f}"
                    for r in rows) + f" = **{score:.4f}**")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    cached_engine()
    options = sidebar()

    st.title("Week 02 — Structured Search Engine")
    st.caption(
        "Probability-based ranking (BIM, BM25), structured field search, the "
        "overlapped and non-overlapped list models, and the proximal node "
        "model — over 20 research papers with title, abstract, methodology "
        "and results."
    )

    tabs = st.tabs(["Search", "Fields & Lists", "Proximity", "Evaluation",
                    "Corpus", "Pipeline"])
    with tabs[0]:
        tab_search(options)
    with tabs[1]:
        tab_compare(options)
    with tabs[2]:
        tab_proximity(options)
    with tabs[3]:
        tab_evaluation()
    with tabs[4]:
        tab_corpus()
    with tabs[5]:
        tab_pipeline()


if __name__ == "__main__":
    main()