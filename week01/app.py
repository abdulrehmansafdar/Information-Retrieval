"""Mini search engine - Streamlit web interface.

    streamlit run app.py

The retrieval logic lives in engine.py, which main.py also uses, so the web
interface and the command line can never disagree about what the engine
returns. This file only handles presentation.

Streamlit re-runs the whole script on every interaction, so the engine is built
once with @st.cache_resource. Without it, each click would rebuild the index.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from demo import demo_sections, renderable_to_text
import engine as ir
from pipeline import (
    STAGE_CAPTIONS_ACT1,
    STAGE_CAPTIONS_ACT2,
    build_stage_model,
)

st.set_page_config(page_title="IR Mini Search Engine", layout="wide")

EXAMPLE_QUERIES = [
    "machine learning healthcare",
    "artificial intelligence",
    "medical imaging",
    "neural network",
    "security attack anomaly",
]

EXAMPLE_BOOLEAN_QUERIES = [
    "machine learning AND healthcare",
    "artificial intelligence OR robotics",
    "machine learning AND NOT finance",
    "healthcare AND diagnosis",
    "computer vision OR medical imaging",
    "security AND machine learning",
    "NLP AND education",
    "(artificial intelligence OR robotics) AND NOT security",
]


@st.cache_resource(show_spinner="Building the index...")
def cached_engine():
    """Built once per server process, not once per interaction."""
    return ir.get_engine()


def hits_frame(response: ir.SearchResponse) -> pd.DataFrame:
    """Ranked hits as a DataFrame, which st.dataframe can sort."""
    rows = []
    for hit in response.hits:
        row = {
            "rank": hit.rank,
            "doc": hit.doc_id,
            "title": hit.title,
            "terms": hit.length,
        }
        if response.mode == "ranked":
            row["cosine"] = round(hit.score, 4)
            row["AND all"] = "yes" if hit.satisfies_and else "no"
            row["matched"] = ", ".join(
                f"{term} ({weight:.2f})" for term, weight in hit.matched_terms
            )
        rows.append(row)
    return pd.DataFrame(rows)


def show_response(response: ir.SearchResponse, key: str) -> None:
    if response.unknown_terms:
        st.warning(f"not in the vocabulary, ignored: "
                   f"{', '.join(response.unknown_terms)}")

    if response.error:
        st.error(response.error)
        return

    if response.terms:
        st.caption(f"preprocessed terms: {', '.join(response.terms)}")

    if not response.hits:
        st.info("nothing matched")
        return

    total = cached_engine().number_of_documents
    if response.mode == "ranked":
        st.success(f"{response.count} of {total} documents  |  mode: ranked  |  "
                   f"strict AND matches: {len(response.boolean_ids)}")
    else:
        st.success(f"{response.count} of {total} documents satisfy the "
                   "Boolean expression")

    frame = hits_frame(response)
    st.dataframe(frame, width="stretch", hide_index=True)

    if response.mode == "ranked" and len(response.boolean_ids) < response.count:
        only = sorted({h.doc_id for h in response.hits} - set(response.boolean_ids))
        st.caption(f"partial matches, dropped by a strict AND reading: "
                   f"{', '.join(only)}")

    with st.expander("why each document scored what it did"):
        for hit in response.hits:
            parts = "  ".join(f"{t} × {w:.3f}" for t, w in hit.matched_terms)
            st.markdown(
                f"**{hit.rank}. {hit.doc_id} - {hit.title}**  \n"
                f"score `{hit.score:.4f}`  |  {hit.length} terms  \n"
                f"{parts or 'no shared query terms'}"
            )

    st.download_button("download as CSV", frame.to_csv(index=False),
                       file_name=f"{response.query.replace(' ', '_')}.csv",
                       mime="text/csv", key=key)


def sidebar() -> None:
    st.sidebar.title("Controls")
    stats = ir.corpus_stats()
    st.sidebar.metric("documents", stats["documents"])
    st.sidebar.metric("vocabulary terms", f"{stats['vocabulary']:,}")
    st.sidebar.metric("postings", f"{stats['postings']:,}")

    top_k = st.sidebar.slider("results to show", 1, 20, 10)
    mode = st.sidebar.radio("retrieval model", ["auto", "ranked", "boolean"],
                            index=0,
                            help="auto: Boolean operators in the query mean "
                                 "Boolean, otherwise cosine ranking")

    st.sidebar.divider()

    ranked_example = st.sidebar.selectbox(
        "example ranked query",
        ["choose one", *EXAMPLE_QUERIES],
        key="ex_ranked",
    )
    if ranked_example != "choose one":
        st.session_state["query"] = ranked_example

    boolean_example = st.sidebar.selectbox(
        "example Boolean query",
        ["choose one", *EXAMPLE_BOOLEAN_QUERIES],
        key="ex_boolean",
    )
    if boolean_example != "choose one":
        st.session_state["bool_query"] = boolean_example
    return top_k, mode


def tab_search(top_k: int, mode: str) -> None:
    st.subheader("Ranked search")
    st.caption("TF-IDF weights and cosine similarity. A query using AND, OR, "
               "NOT or parentheses switches to Boolean retrieval automatically.")

    query = st.text_input("query", key="query",
                          placeholder="machine learning healthcare")
    if query:
        show_response(ir.search(query, top=top_k, mode=mode), "dl_ranked")

    if "history" not in st.session_state:
        st.session_state["history"] = []
    if query and query not in st.session_state["history"]:
        st.session_state["history"].insert(0, query)
        st.session_state["history"] = st.session_state["history"][:8]

    if st.session_state["history"]:
        st.caption("recent: " + "  |  ".join(st.session_state["history"][:6]))


def tab_boolean() -> None:
    st.subheader("Boolean retrieval")
    st.caption("AND, OR, NOT and parentheses. The query is parsed into a tree "
               "before evaluation, so NOT binds tighter than AND, and AND "
               "binds tighter than OR.")

    query = st.text_input("boolean query", key="bool_query",
                          placeholder="machine learning AND NOT finance")
    if query:
        response = ir.search(query, mode="boolean")
        if response.error:
            st.error(response.error)
        elif not response.hits:
            st.info("no document satisfies that expression")
        else:
            st.success(f"{response.count} of "
                       f"{cached_engine().number_of_documents} documents")
            st.dataframe(hits_frame(response), width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Why some of these return nothing")
    st.markdown(
        "`machine AND healthcare` returns **0 documents**, and that is correct. "
        "Preprocessing stored *machine learning* as the single phrase term "
        "`machin learn`, so the free-standing `machin` survives only in the "
        "three documents where *machine* means a physical machine. Write "
        "`machine learning AND healthcare` for the 2 documents you probably "
        "meant."
    )


SAMPLE_SENTENCES = [
    "Hospitals and clinics collect large amounts of medical data, including "
    "laboratory results and patient records.",
    "An autonomous robot builds an internal map of its environment from "
    "sensor data.",
    "A security system uses anomaly detection to block new network attacks.",
]


def tokens_line(items: list[str]) -> str:
    """Render terms as inline code chips joined by spaces."""
    items = [t for t in items if t]
    return " ".join(f"`{t}`" for t in items) if items else "*none*"


def stage_card(number: int, label: str, caption: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{number}. {label}**  \n{caption}")


def tab_pipeline() -> None:
    st.subheader("Pipeline walkthrough")
    st.caption("One sentence runs through the whole engine. Act 1 takes it from "
               "raw text to final terms; Act 2 follows the terms into the "
               "inverted index, IDF weighting, a query and the ranked results. "
               "Every number and table below is computed by the real engine.")

    sentence = st.selectbox("sentence to run through the pipeline",
                            SAMPLE_SENTENCES, key="pip_sentence")
    model = build_stage_model(sentence)
    act1, act2 = model["act1"], model

    st.markdown("#### Act 1  Preprocessing")

    stage_card(1, "Raw", STAGE_CAPTIONS_ACT1[0])
    st.markdown(f"*{act1['raw']}*")

    stage_card(2, "Lowercase", STAGE_CAPTIONS_ACT1[1])
    st.markdown(f"*{act1['lower']}*")

    stage_card(3, "Tokens", STAGE_CAPTIONS_ACT1[2])
    st.markdown(tokens_line(act1["tokens"]))

    stage_card(4, "Clean", STAGE_CAPTIONS_ACT1[3])
    st.markdown(tokens_line(act1["clean"]))

    stage_card(5, "Stems", STAGE_CAPTIONS_ACT1[4])
    st.markdown(tokens_line(act1["stems"]))
    if act1["stops_used"]:
        st.caption("stop words still visible: "
                   f"{tokens_line(act1['stops_used'])}")

    stage_card(6, "Phrases", STAGE_CAPTIONS_ACT1[5])
    if act1["phrases"]:
        st.markdown(tokens_line(act1["phrases"]))
        st.caption("each chip above is now a SINGLE term in the index")
    else:
        st.markdown("*no phrase-list words in this sentence*")

    stage_card(7, "Terms", STAGE_CAPTIONS_ACT1[6])
    st.markdown(tokens_line(act1["final"]))
    if act1["stops_used"]:
        st.caption(f"dropped stop words: {tokens_line(act1['stops_used'])}")

    st.divider()
    st.markdown("#### Act 2  Retrieval")

    stage_card(8, "Index", STAGE_CAPTIONS_ACT2[0])
    st.dataframe(pd.DataFrame(act2["postings_rows"]), width="stretch",
                 hide_index=True)

    stage_card(9, "Weights", STAGE_CAPTIONS_ACT2[1])
    st.dataframe(pd.DataFrame(act2["weight_rows"]), width="stretch",
                 hide_index=True)

    stage_card(10, "Query", STAGE_CAPTIONS_ACT2[2])
    st.markdown(tokens_line(act2["query_terms"]))
    st.caption(f"{act2['scored_count']} of {act2['number_of_documents']} "
               f"documents share at least one query term, so they are all "
               f"scored.")

    stage_card(11, "Results", STAGE_CAPTIONS_ACT2[3])
    st.dataframe(pd.DataFrame(act2["ranked_rows"]), width="stretch",
                 hide_index=True)
    st.caption("cos(q, d) = (q . d) / (|q| x |d|); only documents sharing a "
               "query term score above zero.")

    st.divider()
    with st.expander("static corpus walkthrough"):
        st.caption("The same sections as --demo on the command line, with the "
                   "real corpus numbers.")
        if st.button("show the walkthrough", type="secondary",
                     key="pip_static"):
            with st.spinner("collecting numbers..."):
                sections = demo_sections()
            for title, renderable in sections:
                with st.expander(title, expanded=True):
                    st.code(renderable_to_text(renderable))


def tab_corpus() -> None:
    st.subheader("Corpus")
    stats = ir.corpus_stats()
    low, high = stats["terms_per_document"]

    columns = st.columns(4)
    columns[0].metric("documents", stats["documents"])
    columns[1].metric("terms", f"{stats['total_terms']:,}")
    columns[2].metric("distinct terms", f"{stats['vocabulary']:,}")
    columns[3].metric("hapax terms", f"{stats['hapax']:,}")
    st.caption(f"document length {low}-{high} terms "
               f"(ratio {high / low:.2f}x)  |  idf range "
               f"{stats['idf_min']:.4f} to {stats['idf_max']:.4f}")

    st.markdown("#### Most frequent terms carry the least information")
    engine = cached_engine()
    rows = [{"term": term, "df": df,
             "idf": round(engine.idf_model.idf_for(term), 4)}
            for term, df in stats["top_by_df"]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### Documents")
    st.dataframe(
        pd.DataFrame([{"doc": d.doc_id, "title": d.title,
                       "terms": len(d.terms)} for d in engine.index.documents]),
        width="stretch", hide_index=True)


def main() -> None:
    cached_engine()
    top_k, mode = sidebar()

    st.title("IR Mini Search Engine")
    st.caption("Boolean retrieval and vector-based ranking over a "
               "20-document corpus, sharing one inverted index.")

    tabs = st.tabs(["Ranked search", "Boolean", "Pipeline", "Corpus"])
    with tabs[0]:
        tab_search(top_k, mode)
    with tabs[1]:
        tab_boolean()
    with tabs[2]:
        tab_pipeline()
    with tabs[3]:
        tab_corpus()


if __name__ == "__main__":
    main()




