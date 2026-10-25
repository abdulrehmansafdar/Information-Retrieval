"""Week 02 - structured search engine, web interface.

    python main.py --web
    streamlit run app.py

Every number and every result here is produced by engine.py, which console_ui.py
also uses, so the browser and the terminal can never disagree about an answer.
This file only handles presentation.

Design notes
    No sidebar. Every control sits inline in a bordered bar at the top of the
    page, so nothing is hidden off to the side and the whole screen belongs to
    the content. Cards, metrics and tables use Streamlit's own bordered
    primitives, and the one accent (teal, from .streamlit/config.toml) is applied
    through Streamlit's CSS variables so it survives a switch to dark mode.

Tabs
    Start here      what the app is, and the order to read the tabs in
    Search          one query, and the model it runs under
    Fields & Lists  a single field, and the two list models
    Proximity       questions about the paper tree
    Evaluation      P@k, MRR and MAP for every model
    Corpus          what was indexed, and how
    Pipeline        one sentence through the whole engine
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import engine as ir
from evaluate import SHORT_NAMES, evaluate_all
from proximity import ProximitySyntaxError

st.set_page_config(page_title="Week 2 - Search", layout="wide",
                   initial_sidebar_state="collapsed")

REGION_MODES = ("overlapped", "non-overlapped", "proximal")

MODE_ORDER = ("bm25", "bim", "tfidf", "boolean", "fields", "bm25f",
              "overlapped", "non-overlapped", "proximal")

# Colour by model family, using Radix Colors step 11 (text) and step 3 (subtle
# background). Same discipline as week 3: one hue per family, not decoration.
#   https://www.radix-ui.com/colors/docs/palette-composition/understanding-the-scale
MODE_FAMILY = {
    "bm25": "probability", "bim": "probability", "bm25f": "probability",
    "fields": "probability", "tfidf": "vector", "boolean": "boolean",
    "overlapped": "list", "non-overlapped": "list", "proximal": "node",
}
FAMILY_STYLE = {
    "probability": ("#008573", "#E0F8F3"),   # teal
    "vector": ("#6550B9", "#F4F0FE"),        # violet
    "boolean": ("#60646C", "#F0F0F3"),       # slate
    "list": ("#AB6400", "#FFF7C2"),          # amber
    "node": ("#0D74CE", "#E6F4FE"),          # blue
}

PAGE_CSS = """
<style>
  .stApp { font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue",
           Arial, sans-serif;
           background-image:
             radial-gradient(60% 42% at 12% 0%, rgba(15, 118, 110, 0.055), transparent 62%);
           background-attachment: fixed; }
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

  /* page header */
  .hero { padding: 4px 0 14px 0; }
  .hero h1 { font-size: clamp(1.9rem, 3.4vw, 2.9rem); font-weight: 780;
             letter-spacing: -0.03em; margin: 0 0 10px 0; line-height: 1.07; }
  .hero p { font-size: 1.02rem; margin: 0; opacity: 0.78; max-width: 68ch; }
  .chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { font-size: 0.78rem; font-weight: 600; padding: 3px 11px;
          border-radius: 999px; border: 1px solid rgba(100, 116, 139, 0.35);
          background: var(--secondary-background-color); }
  .badge { display: inline-block; font-size: 0.74rem; font-weight: 700;
           letter-spacing: 0.02em; padding: 2px 9px; border-radius: 999px;
           margin-bottom: 4px; }

  /* styled tables: tinted header, zebra rows, coloured meters */
  .tablewrap { border: 1px solid #E5E5E1; border-radius: 14px;
               overflow: auto; margin: 4px 0 14px 0; background: #FFFFFF; }
  table.tt { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  table.tt thead th { background: var(--secondary-background-color);
      color: var(--accent); font-weight: 700; text-align: left;
      padding: 10px 12px; border-bottom: 2px solid var(--accent);
      white-space: nowrap; }
  table.tt tbody td { padding: 9px 12px;
      border-bottom: 1px solid #EDEDEA; }
  table.tt tbody tr:last-child td { border-bottom: none; }
  table.tt tbody tr:nth-child(even) { background: #FAFAF8; }
  table.tt tbody tr:hover { background: #F2F2EF; }
  .meter { display: inline-block; width: 62px; height: 6px; border-radius: 999px;
           background: rgba(100, 116, 139, 0.18); vertical-align: middle;
           margin-right: 7px; overflow: hidden; }
  .meter > span { display: block; height: 100%; }

  /* lede used to open a tab */
  .lede { border-left: 3px solid var(--primary-color);
          background: var(--secondary-background-color);
          border-radius: 4px 12px 12px 4px;
          padding: 12px 16px; margin: 2px 0 16px 0; }
  .lede .lede-title { display: block; font-weight: 700; font-size: 0.95rem; }
  .lede .lede-body { display: block; font-size: 0.9rem; margin-top: 2px; }

  h1, h2, h3 { letter-spacing: -0.015em; }
  h3 { margin-top: 0.4rem; }

  /* tabs */
  button[data-baseweb="tab"] { font-weight: 600; }
  button[data-baseweb="tab"][aria-selected="true"] { color: var(--primary-color); }
  div[data-baseweb="tab-highlight"] { background-color: var(--primary-color); }

  /* metrics and tables */
  div[data-testid="stMetric"] { background: #FFFFFF;
          border: 1px solid #E5E5E1;
          border-radius: 14px; padding: 14px 16px;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04); }
  div[data-testid="stMetricValue"] { font-size: 1.4rem; }
  div[data-testid="stDataFrame"] { border: 1px solid rgba(100, 116, 139, 0.22);
          border-radius: 12px; overflow: hidden; }

  /* controls */
  code { background: var(--secondary-background-color); border-radius: 6px;
         padding: 1px 5px; }
  div[data-testid="stAlert"] { border-radius: 10px; }
  div[data-testid="stExpander"] { border-radius: 12px; }
  .stButton button { border-radius: 9px;
      transition: transform .18s ease, box-shadow .18s ease; }
  .stButton button:hover { transform: translateY(-1px);
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.10); }
  hr { margin: 1.6rem 0; }
  div[data-testid="stMetric"] { transition: transform .2s ease, box-shadow .2s ease; }
  div[data-testid="stMetric"]:hover { transform: translateY(-2px);
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08); }

  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
    .stButton button:hover, div[data-testid="stMetric"]:hover {
      transform: none; box-shadow: none; }
  }
</style>
"""


# ---------------------------------------------------------------------------
# small presentation helpers
# ---------------------------------------------------------------------------


def inject_styles() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    html = [f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>']
    if chips:
        html.append('<div class="chips">'
                    + "".join(f'<span class="chip">{c}</span>' for c in chips)
                    + "</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def lede(title: str, body: str, color: str = "") -> None:
    """The one-line statement of purpose that opens every tab."""
    style = f' style="border-left-color:{color}"' if color else ""
    st.markdown(
        f'<div class="lede"{style}><span class="lede-title">{title}</span>'
        f'<span class="lede-body">{body}</span></div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def stat_row(items: list[tuple]) -> None:
    """Metric cards. Each item is (label, value) or (label, value, icon)."""
    for column, item in zip(st.columns(len(items)), items):
        icon = item[2] if len(item) > 2 else None
        column.metric(item[0], item[1], border=True, icon=icon)


def html_table(headers: list[str], rows: list[list[str]],
               accent: str = "#0f766e") -> None:
    """A table we can actually colour.

    st.dataframe draws on a canvas, so its cells ignore CSS and stay black and
    white. The scoreboard and the corpus tables carry meaning, so they render as
    real HTML: tinted header, zebra rows, and an accent per section.
    """
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows)
    st.markdown(
        f'<div class="tablewrap" style="--accent:{accent}">'
        f'<table class="tt"><thead><tr>{head}</tr></thead>'
        f'<tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True)


def meter(value: float, color: str, label: str | None = None) -> str:
    """A small coloured bar plus a number.

    `value` drives the bar (0..1); `label` is what is printed, so a raw score can
    keep its own scale while the bar stays relative to the best hit.
    """
    percent = max(0, min(100, round(value * 100)))
    shown = label if label is not None else f"{value:.3f}"
    return (f'<span class="meter"><span style="width:{percent}%;'
            f'background:{color}"></span></span>{shown}')


@st.cache_resource(show_spinner="Reading the papers...")
def cached_engine() -> ir.Engine:
    """Built once per server process, not once per interaction."""
    return ir.get_engine()


@st.cache_data(show_spinner="Scoring every model...")
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


def results_table(response: ir.SearchResponse, key: str | None = None) -> None:
    """The results, with the score drawn as a bar relative to the best hit."""
    frame = hits_frame(response)
    largest = max((h.score for h in response.hits), default=0.0)
    config: dict = {"rank": st.column_config.NumberColumn("#", width="small")}
    if largest > 0.0:
        config["score"] = st.column_config.ProgressColumn(
            "score", format="%.4f", min_value=0.0, max_value=float(largest))
    else:
        config["score"] = st.column_config.NumberColumn("score", format="%.4f")
    st.dataframe(frame, width="stretch", hide_index=True, column_config=config)
    return frame


def show_response(response: ir.SearchResponse, key: str) -> None:
    """Render one SearchResponse inside a bordered card."""
    if response.unknown:
        st.warning(f"not in the collection, ignored: "
                   f"{', '.join(response.unknown)}")
        hint = ir.phrase_hint(response, cached_engine())
        if hint:
            st.info(
                f"{hint}. So {len(response.unknown)} of the words you typed were "
                f"**never searched**, and the results below come from the rest of "
                f"the query. Search the phrase itself, for example "
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
        st.info("Nothing matched. Try a shorter query, or pick an example above.")
        return

    with st.container(border=True):
        papers = len({h.doc_id for h in response.hits})
        st.success(f"{response.count} results in {papers} of "
                   f"{cached_engine().number_of_documents} papers")
        coverage = ir.coverage_note(response)
        if coverage:
            st.caption(coverage)
        frame = results_table(response, key)

        if any(h.snippet for h in response.hits):
            with st.expander("show the matched text",
                         icon=":material/format_quote:"):
                for hit in response.hits:
                    if not hit.snippet:
                        continue
                    snippet = " ".join(hit.snippet.split())
                    st.markdown(
                        f"**{hit.rank}. {hit.region_id}**  \n"
                        f"{hit.doc_id} · {hit.title}  \n"
                        f"`{hit.note}`\n\n"
                        f"> {snippet}\n"
                    )

        st.download_button("Download these results as CSV",
                           frame.to_csv(index=False),
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
        st.caption("Every query term must occur inside **one** region. "
                   "High precision, low coverage.")
        show_response(strict, "dl_overlapped")
    with right:
        st.markdown("#### Non-overlapped")
        st.caption("Query terms may occur in **different** regions. "
                   "High coverage, lower precision.")
        show_response(loose, "dl_non_overlapped")

    only_loose = sorted(set(loose.papers) - set(strict.papers))
    only_strict = sorted(set(strict.papers) - set(loose.papers))
    shared = sorted(set(strict.papers) & set(loose.papers))

    st.markdown("#### The difference")
    rows = [
        ['<span class="badge" style="color:#60646C;background:#F0F0F3">'
         'both models</span>', ", ".join(shared) or "-", str(len(shared))],
        ['<span class="badge" style="color:#AB6400;background:#FFF7C2">'
         'non-overlapped only</span>', ", ".join(only_loose) or "-",
         str(len(only_loose))],
        ['<span class="badge" style="color:#6550B9;background:#F4F0FE">'
         'overlapped only</span>', ", ".join(only_strict) or "-",
         str(len(only_strict))],
    ]
    html_table(["group", "papers", "count"], rows, accent="#008573")
    if only_loose:
        st.info(
            f"`{'`, `'.join(only_loose)}` contain every query term, but never two "
            f"of them in the same region. The overlapped model cannot return them "
            f"at all. That is the coverage you trade away for precision."
        )


# ---------------------------------------------------------------------------
# controls (inline, no sidebar)
# ---------------------------------------------------------------------------


def controls() -> dict:
    """The model and its settings, laid out across the top of the page."""
    options = ir.query_options()
    with st.container(border=True):
        st.markdown("**Model and settings**")
        st.caption("These apply on the **Search** tab. The other tabs have their "
                   "own inputs.")

        mode = st.radio(
            "model", list(MODE_ORDER), index=0, horizontal=True, key="mode",
            help="Each model ranks the same papers differently. Start with bm25.")
        st.caption(ir.MODE_LABELS[mode])

        columns = st.columns([1, 1, 2])
        top_k = columns[0].slider("results to show", 1, 20, 10, key="top",
                                  help="How many rows to list.")

        k1, b = 1.2, 0.75
        if mode in {"bm25", "bm25f", "fields"}:
            k1 = columns[1].slider(
                "k1 (term-frequency saturation)", 0.0, 3.0, 1.2, 0.1, key="k1",
                help="k1 = 0 makes this the Binary Independence Model exactly. "
                     "Larger values let extra occurrences count for more.")
            b = columns[2].slider(
                "b (length normalisation)", 0.0, 1.0, 0.75, 0.05, key="b",
                help="b = 0 ignores paper length entirely. b = 1 normalises it "
                     "fully.")
            st.caption(f"`score = Σ idf(t) · tf(k₁+1) / "
                       f"(tf + k₁((1−b) + b·|d|/avgdl))` with "
                       f"k₁ = {k1:g}, b = {b:g}")
            if k1 == 0.0:
                st.info(
                    "**k₁ = 0**, so the saturation factor is exactly `tf/tf = 1` "
                    "and this mode **is** the Binary Independence Model. The test "
                    "suite asserts the two produce identical scores."
                )

        hierarchy = "paragraph"
        if mode in REGION_MODES:
            hierarchy = columns[1].selectbox(
                "region hierarchy", list(ir.HIERARCHY_NAMES), index=1,
                key="hierarchy",
                help="Which flat view of the text the list models use.")

        if mode in REGION_MODES:
            pool = options["proximal"] if mode == "proximal" else options["ranked"]
        elif mode in {"fields", "bm25f"}:
            pool = options["field"]
        else:
            pool = options["ranked"]

        example = st.selectbox("example query", ["choose one", *pool],
                               key=f"ex_{mode}", help="Fills the query box for you.")
        if example != "choose one":
            st.session_state[f"q_{mode}"] = example

    return {"mode": mode, "top": top_k, "k1": k1, "b": b,
            "hierarchy": hierarchy, "pool": pool}


def current_query(key: str, placeholder: str) -> str:
    return st.text_input("Query", key=key, placeholder=placeholder,
                         icon=":material/search:")


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------


def tab_search(options: dict) -> None:
    mode, top = options["mode"], options["top"]
    family = MODE_FAMILY.get(mode, "boolean")
    text, background = FAMILY_STYLE[family]
    section(f"{mode}: {ir.MODE_LABELS[mode]}")
    st.markdown(
        f'<span class="badge" style="color:{text}; background:{background}">'
        f'{family}</span>', unsafe_allow_html=True)
    lede("What this tab does",
         "Type a query and read the ranked list this model returns. Change the "
         "model in the bar above to see the same query answered a different way.",
         color=text)

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
            "`section with(2) retrieval`, "
            "`sentence before(10) index (paragraph)`, "
            "`paragraph child methodology`, `section parent(2) paragraph`."
        )

    query = current_query(f"q_{mode}", options["pool"][0])
    if not query:
        st.info("Type a query above, or pick one of the examples in the bar above.")
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
        st.caption("Recent queries: " + " | ".join(st.session_state["history"][:5]))


def tab_compare(options: dict) -> None:
    section("Fields and lists, side by side",
            "Two questions the same engine answers differently.")
    lede("Two questions, one engine",
         "Left: which field does a term live in? Right: do the query terms ever "
         "meet, and how strictly should that be enforced?")

    left, right = st.columns(2)

    with left, st.container(border=True):
        st.markdown("#### Structured field search")
        st.caption("`title:...` and `results:...` restrict the ranking to one "
                   "field. `bm25f` weights every field instead.")
        field_query = st.text_input("field query", key="q_fields_compare",
                                    placeholder="title:search",
                                    icon=":material/search:")
        design = st.radio("design", ["restrict", "filter", "bm25f"],
                          horizontal=True, key="design_compare",
                          help="Restrict scores only the field. Filter ranks the "
                               "whole paper, then discards. BM25F weights every "
                               "field.")
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
                st.info("Nothing matched.")
            else:
                largest = max((h.score for h in result.hits), default=0.0) or 1.0
                html_table(
                    ["rank", "paper", "field", "score", "matched"],
                    [[str(h.rank), f"<b>{h.doc_id}</b>", h.field,
                      meter(h.score / largest, "#008573", f"{h.score:.4f}"),
                      f"{h.matched}/{h.total_query_terms}"]
                     for h in result.hits], accent="#008573")
                st.caption(
                    "`restrict` and `filter` return the same papers in a "
                    "different order. Filter ranks the whole paper and then "
                    "discards, so the order reflects text outside the field."
                )

    with right, st.container(border=True):
        st.markdown("#### Overlapped vs non-overlapped")
        st.caption("Precision against coverage, over the chosen hierarchy.")
        list_query = st.text_input("list query", key="q_list_compare",
                                   placeholder="document index positions",
                                   icon=":material/search:")
        if list_query:
            show_comparison(list_query, options["top"], options["hierarchy"])


def tab_proximity(options: dict) -> None:
    section("Proximal node model")
    lede("Questions about the tree",
         "A paper is a tree: paper, then section, then paragraph, then sentence. "
         "These operators ask about that tree instead of whole documents.")

    query = current_query("q_prox_tab", "paragraph in methodology")
    if not query:
        st.info("Type an operator expression, for example "
                "`paragraph in methodology`.")
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
        st.info("No node satisfies that query.")
        return

    st.success(f"{result.count} node(s) in {len(result.papers)} papers")
    st.info("This model returns a **set** of nodes, not a ranking. There is no "
            "relevance score, so the order shown is document order.")

    top = options["top"]
    html_table(
        ["#", "paper", "node", "kind", "label", "start", "end", "terms", "text"],
        [[str(position), f"<b>{node.doc_id}</b>",
          f"<code>{node.region_id}</code>", node.kind, node.label,
          str(node.start), str(node.end), str(node.length),
          " ".join(node.text.split())[:160]]
         for position, node in enumerate(result.nodes[:top], 1)],
        accent="#0D74CE")

    if len(result.nodes) > top:
        st.caption(f"Showing {top} of {len(result.nodes)} nodes.")

    with st.expander("the operators this query language accepts",
                     icon=":material/help:"):
        st.markdown(
            "| operator | meaning | example |\n"
            "|---|---|---|\n"
            "| `P in Q` | P nodes inside some Q node | `paragraph in methodology` |\n"
            "| `P beginin Q` | only P's start inside Q | `sentence beginin results` |\n"
            "| `P with(k) T` | P contains at least k occurrences of T | "
            "`section with(2) retrieval` |\n"
            "| `P child Q` | direct children only | `paragraph child methodology` |\n"
            "| `P parent(k) Q` | P is parent of at least k Q nodes | "
            "`section parent(2) paragraph` |\n"
            "| `P before(k) T (C)` | P ends at most k terms before T, inside C | "
            "`sentence before(10) index (paragraph)` |\n"
            "| `P after(k) T (C)` | P starts at most k terms after T, inside C | "
            "`sentence after(5) index (paragraph)` |\n"
        )


def tab_evaluation() -> None:
    section("Effectiveness")
    lede("The scoreboard",
         "Twelve hand-made queries, judged by hand. This shows how to measure a "
         "model, not which model wins in general.")

    scores = cached_scores()
    best = max(scores, key=lambda s: s.mrr)
    worst = min(scores, key=lambda s: s.mrr)
    stat_row([
        ("best model", SHORT_NAMES.get(best.model, best.model),
         ":material/emoji_events:"),
        ("best MRR", f"{best.mrr:.3f}", ":material/trending_up:"),
        ("weakest MRR", f"{worst.mrr:.3f}", ":material/trending_down:"),
        ("spread", f"{best.mrr - worst.mrr:.3f}", ":material/straighten:"),
    ])

    overall = pd.DataFrame([{
        "model": SHORT_NAMES.get(s.model, s.model),
        "P@1": round(s.mean("precision", 1), 3),
        "P@5": round(s.mean("precision", 5), 3),
        "R@10": round(s.mean("recall", 10), 3),
        "MRR": round(s.mrr, 3),
        "MAP": round(s.map, 3),
        "queries with no relevant hit": len(s.queries_with_no_hit),
    } for s in scores])
    html_table(
        ["model", "P@1", "P@5", "R@10", "MRR", "MAP", "no hit"],
        [[f"<b>{SHORT_NAMES.get(s.model, s.model)}</b>",
          meter(s.mean("precision", 1), "#008573"),
          meter(s.mean("precision", 5), "#008573"),
          meter(s.mean("recall", 10), "#008573"),
          meter(s.mrr, "#6550B9"),
          meter(s.map, "#6550B9"),
          str(len(s.queries_with_no_hit))]
         for s in scores])

    st.markdown("#### Query by query, MRR per model")
    judgements = ir.get_judgements()
    models = [SHORT_NAMES.get(s.model, s.model) for s in scores]
    rows = []
    for query_id in judgements.query_ids:
        cells = [f"<b>{query_id}</b>", judgements.queries[query_id],
                 str(len(judgements.relevant[query_id]))]
        for score in scores:
            match = next(q for q in score.queries if q.query_id == query_id)
            rr = match.rr
            color = ("#218358" if rr >= 0.99
                     else "#CB1D63" if rr == 0 else "#60646C")
            cells.append(f'<span style="color:{color}; font-weight:650">'
                         f'{rr:.2f}</span>')
        rows.append(cells)
    html_table(["query", "text", "rel", *models], rows, accent="#6550B9")
    st.caption("1.00 means the correct paper came first. 0.00 means it never "
               "appeared at all. Read down a column to see which query type a "
               "model handles.")

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
    st.download_button("Download the table", overall.to_csv(index=False),
                       file_name="week02-evaluation.csv", mime="text/csv",
                       key="dl_eval")


def tab_corpus() -> None:
    section("Corpus and structure")
    lede("What was indexed",
         "The twenty papers, the terms the engine kept, and the region tree each "
         "paper is cut into.")

    engine = cached_engine()
    stats = ir.corpus_stats()

    stat_row([
        ("papers", str(stats["papers"]), ":material/description:"),
        ("total terms", f"{stats['total_terms']:,}", ":material/text_fields:"),
        ("distinct terms", f"{stats['vocabulary']:,}", ":material/tag:"),
        ("highest idf", f"{stats['idf_max']:.3f}", ":material/insights:"),
    ])
    st.caption(
        f"paper length {stats['length_min']} to {stats['length_max']} terms "
        f"(average {stats['avg_length']:.1f}) | "
        f"{stats['sections']} sections | {stats['paragraphs']} paragraphs | "
        f"{stats['sentences']} sentences"
    )

    st.markdown("#### Papers")
    html_table(
        ["paper", "title", "year", "venue", "terms", "fields", "paragraphs"],
        [[f"<b>{p.doc_id}</b>", p.title, str(p.year), p.venue, str(p.length),
          ", ".join(sorted(p.fields)),
          str(sum(len(v) for v in p.paragraphs.values()))]
         for p in engine.index.papers], accent="#0D74CE")

    st.markdown("#### Most frequent terms carry the least information")
    max_idf = max(engine.idf_table.idf_for(term)
                  for term, _ in stats["top_by_df"]) or 1.0
    html_table(
        ["term", "df", "share", "idf"],
        [[f"<b>{term}</b>", str(df),
          f'<span class="meter"><span style="width:'
          f'{round(df / stats["papers"] * 100)}%;'
          f'background:#0D74CE"></span></span>{df}',
          meter(engine.idf_table.idf_for(term) / max_idf, "#AB6400")]
         for term, df in stats["top_by_df"]], accent="#AB6400")

    st.markdown("#### Regions of one paper")
    doc_id = st.selectbox("paper", [p.doc_id for p in engine.index.papers],
                          key="corpus_paper",
                          help="Pick a paper to see its section, paragraph and "
                               "sentence regions.")
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
    with st.container(border=True):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def tab_pipeline() -> None:
    section("Pipeline walkthrough")
    lede("One sentence, the whole engine",
         "Watch a sentence become terms, then terms become a ranking. Every "
         "number here is computed live, not written down.")

    sentence = st.text_input("sentence", key="pipe_sentence",
                             value="Machine Learning improves clinic workflow "
                                   "in patient records",
                             icon=":material/edit_note:")
    if not sentence.strip():
        return

    from preprocess import preprocess_with_positions

    terms, positions = preprocess_with_positions(sentence)
    with st.container(border=True):
        st.markdown("#### 1  Preprocessing: positions, not just counts")
        st.markdown("`" + "`, `".join(f"{t}" for t in terms) + "`")
        st.caption(f"{len(terms)} terms at dense positions "
                   f"0 to {max(len(terms) - 1, 0)}. A phrase term keeps the "
                   f"position of its first word, which is what makes regions "
                   f"work. Week 1's `preprocess_text()` returns unigrams then "
                   f"phrases, so it loses that order.")

    query = st.text_input("run it as a query", key="pipe_query",
                          value=sentence, icon=":material/search:")
    if query.strip():
        from ranking import parse_query as parse_ranked_query

        parsed = parse_ranked_query(query, cached_engine().idf_table)
        columns = st.columns(2)
        for column, mode in zip(columns, ("bim", "bm25")):
            with column, st.container(border=True):
                response = ir.search(query, mode=mode, top=5)
                st.markdown(f"#### 2  {mode.upper()}")
                if not response.hits:
                    st.info("No paper shares a term with that sentence.")
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
# start here
# ---------------------------------------------------------------------------


def tab_start() -> None:
    section("Start here")
    lede("What this app is",
         "Type a query and watch several textbook search models answer it over "
         "the same 20 research papers. The papers never change, so every "
         "difference you see is the model.")

    st.markdown("#### How to read any screen")
    st.markdown(
        "1. **Set up** a query in the bar at the top: pick a model, the number "
        "of results, and an example if you want one.\n"
        "2. **Read the answer** in the main panel. The green line counts the "
        "results; the table is sortable.\n"
        "3. **Change one thing** (the model, or `k1`) and see what moves. That "
        "is the whole method of the week."
    )

    st.markdown("#### What each tab is for")
    html_table(
        ["tab", "purpose"],
        [["<b>Search</b>",
          "one query under any model; the k1 and b sliders live in the top bar"],
         ["<b>Fields &amp; Lists</b>",
          "search inside one field, and compare the two list models"],
         ["<b>Proximity</b>",
          "questions about the paper tree: paper, section, paragraph, sentence"],
         ["<b>Evaluation</b>",
          "the scoreboard: P@1, P@5, R@10, MRR, MAP for every model"],
         ["<b>Corpus</b>", "what got indexed, and how"],
         ["<b>Pipeline</b>",
          "one sentence through the whole engine, number by number"]],
        accent="#008573")

    st.markdown("#### The nine models, one line each")
    for mode, label in ir.MODE_LABELS.items():
        st.markdown(f"- **{mode}**: {label}")

    st.markdown("#### A five-minute tour")
    st.markdown(
        "1. **Search**: type `proximity window`. Paper **p04** comes first.\n"
        "2. Still on **Search**: drag **k1** to `0`. The app says it has become "
        "the Binary Independence Model, and the list matches the `bim` mode.\n"
        "3. **Fields & Lists**: run `title:search`, then flip *restrict* and "
        "*filter*. Same papers, different order.\n"
        "4. **Proximity**: run `section with(2) retrieval`. The answer is a "
        "*set* of nodes, not a ranking.\n"
        "5. **Evaluation**: BM25 sits at MRR `1.000`. Read the `nohit` column "
        "before the MRR column."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    inject_styles()
    engine = cached_engine()

    page_header(
        "Search the papers, several ways",
        "A query in, several ranking models out, over the same 20 papers. "
        "Change one setting and watch what moves.",
    )

    options = controls()

    tabs = st.tabs(["Start here", "Search", "Fields & Lists", "Proximity",
                    "Evaluation", "Corpus", "Pipeline"])
    with tabs[0]:
        tab_start()
    with tabs[1]:
        tab_search(options)
    with tabs[2]:
        tab_compare(options)
    with tabs[3]:
        tab_proximity(options)
    with tabs[4]:
        tab_evaluation()
    with tabs[5]:
        tab_corpus()
    with tabs[6]:
        tab_pipeline()


if __name__ == "__main__":
    main()
