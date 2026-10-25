"""Week 06 - fuzzy and extended Boolean search over the shared clinical corpus.

    python main.py --web
    streamlit run app.py

Every score, count and match here comes from fuzzy_search.py, which console_ui.py
also uses, so the browser and the terminal can never disagree. This file only
handles presentation.

Design notes
    No sidebar. The mode and the settings sit on the Search page, where the user
    can see what they change.

    The sections are chosen from a segmented bar, not st.tabs, because a tab cannot
    be switched from code. The example queries on Start here move the user to Search
    with the query filled in, so navigation actually navigates.

    Scores are shown as meters from 0 to 1, with the working behind each one in a
    "why" panel, so a reader can see how a number was reached.

Sections
    Start here   what the app is, the six requirements, and three example queries
    Search       the query box, the three modes, the ranked results and their working
    Compare      one query in all three modes, side by side
    Membership   mu of one word under the three membership functions, and partial matches
    Set view     (A OR B) AND NOT C checked against (A u B) - C
    Articles     the corpus: specialty, code, sections
"""

from __future__ import annotations

import html
import math
import re

import streamlit as st

from fuzzy_logic import plain_english
from fuzzy_search import MODES, Settings, get_engine
from membership import FUNCTIONS, DEFAULT_TAU
from query_parser import QueryError

st.set_page_config(page_title="Week 6 - Fuzzy search", layout="wide",
                   initial_sidebar_state="collapsed")

PAGES = ("Start here", "Search", "Compare", "Membership", "Set view", "Articles")
PAGE_ICONS = {"Start here": "home", "Search": "search", "Compare": "compare_arrows",
              "Membership": "equalizer", "Set view": "donut_large", "Articles": "menu_book"}

EXAMPLES = (
    "hypertension treatment",
    "hypertension AND NOT diabetes",
    "heart failure BUT NOT (asthma OR -fever)",
)

P_CHOICES = {"1 (average)": 1.0, "2": 2.0, "5": 5.0, "10": 10.0, "infinity (min/max)": math.inf}

PAGE_CSS = """
<style>
  .stApp { font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue",
           Arial, sans-serif;
           background-image:
             radial-gradient(60% 42% at 12% 0%, rgba(15, 118, 110, 0.055), transparent 62%);
           background-attachment: fixed; }
  /* the top bar is its own white strip, and the page bar sits clear of it */
  header[data-testid="stHeader"] { background: #FFFFFF;
    border-bottom: 1px solid #E5E5E1; }
  .block-container { padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1200px; }

  .hero { padding: 4px 0 12px 0; }
  .hero h1 { font-size: clamp(1.9rem, 3.4vw, 2.9rem); font-weight: 780;
             letter-spacing: -0.03em; margin: 0 0 10px 0; line-height: 1.07; }
  .hero p { font-size: 1.02rem; margin: 0; opacity: 0.78; max-width: 68ch; }
  .chips { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { font-size: 0.78rem; font-weight: 600; padding: 3px 11px;
          border-radius: 999px; border: 1px solid rgba(100, 116, 139, 0.35);
          background: var(--secondary-background-color); }

  .lede { border-left: 3px solid var(--primary-color);
          background: var(--secondary-background-color);
          border-radius: 4px 12px 12px 4px;
          padding: 12px 16px; margin: 2px 0 16px 0; }
  .lede .lede-title { display: block; font-weight: 700; font-size: 0.95rem; }
  .lede .lede-body { display: block; font-size: 0.9rem; margin-top: 2px; }

  .req { background: #FFFFFF; border: 1px solid #E5E5E1; border-radius: 14px;
         padding: 14px 16px; height: 100%; }
  .req .num { display: inline-block; width: 26px; height: 26px; line-height: 26px;
              text-align: center; border-radius: 50%; background: #0f766e;
              color: #FFFFFF; font-weight: 700; font-size: 0.85rem; }
  .req .rname { font-weight: 700; margin: 6px 0 3px 0; }
  .req .rbody { font-size: 0.86rem; color: #4B4F57; }

  /* a result row: rank, title, score meter, matched words */
  .result { background: #FFFFFF; border: 1px solid #E5E5E1; border-radius: 14px;
            padding: 12px 16px 10px 16px; margin: 0 0 10px 0; }
  .result .rtitle { font-weight: 700; font-size: 0.98rem; }
  .result .rcode { font-size: 0.78rem; color: #60646C; margin-left: 6px; }
  .result .rsub { font-size: 0.84rem; color: #4B4F57; margin-top: 4px; }
  .meter { display: inline-block; width: 150px; height: 8px; border-radius: 999px;
           background: rgba(100, 116, 139, 0.18); vertical-align: middle;
           margin-right: 8px; overflow: hidden; }
  .meter > span { display: block; height: 100%; background: #0f766e; }
  .score { font-weight: 700; font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; font-size: 0.74rem; font-weight: 700;
           padding: 2px 9px; border-radius: 999px; margin-right: 6px;
           background: #E0F8F3; color: #0f5f58; }
  .badge.partial { background: #FFF4D6; color: #6b4a00; }

  /* query words highlighted in the result list */
  mark { background: #FFF7C2; color: #4F3422; padding: 0 3px;
         border-radius: 4px; font-weight: 650; }

  /* page bar and model picker: a teal tint, filled teal when chosen */
  div[data-testid="stSegmentedControl"] button {
    background: #E6F4F1; color: #0f5f58; border: 1px solid #B9E4DC; font-weight: 600; }
  div[data-testid="stSegmentedControl"] button:hover { background: #CDEFE8; }
  div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: #0f766e; color: #FFFFFF; border-color: #0f766e; }

  /* buttons are filled teal, never white */
  .stButton button { background: #0f766e; color: #FFFFFF; border: 1px solid #0f766e;
    font-weight: 600; }
  .stButton button:hover, .stButton button:focus {
    background: #0b5d57; color: #FFFFFF; border-color: #0b5d57; }

  /* cards: a white surface lifting off the off-white page */
  div[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-radius: 14px; }

  /* expanders: tinted header, teal icon */
  div[data-testid="stExpander"] summary { background: #E6F4F1; border-radius: 12px; }
  div[data-testid="stExpander"] summary:hover { background: #D3EEE8; }
  span[data-testid="stIconMaterial"] { color: #0f766e; }

  /* text inputs: teal focus border */
  div[data-baseweb="input"]:focus-within { border-color: #0f766e !important; }

  /* coloured tables: tinted header, zebra rows, meters */
  .tablewrap { border: 1px solid #E5E5E1; border-radius: 14px; overflow: auto;
    background: #FFFFFF; margin: 4px 0 14px 0; }
  table.tt { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  table.tt thead th { background: #E6F4F1; color: #0f5f58; font-weight: 700;
    text-align: left; padding: 10px 12px; border-bottom: 2px solid #0f766e;
    white-space: nowrap; }
  table.tt tbody td { padding: 9px 12px; border-bottom: 1px solid #EDEDEA; }
  table.tt tbody tr:nth-child(even) { background: #FAFAF8; }
  table.tt tbody tr:hover { background: #F0F8F6; }

  .verdict { font-weight: 700; }
  div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E5E5E1;
          border-radius: 14px; padding: 14px 16px;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04); }
  div[data-testid="stMetricValue"] { font-size: 1.4rem; }
  code { background: var(--secondary-background-color); border-radius: 6px;
         padding: 1px 5px; }
  .stButton button { border-radius: 9px; }
  hr { margin: 1.6rem 0; }
</style>
"""


# ---------------------------------------------------------------------------
# small presentation helpers
# ---------------------------------------------------------------------------


def inject_styles() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    parts = [f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>']
    if chips:
        parts.append('<div class="chips">'
                     + "".join(f'<span class="chip">{c}</span>' for c in chips)
                     + "</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def html_table(headers: list[str], rows: list[list[str]]) -> None:
    """A table styled in the app's colours. Cells are already escaped HTML."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                   for row in rows)
    st.markdown(f'<div class="tablewrap"><table class="tt"><thead><tr>{head}</tr>'
                f'</thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def lede(title: str, body: str) -> None:
    st.markdown(f'<div class="lede"><span class="lede-title">{title}</span>'
                f'<span class="lede-body">{body}</span></div>', unsafe_allow_html=True)


def highlight(text: str, words: list[str]) -> str:
    """Escape the text, then mark each query word wherever it occurs."""
    escaped = html.escape(text)
    for word in sorted({w for w in words if len(w) > 2}, key=len, reverse=True):
        # mark the whole word that contains the stem, so "failur" marks "failure"
        pattern = re.compile(rf"\b\w*{re.escape(word)}\w*\b", re.IGNORECASE)
        escaped = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)
    return escaped


def meter(value: float) -> str:
    width = max(0.0, min(1.0, value)) * 100
    return f'<span class="meter"><span style="width:{width:.0f}%"></span></span>'


def use_example(query: str) -> None:
    st.session_state["query"] = query
    st.session_state["page"] = "Search"


# ---------------------------------------------------------------------------
# the engine and the settings
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Reading the clinical articles...")
def cached_engine():
    return get_engine()


def read_settings() -> Settings:
    """The settings the user chose on the Search page."""
    return Settings(
        mode=st.session_state.get("mode", "fuzzy"),
        implicit=st.session_state.get("implicit", "AND"),
        function=st.session_state.get("function", "log"),
        partial=st.session_state.get("partial", True),
        tau=st.session_state.get("tau", DEFAULT_TAU),
        p=P_CHOICES.get(st.session_state.get("p", "2"), 2.0),
        alpha=st.session_state.get("alpha", 0.0),
    )


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------


def start_here() -> None:
    engine = cached_engine()
    page_header(
        "Fuzzy search over clinical articles",
        "A search that gives every article a score between 0 and 1, instead of a "
        "yes or no. Partial words still count, and AND, OR and NOT still work.",
        [f"{len(engine.articles)} articles", "10 specialties", "3 modes"],
    )
    lede("What to do first",
         "Open Search, run one of these examples, and read the score bar. Then open "
         "the 'why' panel under a result to see how the score was reached.")

    cols = st.columns(len(EXAMPLES))
    for column, example in zip(cols, EXAMPLES):
        column.button(f"Try: {example}", key=f"ex_{example}", icon=":material/play_arrow:",
                      on_click=use_example, args=(example,))

    st.subheader("The six requirements")
    requirements = (
        ("User query input", "Type a query in plain words, with AND, OR, NOT and brackets."),
        ("Term matching", "Words match partly too: 'hypertenion' still reaches hypertension."),
        ("Relevance score 0 to 1", "Each article gets a membership value, shown as a meter."),
        ("Fuzzy AND / OR / NOT", "AND takes the weaker score, OR the stronger, NOT flips it."),
        ("Extended Boolean", "p-norm operators, plus exact include and exclude sets."),
        ("Ranked results", "Articles are sorted by score, with the matched words shown."),
    )
    icons = ("keyboard", "spellcheck", "speed", "join_inner", "tune", "format_list_numbered")
    for row in range(0, len(requirements), 3):
        cols = st.columns(3)
        for offset, column in enumerate(cols):
            index = row + offset
            name, body = requirements[index]
            with column.container(border=True):
                st.markdown(f":material/{icons[index]}: **{index + 1}. {name}**")
                st.caption(body)

    st.subheader("Three ways to score")
    st.markdown(
        "- **Boolean**: a word is either in an article or not. Results are unranked.\n"
        "- **Fuzzy**: a score for each word, combined with min (AND), max (OR) and 1 minus "
        "the score (NOT).\n"
        "- **Extended Boolean**: the same words, but the operators are p-norms, so a "
        "strong match on one word and a weak match on another can still score well."
    )


def search_page() -> None:
    engine = cached_engine()
    page_header("Search", "Type a query, pick a model, and read the ranked articles.")

    query = st.text_input("Query", key="query", placeholder="e.g. hypertension AND NOT diabetes",
                          label_visibility="collapsed")

    st.segmented_control("Model", MODES, default="fuzzy", key="mode",
                         format_func=lambda m: {"boolean": "Boolean",
                                                "fuzzy": "Fuzzy",
                                                "extended": "Extended Boolean"}[m])
    with st.expander("Settings", icon=":material/tune:"):
        left, right = st.columns(2)
        with left:
            st.radio("Two words side by side mean", ("AND", "OR"), horizontal=True,
                     key="implicit")
            st.selectbox("Membership function", FUNCTIONS, index=2, key="function",
                         help="binary = present or not; linear = proportional; "
                              "log = damps repeated words.")
            st.toggle("Partial matching (fuzzy word matches)", value=True, key="partial")
        with right:
            st.slider("Partial match threshold (tau)", 0.3, 1.0, DEFAULT_TAU, 0.05,
                      key="tau", help="Lower accepts looser spellings.")
            st.select_slider("p-norm exponent p (extended only)", list(P_CHOICES),
                             value="2", key="p")
            st.slider("Alpha cut (extended only)", 0.0, 0.9, 0.0, 0.05, key="alpha",
                      help="An article is in a word's set when its membership is above alpha.")

    settings = read_settings()
    if not query.strip():
        st.info("Type a query above, or pick an example on Start here.")
        return

    response = engine.search(query, settings)
    if not response.ok:
        st.error(response.message, icon=":material/error:")
        return

    st.markdown(f"Parsed as **{html.escape(response.parsed)}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Articles", f"{response.count} of {response.considered}",
              border=True, icon=":material/description:")
    c2.metric("Words", str(len(response.terms)), border=True, icon=":material/search:")
    c3.metric("Model", settings.mode.title(), border=True, icon=":material/functions:")

    if not response.hits:
        st.warning("No article matches this query. Try a shorter query, or turn on "
                   "partial matching.", icon=":material/search_off:")
        return

    words = list(response.terms)
    for rank, hit in enumerate(response.hits[:10], start=1):
        badges = "".join(
            f'<span class="badge partial">{html.escape(m.matched_as)}</span>'
            for m in hit.matches
            if m.matched_as and m.matched_as != m.stem)
        exact = "" if badges else '<span class="badge">exact words</span>'
        st.markdown(
            f'<div class="result"><div><span class="rtitle">{rank}. '
            f'{html.escape(hit.title)}</span><span class="rcode">{hit.code} '
            f'&middot; {html.escape(hit.specialty)}</span></div>'
            f'<div class="rsub">{meter(hit.score)}<span class="score">{hit.score:.2f}'
            f'</span> &nbsp; {exact}{badges}</div></div>',
            unsafe_allow_html=True)
        with st.expander(f"why {hit.score:.2f}", icon=":material/insights:"):
            st.caption("Each bullet is one step. 'All must match' takes the lowest score, "
                       "'any one is enough' takes the highest, and NOT flips a score.")
            st.markdown("\n".join(plain_english(hit.working)))
            article = engine.by_id[hit.doc_id]
            overview = article.sections[0].text
            st.markdown(highlight(overview, words), unsafe_allow_html=True)
    if response.count > 10:
        st.caption(f"Showing the top 10 of {response.count}.")


def compare_page() -> None:
    engine = cached_engine()
    page_header("Compare", "One query in all three models, side by side.")
    query = st.text_input("Query to compare", key="cmp_query",
                          value=st.session_state.get("query", "asthma OR COPD"))
    if not query.strip():
        st.info("Type a query to compare.")
        return
    results = engine.compare(query, read_settings())
    columns = st.columns(3)
    labels = {"boolean": "Boolean", "fuzzy": "Fuzzy", "extended": "Extended Boolean"}
    for column, mode in zip(columns, MODES):
        response = results[mode]
        with column:
            st.subheader(labels[mode])
            if not response.ok:
                st.error(response.message)
                continue
            st.metric("Articles", f"{response.count}", border=True)
            if not response.hits:
                st.caption("No article matches.")
                continue
            for hit in response.hits[:5]:
                st.markdown(f"**{html.escape(hit.doc_id)}** {html.escape(hit.title)} "
                            f"&middot; {hit.score:.2f}", unsafe_allow_html=True)
    lede("What to look for",
         "Boolean returns the exact words only. Fuzzy and Extended Boolean also return "
         "partial matches, and they rank them. Extended Boolean uses the p-norm for the "
         "order, so the order can differ from Fuzzy.")


def membership_page() -> None:
    engine = cached_engine()
    page_header("Membership", "How strongly one word belongs to each article.")
    word = st.text_input("Word", key="mem_word", value="hypertension",
                         placeholder="e.g. hypertenion (typo)")
    if not word.strip():
        st.info("Type one word.")
        return
    try:
        rows = {fn: engine.membership_table(word, fn) for fn in FUNCTIONS}
    except QueryError as error:
        st.error(str(error))
        return

    st.subheader("Top articles under each membership function")
    lede("Reading the table",
         "Each column is mu, a value from 0 to 1. Binary gives 1 to every article that "
         "contains the word. Linear and log scale by how often the word appears, and "
         "the article that uses it most gets 1.")
    body = []
    for doc_id, title, _ in rows["log"][:10]:
        cells = [f"<strong>{html.escape(doc_id)}</strong> {html.escape(title)}"]
        for fn in FUNCTIONS:
            mu = next(value for d, _, value in rows[fn] if d == doc_id)
            cells.append(f'{meter(mu)}<span class="score">{mu:.2f}</span>')
        body.append(cells)
    html_table(["Article", *FUNCTIONS], body)

    matches = engine.partial_matches(word)
    st.subheader("Partial matches")
    if matches:
        st.write("This word is matched as: " + ", ".join(
            f"`{stem}` ({similarity:.2f})" for stem, similarity in matches))
    else:
        st.write("No partial match above the threshold.")


def set_page() -> None:
    engine = cached_engine()
    page_header("Set view", "Lecture section 3: OR is union, AND NOT is difference.")
    left, middle, right = st.columns(3)
    a = left.text_input("A", key="set_a", value="diabetes")
    b = middle.text_input("B", key="set_b", value="hypertension")
    c = right.text_input("C", key="set_c", value="kidney")
    try:
        view = engine.set_view(a, b, c, read_settings())
    except QueryError as error:
        st.error(str(error))
        return

    cols = st.columns(4)
    cols[0].metric("A", len(view["A"]), border=True)
    cols[1].metric("A or B (union)", len(view["union"]), border=True)
    cols[2].metric("C", len(view["C"]), border=True)
    cols[3].metric("(A or B) minus C", len(view["result"]), border=True)

    st.markdown(f"**A or B minus C** is {', '.join(sorted(view['result'])) or 'empty'}")
    if view["equal"]:
        st.success("The query '((A) OR (B)) AND NOT (C)' returns the same set.",
                   icon=":material/check_circle:")
    else:
        st.error("The query and the set arithmetic disagree. This should never happen.",
                 icon=":material/error:")


def articles_page() -> None:
    engine = cached_engine()
    page_header("Articles", "The shared corpus, one clinical article at a time.",
                [f"{len(engine.articles)} articles", "4 sections each"])
    ids = [a.doc_id for a in engine.articles]
    chosen = st.selectbox("Article", ids, key="article_id",
                          format_func=lambda d: f"{d}  {engine.by_id[d].title}")
    article = engine.by_id[chosen]
    st.subheader(article.title)
    st.caption(f"{article.specialty} | ICD-10 {article.code} | updated {article.updated}")
    for section in article.sections:
        with st.expander(section.name, expanded=section.name == "Overview"):
            st.write(section.text)


# ---------------------------------------------------------------------------
# the page bar
# ---------------------------------------------------------------------------


def main() -> None:
    inject_styles()
    st.session_state.setdefault("page", PAGES[0])
    page = st.segmented_control("Section", PAGES, key="page",
                                label_visibility="collapsed",
                                format_func=lambda p: f":material/{PAGE_ICONS[p]}: {p}")
    if page is None:
        page = PAGES[0]
    st.divider()
    {
        "Start here": start_here,
        "Search": search_page,
        "Compare": compare_page,
        "Membership": membership_page,
        "Set view": set_page,
        "Articles": articles_page,
    }[page]()


main()
