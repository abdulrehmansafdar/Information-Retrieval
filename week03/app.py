"""Week 03 - FYP browsing / navigation architecture, web interface.

    python main.py --web
    streamlit run app.py

Every number and every result here is produced by browsing.py, which console_ui.py
also uses, so the browser and the terminal can never disagree. This file only
handles presentation.

Design notes
    No sidebar. The page bar, the breadcrumb and the filters all sit along the
    top, so the whole screen belongs to the content.

    The sections are chosen from a segmented bar rather than st.tabs, because a
    tab cannot be switched from code: a "browse" button on the Home page changed
    the state but left the user staring at the Home page. A page selector can be
    driven from a button, so navigation actually navigates.

    Every document list ends each row with a **preview** button, and the document
    opens inline, directly under its row, instead of being sent to another
    section. The Document section still exists for the full page.

Sections
    Start here  what the app is, and the seven things it demonstrates
    Home        what the collection holds, and the three browsing models
    Browse      the category hierarchy, breadcrumb and document list, with filters
    Search      the search box, with results that preview inline
    Document    the full preview: sections, related links, internal links
    Hypertext   H = (N, L): the graph, a node's links, and shortest paths
    History     the navigation history and the recently viewed list
"""

from __future__ import annotations

import html
import re

import streamlit as st

import browsing as browse
from browsing import Filters, NavigationState
from hypertext import document_node

st.set_page_config(page_title="Week 3 - Browsing", layout="wide",
                   initial_sidebar_state="expanded")

PAGES = ("Start here", "Home", "Browse", "Search", "Document", "Hypertext",
         "History")

PAGE_CSS = """
<style>
  .stApp { font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue",
           Arial, sans-serif;
           background-image:
             radial-gradient(60% 42% at 12% 0%, rgba(15, 118, 110, 0.055), transparent 62%);
           background-attachment: fixed; }
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

  .hero { padding: 4px 0 12px 0; }
  .hero h1 { font-size: clamp(1.9rem, 3.4vw, 2.9rem); font-weight: 780;
             letter-spacing: -0.03em; margin: 0 0 10px 0; line-height: 1.07; }
  .hero p { font-size: 1.02rem; margin: 0; opacity: 0.78; max-width: 68ch; }
  .chips { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { font-size: 0.78rem; font-weight: 600; padding: 3px 11px;
          border-radius: 999px; border: 1px solid rgba(100, 116, 139, 0.35);
          background: var(--secondary-background-color); }

  /* coloured category cards and inline badges */
  .dcard { border: 1px solid; border-left-width: 5px; border-radius: 12px;
           padding: 12px 14px 10px 14px; }
  .dcard .dname { font-size: 0.95rem; font-weight: 700; line-height: 1.25; }
  .dcard .dcount { font-size: 1.8rem; font-weight: 780; line-height: 1.2;
                   margin-top: 2px; }
  .dcard .dlabel { font-size: 0.74rem; text-transform: uppercase;
                   letter-spacing: 0.09em; color: #60646C; }
  .badge { display: inline-block; font-size: 0.74rem; font-weight: 700;
           padding: 2px 9px; border-radius: 999px; margin-right: 6px; }
  .catdot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            margin-right: 7px; vertical-align: middle; }

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

  /* the query words, wherever they occur */
  mark { background: #FFF7C2; color: #4F3422; padding: 0 3px;
         border-radius: 4px; font-weight: 650; }
  .meta { font-size: 0.85rem; color: #60646C; margin: -2px 0 10px 0; }

  /* inline hypertext links, the H = (N, L) graph rendered as prose */
  a.ilink { color: #0f766e; text-decoration: underline;
            text-underline-offset: 2px; font-weight: 600; }
  a.ilink:hover { background: #E0F8F3; border-radius: 4px; }

  .lede { border-left: 3px solid var(--primary-color);
          background: var(--secondary-background-color);
          border-radius: 4px 12px 12px 4px;
          padding: 12px 16px; margin: 2px 0 16px 0; }
  .lede .lede-title { display: block; font-weight: 700; font-size: 0.95rem; }
  .lede .lede-body { display: block; font-size: 0.9rem; margin-top: 2px; }

  /* breadcrumb: one line, chevron separators, subtle until hovered */
  .crumbsep { color: #94a3b8; font-weight: 400; padding: 0 6px;
              font-size: 1.05rem; line-height: 1; }
  .crumbnow { font-weight: 700; color: var(--primary-color);
              padding: 6px 0 0 4px; }
  .st-key-crumbs div[data-testid="stButton"] button {
      background: transparent; border: none; box-shadow: none;
      color: #64748b; font-size: 0.98rem; font-weight: 500;
      padding: 0 2px; min-height: 0; height: auto; }
  .st-key-crumbs div[data-testid="stButton"] button:hover {
      color: var(--primary-color); text-decoration: underline;
      background: transparent; }

  h1, h2, h3 { letter-spacing: -0.015em; }
  h3 { margin-top: 0.4rem; }

  div[data-testid="stMetric"] { background: #FFFFFF;
          border: 1px solid #E5E5E1;
          border-radius: 14px; padding: 14px 16px;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04); }
  div[data-testid="stMetricValue"] { font-size: 1.4rem; }
  div[data-testid="stDataFrame"] { border: 1px solid rgba(100, 116, 139, 0.22);
          border-radius: 12px; overflow: hidden; }

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


def lede(title: str, body: str) -> None:
    """The one-line statement of purpose that opens every section."""
    st.markdown(
        f'<div class="lede"><span class="lede-title">{title}</span>'
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
    white. For the tables that carry meaning (the scoreboard, the categories,
    the link kinds) this renders real HTML instead: tinted header, zebra rows,
    and an accent taken from the section.
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


def meter(value: float, color: str) -> str:
    """A small coloured bar plus the number, for a 0..1 metric."""
    percent = max(0, min(100, round(value * 100)))
    return (f'<span class="meter"><span style="width:{percent}%;'
            f'background:{color}"></span></span>{value:.3f}')


def annotate(text: str, query_terms: list[str] | tuple[str, ...] | None = None,
             links: dict[str, str] | None = None) -> str:
    """Escape `text`, mark the query words, and turn link words into anchors.

    Both substitutions happen in one pass, so a word is never wrapped twice and
    no tag can end up inside another tag. A query word wins over a link, because
    the user is already looking for it.
    """
    escaped = html.escape(text)
    query = {term.lower() for term in (query_terms or []) if term}
    link_map = {term.lower(): target for term, target in (links or {}).items()
                if term and term.lower() not in query}
    vocabulary = sorted(set(query) | set(link_map), key=len, reverse=True)
    if not vocabulary:
        return escaped

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in vocabulary) + r")\b",
        re.IGNORECASE)

    def replace(match: re.Match) -> str:
        word = match.group(0).lower()
        if word in link_map:
            return (f'<a class="ilink" href="?doc={link_map[word]}">'
                    f'{match.group(0)}</a>')
        return f"<mark>{match.group(0)}</mark>"

    return pattern.sub(replace, escaped)


def highlight(text: str, terms: list[str] | tuple[str, ...] | None) -> str:
    """The query-only case, kept for the result titles."""
    return annotate(text, terms)


def inline_links(doc_id: str, limit: int = 8) -> dict[str, str]:
    """Term -> related document, for the inline hypertext links.

    The links are DERIVED, not authored: there are no citations in this corpus,
    so a word is linked when it occurs in both this paper and a related one and
    is distinctive enough to mean something (idf above 1, so it is in fewer than
    about a third of the papers). That is stated as such in the docs and the
    viva, rather than pretending the papers cite each other.
    """
    engine_obj = engine()
    own = set(engine_obj.by_id[doc_id].terms)
    links: dict[str, str] = {}
    for other, _title, _score in engine_obj.related(doc_id, 3):
        shared = own & set(engine_obj.by_id[other].terms)
        candidates = sorted(
            (term for term in shared
             if len(term) >= 6
             and engine_obj.idf.get(term, 0.0) > 1.0
             and term not in links),
            key=lambda term: -engine_obj.idf.get(term, 0.0))
        for term in candidates[:3]:
            links[term] = other
        if len(links) >= limit:
            break
    return dict(list(links.items())[:limit])


@st.cache_resource(show_spinner="Loading the collection...")
def engine() -> browse.BrowseEngine:
    return browse.get_engine()


def session() -> dict:
    """The per-user browsing state: where we are, what is filtered."""
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "nav" not in st.session_state:
        st.session_state.nav = NavigationState()
    if "node" not in st.session_state:
        st.session_state.node = "root"
    if "doc" not in st.session_state:
        st.session_state.doc = ""
    if "preview_open" not in st.session_state:
        st.session_state.preview_open = ""
    if "preview_terms" not in st.session_state:
        st.session_state.preview_terms = []
    if "filters" not in st.session_state:
        st.session_state.filters = Filters()
    if "query" not in st.session_state:
        st.session_state.query = ""
    return st.session_state


# --- navigation ------------------------------------------------------------


def go(node_id: str) -> None:
    st.session_state.node = node_id
    st.session_state.nav.visit(node_id)


def goto(page: str, node_id: str | None = None) -> None:
    """Move to a section, optionally changing the node first.

    Written as an on_click callback, never called inline: the state changes at
    the start of the next run, before the page selector is built, which is the
    one moment Streamlit allows a widget's value to be set from code.
    """
    if node_id:
        go(node_id)
    st.session_state.page = page


def show_on_document_page(doc_id: str) -> None:
    """Open a document as a full page in the Document section."""
    st.session_state.doc = doc_id
    st.session_state.node = document_node(doc_id)
    st.session_state.nav.visit(document_node(doc_id))
    st.session_state.page = "Document"


def go_back() -> None:
    st.session_state.node = st.session_state.nav.back()


def go_forward() -> None:
    st.session_state.node = st.session_state.nav.forward()


def criteria(filters: Filters) -> str:
    return filters.describe()


# ---------------------------------------------------------------------------
# colour: one hue per domain, taken from the Radix Colors scales
# ---------------------------------------------------------------------------
#
# Radix gives every hue a 12-step scale and assigns each step a job. Step 3 is a
# subtle background, step 6 a subtle border, step 9 the solid fill, and step 11
# low-contrast text that is contrast-checked on a step-2 background. Using those
# steps means the domain colours are systematic rather than picked by eye.
#
#   https://www.radix-ui.com/colors/docs/palette-composition/understanding-the-scale

DOMAIN_STYLE = {
    "root/ir": ("#008573", "#E0F8F3", "#A1DED2"),          # teal
    "root/ml": ("#6550B9", "#F4F0FE", "#D4CAFE"),          # violet
    "root/applied": ("#AB6400", "#FFF7C2", "#F3D673"),     # amber
    "root/systems": ("#0D74CE", "#E6F4FE", "#ACD8FC"),     # blue
    "root/evaluation": ("#CB1D63", "#FFE9F0", "#F3BED1"),  # crimson
    "root/hci": ("#218358", "#E6F6EB", "#ADDDC0"),         # green
}
FALLBACK_STYLE = ("#60646C", "#F0F0F3", "#D9D9E0")         # slate

LINK_KIND_STYLE = {
    "hierarchy": ("#60646C", "#F0F0F3"),
    "internal": ("#0D74CE", "#E6F4FE"),
    "related": ("#008573", "#E0F8F3"),
    "shared-venue": ("#AB6400", "#FFF7C2"),
    "shared-year": ("#6550B9", "#F4F0FE"),
}


def domain_style(node_id: str) -> tuple[str, str, str]:
    """(text, background, border) for a taxonomy node's domain."""
    if node_id in DOMAIN_STYLE:
        return DOMAIN_STYLE[node_id]
    parts = node_id.split("/")
    if len(parts) >= 2:
        return DOMAIN_STYLE.get("/".join(parts[:2]), FALLBACK_STYLE)
    return FALLBACK_STYLE


def domain_of(doc_id: str) -> str:
    return engine().taxonomy.category_of(doc_id).parent_id


def trail_for_current_node() -> tuple[list, str]:
    """The taxonomy trail to the current node, and the document, if any."""
    taxonomy = engine().taxonomy
    graph = engine().graph
    node = st.session_state.node
    if node in taxonomy:
        return taxonomy.breadcrumb(node), ""
    if node in graph and graph.find(node).doc_id:
        document = graph.find(node).doc_id
        return taxonomy.breadcrumb(taxonomy.leaf_of(document)), document
    return taxonomy.breadcrumb("root"), ""


# ---------------------------------------------------------------------------
# breadcrumb and controls
# ---------------------------------------------------------------------------


def breadcrumb_bar() -> None:
    """A clickable trail, one line, chevron separated.

    Every crumb above you jumps straight to that category; the crumb you are
    standing on is plain text. Rendered in a keyed horizontal container so the
    styles in PAGE_CSS can make the buttons read as links rather than buttons.
    """
    trail, document = trail_for_current_node()
    with st.container(horizontal=True, key="crumbs"):
        for position, category in enumerate(trail):
            if position:
                st.markdown('<span class="crumbsep">&#8250;</span>',
                            unsafe_allow_html=True)
            last = position == len(trail) - 1 and not document
            if last:
                text, _, _ = domain_style(category.node_id)
                st.markdown(
                    f'<span class="crumbnow" style="color:{text}">'
                    f'{category.name}</span>', unsafe_allow_html=True)
            else:
                st.button(category.name, key=f"crumb_{category.node_id}",
                          type="tertiary", on_click=goto,
                          args=("Browse", category.node_id))
        if document:
            st.markdown('<span class="crumbsep">&#8250;</span>',
                        unsafe_allow_html=True)
            st.markdown(f'<span class="crumbnow">{document}</span>',
                        unsafe_allow_html=True)


def sidebar(state) -> None:
    """The filters, and the history controls, kept off the page itself.

    In the sidebar they are always at hand without pushing the content down, and
    every control in here applies to whatever section you are standing on.
    """
    engine_obj = engine()
    with st.sidebar:
        st.markdown("### Filters")
        st.caption("These apply to every document list in the app.")

        facets = engine_obj.filters_available()
        domains = ["(all)"] + list(facets["domain_nodes"])
        domain_name = st.selectbox("domain", domains, key="f_domain")
        domain_id = "" if domain_name == "(all)" else \
            facets["domain_nodes"][domain_name]

        subtopic_name = "(all)"
        if domain_id:
            subtopics = [n.name for n in engine_obj.taxonomy.subtopics(domain_id)]
            subtopic_name = st.selectbox("subtopic", ["(all)"] + subtopics,
                                         key="f_subtopic")
        years = ["(any)"] + list(facets["year"])
        year = st.selectbox("year", years, key="f_year")
        venues = ["(any)"] + list(facets["venue"])
        venue = st.selectbox("venue", venues, key="f_venue")

        subtopic_id = ""
        if subtopic_name != "(all)":
            subtopic_id = next(
                n.node_id for n in engine_obj.taxonomy.subtopics(domain_id)
                if n.name == subtopic_name)
        st.session_state.filters = Filters(
            domain=domain_id, subtopic=subtopic_id,
            year=int(year) if year != "(any)" else 0,
            venue="" if venue == "(any)" else venue)
        if st.session_state.filters.active:
            st.caption(f"active: {criteria(st.session_state.filters)}")

        st.divider()
        st.markdown("### Move around")
        st.caption("Step through the pages you have visited, or use the "
                   "breadcrumb at the top of the page.")
        st.button("back", disabled=not st.session_state.nav.can_go_back(),
                  on_click=go_back, use_container_width=True,
                  icon=":material/arrow_back:",
                  help="Step back through the pages you have visited.")
        st.button("forward", disabled=not st.session_state.nav.can_go_forward(),
                  on_click=go_forward, use_container_width=True,
                  icon=":material/arrow_forward:",
                  help="Step forward again, if you have gone back.")


# ---------------------------------------------------------------------------
# document rows and the inline preview
# ---------------------------------------------------------------------------


def inline_preview(doc_id: str) -> None:
    """The document, opened under its own row.

    Query words are marked, and distinctive words that also occur in a related
    paper become real links (?doc=...), so the preview is navigable the way a
    wiki page is.
    """
    preview = engine().preview(doc_id)
    terms = st.session_state.get("preview_terms", [])
    links = inline_links(doc_id)
    meta = (f"{preview.doc_id} | {preview.year} | {preview.venue} | "
            f"{preview.category}")
    with st.container(border=True):
        head = st.columns([5, 1])
        with head[0]:
            st.markdown(f"#### {highlight(preview.title, terms)}",
                        unsafe_allow_html=True)
            st.markdown(f'<div class="meta">{highlight(meta, terms)}</div>',
                        unsafe_allow_html=True)
        head[1].button("full page", key=f"full_{doc_id}",
                       on_click=show_on_document_page, args=(doc_id,),
                       use_container_width=True, icon=":material/open_in_new:",
                       help="Open this document in the Document section.")
        for name, text in preview.sections:
            with st.expander(name, expanded=(name.lower() == "abstract"),
                             icon=":material/article:"):
                st.markdown(annotate(text, terms, links), unsafe_allow_html=True)
        if preview.related:
            st.markdown("**See also**")
            for other, title, score in preview.related:
                st.markdown(
                    f'<a class="ilink" href="?doc={other}">{other}</a> '
                    f'<span style="color:#60646C">{title} ({score:.2f})</span>',
                    unsafe_allow_html=True)


def document_rows(rows: list[tuple], prefix: str,
                  terms: list[str] | None = None) -> None:
    """One row per document, ending in a preview button.

    rows: (doc_id, title, category, score_text or None).
    `terms` are the query words to highlight, so a search result shows why it
    matched. A click toggles the preview under that row; nothing is sent to
    another section unless you ask for the full page.
    """
    for doc_id, title, category, score in rows:
        columns = (st.columns([1, 4.5, 2, 1.1, 1.2]) if score is not None
                   else st.columns([1, 5, 2.2, 1.2]))
        columns[0].markdown(f"`{doc_id}`")
        columns[1].markdown(highlight(title, terms), unsafe_allow_html=True)
        text, _, _ = domain_style(domain_of(doc_id))
        columns[2].markdown(
            f'<span class="catdot" style="background:{text}"></span>'
            f'<span style="color:{text}; font-size:0.84rem; font-weight:600">'
            f'{category}</span>', unsafe_allow_html=True)
        if score is not None:
            columns[3].markdown(f"`{score}`")
        open_now = st.session_state.get("preview_open") == doc_id
        if columns[-1].button("hide" if open_now else "preview",
                              key=f"{prefix}_preview_{doc_id}",
                              use_container_width=True, type="primary",
                              icon=(":material/visibility_off:" if open_now
                                    else ":material/visibility:")):
            if open_now:
                st.session_state.preview_open = ""
                st.session_state.preview_terms = []
            else:
                st.session_state.preview_open = doc_id
                st.session_state.doc = doc_id
                st.session_state.preview_terms = list(terms or [])
            st.rerun()
        if st.session_state.get("preview_open") == doc_id:
            inline_preview(doc_id)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def tab_home(state) -> None:
    section("Home")
    lede("What the collection holds",
         "Twenty papers, grouped into six domains that were derived from their "
         "own titles and abstracts.")

    home = engine().home(st.session_state.nav)
    stats = home["stats"]

    stat_row([
        ("documents", str(stats["documents"]), ":material/description:"),
        ("sections", str(stats["sections"]), ":material/segment:"),
        ("words", f"{stats['words']:,}", ":material/text_fields:"),
        ("domains", str(len(home["domains"])), ":material/account_tree:"),
    ])
    st.caption(
        f"{len(stats['years'])} years ({stats['years'][0]} to {stats['years'][-1]}) "
        f"| {len(stats['venues'])} venues | {stats['vocabulary']:,} distinct terms "
        f"| sections: {', '.join(stats['section_names'])}")

    st.markdown("#### Categories")
    st.caption("Pick a domain to browse into it. That opens the Browse section "
               "at that domain.")
    for row in [home["domains"][i:i + 3] for i in range(0, len(home["domains"]), 3)]:
        columns = st.columns(3)
        for column, domain in zip(columns, row):
            text, background, border = domain_style(domain["node_id"])
            with column:
                st.markdown(
                    f'<div class="dcard" style="background:{background};'
                    f'border-color:{border};">'
                    f'<div class="dname" style="color:{text}">'
                    f'{domain["name"]}</div>'
                    f'<div class="dcount" style="color:{text}">'
                    f'{domain["count"]}</div>'
                    f'<div class="dlabel">documents</div></div>',
                    unsafe_allow_html=True)
                st.button("browse", key=f"home_{domain['node_id']}",
                          on_click=goto, args=("Browse", domain["node_id"]),
                          use_container_width=True, type="primary",
                          icon=":material/arrow_forward:")

    if home["recent"]:
        st.markdown("#### Recently viewed")
        st.markdown('<div class="chips">'
                    + "".join(f'<span class="chip">{r["label"]}</span>'
                              for r in home["recent"])
                    + "</div>", unsafe_allow_html=True)

    st.markdown("#### The three browsing models")
    st.caption("Flat and structured-guided are sections here. The hypertext "
               "model is the graph in the Hypertext section.")
    model_colors = ("#008573", "#6550B9", "#AB6400")
    html_table(
        ["model", "idea", "best for", "in this app"],
        [[f'<b style="color:{color}">{model["name"]}</b>',
          model["idea"], model["best_for"], model["here"]]
         for model, color in zip(home["models"], model_colors)])


def tab_browse(state) -> None:
    section("Browse the hierarchy",
            "Move down the tree one level at a time.")
    lede("Walk the categories",
         "The trail at the top is the breadcrumb. Each document row has a "
         "preview button that opens the paper underneath the row.")

    engine_obj = engine()
    node = st.session_state.node if st.session_state.node in engine_obj.taxonomy \
        else "root"

    if st.session_state.filters.active:
        st.caption(f"filters: {criteria(st.session_state.filters)}")

    node_obj = engine_obj.taxonomy.find(node)
    if node_obj.children:
        st.markdown("#### " + ("Domains" if node == "root" else "Subtopics"))
        columns = st.columns(min(len(node_obj.children), 4))
        for column, child_id in zip(columns, node_obj.children):
            child = engine_obj.taxonomy.find(child_id)
            with column, st.container(border=True):
                st.metric(child.name, child.count, border=False)
                label = "open" if node != "root" else "browse"
                st.button(label, key=f"br_{child_id}", on_click=goto,
                          args=("Browse", child_id), use_container_width=True,
                          type="primary", icon=":material/arrow_forward:")
        up = node_obj.parent_id
        if up:
            st.button("up to " + engine_obj.taxonomy.find(up).name,
                      on_click=goto, args=("Browse", up))

    summaries = engine_obj.documents_in(node, st.session_state.filters)
    st.markdown(f"#### Documents ({len(summaries)})")
    if summaries:
        document_rows([(s.doc_id, s.title, s.category, None)
                       for s in summaries], "browse")
    else:
        st.info("No documents here. Try clearing a filter in the bar above.")


def tab_search(state) -> None:
    section("Search, then browse")
    lede("A way into the collection",
         "Type a query, then press **preview** on any row to open the paper right "
         "under it.")

    query = st.text_input("query", value=st.session_state.query,
                          placeholder="proximity window", key="search_box",
                          icon=":material/search:")
    st.session_state.query = query
    if not query.strip():
        st.info("Type a query above. For example: `proximity window`.")
        return

    response = engine().search(query, st.session_state.filters, top=20)
    if response.error:
        st.error(response.error)
        return
    if response.filters.active:
        st.caption(f"filters: {criteria(response.filters)}")
    if not response.hits:
        st.info("Nothing matched. Try a single distinctive word.")
        return

    st.success(f"{response.count} results in {response.considered} candidate "
               f"documents. Query words are highlighted below.")
    document_rows([(h.doc_id, h.title, h.category, f"{h.score:.3f}")
                   for h in response.hits], "search", terms=response.terms)


def tab_document(state) -> None:
    section("Document preview")
    doc_id = st.session_state.doc
    if not doc_id:
        st.info("No document open yet. Open one from Browse or Search with its "
                "preview button, or pick one below.")
        options = [d.doc_id for d in engine().documents]
        chosen = st.selectbox("document", options, key="doc_pick")
        if st.button("open", key="doc_pick_btn", type="primary",
                     icon=":material/open_in_new:"):
            show_on_document_page(chosen)
            st.rerun()
        return
    if doc_id not in engine().by_id:
        st.error(f"No such document: {doc_id}")
        return

    preview = engine().preview(doc_id)
    lede(preview.doc_id, f"{preview.year} | {preview.venue} | {preview.category}")
    st.header(preview.title)

    terms = st.session_state.get("preview_terms", [])
    links = inline_links(doc_id)
    left, right = st.columns([2, 1])
    with left, st.container(border=True):
        st.markdown("#### The paper")
        st.caption("Underlined words are hypertext links to the paper they "
                   "share that topic with. Click one to follow it.")
        for name, text in preview.sections:
            with st.expander(name, expanded=(name.lower() == "abstract"),
                             icon=":material/article:"):
                st.markdown(annotate(text, terms, links), unsafe_allow_html=True)
    with right, st.container(border=True):
        st.markdown("#### Links out")
        st.caption("Everything this document points at.")
        st.markdown("**Related documents**")
        for other, title, score in preview.related:
            st.button(f"{other} · {score:.2f}", key=f"rel_{other}",
                      on_click=show_on_document_page, args=(other,),
                      use_container_width=True, help=title)
        st.markdown("**Same venue**")
        if preview.peers_venue:
            for other, _ in preview.peers_venue:
                st.button(other, key=f"ven_{other}",
                          on_click=show_on_document_page, args=(other,),
                          use_container_width=True)
        else:
            st.caption("none")
        st.markdown("**Same year**")
        for other, _ in preview.peers_year:
            st.button(other, key=f"yr_{other}", on_click=show_on_document_page,
                      args=(other,), use_container_width=True)
        st.markdown("**Internal links**")
        st.caption(", ".join(preview.internal))


def tab_hypertext(state) -> None:
    section("Hypertext model")
    lede("H = (N, L)",
         "Every page, section and document is a node (N); every move between them "
         "is a link (L). This section shows the graph and finds a path through it.")

    graph = engine().graph
    stats = graph.graph_stats()
    stat_row([
        ("nodes", str(stats["nodes"]), ":material/hub:"),
        ("links", str(stats["links"]), ":material/link:"),
        ("documents", str(stats["documents"]), ":material/description:"),
        ("categories", str(stats["categories"]), ":material/category:"),
        ("sections", str(stats["sections"]), ":material/segment:"),
    ])

    st.markdown("#### Links by kind")
    meanings = {
        "hierarchy": "repository into domain into subtopic into document",
        "internal": "document to its sections, sections in reading order",
        "related": "document to document, by shared wording",
        "shared-venue": "document to document, same publication venue",
        "shared-year": "document to document, same year",
    }
    with st.container(border=True):
        for kind, count in stats["by_kind"].items():
            text, background = LINK_KIND_STYLE.get(kind, FALLBACK_STYLE[:2])
            st.markdown(
                f'<div style="padding:5px 0;">'
                f'<span class="badge" style="color:{text};'
                f'background:{background}">{kind}</span>'
                f'<span style="font-weight:750">{count}</span>'
                f'<span style="color:#60646C; margin-left:12px">'
                f'{meanings.get(kind, "")}</span></div>',
                unsafe_allow_html=True)

    st.markdown("#### Out-links of a node")
    node = st.selectbox("node", sorted(graph.nodes), key="graph_node",
                        help="Pick any node to see where it can go in one move.")
    if node:
        links = graph.out_links(node)
        st.caption(f"{graph.find(node).label}: degree {graph.degree(node)}, "
                   f"{len(links)} out-link(s)")
        if links:
            rows = []
            for link in links:
                text, background = LINK_KIND_STYLE.get(link.kind,
                                                       FALLBACK_STYLE[:2])
                rows.append([
                    f'<span class="badge" style="color:{text};'
                    f'background:{background}">{link.kind}</span>',
                    f'<code>{link.target}</code>',
                    graph.find(link.target).label,
                    f"{link.weight:.3f}" if link.weight else "",
                ])
            html_table(["kind", "target", "label", "weight"], rows)

    st.markdown("#### Shortest navigation path")
    st.caption("Ask how to get from one paper to another. The answer is the "
               "fewest moves, and every move is a real link.")
    documents = [d.doc_id for d in engine().documents]
    columns = st.columns(2)
    start = columns[0].selectbox("from", documents, index=0, key="path_a")
    goal = columns[1].selectbox("to", documents, index=len(documents) - 1,
                                key="path_b")
    if st.button("find path", key="path_btn", type="primary",
                 icon=":material/route:"):
        path = engine().path_between(start, goal)
        hops = engine().describe_path(path)
        if not hops:
            st.warning("No path exists between those two papers.")
        else:
            st.success(f"{len(hops)} move(s) from {start} to {goal}")
            rows = []
            for label, kind, target in hops:
                text, background = LINK_KIND_STYLE.get(kind, FALLBACK_STYLE[:2])
                rows.append([
                    label,
                    f'<span class="badge" style="color:{text};'
                    f'background:{background}">{kind}</span>',
                    target,
                ])
            html_table(["from", "link", "to"], rows, accent="#008573")
            st.caption("Every hop is a real link in the out-links table above.")


def tab_history(state) -> None:
    section("Navigation history")
    lede("Where you have been",
         "The back and forward trail, and the list of documents seen most "
         "recently. This is the flat side of browsing.")

    nav = st.session_state.nav
    if nav.history:
        rows = []
        for position, node in enumerate(nav.history):
            here = position == nav.position
            rows.append([
                str(position),
                f"<code>{node}</code>",
                f"<b>{engine().label(node)}</b>" if here else engine().label(node),
                ('<span class="badge" style="color:#008573;background:#E0F8F3">'
                 'here</span>') if here else "",
            ])
        html_table(["position", "node", "label", ""], rows)
    else:
        st.info("Nothing visited yet. Open a category or a document to start "
                "a trail.")

    if nav.recent():
        st.markdown("#### Recently viewed")
        st.markdown('<div class="chips">'
                    + "".join(f'<span class="chip">{engine().label(n)}</span>'
                              for n in nav.recent(8))
                    + "</div>", unsafe_allow_html=True)

    st.markdown("#### The whole collection, as one list")
    st.caption("Flat browsing: every document, no hierarchy. Each row has a "
               "preview button.")
    summaries = engine().documents_in("root", st.session_state.filters)
    if summaries:
        document_rows([(s.doc_id, s.title, s.category, None)
                       for s in summaries], "history")
    else:
        st.info("No documents match the current filters.")


def tab_start(state) -> None:
    section("Start here")
    lede("What this app is",
         "A browsing app over a collection of 20 research papers. Instead of only "
         "typing a search, you move around the collection: by category, by "
         "search, by following links between papers, with a trail of where you "
         "have been.")

    st.markdown("#### How to read any screen")
    st.markdown(
        "1. **Where you are** is the breadcrumb under the section bar, and every "
        "crumb in it is clickable.\n"
        "2. **What is here** is the main panel: a list, a document, or a graph.\n"
        "3. **Where can I go** is a button, or the **preview** button at the end "
        "of a document row, which opens the paper in place."
    )

    st.markdown("#### The seven things this app demonstrates")
    html_table(
        ["#", "what", "where", "what you see"],
        [["<b>1</b>", "Home page", "Home",
          "20 papers, 6 domains with counts, recently viewed"],
         ["<b>2</b>", "Category hierarchy", "Browse",
          "repository, then domain, then subtopic, then paper"],
         ["<b>3</b>", "Document list", "Browse",
          "the papers inside the category you chose"],
         ["<b>4</b>", "Search + browse", "Search",
          "type <code>proximity window</code>, then preview a result"],
         ["<b>5</b>", "Breadcrumb", "every section, along the top",
          "a clickable trail, for example Repository / Information Retrieval"],
         ["<b>6</b>", "Related document links", "Document",
          "the 3 papers closest in wording to this one"],
         ["<b>7</b>", "Document preview", "any row, and Document",
          "title, year, venue, and the three sections"]],
        accent="#008573")

    st.caption("Two more the course asks for: filters (the top bar) and a "
               "navigation history (the History section, plus back and forward "
               "in the top bar).")

    st.markdown("#### The three browsing models, one line each")
    for model in browse.BROWSING_MODELS:
        st.markdown(f"- **{model['name']}**: {model['idea']}")

    st.markdown("#### A five-minute tour")
    st.markdown(
        "1. **Home**: see the six domains and how many papers each holds, then "
        "press **browse** on one.\n"
        "2. **Browse**: the breadcrumb grows under the section bar. Open "
        "*Ranking & Scoring*.\n"
        "3. **The list**: press **preview** on any row. The paper opens right "
        "below it.\n"
        "4. **Search**: type `proximity window`. Paper **p04** should be first.\n"
        "5. **Hypertext**: ask for a path from **p01** to **p18**. It is two "
        "moves, and each move is a real link.\n"
        "6. **History**: look at the trail you just left, then press back in the "
        "top bar."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    inject_styles()
    engine()
    state = session()

    # A hypertext link is a real address (?doc=p04). Consume it once, so the
    # reload lands on that paper and the jump is recorded in the history.
    link_target = st.query_params.get("doc")
    if link_target:
        del st.query_params["doc"]
        if link_target in engine().by_id:
            show_on_document_page(link_target)

    page_header(
        "Browsing the paper collection",
        "Move around 20 papers by category, by search, and by the links "
        "between them.",
    )

    # No `default` here: session() already puts "Home" in the key, and passing
    # both makes Streamlit warn about a widget default that is also set by state.
    page = st.segmented_control("section", list(PAGES), key="page",
                                selection_mode="single",
                                label_visibility="collapsed") or "Home"

    breadcrumb_bar()
    sidebar(state)

    if page == "Start here":
        tab_start(state)
    elif page == "Home":
        tab_home(state)
    elif page == "Browse":
        tab_browse(state)
    elif page == "Search":
        tab_search(state)
    elif page == "Document":
        tab_document(state)
    elif page == "Hypertext":
        tab_hypertext(state)
    else:
        tab_history(state)


if __name__ == "__main__":
    main()
