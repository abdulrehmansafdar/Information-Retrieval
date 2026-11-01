"""Week 07 - semantic ranking with GVSM and LSI over the shared clinical corpus.

    python main.py --web
    streamlit run app.py

Every score and step here comes from semantic_search.py, which console_ui.py also uses,
so the browser and the terminal can never disagree. This file only handles presentation.

Design notes
    Two sections only: searching with the two models, and the documents being searched.
    The matrices live in code (term_matrix.py, lsi_model.py), not on screen.

    Cosine is drawn as what it is, an angle. Any two vectors lie on one flat plane, so
    the angle between a query and a document can be drawn exactly, however many
    dimensions they have. Arrow length is not to scale, because cosine ignores length.

Sections
    Search      the query, the model, the angle of every result, and the working of each
    Documents   the clinical articles, by specialty
"""

from __future__ import annotations

import html
import math
import re

import streamlit as st

from semantic_search import DEFAULT_K, Settings, get_engine
from term_matrix import WEIGHTINGS
from vector_cosine import angle_degrees

st.set_page_config(page_title="Week 7 - Semantic ranking", layout="wide",
                   initial_sidebar_state="collapsed")

PAGES = ("Search", "Documents")
PAGE_ICONS = {"Search": "search", "Documents": "menu_book"}

EXAMPLES = ("wheezing", "mosquito bite", "chest pain and blood vessels")
MODEL_LABELS = {"gvsm": "GVSM", "lsi": "LSI"}
WEIGHT_LABELS = {"tfidf": "TF-IDF", "count": "Count", "binary": "Binary"}

QUERY_COLOUR = "#6d28d9"
DOCUMENT_COLOUR = "#0f766e"
NO_WORD_COLOUR = "#b45309"

PLOT_WIDTH, PLOT_HEIGHT = 390, 330
ORIGIN_X, ORIGIN_Y, RADIUS = 34, 296, 270
GUIDE_ANGLES = (30, 60, 90)
# markers closer than this many degrees move to an inner ring so that both stay readable
MARKER_GAP = 7
RING_STEP = 34

PAGE_CSS = """
<style>
  .stApp { font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue",
           Arial, sans-serif;
           background-image:
             radial-gradient(60% 42% at 12% 0%, rgba(15, 118, 110, 0.055), transparent 62%);
           background-attachment: fixed; }
  header[data-testid="stHeader"] { background: #FFFFFF;
    border-bottom: 1px solid #E5E5E1; }
  .block-container { padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1200px; }

  .hero { padding: 4px 0 12px 0; }
  .hero h1 { font-size: clamp(1.9rem, 3.4vw, 2.9rem); font-weight: 780;
             letter-spacing: -0.03em; margin: 0 0 10px 0; line-height: 1.07; }
  .hero p { font-size: 1.02rem; margin: 0; opacity: 0.78; max-width: 68ch; }

  .lede { border-left: 3px solid var(--primary-color);
          background: var(--secondary-background-color);
          border-radius: 4px 12px 12px 4px;
          padding: 12px 16px; margin: 2px 0 16px 0; font-size: 0.9rem; }

  .result { background: #FFFFFF; border: 1px solid #E5E5E1; border-radius: 14px;
            padding: 12px 16px 10px 16px; margin: 0 0 10px 0; }
  .result .rtitle { font-weight: 700; font-size: 0.98rem; }
  .result .rcode { font-size: 0.78rem; color: #60646C; margin-left: 6px; }
  .result .rsub { font-size: 0.84rem; color: #4B4F57; margin-top: 4px; }
  .meter { display: inline-block; width: 150px; height: 8px; border-radius: 999px;
           background: rgba(100, 116, 139, 0.18); vertical-align: middle;
           margin-right: 8px; overflow: hidden; }
  .meter > span { display: block; height: 100%; background: #0f766e;
                  border-radius: 0 4px 4px 0; }
  .score { font-weight: 700; font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; font-size: 0.74rem; font-weight: 700;
           padding: 2px 9px; border-radius: 999px; margin-right: 6px;
           background: #E0F8F3; color: #0f5f58; }
  .badge.meaning { background: #FFF4D6; color: #6b4a00; }

  mark { background: #FFF7C2; color: #4F3422; padding: 0 3px;
         border-radius: 4px; font-weight: 650; }

  div[data-testid="stSegmentedControl"] button {
    background: #E6F4F1; color: #0f5f58; border: 1px solid #B9E4DC; font-weight: 600; }
  div[data-testid="stSegmentedControl"] button:hover { background: #CDEFE8; }
  div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: #0f766e; color: #FFFFFF; border-color: #0f766e; }

  .stButton button { background: #0f766e; color: #FFFFFF; border: 1px solid #0f766e;
    font-weight: 600; border-radius: 9px; }
  .stButton button:hover, .stButton button:focus {
    background: #0b5d57; color: #FFFFFF; border-color: #0b5d57; }

  div[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-radius: 14px; }
  div[data-testid="stExpander"] summary { background: #E6F4F1; border-radius: 12px; }
  div[data-testid="stExpander"] summary:hover { background: #D3EEE8; }
  span[data-testid="stIconMaterial"] { color: #0f766e; }
  div[data-baseweb="input"]:focus-within { border-color: #0f766e !important; }

  .tablewrap { border: 1px solid #E5E5E1; border-radius: 14px; overflow: auto;
    background: #FFFFFF; margin: 4px 0 14px 0; }
  table.tt { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  table.tt thead th { background: #E6F4F1; color: #0f5f58; font-weight: 700;
    text-align: left; padding: 10px 12px; border-bottom: 2px solid #0f766e;
    white-space: nowrap; }
  table.tt tbody td { padding: 9px 12px; border-bottom: 1px solid #EDEDEA; }
  table.tt tbody tr:nth-child(even) { background: #FAFAF8; }

  /* the numbered boxes of the cosine working */
  .step { background: #FFFFFF; border: 1px solid #E5E5E1; border-radius: 14px;
          padding: 12px 16px; margin-bottom: 10px; }
  .step .num { display: inline-block; width: 24px; height: 24px; line-height: 24px;
               text-align: center; border-radius: 50%; background: #0f766e;
               color: #FFFFFF; font-weight: 700; font-size: 0.8rem; margin-right: 8px; }
  .step .sname { font-weight: 700; }
  .step .swork { font-family: "Cascadia Mono", Consolas, monospace; font-size: 0.88rem;
                 margin-top: 6px; overflow-wrap: anywhere; }

  .plot svg { max-width: 100%; height: auto; background: #FFFFFF;
              border: 1px solid #E5E5E1; border-radius: 14px; }

  div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E5E5E1;
          border-radius: 14px; padding: 14px 16px;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04); }
  div[data-testid="stMetricValue"] { font-size: 1.4rem; }
  hr { margin: 1.6rem 0; }
</style>
"""


# ---------------------------------------------------------------------------
# small presentation helpers
# ---------------------------------------------------------------------------


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def html_table(headers: list[str], rows: list[list[str]]) -> None:
    """A table styled in the app's colours. Cells are already escaped HTML."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                   for row in rows)
    st.markdown(f'<div class="tablewrap"><table class="tt"><thead><tr>{head}</tr>'
                f'</thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def meter(value: float) -> str:
    width = max(0.0, min(1.0, value)) * 100
    return f'<span class="meter"><span style="width:{width:.0f}%"></span></span>'


def highlight(text: str, stems: tuple[str, ...]) -> str:
    """Escape the text, then mark every word that begins with a query stem."""
    escaped = html.escape(text)
    for stem in sorted(set(stems), key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(stem)}\w*\b", re.IGNORECASE)
        escaped = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)
    return escaped


def use_example(query: str) -> None:
    st.session_state["query"] = query


# ---------------------------------------------------------------------------
# the cosine, drawn as an angle
# ---------------------------------------------------------------------------


def polar(degrees: float, radius: float) -> tuple[float, float]:
    """A point at this angle above the query's arrow, which lies along the x axis."""
    theta = math.radians(degrees)
    return ORIGIN_X + radius * math.cos(theta), ORIGIN_Y - radius * math.sin(theta)


def arrow(degrees: float, radius: float, colour: str) -> str:
    x, y = polar(degrees, radius)
    return (f'<line x1="{ORIGIN_X}" y1="{ORIGIN_Y}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="2" stroke-linecap="round"/>')


def plot_frame(heading: str, marks: str, label: str) -> str:
    """The quarter circle, the guide lines with their cosines, and the query's arrow."""
    guides = [f'<text x="{ORIGIN_X + RADIUS + 2}" y="{ORIGIN_Y + 16}" text-anchor="end" '
              f'font-size="11" fill="#60646C">cos 0° = 1</text>']
    for degrees in GUIDE_ANGLES:
        x, y = polar(degrees, RADIUS)
        # clear of the markers on the outer ring, which can sit right on a guide line
        text_x, text_y = polar(degrees, RADIUS + 15)
        guides.append(f'<line x1="{ORIGIN_X}" y1="{ORIGIN_Y}" x2="{x:.1f}" y2="{y:.1f}" '
                      f'stroke="#E5E5E1"/>')
        guides.append(f'<text x="{text_x:.1f}" y="{text_y + 4:.1f}" text-anchor="start" '
                      f'font-size="11" fill="#60646C">{degrees}° = '
                      f'{math.cos(math.radians(degrees)):.2f}</text>')
    end_x, end_y = polar(90, RADIUS)
    start_x, start_y = polar(0, RADIUS)
    circle = (f'<path d="M {start_x:.1f} {start_y:.1f} A {RADIUS} {RADIUS} 0 0 0 '
              f'{end_x:.1f} {end_y:.1f}" fill="none" stroke="#D4D4CF"/>')
    query_x, query_y = polar(0, RADIUS)
    query = (f'{arrow(0, RADIUS, QUERY_COLOUR)}'
             f'<circle cx="{query_x:.1f}" cy="{query_y:.1f}" r="5" fill="{QUERY_COLOUR}" '
             f'stroke="#FFFFFF" stroke-width="2"/>'
             f'<text x="{query_x - 8:.1f}" y="{query_y - 9:.1f}" text-anchor="end" '
             f'font-size="13" font-weight="700" fill="#1A1A1A">Q (query)</text>')
    # no blank lines: a blank line would end the HTML block in markdown
    return (f'<div class="plot"><svg viewBox="0 0 {PLOT_WIDTH} {PLOT_HEIGHT}" '
            f'width="{PLOT_WIDTH}" role="img" aria-label="{html.escape(label)}">'
            f'<text x="{PLOT_WIDTH - 12}" y="20" text-anchor="end" font-size="13" '
            f'font-weight="700" fill="#1A1A1A">{html.escape(heading)}</text>'
            f'{"".join(guides)}{circle}{query}{marks}</svg></div>')


def fan_plot(hits) -> str:
    """Every result's angle from the query, on one picture. Numbers are the ranks."""
    placed: dict[float, list[float]] = {}
    marks = []
    for rank, hit in enumerate(hits, start=1):
        degrees = angle_degrees(hit.score)
        radius = RADIUS
        # the outermost ring with room for this marker
        while (radius - RING_STEP > 40
               and any(abs(degrees - a) < MARKER_GAP for a in placed.get(radius, []))):
            radius -= RING_STEP
        placed.setdefault(radius, []).append(degrees)
        colour = DOCUMENT_COLOUR if hit.shared_terms else NO_WORD_COLOUR
        x, y = polar(degrees, radius)
        marks.append(
            f'<g><title>{rank}. {html.escape(hit.title)}: cosine {hit.score:.3f}, '
            f'angle {degrees:.1f}°</title>{arrow(degrees, radius, colour)}'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{colour}" stroke="#FFFFFF" '
            f'stroke-width="2"/><text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
            f'font-size="11" font-weight="700" fill="#FFFFFF">{rank}</text></g>')
    return plot_frame("Numbers are the ranks below", "".join(marks),
                      "The angle between the query and each result")


def document_label(degrees: float, x: float, y: float) -> str:
    """D's label: inside the circle when steep, clear of the guide labels outside it,
    and lifted above Q's label when the two arrows nearly meet."""
    if degrees > 45:
        along_x, along_y = polar(degrees, RADIUS * 0.62)
        label_x, label_y, anchor = along_x + 10, along_y, "start"
    else:
        label_x, label_y, anchor = x - 8, min(y - 9, ORIGIN_Y - 30), "end"
    return (f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" font-size="13" '
            f'font-weight="700" fill="#1A1A1A">D (document)</text>')


def pair_plot(score: float) -> str:
    """One document against the query, with the angle between them marked."""
    degrees = angle_degrees(score)
    x, y = polar(degrees, RADIUS)
    arc_x, arc_y = polar(degrees, 58)
    marks = (f'{arrow(degrees, RADIUS, DOCUMENT_COLOUR)}'
             f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{DOCUMENT_COLOUR}" '
             f'stroke="#FFFFFF" stroke-width="2"/>'
             f'{document_label(degrees, x, y)}'
             f'<path d="M {ORIGIN_X + 58} {ORIGIN_Y} A 58 58 0 0 0 {arc_x:.1f} {arc_y:.1f}" '
             f'fill="none" stroke="#1A1A1A" stroke-width="1.5"/>')
    return plot_frame(f"θ = {degrees:.1f}°, cos θ = {score:.3f}", marks,
                      f"The angle between the query and this document is {degrees:.1f} degrees")


def show_steps(steps) -> None:
    """The numbered boxes, one per step of the formula."""
    for number, (name, working) in enumerate(steps.lines(), start=1):
        st.markdown(f'<div class="step"><span class="num">{number}</span>'
                    f'<span class="sname">{html.escape(name)}</span>'
                    f'<div class="swork">{html.escape(working)}</div></div>',
                    unsafe_allow_html=True)


def show_article(article, stems: tuple[str, ...]) -> None:
    """Everything the counts come from: the title and every section, query words marked."""
    st.caption("The whole article is counted: its title and all four sections.")
    st.markdown(f"<strong>{highlight(article.title, stems)}</strong>", unsafe_allow_html=True)
    for name, text in article.sections:
        st.markdown(f"<strong>{html.escape(name)}</strong><br>{highlight(text, stems)}",
                    unsafe_allow_html=True)


def show_vectors(engine, response, steps) -> None:
    """The numbers the working used: the query's terms, or the k concepts."""
    settings = response.settings
    if settings.model == "gvsm":
        matrix = engine.matrix(settings.weighting)
        rows = [[html.escape(t), f"{steps.q[matrix.row_of[t]]:.3f}",
                 f"{steps.d[matrix.row_of[t]]:.3f}"] for t in response.terms]
        st.caption(f"Each vector has {matrix.shape[0]:,} numbers, one per term. Only the "
                   "query's terms add to the dot product:")
        html_table(["Term", "Q", "D"], rows)
    else:
        rows = [[f"concept {c + 1}", f"{steps.q[c]:.3f}", f"{steps.d[c]:.3f}"]
                for c in range(settings.k)]
        st.caption(f"Each vector has {settings.k} numbers, one per hidden concept:")
        html_table(["Dimension", "Q", "D"], rows)


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Reading the clinical articles...")
def cached_engine():
    return get_engine()


def read_settings() -> Settings:
    return Settings(model=st.session_state.get("model") or "lsi",
                    weighting=st.session_state.get("weighting", "tfidf"),
                    k=st.session_state.get("k", DEFAULT_K))


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------


def search_page() -> None:
    engine = cached_engine()
    page_header("Semantic search",
                "Each article is ranked by the cosine between its vector and the query's.")
    query = st.text_input("Query", key="query", placeholder="Type a query, e.g. wheezing",
                          label_visibility="collapsed")
    st.segmented_control("Model", ("lsi", "gvsm"), key="model", format_func=MODEL_LABELS.get)
    with st.expander("Settings", icon=":material/tune:"):
        left, right = st.columns(2)
        left.radio("Weighting", WEIGHTINGS, horizontal=True, key="weighting",
                   format_func=WEIGHT_LABELS.get, help="What each number in the matrix means.")
        right.slider("LSI concepts to keep (k)", 2, 30, key="k",
                     help="Few concepts blur topics together. All 30 give GVSM's order.")

    settings = read_settings()
    if not query.strip():
        for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            column.button(f"Try: {example}", key=f"ex_{example}",
                          icon=":material/play_arrow:", on_click=use_example, args=(example,))
        return
    response = engine.search(query, settings)
    if not response.ok:
        st.error(response.message, icon=":material/error:")
        return
    if response.unknown:
        st.caption(f"Not in the collection, so ignored: {', '.join(response.unknown)}")
    if not response.hits:
        st.warning("No article scores above 0.", icon=":material/search_off:")
        return

    left, right = st.columns([3, 2])
    with left:
        st.markdown(fan_plot(response.hits), unsafe_allow_html=True)
        st.caption("A smaller angle to Q means a higher cosine. Each arrow's angle to Q is "
                   "exact, however many dimensions the vectors have. Arrow length is not to "
                   "scale, and angles between two articles are not shown.")
    with right:
        st.metric("Articles above 0", f"{response.count} of {len(engine.articles)}",
                  border=True, icon=":material/description:")
        st.metric("Query terms", ", ".join(response.terms), border=True,
                  icon=":material/search:")
        st.metric("Model", MODEL_LABELS[settings.model]
                  + (f", k = {settings.k}" if settings.model == "lsi" else ""),
                  border=True, icon=":material/functions:")

    no_word = sum(not h.shared_terms for h in response.hits)
    if no_word:
        st.markdown(f'<div class="lede">{no_word} of these {len(response.hits)} share no word '
                    "with the query (shown in amber). LSI reaches them because their words "
                    "load on the same hidden concepts as the query's. Some will be right and "
                    "some wrong.</div>", unsafe_allow_html=True)

    for rank, hit in enumerate(response.hits, start=1):
        badges = ("".join(f'<span class="badge">{html.escape(t)}</span>'
                          for t in hit.shared_terms)
                  or '<span class="badge meaning">no shared word: matched on meaning</span>')
        st.markdown(
            f'<div class="result"><div><span class="rtitle">{rank}. '
            f'{html.escape(hit.title)}</span><span class="rcode">{hit.code} '
            f'&middot; {html.escape(hit.specialty)}</span></div>'
            f'<div class="rsub">{meter(hit.score)}<span class="score">{hit.score:.3f}'
            f'</span> &nbsp; {angle_degrees(hit.score):.1f}° &nbsp; {badges}</div></div>',
            unsafe_allow_html=True)
        with st.expander(f"why {hit.score:.3f}", icon=":material/insights:"):
            steps = engine.steps(query, hit.doc_id, settings)
            plot_column, steps_column = st.columns([2, 3])
            with plot_column:
                st.markdown(pair_plot(hit.score), unsafe_allow_html=True)
            with steps_column:
                show_steps(steps)
            show_vectors(engine, response, steps)
            show_article(engine.by_id[hit.doc_id], hit.shared_terms)
    if response.count > len(response.hits):
        st.caption(f"Showing the top {len(response.hits)} of {response.count}.")


def documents_page() -> None:
    engine = cached_engine()
    page_header("Documents", f"The {len(engine.articles)} clinical articles being searched.")
    specialties = sorted({a.specialty for a in engine.articles})
    for specialty in specialties:
        st.subheader(specialty)
        for article in engine.articles:
            if article.specialty != specialty:
                continue
            with st.expander(f"{article.doc_id}  {article.title}  ({article.code})",
                             icon=":material/description:"):
                for name, text in article.sections:
                    st.markdown(f"**{name}**")
                    st.write(text)


# ---------------------------------------------------------------------------
# the page bar
# ---------------------------------------------------------------------------


def main() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("page", PAGES[0])
    st.session_state.setdefault("model", "lsi")
    st.session_state.setdefault("weighting", "tfidf")
    st.session_state.setdefault("k", DEFAULT_K)
    page = st.segmented_control("Section", PAGES, key="page", label_visibility="collapsed",
                                format_func=lambda p: f":material/{PAGE_ICONS[p]}: {p}")
    st.divider()
    {"Search": search_page, "Documents": documents_page}[page or PAGES[0]]()


main()
