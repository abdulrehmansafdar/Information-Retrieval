# Console and Web Interfaces

The project has one search backend and two separate user interfaces. `main.py` is
only the launcher: it selects the console or Streamlit front end.

```text
                       +--> console_ui.py -- Rich terminal output
main.py -- choose -----|
                       +--> app.py -------- Streamlit web UI
                                  |
                                  v
                              engine.py
                       common search logic
                                  |
              index + Boolean + TF-IDF + cosine
```

The key rule is: **the interfaces never implement retrieval themselves**. They call
`engine.search()` and display its response. That keeps the console and web results
consistent.

---

## 1. Run the project

Activate the project virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Console

```powershell
python main.py --console
python main.py --console "medical imaging"
python main.py --console --mode ranked "machine learning healthcare"
python main.py --console --mode boolean "healthcare AND diagnosis"
python main.py --console --demo
python main.py --console --stats
```

The `--console` flag is required so the launcher never guesses which interface to
start.

### Web UI

```powershell
python main.py --web
```

This starts Streamlit and prints the local URL, normally `http://localhost:8501`.
Streamlit options can be forwarded after `--web`, for example:

```powershell
python main.py --web --server.port 8502 --server.headless true
```

You can also invoke Streamlit directly:

```powershell
streamlit run app.py
```

---

## 2. File responsibilities

| File | Responsibility |
|---|---|
| `engine.py` | Shared data API: cached corpus build, auto-mode search, ranked hits, Boolean ids, corpus statistics |
| `main.py` | Small dispatcher: `--console` or `--web` |
| `console_ui.py` | Terminal prompt, Rich tables, text/HTML/JSON output and `--demo` |
| `app.py` | Streamlit controls, tabs, sortable result table, query history, CSV download |
| `demo.py` | Shared corpus walkthrough renderables used by console and web |
| `pipeline.py` | Real stage data for the Pipeline tab (preprocessing + retrieval)
| `search.py` | Earlier standalone terminal demo, kept as a secondary entry point |

`app.py` does not import `console_ui.py` or `main.py`. `console_ui.py` does not import
`app.py`. The two front ends share `engine.py` and the presentation-only walkthrough
in `demo.py`.

---

## 3. Shared search API

```python
from engine import search

response = search("machine learning healthcare")
response.mode          # 'ranked'
response.terms         # ['healthcar', 'machin learn']
response.hits          # 10 cosine-ranked Hit objects
response.boolean_ids   # ['doc03', 'doc04'] - strict AND interpretation
response.hits[0].doc_id        # 'doc04'
response.hits[0].score         # 0.2447...
response.hits[0].satisfies_and # True
```

The engine returns data only. It prints nothing, so the same result can be used in
the terminal or rendered by Streamlit.

### Automatic mode

| Input | Selected mode | Result |
|---|---|---|
| `machine learning healthcare` | ranked | Cosine-ranked list plus a strict-AND marker per result |
| `healthcare AND diagnosis` | Boolean | Exact set; no score or ranking |

You can also force a mode:

```python
search("machine learning healthcare", mode="ranked")
search("healthcare AND diagnosis", mode="boolean")
```

Ranked queries show both retrieval models at once. For example, the ranked query
returns 10 partial or full matches, while the strict AND interpretation marks only
doc03 and doc04.

---

## 4. Web UI features

| Tab | What it demonstrates |
|---|---|
| **Ranked search** | Search box, cosine score, strict-AND marker, matched-term explanation, sortable results and CSV download |
| **Boolean** | AND / OR / NOT, parentheses, exact result set, plus the phrase-extraction zero-result example |
| **Pipeline** | Text-based stepper for the whole engine: Act 1 (sentence to final terms as cards) plus Act 2 (inverted index, weights, query, ranked results as tables), then the static `demo_sections()` corpus summary |
| **Corpus** | 20 documents, vocabulary/posting counts, length range, IDF range, common terms and document table |

The sidebar has example queries for both models and a result-count control.

### Theme

`.streamlit/config.toml` sets the palette, so the whole app reads as one system:

- **One accent**: teal-700 `#0f766e`. It is the only accent on the page, used for
  buttons, links, the slider, the active radio state and metrics.
- **Why teal-700**: checked against WCAG. White text on it is **5.5:1** (AA for
  buttons), it reads as a link on white at **5.5:1**, and on Streamlit's dark
  surface at **3.3:1** (the UI minimum). Brighter teals passed the button check
  but failed as links; a dark blue passed the button check but collapsed to 2:1
  on dark surfaces.
- **Neutral base**: slate. Near-white `#fafafa` surfaces, `#f1f5f9` panels,
  `#0f172a` text. No pure white and no pure black, which keeps depth in both
  modes.
- **Dark mode**: `base = "light"` is the default. The Streamlit toolbar theme
  picker flips the app to dark; the same accent sits on the framework's
  off-black surfaces, so nothing inverts mid-page and the accent stays
  recognisable. Test each query in both modes before submission.

The example shortcuts are two dropdowns on the sidebar, not a wall of buttons.

### Pipeline walkthrough

The Pipeline tab shows the **whole engine as a text-based stepper**: Act 1 takes
a chosen sentence from raw text to final terms, Act 2 follows those terms into
the inverted index, weighting, a query and the ranked results. `pipeline.py`
computes every stage with the **actual** preprocessing functions
(`wordpunct_tokenize`, `stem_tokens`, `extract_phrases`, `get_stop_words`,
`preprocess_text`) and the **real engine** (inverted index, IDF table, TF-IDF,
cosine ranking), so the numbers cannot disagree with the rest of the app.

Act 1 is a numbered card per stage, with the tokens shown as inline code chips:

| # | Stage | Card shows |
|---|---|---|
| 1 | Raw | the sentence as written |
| 2 | Lowercase | the sentence lowercased |
| 3 | Tokens | chips including punctuation as its own token |
| 4 | Clean | chips with punctuation removed |
| 5 | Stems | chips after Snowball (plus the stop words still visible) |
| 6 | Phrases | the phrase terms that became single index terms |
| 7 | Terms | final terms, with the dropped stop words listed |

Act 2 uses real data from the engine, rendered as sortable tables:

| # | Stage | Shows |
|---|---|---|
| 8 | Index | term x df x postings (doc:tf) from the inverted index |
| 9 | Weights | term x df x idf x highest tf-idf |
| 10 | Query | the final terms as the query, and how many of the 20 documents share a term |
| 11 | Results | ranked documents with real cosine scores |

The ranking in stage 11 is not staged: the query is the sentence's own final
terms and the scores come from the actual TF-IDF and cosine implementation.

This replaces an earlier animated diagram. It was dropped deliberately: it
relied on `st.components.v1.html`, which Streamlit flags as deprecated
(advising `st.iframe`), and the iframe restarted on every unrelated widget
change, which fought a live presentation. A plain stepper of cards and tables
is simpler to explain, never resets, and is native to Streamlit, so it behaves
exactly like the rest of the app.

The static corpus walkthrough (`--demo`) remains available below it for the
collection-level numbers.

### Why `@st.cache_resource` is used

Streamlit reruns `app.py` after most user interactions. Building the corpus every
time would repeat preprocessing, index construction, IDF, and TF-IDF. `cached_engine()`
uses `@st.cache_resource` to keep one `Engine` object for the server process. The engine
also has `@lru_cache`, so command-line callers get the same once-built object.

---

## 5. Console output formats

These are still available through the console interface:

```powershell
python main.py --console --text "medical imaging"
python main.py --console --json "medical imaging"
python main.py --console --html "medical imaging"
```

- `--text` is convenient for piping or quick inspection.
- `--json` is structured data for scripts or a future API.
- `--html` is a standalone Rich-rendered HTML document; Rich's tables are exported
  as styled `<pre><code>` content rather than semantic HTML `<table>` elements.

These formats do not start Streamlit. Use `--web` when you want the interactive UI.

---

## 6. Setup and verification

Install dependencies into the active venv:

```powershell
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
```

The assignment scripts and the Streamlit app are smoke-tested with:

```powershell
python main.py --console "medical imaging"
python main.py --console "healthcare AND diagnosis"
python main.py --web
```

Streamlit's `AppTest` also verifies that the app has four tabs, produces no
exceptions, returns 10 rows for `machine learning healthcare`, returns 7 Boolean
results for `machine learning AND NOT finance`, and renders the pipeline walkthrough.

The standalone verification scripts live outside the project and are not required to
run the application. They check the inverted index, Boolean parser, IDF, TF-IDF,
cosine formula, ranking order, and the five assignment features.

---

## 7. Scope boundary

This is a local university mini-project. It deliberately has:

- No database; the 20 source documents remain in `data/documents/`.
- No upload endpoint; the assignment says to upload **or store** documents.
- No user accounts or authentication.
- No network API; JSON output is available from the CLI, but no server endpoint is
  needed for the Streamlit assignment UI.

The Streamlit UI is the web interface; `main.py --console` remains the console
interface. Both use the same backend.
