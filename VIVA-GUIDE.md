# VIVA — exploring the three web apps, and what to say

This is a rehearsal sheet, not a reference. It tells you **which app to open, what
to click, what should appear, and the one sentence to say out loud.** Explore in
this order; by the end you will have seen every model and every feature the course
covers.

Week 1 does not have a "Start here" tab (it is deliberately frozen), but week 2
and week 3 each open on a **Start here** tab that repeats this for that app.

---

## 0. Before you start

```powershell
.\.venv\Scripts\python.exe -m streamlit run week01\app.py   # week 1
.\.venv\Scripts\python.exe week02\main.py --web             # week 2
.\.venv\Scripts\python.exe week03\main.py --web             # week 3
```

Each opens in a browser (normally `http://localhost:8501`; run them one at a time
or pass different ports). Every number on every page is computed live from the
papers — nothing is hard-coded, so if you change a slider the number moves.

---

## 1. The story in five sentences

Say this first; it frames the whole viva.

1. **Week 1** built one retrieval model from scratch: preprocess text, build an
   inverted index, and rank with TF-IDF cosine.
2. **Week 2** asked whether a *better* model helps, and answered it by measuring:
   BIM, BM25, BM25F, two list models and a node model, scored on judged queries.
3. **Week 2 also showed** that some questions are about *where* a word sits, which
   is why the index became positional.
4. **Week 3** changed the question entirely: not "find me X" but "let me explore",
   and built the three browsing models of the lecture — flat, structured-guided and
   hypertext.
5. Throughout, the **front ends never implement retrieval**; they call one engine,
   so the web page and the command line can never disagree.

---

## 2. Week 1 — one model, end to end

Four tabs: **Ranked search**, **Boolean**, **Pipeline**, **Corpus**.

| click | you should see | say |
|---|---|---|
| Ranked search → type `machine learning healthcare` | a cosine-ranked list of ~10 documents with scores, and a strict-AND marker on doc03/doc04 | "One query, one model. The score is TF-IDF cosine; the marker is the exact AND interpretation of the same words." |
| Boolean → type `healthcare AND diagnosis` | an exact set, no scores | "Boolean has no notion of 'best'. A document is in or out." |
| Pipeline → pick a sentence | raw → lowercase → tokens → stems → phrases → terms, each on a card | "This is the whole engine as a stepper, and every stage runs the real function." |
| Corpus | 20 documents, vocabulary and posting counts, length range, IDF range | "The corpus and its statistics." |

**The one thing to be able to defend:** why a phrase becomes **one** index term,
and why that matters (it changes what a region boundary means later).

---

## 3. Week 2 — many models, measured

Open on **Start here**, then visit the tabs. Seven tabs: Start here, Search,
Fields & Lists, Proximity, Evaluation, Corpus, Pipeline. There is no sidebar: the
model, the result count, and `k1` / `b` all sit in the bar at the top of the page.

| click | you should see | say |
|---|---|---|
| Search → `proximity window`, model `bm25` | **p04** first | "BM25 finds the paper about proximity operators." |
| Search → drag **k1** to `0` | a message: it is now the Binary Independence Model, and the list equals the `bim` mode | "BM25 with k₁ = 0 *is* BIM. It is an identity, not an approximation." |
| Fields & Lists → `title:search` under *restrict*, then *filter* | the **same five papers in a different order** (p18 jumps to the top) | "Restrict scores only the title; filter ranks the whole paper and then discards. Same set, different order — and the order comes from text you said you were excluding." |
| Fields & Lists → `document index positions` | Overlapped returns 1 region; Non-overlapped returns 4 regions in 2 papers, adding p11 | "Overlapped needs all terms in one region; non-overlapped allows them in different regions. That is precision traded for coverage." |
| Proximity → `section with(2) retrieval` | a **set** of nodes, scores all 0 | "This model answers 'which', not 'in what order', so there is no score." |
| Proximity → `paragraph methodology` | a readable error, no crash | "Malformed input is a message, not a traceback." |
| Evaluation | the table: BM25 MRR `1.000`, BIM `0.917`, list models `0.750`/`0.833` | "Read `nohit` before MRR: it is how you see a model that is fine on easy queries and useless on hard ones." |

**Numbers worth having ready:** 20 papers · 966 terms · 2,134 postings · 3,434
positions · 80 sections / 120 paragraphs / 299 sentences · qrels = 12 queries,
18 positives, 61 negatives.

**The one thing to be able to defend:** the k₁ = 0 bridge
(`tf(k₁+1)/(tf + k₁·K)` → `tf/tf = 1`), and *why* restrict and filter return the
same papers in a different order.

---

## 4. Week 3 — browsing and navigation

Open on **Start here**, then pick a section from the bar at the top. Seven
sections: Start here, Home, Browse, Search, Document, Hypertext, History. Filters
and the back/forward buttons live in the sidebar; the breadcrumb is the clickable
trail under the section bar.

| click | you should see | say |
|---|---|---|
| **Home** | 20 papers, **6** domains with counts, recently viewed | "The landing page: what the collection holds." |
| **Browse** → *Information Retrieval* → *Ranking & Scoring* | the breadcrumb grows; the list narrows to p03, p16 | "The category tree, derived from the papers' topics, not hand-listed." |
| **Search** → `proximity window` | p04 first; click a row to open the paper | "Search is a way *into* the collection; you then browse from the result." |
| **Document** → open p17 | the trail `Repository / … / p17`, the three expandable sections, and underlined **inline links** to related papers | "The preview: metadata plus the paper's own sections. The underlined words are the hypertext links, rendered as prose instead of a graph." |
| **Document** → *Related documents* | p14 first, then p04, p13 | "The hypertext links — a cosine over title + abstract." |
| **Hypertext** → path `p01` → `p18` | **two** hops, both marked `related` | "H = (N, L). The path is built from the same links shown in the table." |
| **History** | the trail you just left; the **back** button in the sidebar walks it | "Navigation history: where the user has been." |

**Numbers worth having ready:** 20 documents · 60 sections · 6 domains · 11
subtopics · `|N| = 98` nodes · `|L| = 261` links (hierarchy 37 + internal 100 +
related 60 + shared-venue 6 + shared-year 58).

**The one thing to be able to defend:** the three browsing models and where each
lives — **flat** = the plain document list / History; **structured guided** = the
taxonomy tree + breadcrumb; **hypertext** = the `H = (N, L)` graph.

---

## 4b. Week 6 — fuzzy and extended Boolean search

Open on **Start here**, then pick a section from the bar at the top: Start here,
Search, Compare, Membership, Set view, Articles. The search runs over **30 clinical
articles** in `corpus/articles`, the one shared corpus that every module from week 6
onward reads. The mode is chosen on the Search page: Boolean, Fuzzy or Extended Boolean.

| click | you should see | say |
|---|---|---|
| **Start here** → *Try: hypertension AND NOT diabetes* | Search opens, fuzzy mode: **7 of 30**, Hypertension first at **1.00** | "Fuzzy: every article gets a score. NOT is soft, so diabetes lowers a score rather than removing the article." |
| Search → mode **Extended Boolean** (same query) | **1 article**: Hypertension | "Extended Boolean: NOT is a hard exclusion. The set is (A or B) minus C." |
| Search → `hypertenion` (a typo), Fuzzy | **7 of 30**; Hypertension first at **0.70**, *matched as hypertens* | "Partial matching: the typo still reaches the right article, with a lower score for a looser match." |
| Search → `hypertenion`, mode **Boolean** | **0 articles** | "Classical Boolean: an exact word or nothing. That is the weakness the fuzzy model fixes." |
| Search → `hypertension treatment` → open *why* on Hypertension | `AND` 0.50, over 1.00 and 0.50 | "A fuzzy AND is a min. You can show the arithmetic." |
| Compare → `asthma OR COPD` | Boolean 2, Fuzzy 2, Extended 2; top scores 1.00, 1.00, 0.71 | "Same articles, different scores. The p-norm changes the order, not the set." |
| Membership → `hypertension` | a table with binary, linear and log columns; Coronary Artery Disease 1.00 / 0.40 / 0.61 | "The membership function is a design choice. Binary is Boolean, linear is proportional, log damps repeats." |
| Set view (defaults: A = diabetes, B = hypertension, C = kidney) | 11, 7, 11; union 12; (A or B) minus C = **5** articles; green "same set" | "Lecture section 3, checked on real sets: (A OR B) AND NOT C equals (A u B) minus C." |

**Numbers worth having ready:** 30 articles · 120 sections · 10 specialties ·
1,135 stems · 3 membership functions · title weight 3 · default p = 2 · default
threshold tau = 0.6.

**The one thing to be able to defend:** every article gets a membership value between
0 and 1 for each word. Fuzzy AND, OR and NOT are min, max and 1 minus x. Boolean is the
same evaluator with 0/1 values. Extended Boolean keeps set logic for *which* articles
(the alpha cut) and uses the p-norm for *order*.

---

## 5. The thread that ties them together

- **One engine per week, two front ends.** Console and web call the same code, so
  they cannot disagree. This is a design decision you can state and defend.
- **Each week is self-contained.** Week 2 does not import week 1; week 3 does not
  import week 2. They share ideas and a corpus format, not code — so editing one
  cannot break another.
- **The corpus is the same 20-paper idea throughout**, which is what lets you say
  "the same collection, three different questions": rank it (W2), explore it (W3).
  Week 6 breaks the pattern on purpose: it is a different domain (clinical articles)
  and its data lives once, at `corpus/`, so no week keeps its own copy.
- **Documentation is executable.** `check_doc_lines.py` proves every `file.py:NN`
  in the docs still points at the code it claims, and `check_doc_claims.py`
  re-derives every quoted number from the corpus. Both run inside `verify.py`.

---

## 6. Likely viva questions, with one-line answers

| question | answer to give |
|---|---|
| Difference between search and browsing? | Searching asks directly; browsing explores step by step, guided by what you have just seen. |
| What is flat browsing's weakness? | No structure, so in a large collection the user loses context. |
| Why structured guided browsing? | It fits content that already has categories or chapters, and the breadcrumb keeps the user oriented. |
| What do N and L mean in the hypertext model? | N is the set of nodes (pages/sections/documents); L is the set of links between them. |
| TF-IDF in one line? | A term counts more if it occurs often in a document but rarely across the collection. |
| BM25 vs BIM? | BM25 saturates term frequency and normalises length; set k₁ = 0 and it collapses to BIM exactly. |
| Why a positional index? | To answer *where* a term occurs — needed for phrases, proximity and regions. |
| Why search inside a field? | To raise precision by ignoring text the user did not ask about. |
| How do you know a model is better? | On judged queries, with a metric — and read `nohit`, not only MRR. |
| What is the search + browse hybrid? | The search box gets you *into* the collection; the graph and breadcrumb let you continue without retyping. |
| What is a membership value? | A number between 0 and 1 for how strongly an article belongs to a word: 1 is fully about it, 0 is not at all. |
| Fuzzy AND, OR and NOT? | Min, max and 1 minus the value. Boolean is the same rules with only 0 and 1. |
| How does a typo still find an article? | Partial matching: trigram Dice similarity between word stems, discounted by how close the match is, and only above the threshold tau. |
| Why have both a fuzzy and an extended Boolean model? | Fuzzy ranks every partial match. Extended Boolean keeps exact include and exclude sets and ranks inside them with a p-norm. |
| What does p do in the p-norm? | p = 1 is a plain average, p = infinity is min and max, and larger p makes the operators stricter. |

---

## 7. If something goes wrong in the demo

- **A number looks different from this sheet** — say so out loud, then trust the
  screen: every figure is computed live, and the sheet is only a snapshot.
- **A malformed query shows an error** — that is the intended behaviour, not a
  crash. Point at it: "malformed input is a message, not a traceback."
- **You forget a figure** — you do not need it. Say *what* the app is doing and
  open the tab; the number is on screen.
