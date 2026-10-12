# Week 02 — Every Formula and Where It Is Computed

One page that answers a single question: **for each model in this week, what is
the formula, and which line of code turns a `.txt` paper into that number?**

The six guides in this folder explain *why* each model is the way it is. This
page answers *where*, so a viva question like "how is the BM25 saturation
actually computed?" has a file and a line number, not a guess.

Every number quoted was measured by running the code, not read off a slide.

---

## 0. The master table

| Step | Model | Main formula | Computed by | Line |
|---|---|---|---|---|
| 1 | Positional preprocessing | `raw → terms at dense positions` | `preprocess_with_positions()` | `preprocess.py:259` |
| 2 | Positional index | `postings[t] = [(d, tf, [positions])]` | `build_index()` | `structured_index.py:274` |
| 2 | Region | `[start, end)` half-open token range | `Region` | `preprocess.py:336` |
| 3 | IDF | `idf(t) = log(1 + (N − df + 0.5)/(df + 0.5))` | `IdfTable._idf_value()` | `ranking.py:101` |
| 3 | **BIM** | `score(q,d) = Σ idf(t) · 1[t ∈ d]` | `BinaryIndependenceModel.score_paper()` | `ranking.py:238` |
| 3 | **BM25** | `score(q,d) = Σ idf(t) · tf(k₁+1) / (tf + k₁·K(d))` | `Bm25Model.score_paper()` | `ranking.py:327` |
| 3 | saturation | `tf(k₁+1) / (tf + k₁·K(d))` | `Bm25Model.saturation()` | `ranking.py:312` |
| 3 | length norm | `K(d) = (1−b) + b·\|d\|/avgdl` | `Bm25Model.length_norm()` | `ranking.py:308` |
| 4 | **BM25F** | `score(q,d) = Σ_f weight_f · score_BM25(q, f)` | `FieldSearch.weighted()` | `fields.py:628` |
| 4 | field idf | `log(1 + (N_f − df_f + 0.5)/(df_f + 0.5))` | `FieldStats.__post_init__()` | `fields.py:375` |
| 4 | field BM25 | same as BM25, per field | `FieldSearch._bm25()` | `fields.py:427` |
| 5 | **collapse** | matches → disjoint minimal `[start,end)` | `collapse()` | `lists.py:118` |
| 5 | region score | BM25 inside one region | `ListModels._region_score()` | `lists.py:205` |
| 5 | **Overlapped** | all query terms in **one** region | `ListModels.overlapped()` | `lists.py:228` |
| 5 | **Non-overlapped** | terms in **separate** regions, union | `ListModels.non_overlapped()` | `lists.py:279` |
| 6 | **innermost component** | smallest node containing a position | `innermost_component()` | `proximity.py:162` |
| 6 | `P with(k) T` | P contains ≥ k occurrences of T | `_op_with()` | `proximity.py:354` |
| 6 | `P before(k) T` | P ends ≤ k terms before T | `_op_near()` | `proximity.py:394` |
| 7 | Precision@k | hits in first k / k | `precision_at_k()` | `evaluate.py:121` |
| 7 | MRR | 1 / rank of first relevant hit | `reciprocal_rank()` | `evaluate.py:142` |
| 7 | MAP | mean of P@rank over relevant hits | `average_precision()` | `evaluate.py:150` |

---

## 1. Step 1 — Positional preprocessing

There is no arithmetic here, so "the formula" is the pipeline:

```
raw text → tokens → stems → phrases → stop words → terms AT POSITIONS
```

**Every stage, and the line that runs it**

| Stage | Does | Computed by | Line |
|---|---|---|---|
| tokenize | lowercase, split, strip punctuation | `tokenize()` | `preprocess.py:216` |
| stem | `"networks"` → `"network"` | `stem_token()` | `preprocess.py:227` |
| phrases | `"machine learning"` → one term | `_build_phrase_lookup()` | `preprocess.py:206` |
| stop words | drop `the`, `of`, … | `preprocess_with_positions()` | `preprocess.py:314` |
| **assign positions** | dense `0..n−1`, phrases inline | `preprocess_with_positions()` | `preprocess.py:259` |

**Why this file is not a copy of `week01/preprocess.py`**

week01's `preprocess_text()` returns `unigrams + phrases` — phrases are appended
at the end, so list order no longer matches word order. That is harmless for
TF-IDF, which only counts. It is fatal for regions.

```powershell
python -c "import preprocess; print(preprocess.preprocess_with_positions('Deep learning improves clinic workflow'))"
# (['deep learn', 'improv', 'clinic', 'workflow'], [0, 1, 2, 3])
```

The phrase is at position 0, where its words actually were. That is the whole
difference, and everything from step 2 onward depends on it.

---

## 2. Step 2 — The positional index

```
postings[t] = [ (doc, tf, [positions]), … ]
```

**Symbol table**

| Symbol | Meaning | Computed by | Line |
|---|---|---|---|
| `postings[t]` | postings list for term *t* | `build_index()` | `structured_index.py:274` |
| `tf(t,d)` | occurrences in *d* | `len(places)` | `structured_index.py:285` |
| `positions` | where, ascending, from 0 | `tuple(sorted(grouped[term]))` | `structured_index.py:285` |
| `N` | papers in the corpus | `number_of_documents` | `structured_index.py:92` |
| `avgdl` | mean paper length | `average_document_length` | `structured_index.py:105` |
| `enclosing(kind, doc, pos)` | smallest region holding a position | `enclosing()` | `structured_index.py:166` |

**The region**

```
[start, end)      half-open, dense, token offsets
```

A sentence's range lies inside its paragraph's, which lies inside its section's.
`check_invariants()` (`structured_index.py:311`) proves that rather than
trusting it, and `tests/test_corpus.py` asserts it.

**Measured on this corpus**

| | |
|---|---|
| papers | 20 |
| vocabulary | 966 |
| postings | 2,134 |
| stored positions | 3,434 |
| paper length | 144–197 terms, average 171.7 |
| sections / paragraphs / sentences | 80 / 120 / 299 |

**Verify it yourself**

```powershell
python structured_index.py
python -c "import structured_index as s; i=s.build_index(); print(s.check_invariants(i) or 'all invariants hold')"
```

---

## 3. Step 3 — Probability-based ranking

Two formulas, one function apart.

### BIM

```
score(q, d) = Σ over t in q of  idf(t) · 1[t ∈ d]
```

| Symbol | On this corpus | Computed by | Line |
|---|---|---|---|
| `idf(t)` | `2.1282` for `field` | `IdfTable._idf_value()` | `ranking.py:101` |
| `N` | `20` | `number_of_documents` | `ranking.py:94` |
| `df(t)` | `2` | `len(index.lookup(term))` | `ranking.py:97` |
| `1[t ∈ d]` | `1` or `0` | `if tf <= 0: return 0.0` | `ranking.py:234` |
| the sum | `total += idf` | `score_paper()` | `ranking.py:248` |

### BM25

```
score(q, d) = Σ over t in q of  idf(t) · tf·(k₁+1) / (tf + k₁·K(d))
K(d) = (1 − b) + b · |d| / avgdl
```

| Symbol | On this corpus | Computed by | Line |
|---|---|---|---|
| `tf` | `15` for `field` in p19 | `term_frequency_in()` | `ranking.py:333` |
| `k₁+1` | `2.2` at the default | `self.k1 + 1.0` | `ranking.py:308` |
| `K(d)` | `0.9533` for p19 | `length_norm()` | `ranking.py:308` |
| saturation | `2.0441` at tf=15 | `saturation()` | `ranking.py:308` |
| contribution | `2.1282 × 2.0441 = 4.3503` | `idf * sat` | `ranking.py:338` |
| the sum | `total += contribution` | `score_paper()` | `ranking.py:339` |

**Call chain** — every hop from a `.txt` file to `4.3494`

```
ranking.py:339     total += contribution
 ├─ contribution ← ranking.py:338   idf(t) * saturation(tf, |d|)
 │   ├─ idf       ← ranking.py:336 → ranking.py:101  log(1 + (N-df+.5)/(df+.5))
 │   │                                ├─ N  ← ranking.py:94   index.number_of_documents
 │   │                                └─ df ← ranking.py:97   len(index.lookup(term))
 │   │                                                  ← structured_index.py:274  build_index()
 │   └─ saturation ← ranking.py:308   tf(k1+1)/(tf + k1*length_norm)
 │                     ├─ tf         ← ranking.py:333   index.term_frequency_in(doc, term)
 │                     │                                ← structured_index.py:283  positions
 │                     │                                  ← preprocess.py:259  preprocess_with_positions
 │                     └─ length_norm ← ranking.py:308  (1-b) + b*|d|/avgdl
```

**Worked example** — query `structured field search`, paper p19

| term | tf | idf | saturation | = idf × sat |
|---|---|---|---|---|
| `structur field` | 1 | 2.6391 | 1.0262 | 2.7081 |
| `search` | 2 | 1.1727 | 1.3995 | 1.6413 |

```
|p19| = 161,  avgdl = 171.7
K(d)  = (1 − 0.75) + 0.75 × 161 / 171.7 = 0.9533
score = 2.7081 + 1.6413 = 4.3494
```

**The bridge — BIM is BM25 at k₁ = 0**

Substituting `k₁ = 0`: `tf·1 / (tf + 0) = 1`. Measured, not asserted:

```
max |saturation(tf, |d|) − 1| at k1 = 0, tf ≥ 1     = 0.0
max |BIM − BM25(k1=0)| over all 20 papers            = 0.0
```

`tests/test_ranking.py::TestTheBridge` asserts it on every paper and every
query term, and the Streamlit UI says so when the k₁ slider hits 0.

**Verify it yourself**

```powershell
python ranking.py
python -c "import structured_index as s, ranking as r; i=s.build_index(); print(r.Bm25Model(i,k1=0.0).saturation(15,161))"
# 1.0
```

---

## 4. Step 4 — Structured field search

### BM25F

```
score_BM25F(q, d) = Σ over fields f of  weight_f · score_BM25(q, f)

    title 3.0 | abstract 2.0 | methodology 1.5 | results 1.0
```

| Symbol | Meaning | Computed by | Line |
|---|---|---|---|
| `weight_f` | per-field weight | `DEFAULT_WEIGHTS` | `fields.py:50` |
| `score_BM25(q,f)` | BM25 inside one field | `_bm25()` | `fields.py:427` |
| the sum | `total += weight * part` | `weighted()` | `fields.py:657` |
| field `df` | documents with the term **in that field** | `FieldStats.__post_init__()` | `fields.py:365` |
| field `idf` | `log(1 + (N_f - df_f + .5)/(df_f + .5))` | `FieldStats.__post_init__()` | `fields.py:376` |
| per-field length norm | `|f| / avg|f|` | `length_norm()` | `fields.py:394` |

**Why per-field normalisation is not optional**

| field | average length | default weight |
|---|---|---|
| title | 3.45 | 3.0 |
| abstract | 34.85 | 2.0 |
| methodology | 66.20 | 1.5 |
| results | 67.20 | 1.0 |

A title is **19× shorter** than a results section. Without dividing each field by
its own average, a results match outscores a title match for no good reason and
a weight of 3.0 on the title does almost nothing.

**Worked example** — `search` in p19's title

```
df_title(search) = 5 of 20 titles
idf_title = log(1 + (20−5+0.5)/(5+0.5)) = log(1 + 2.818) = 1.3398
avg title length = 3.45
tf(search) in p19 title = 1
```

Note the field idf, `1.3398`, is not the paper idf, `1.1727`. A term in 5 of 20
*titles* is rarer than a term in 5 of 20 *papers*, because a title is a much
smaller collection.

**Verify it yourself**

```powershell
python fields.py
```

---

## 5. Step 5 — Overlapped and non-overlapped lists

There is no arithmetic, so the "formula" is a rule and an operator.

### `M collapse M'`

```
matches  →  disjoint minimal half-open ranges
```

| Step | Does | Computed by | Line |
|---|---|---|---|
| take every occurrence | `Match(term, position)` per hit | `overlapped()` | `lists.py:246` |
| sort by position | `sorted(matches, key=…)` | `collapse()` | `lists.py:134` |
| merge runs | `if match.position <= end` | `collapse()` | `lists.py:139` |
| close a run | `spans.append((start, end))` | `collapse()` | `lists.py:142` |
| `M subtract M'` | cut holes out of spans | `subtract()` | `lists.py:149` |

`end` is **exclusive**, so `position <= end` is the adjacency test: offset 11
continues a run that already covers 10..10.

**Worked example**

```
matches   index@10, index@22, phrase@12, phrase@23, query@14
collapse  [(10,11), (12,13), (14,15), (22,24)]
             └── one run ─┘             └── 22 and 23 are adjacent, so merged
```

### The two models

| Model | Rule | Computed by | Line |
|---|---|---|---|
| Overlapped | every term in ONE region | `overlapped()` | `lists.py:228` |
| Non-overlapped | terms in DIFFERENT regions, union | `non_overlapped()` | `lists.py:279` |
| region score | BM25 inside one region | `_region_score()` | `lists.py:205` |

**Worked example** — query `document index positions`, paragraphs

```
OVERLAPPED       1 region
  p06:results:p2      [150,180)  30 terms  document, index, posit   4.9830

NON-OVERLAPPED   4 regions
  p11:results:p2      [147,178)  31 terms  index                     3.5282
  p06:methodology:p1  [ 43, 75)  32 terms  document, posit           3.2194
  p11:abstract:p1     [  2, 39)  37 terms  document, posit           2.9843
  p06:abstract:p1     [  3, 43)  40 terms  index                     1.7090
```

p11 contains all three terms but never two of them in one paragraph. The strict
model cannot return it at all. That is the trade.

**Verify it yourself**

```powershell
python lists.py
python -c "from structured_index import build_index; from lists import ListModels; m=ListModels(build_index()); print([(r.region_id, r.terms) for r in m.overlapped('document index positions','paragraph').regions])"
```

---

## 6. Step 6 — Proximal node model

No arithmetic either — the cost model is the interesting part.

### The bottom-up step

```
occurrence position  →  innermost node of kind P containing it
```

| Step | Does | Computed by | Line |
|---|---|---|---|
| read the postings list | `positions_of()` | `structured_index.py:131` |
| find the smallest enclosing node | `innermost_component()` | `proximity.py:162` |
| → | `enclosing()` | `structured_index.py:166` |
| count per node | `counts[node.region_id] += 1` | `proximity.py:368` |
| keep nodes with ≥ k | `if count >= k` | `proximity.py:370` |

**The cost argument, measured**

For `section with(2) retrieval`:

| | |
|---|---|
| occurrences of `retriev` | 28, in 9 papers |
| sections in the corpus | 80 |
| top-down: node × occurrence | 80 × 28 = **2,240** containment tests |
| bottom-up: walk the postings | **28** upward steps |
| speed-up | **80×** |

Both paths return the same 5 nodes. The upward step is `innermost_component()`,
and it is the entire performance argument for the model.

### The operators

| Operator | Meaning | Computed by | Line |
|---|---|---|---|
| `P in Q` | P inside Q | `_op_in()` | `proximity.py:280` |
| `P beginin Q` | only P's **start** inside Q | `_op_beginin()` | `proximity.py:295` |
| `P child Q` | direct children only | `_op_child()` | `proximity.py:306` |
| `P parent(k) Q` | P is parent of ≥ k Q nodes | `_op_parent()` | `proximity.py:372` |
| `P with(k) T` | P holds ≥ k occurrences of T | `_op_with()` | `proximity.py:354` |
| `P before(k) T (C)` | P ends ≤ k terms before T | `_op_near()` | `proximity.py:394` |
| context `(C)` | both inside one C node | `_in_context()` | `proximity.py:426` |

**Worked example** — the distance dial, `sentence before(k) index (paragraph)`

| k | nodes | papers |
|---|---|---|
| 0 | 2 | 2 |
| 2 | 3 | 3 |
| 5 | 6 | 5 |
| 10 | 9 | 6 |
| 25 | 17 | 6 |
| 60 | 33 | 6 |

Monotone by construction, and distance is measured in **index terms**, so the
number means the same thing in a 4-term title and a 70-term section.

**Verify it yourself**

```powershell
python proximity.py
```

---

## 7. Step 7 — Evaluation

| Metric | Formula | Computed by | Line |
|---|---|---|---|
| Precision@k | hits in first k ÷ k | `precision_at_k()` | `evaluate.py:121` |
| Recall@k | hits in first k ÷ all relevant | `recall_at_k()` | `evaluate.py:131` |
| MRR | 1 ÷ rank of the first relevant hit | `reciprocal_rank()` | `evaluate.py:142` |
| MAP | Σ P@rank over relevant hits ÷ #relevant | `average_precision()` | `evaluate.py:150` |

**Measured** — 12 judged queries, 20 papers

| model | P@1 | P@5 | R@10 | MRR | MAP | queries with no hit |
|---|---|---|---|---|---|---|
| boolean AND | 0.833 | 0.833 | 0.667 | 0.833 | 0.667 | 2 |
| tfidf cosine | 0.917 | 0.617 | 0.958 | 0.958 | 0.883 | 0 |
| BIM | 0.833 | 0.633 | 0.958 | 0.917 | 0.875 | 0 |
| **BM25** | **1.000** | 0.617 | 0.958 | **1.000** | **0.925** | 0 |
| BM25F | 0.917 | 0.617 | 0.958 | 0.958 | 0.904 | 0 |
| overlapped | 0.750 | 0.750 | 0.542 | 0.750 | 0.542 | 3 |
| non-overlapped | 0.833 | 0.833 | 0.667 | 0.833 | 0.667 | 2 |
| proximal | 0.750 | 0.683 | 0.708 | 0.764 | 0.609 | 2 |

BM25 wins because it ranks **whole papers**. The list models only return a paper
when one **region** holds every term, which is exactly the coverage they trade
away — and on paper-level judgements that trade does not pay.

Read that as a fact about twelve queries, not as a general claim. The qrels file
says so in its own header.

**Verify it yourself**

```powershell
python evaluate.py
```

---

## 8. Where the user sees all of this

| Surface | Entry point | Line |
|---|---|---|
| Command line | `python main.py --console "query"` | `main.py:34` |
| Web UI | `python main.py --web` → `app.py` | `main.py:42` |
| Shared engine | `search(query, mode, top, k1, b, …)` | `engine.py:277` |
| BM25 handler | `_bm25()` | `engine.py:370` |
| BIM handler | `_bim()` | `engine.py:363` |
| field handler | `_fields()` / `_bm25f()` | `engine.py:378` / `engine.py:385` |
| list handlers | `_overlapped()` / `_non_overlapped()` | `engine.py:408` / `engine.py:414` |
| node handler | `_proximal()` | `engine.py:443` |

`engine.py` holds no scoring logic of its own — it only decides which model runs
and wraps the result in a `SearchResponse`. That is why `app.py` and
`console_ui.py` can never disagree about an answer: they both call
`engine.search()`.

### Saying what was *not* searched

Four behaviours exist because a zero-hit answer is ambiguous, and the interface
has to pick one reading and admit it. Each of these was a real bug first.

| what the user typed | what happened before | now |
|---|---|---|
| `zzzznotaword` | silently 0 hits, or **20/20** in Boolean mode | `coverage_note()`: "1 word in your query was never searched" |
| `proximity` | 0 hits, no clue why | `phrase_hint()`: "'proxim' only occurs inside 'proxim oper' or 'proxim window'" |
| `the` | **20/20** — every paper trivially contains zero terms | "none of the words in 'the' were searched - they are all stop words" |
| `title:neural` in BM25 mode | searched for the *word* "title", silently | `misrouted_field_prefix()`: use `fields` or `bm25f` |
| `machine learning` | "1/1" — looked like a full match | `matched / total_terms + len(unknown)` |

The 20/20 case is the clearest. `boolean` requires every query term to be
present, so with zero terms every paper satisfies it; that is a vacuous truth,
not a match. The check that closes it lives in `search()`, so **all** modes agree
rather than each one inventing its own answer.

`coverage_note()` counts only words that were **dropped**. An earlier version
counted searched *terms* against typed *words*, which made a legitimate phrase
look like a partial miss — "1 of 2 words" for a perfect hit on `inverted index`
is its own lie.

---

## 9. Maintenance note

Line numbers in these pages were **wrong twice** before they were checked, for two
different reasons: a line moved, and a number that had never been measured at all
(`bm25f "title:search AND results:precision"` was documented as 6 papers and
actually returns 10). So they are executable now:

```powershell
python check_doc_lines.py    # every file.py:NN points at code, and says what it must contain
python check_doc_claims.py   # every number quoted in these pages is re-derived from the corpus
```

Both are wired into `verify.py`. The first checks that each cited line *contains
the thing the page claims it contains* — checking only "is this a line of code"
let `fields.py:217` through, a real line and the wrong one. The second catches
measurements that were never taken.

The **function names** remain the stable reference; the line numbers are a
convenience. The test suite is the stronger guarantee — it re-derives every
identity quoted here rather than trusting the page.