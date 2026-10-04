# Step 5 — TF, DF and IDF

Guide to `idf.py`: how term frequency fails on its own, and how IDF fixes it.

---

## 1. The core idea

Step 3 stored one extra number in every posting without using it. This step is about
that number, plus two more:

| Measure | Definition | Cost |
|---|---|---|
| **tf(t,d)** | how many times term *t* occurs in document *d* | free — already in every posting |
| **df(t)** | how many documents contain *t* | free — the postings-list length |
| **idf(t)** | how much the term is worth: `log(N / df(t))` | the only new computation |

Only IDF is new. That is why this step is cheap — 608 idf values from 608 postings-list
lengths.

---

## 2. Why TF alone is not enough

Here is doc03's own data, which is a complete argument for IDF:

| term | tf | df | idf |
|---|---|---|---|
| `use` | **4** | 17 | 0.1625 |
| `data` | 2 | 17 | 0.1625 |
| `artifici intellig` | 2 | 12 | 0.5108 |
| `healthcar` | 2 | 3 | 1.8971 |
| `clinic` | 2 | 2 | **2.3026** |
| `patient` | 2 | 2 | **2.3026** |

Read the tf column: **`use` wins.** It is the most frequent term in the document.

Read the idf column: **`clinic` and `patient` win, by 14×.** They appear in only 2 of
20 documents, so finding them is meaningful. `use` appears in 17 of 20 — it tells you
almost nothing about which document you are looking at.

TF promotes the word that says least.

### The same tf means opposite things in different documents

| Document | Most frequent term | tf | df | idf |
|---|---|---|---|---|
| doc03 | `use` | 4 | 17 | 0.1625 |
| doc13 | `autonom` | 3 | 1 | **2.9957** |

doc13 (Autonomous Robots) is *about* autonomous systems, and `autonom` occurs in exactly
one document in the whole corpus. doc03 uses `use` four times as ordinary filler. A raw
tf score would rank these close together. IDF separates them by **18×**.

---

## 3. The formula, and why this one

```
idf(t) = log(N / df(t))          N = 20
```

- A term in **every** document → `df = N` → `log(1) = 0` → worth nothing. Correct.
- A term in **one** document → `log(20)` = 2.996 → maximum. Correct.
- In between, the logarithm makes rare terms count much more than common ones.

Alternatives seen in the literature: `log(N/(1+df))`, `1 + log(N/df)` (Lucene),
`log((N+1)/(df+1)) + 1` (sklearn). All exist mainly to avoid a zero idf.

### This corpus needs no smoothing

| Property | Value |
|---|---|
| Highest df in the corpus | 17 |
| N | 20 |
| Any idf equal to 0? | **no** |
| Any idf negative? | **no** |

Because no term appears in all 20 documents, `log(N/df)` stays strictly positive, so the
plain textbook formula is safe and the simplest to defend. If a term *did* reach df = N,
its idf would be exactly 0 — which is the right answer, not a bug — and smoothing would
only paper over that.

---

## 4. The measured table

| df | terms | idf |
|---|---|---|
| 1 | 309 | 2.9957 |
| 2 | 139 | 2.3026 |
| 3 | 56 | 1.8971 |
| 4 | 39 | 1.6094 |
| 5 | 17 | 1.3863 |
| 6 | 19 | 1.2040 |
| 7 | 10 | 1.0498 |
| 8 | 2 | 0.9163 |
| 9 | 5 | 0.7985 |
| 10 | 3 | 0.6931 |
| 11 | 1 | 0.5978 |
| 12 | 2 | 0.5108 |
| 13 | 1 | 0.4308 |
| 15 | 1 | 0.2877 |
| 16 | 1 | 0.2231 |
| 17 | 3 | 0.1625 |

16 distinct df values across 608 terms. Read the idf column downward: as df rises, idf
falls. **That is the entire idea.**

Note there is no row for df = 14 or df = 18–20. No term occurs in exactly 14 documents,
and none occurs in all 20.

---

## 5. Key terms, weighted

| term | df | idf | verdict |
|---|---|---|---|
| `data` | 17 | 0.1625 | worthless for ranking |
| `system` | 17 | 0.1625 | worthless for ranking |
| `use` | 17 | 0.1625 | worthless for ranking |
| `model` | 16 | 0.2231 | worthless for ranking |
| `softwar` | 15 | 0.2877 | worthless for ranking |
| `artifici intellig` | 12 | 0.5108 | weak |
| `machin learn` | 9 | 0.7985 | weak |
| `secur` | 6 | 1.2040 | useful |
| `comput vision` | 3 | 1.8971 | very useful |
| `healthcar` | 3 | 1.8971 | very useful |
| `nlp` | 2 | 2.3026 | very useful |
| `anomali detect` | 2 | 2.3026 | very useful |
| `autonom` | 1 | 2.9957 | identifies its document |
| `iot` | 1 | 2.9957 | identifies its document |

Total spread: **0.1625 → 2.9957, an 18.4× ratio.**

This is precisely the frequency gradient step 3 displayed as postings-list lengths,
now turned into numbers that can be multiplied against tf.

---

## 6. Honest note on TF's second weakness

TF is also known to favour **long documents** — a long document has more chances to
accumulate a high count for any term. That is usually taught alongside rarity blindness.

It is worth being straight about this corpus:

```
document lengths: 71 to 90 terms
ratio: 1.27x
```

The bias is real but **mild here**, because the documents are nearly uniform in length —
they were all written to the same 120–180 word specification. On a corpus of mixed
article lengths, where one document is twenty times another, length bias would dominate
this table. Do not claim a dramatic length effect on this dataset; there isn't one.

---

## 7. API

```python
from idf import build_idf_model, term_frequencies

counts = term_frequencies(document.terms)      # Counter, one document
counts["use"]                                 # 4

model = build_idf_model(index)
model.number_of_documents                      # 20
model.idf_for("clinic")                        # 2.3026
model.df_for("clinic")                         # 2
model.rarest_terms(6)                          # highest idf first
model.most_common_terms(6)                     # lowest idf first
model.distinct_df_values()                     # [(df, count, idf), ...]
```

### `idf_for` on a term outside the corpus

```python
model.idf_for("blockchain")    # 2.9957
```

A term with no postings has no measured df. The convention used here is to give it the
weight of `df = 1` — the maximum available, since it can appear in at most one document.

**This never affects a result**, because a term absent from the index contributes to no
document's score. The convention exists only so the function has no `None` case and no
division by zero. It is a convention, not a measurement, and is documented as such.

---

## 8. Verification

`python idf.py` prints seven sections. All of the following pass:

| # | Check | Result |
|---|---|---|
| 1 | All 608 idf values equal `log(N/df)` | PASS |
| 2 | 14 measured values match to 4 decimal places (`data` 0.1625 … `autonom` 2.9957) | PASS |
| 3 | No idf is zero or negative; max df 17 < N 20; strictly decreasing in df; 18.43× spread | PASS |
| 4 | Every df equals `len(postings[term])`; 16 distinct df values; df=1 → 309 terms; df counts sum to 608 | PASS |
| 5 | tf in every posting equals `term_frequencies(doc)` for all 20 documents; per-document tf sums equal document length; collection total 1,624 | PASS |
| 6 | The TF-fails demo is factually true (doc03 top term `use` tf=4 df=17; `clinic` tf=2 df=2 and 14.2× more valuable; doc13 top term `autonom` tf=3 df=1) | PASS |
| 7 | Unknown terms return a float and never raise | PASS |
| 8 | Earlier steps untouched — 20 docs, 1,624 terms, 608 vocab, 1,401 postings | PASS |

Check 5 is the important one: it proves the tf already stored in the postings is
**exactly** what `term_frequencies()` computes independently. Step 5 therefore analyses
the number rather than trusting a copy of it.

---

## 9. What this step deliberately does **not** do

| Feature | Step |
|---|---|
| **TF-IDF** = tf × idf | **6** |
| Document length normalisation | 6 |
| Vector Space Model | 8 |
| Cosine similarity, ranked results | 9 |

TF and IDF are shown **side by side and deliberately never multiplied.** That product
is TF-IDF, it belongs to step 6, and computing it here would blur the two ideas this
step exists to separate.

---

## 10. Honest limitations

- **Logarithms have no natural unit**, so an idf of 2.3026 is not "twice as much as"
  1.6094 in any meaningful sense. Only the ordering and rough magnitude are meaningful.
- **The formula was chosen partly because it is convenient here.** With only 20
  documents, all idf values sit in a narrow 0.16–3.00 band, which compresses
  differences between moderately common terms.
- **No smoothing variant is implemented**, so this code would need revisiting for a
  corpus where some term appears in every document. The change would be one line in
  `build_idf_model`.
- **idf ignores tf entirely.** Two documents containing a term once each get identical
  idf for it, however differently important it is to each. That gap is what step 6 fills.
- **`rarest_terms` / `most_common_terms` are alphabetical within a tie.** With 309 terms
  sharing the maximum idf 2.9957, "the rarest term" is not a unique answer, and the
  function does not pretend otherwise.