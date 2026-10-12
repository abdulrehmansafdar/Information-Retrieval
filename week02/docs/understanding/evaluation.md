# Step 7 — Evaluating the Models

Guide to `evaluate.py` and `data/relevance/qrels.txt`: whether any of this
actually works, and — just as important — what the resulting numbers do **not**
license you to claim.

Every number printed by steps 3 to 6 is a property of the **model**. A system
that ranks p01 first for *"proximity window phrase query"* is not more right than
one that ranks p19 first unless someone has said which paper answers the
question. That is what this step is for.

---

## 1. The judgements

`data/relevance/qrels.txt`, in TREC format:

```
<query_id> <paper_id> <0 or 1>
```

A `1` means an assessor judged that paper **answers** that query. A `0` means an
assessor looked at it and judged that it **does not**.

**The 0s are the hard part of a qrels file and the part most often left out.**
Without them:

- a system that returns one paper and happens to be right scores **1.0**;
- recall and MAP are simply **undefined**.

This file has **18 positives and 18 negatives** across 12 queries.

| | queries |
|---|---|
| `q01` | proximity operator structured text |
| `q02` | inverted index compression postings |
| `q03` | boolean retrieval |
| `q04` | cross encoder reranking retrieved passages |
| `q05` | document index positions |
| `q06` | term weighting saturation length |
| `q07` | mean average precision reciprocal rank |
| `q08` | learning to rank |
| `q09` | anomaly detection clinical sensor |
| `q10` | graph convolution recommendation |
| `q11` | proximity window phrase query |
| `q12` | metadata structured field search |

They were chosen to cover the cases the earlier steps **claim** to differ on —
single clear topic, two-word topic, title-only, results-only, terms spread over
a paper, terms adjacent in one place, and a term in exactly one paper.

### How they were judged, and the hard call

Binary relevance, judged by hand against each paper's title, abstract,
methodology and results.

The judgement worth defending: **`q07` asks about the evaluation metrics, and
`p16` is marked 0** — even though its results section literally reports *"nine
points of mean average precision"*. `p16` is about term weighting and **borrows**
the metric to report its own numbers. Using a term is not discussing it.

A second: **`q12` is about structured field search and `p15` is marked 0**,
though p15 is about faceted search. `p15` never uses the word *metadata* and its
title says nothing about structure.

These two calls are where a different assessor might disagree, and the whole
0.042 MRR spread in §4 is not large enough to survive them.

---

## 2. The metrics

| Metric | Formula | Computed by | Line |
|---|---|---|---|
| Precision@k | hits in first k ÷ k | `precision_at_k()` | `evaluate.py:121` |
| Recall@k | hits in first k ÷ all relevant | `recall_at_k()` | `evaluate.py:131` |
| MRR | 1 ÷ rank of the first relevant hit | `reciprocal_rank()` | `evaluate.py:142` |
| MAP | Σ P@rank over relevant hits ÷ #relevant | `average_precision()` | `evaluate.py:150` |

Two details that are easy to get wrong, and are pinned by
`tests/test_evaluation.py`:

**Precision divides by what there *was*, not by k.** Returning one relevant
result out of one tried is **1.0**. Padding the ranking with non-results to reach
k would hide that.

**Recall with nothing relevant is 0.0, not undefined.** The caller excludes such
queries from the average instead. All 12 queries here have at least one relevant
paper.

### Why report both MRR and MAP

They fail in opposite directions.

- **MRR only looks at the first hit.** A system that puts one good answer first
  and then fills with noise scores 1.0.
- **MAP only counts hits.** It is the untruncated average of Precision@rank over
  every relevant result found, so a system that finds everything but buries half
  of it is punished.

Reporting one alone is reporting the flattering half.

---

## 3. Overall, every model on every query

| model | P@1 | P@5 | R@10 | MRR | MAP | queries with no hit |
|---|---|---|---|---|---|---|
| boolean AND | 0.833 | 0.833 | 0.667 | 0.833 | 0.667 | 2 |
| tfidf cosine | 0.917 | 0.617 | 0.958 | 0.958 | 0.883 | 0 |
| BIM | 0.833 | 0.633 | 0.958 | 0.917 | 0.875 | 0 |
| **BM25** | **1.000** | 0.617 | 0.958 | **1.000** | **0.925** | 0 |
| BM25F | 0.917 | 0.617 | 0.958 | 0.958 | 0.904 | 0 |
| overlapped | 0.750 | 0.750 | 0.542 | 0.750 | 0.542 | **3** |
| non-overlapped | 0.833 | 0.833 | 0.667 | 0.833 | 0.667 | 2 |
| proximal | 0.750 | 0.683 | 0.708 | 0.764 | 0.609 | 2 |

**The spread between best and worst is 0.250 MRR.** That is the only thing these
numbers support.

### `no hit` is the column that matters most

It counts queries where the model **never returned a judged-relevant paper at
all**. MRR is blind to it by construction — a model scoring 0.0 on a query looks
exactly like one that scored 0.4.

The overlapped list model fails on **3 of 12** queries. Those are the queries
whose terms are spread across different paragraphs.

---

## 4. Query by query, MRR per model

| q | bool | tfidf | bim | bm25 | bm25f | overlap | non-ovl | proxim | rel |
|---|---|---|---|---|---|---|---|---|---|
| q01 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2 |
| q02 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2 |
| q03 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q04 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2 |
| q05 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2 |
| q06 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.17** | 1 |
| q07 | **0.00** | 1.00 | 0.50 | 1.00 | 0.50 | **0.00** | **0.00** | **0.00** | 2 |
| q08 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q09 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.00** | 1.00 | 1.00 | 1 |
| q11 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2 |
| q12 | **0.00** | 0.50 | 0.50 | 1.00 | 1.00 | **0.00** | **0.00** | **0.00** | 1 |

Ten of twelve queries are easy for every model — the corpus is small and the
topics are distinct. **The evaluation is decided by two queries**, and both have a
mechanism worth understanding. Neither is about the parameters k₁ and b.

### `q07` — phrase extraction starved a query term

Query *"mean average precision reciprocal rank"* →
terms `mean averag precis`, `reciproc`, `rank`.

| term | df | idf | papers |
|---|---|---|---|
| `mean averag precis` | 2 | 2.1282 | p09, p16 |
| **`reciproc`** | **1** | **2.6391** | **p12 only** |
| `mean reciproc rank` | 5 | 1.3398 | p03, p05, p09, p12, p14 |
| `rank` | — | — | many |

**`reciproc` survives as a standalone term in exactly one paper.** Everywhere
else, "reciprocal" is absorbed by the phrase `mean reciprocal rank` and never
becomes a term of its own. So it carries the *highest* idf in the query and it
matches only p12 — which does not contain `mean averag precis`.

```
boolean / overlapped / non-overlapped   →  []   no paper has all three terms
bim                                    →  p12 first, p09 second   MRR 0.50
tfidf, bm25                            →  p09 first                MRR 1.00
```

Three models return **nothing**, because a query can only be answered if some
unit contains every term, and no unit does. This is the clearest failure in the
whole evaluation and it is a *preprocessing* failure, not a ranking failure.

The fix is not a better ranking function. It is a better phrase list: if
`mean reciprocal rank` were not a phrase, `reciproc` would appear wherever the
word does and q07 would behave like q01.

### `q12` — length normalisation breaking a tie

Query *"metadata structured field search"* → terms `metadata`,
`structur field`, `search`. One relevant paper: **p19**.

```
p15   len=167   metadata tf=1 (in the METHODOLOGY)   structur field tf=0   search tf=2
p19   len=161   metadata tf=0                        structur field tf=1 (in the TITLE)   search tf=2
```

Both papers match exactly two terms, with **identical idf and identical tf**.

```
BIM   p15 = 3.811778
      p19 = 3.811778          ← an EXACT TIE
```

BIM has no length term and no saturation, so it genuinely cannot separate them.
The tie breaks on `doc_id`, and `p15` < `p19` alphabetically. MRR 0.50.

BM25 separates them on length:

```
p15   K(d) = 0.9795   score 4.2939
p19   K(d) = 0.9533   score 4.3494      ← shorter paper, higher saturation
```

p19 is 6 terms shorter than average-average, so its length norm is lower, its
saturation is higher, and it wins. MRR 1.00.

**This is the single clearest demonstration of what b is for on this corpus** —
and it is worth noting that the "length effect is mild here" caveat in
[bm25.md](bm25.md) §3 still holds: one query out of twelve, 0.042 MRR.

BM25F also gets p19 first, for a different reason: `structur field` is in p19's
**title**, weighted 3.0, while `metadata` is in p15's **methodology**, weighted
1.5. That is the field weighting earning its keep — on one query.

---

### The lesson, stated properly

> The evaluation is decided by **how terms survive preprocessing** (`q07`) and
> **whether a tie can be broken at all** (`q12`) — not by the ranking function
> and not by k₁ or b, which are completely flat above zero.

Both are more interesting than "BM25 scored highest".

---

## 5. Precision and recall at every cutoff

Precision at 10 tells a different story from precision at 1:

| model | P@1 | R@10 |
|---|---|---|
| BM25 | **1.000** | 0.958 |
| tfidf cosine | 0.917 | 0.958 |
| BM25F | 0.917 | 0.958 |
| BIM | 0.833 | 0.958 |
| boolean AND | 0.833 | 0.667 |
| non-overlapped | 0.833 | 0.667 |
| overlapped | 0.750 | 0.542 |
| proximal | 0.750 | 0.708 |

BM25 has the **highest P@1 and the lowest P@5** of the graded models (0.617,
tied with cosine and BM25F). It is better at the very top and no better in the
tail. Reporting only MRR would hide that entirely.

---

## 6. What these numbers do and do not support

They **do** support:

- On these 12 queries, with these judgements, **BM25 ranked a judged-relevant
  paper first for every query** (MRR 1.000, MAP 0.925).
- The two list models and the proximal model **returned nothing at all** for
  3, 2 and 2 queries respectively.
- The difference is driven by **two specific queries**, for two specific
  reasons: phrase extraction starving a query term (q07), and BIM being unable
  to break an exact tie that BM25's length normalisation breaks (q12).
- The k₁ and b parameters make **no difference at all** above k₁ = 0.

They **do not** support:

- **"BM25 is better than cosine."** The gap is 0.042 MRR on 12 queries with
  hand-made judgements, and it comes down to a **single query (q12)** where two
  papers tie. `qrels.txt` names two contested calls that could move it.
- **"Field weighting helps."** BM25F matches plain BM25 on MRR here and never
  discards anything, so there was almost nothing for the weighting to fix. It
  wins exactly one query, q12, and for the same tie-breaking reason.
- **Anything about a real collection.** 12 queries over 20 papers is a
  demonstration of *how to measure*, not a benchmark.
- **A tuned k₁ or b.** They are literature defaults. Fitting them on these 12
  queries and reporting the result on those same 12 queries would be circular —
  and the sweep in [bm25.md](bm25.md) §2 shows there is nothing to tune.

### The obvious next step

The list models are being judged on **paper-level** judgements while retrieving
**regions**. A region-level qrels file — where each judgement says whether *this
paragraph* answers the query — would be the fair test for them, and would likely
reverse the ordering. That is the first thing to add if this continues.

---

## 7. Reproducing it

```powershell
python evaluate.py
```

Deterministic: ties break on `doc_id`, and
`tests/test_evaluation.py::TestEvaluationRun::test_the_results_are_reproducible`
asserts that two runs agree exactly.

The judgements live in `data/relevance/qrels.txt`, in plain text, with the
reasoning in comments. They can be edited, re-run, and argued with without
opening a line of Python.