# Step 5 — Overlapped and Non-Overlapped Lists

Guide to `lists.py`: retrieving **regions** instead of documents, and the two
different answers to the question *"have the query terms ever met?"*

Week 01 returned whole documents. Everything since step 2 has been about **where**
in a paper the evidence is. This step returns the region, and asks a question that
sounds like a technicality and is not.

---

## 1. The two models

| | Overlapped | Non-overlapped |
|---|---|---|
| **Rule** | every query term must occur inside **ONE** region | terms may occur in **DIFFERENT** regions; the answer is the union |
| **Operator** | `M collapse M'` | one region per term, unioned |
| **Gains** | precision — a returned region really does discuss the whole query | coverage — the paper is found wherever the terms are |
| **Costs** | a paper with term A in its methodology and term B in its results is **invisible** | the regions returned together may be **unrelated** |
| **Best for** | exact multi-term search | exploratory structured search |

| | Computed by | Line |
|---|---|---|
| overlapped | `ListModels.overlapped()` | `lists.py:228` |
| non-overlapped | `ListModels.non_overlapped()` | `lists.py:279` |
| `M collapse M'` | `collapse()` | `lists.py:118` |
| `M subtract M'` | `subtract()` | `lists.py:149` |
| region score | `_region_score()` | `lists.py:205` |

---

## 2. `M collapse M'` — the operator both models share

Matches are individual **term occurrences**. Collapse turns them into disjoint
**half-open ranges** `[start, end)`.

| Step | Does | Line |
|---|---|---|
| take every occurrence | `Match(term, position)` | `lists.py:246` |
| sort by position | `ordered = sorted(matches, …)` | `lists.py:134` |
| merge a run | `if match.position <= end:` | `lists.py:139` |
| close a run | `spans.append((start, end))` | `lists.py:142` |

**Adjacent occurrences merge; occurrences with a gap do not.**

```
matches   index@10, index@22, phrase@12, phrase@23, query@14
collapse  [(10,11), (12,13), (14,15), (22,24)]
             └── three separate runs ─┘  └─ 22 and 23 are adjacent, so merged
```

Positions 10 and 11 have nothing between them, so a region containing both holds
no term it does not have. Positions 10 and 13 have two unmatched terms between
them, so merging them would claim a stretch of text **no query term touched**.

The test is `position <= end`, not `<`, because `end` is **exclusive**: offset
11 continues a run that already covers offsets 10..10.

**The first version of this file used `<`, and adjacent terms never merged** —
three consecutive query terms produced three one-term regions instead of one
three-term region. `tests/test_lists.py::TestCollapse` now pins the behaviour in
five tests.

---

## 3. The three hierarchies

A hierarchy is one flat, non-overlapping view of the text:

| hierarchy | regions per corpus | mean length |
|---|---|---|
| `section` | 80 | 42.9 |
| `paragraph` | 120 | 28.6 |
| `sentence` | 299 | 11.5 |

Within one hierarchy the regions **tile** the paper: contiguous, no gaps, no
overlaps. `structured_index.check_invariants()` proves that rather than trusting
it, and `tests/test_corpus.py` asserts it.

**Across** hierarchies they do overlap — a sentence sits inside a paragraph,
which sits inside a section. That *is* the nesting. The names describe the
structure, not the query behaviour, and confusing those two readings is the most
common way to misdescribe this step.

**Choosing a hierarchy is choosing the granularity of the answer.** The
sentence hierarchy returns the most regions and the least text in each; the
section hierarchy returns the fewest regions and the most text.

---

## 4. The same query, both models

Query `document index positions`, paragraphs.

**OVERLAPPED — 1 region**

| region | range | terms | answered | score |
|---|---|---|---|---|
| `p06:results:p2` | `[150,180)` | 30 | document, index, posit | 4.9830 |

**NON-OVERLAPPED — 4 regions**

| region | range | terms | answered | score |
|---|---|---|---|---|
| `p11:results:p2` | `[147,178)` | 31 | index | 3.5282 |
| `p06:methodology:p1` | `[43,75)` | 32 | document, posit | 3.2194 |
| `p11:abstract:p1` | `[2,39)` | 37 | document, posit | 2.9843 |
| `p06:abstract:p1` | `[3,43)` | 40 | index | 1.7090 |

```
both                 p06
non-overlapped only  p11        ← contains all three terms, never two in one paragraph
overlapped only      none
```

**p11 is exactly the case the distinction exists for.** It is about positional
indexes for phrase queries, so it is genuinely relevant. It mentions *index* in
its results, *document* and *position* in its abstract, and **never two query
terms in the same paragraph**. The overlapped model cannot return it at all.

---

## 5. When the strict model returns nothing

Query `ranking web search`:

```
overlapped       0 regions
non-overlapped   4 regions in 2 papers
```

That is not a bug. No paragraph holds *ranking*, *web* and *search* together.

A system that hid this behind a zero-result page would look broken, so the
interface has to offer the broader model **next to** the strict one. That is the
whole argument for building both:

```powershell
python main.py --console "both: ranking web search"
```

prints them side by side with the set difference underneath.

---

## 6. Coverage across seven queries

| query | ovlp regions | ovlp papers | non regions | non papers |
|---|---|---|---|---|
| document index positions | 1 | 1 | 4 | 2 |
| ranking web search | 0 | 0 | 4 | 2 |
| passage retrieval recall | 0 | 0 | 2 | 1 |
| evaluation precision metrics | 0 | 0 | 2 | 1 |
| query expansion embeddings | 0 | 0 | 2 | 1 |
| mean average precision | 5 | 2 | 2 | 2 |
| term weighting saturation | 3 | 1 | 2 | 1 |

Note the last two rows: the overlapped model finds **more** regions there. When
the query terms genuinely do co-occur, collapsing them into one region is a
*compression*, and it is a win.

The coverage direction is not one-way. It depends entirely on whether the corpus
talks about the query in one place or several.

---

## 7. Region scoring

Regions are **returned** by the rule above and **ordered** by BM25 computed
inside the region (`lists.py:205`).

```python
sat = (tf * 2.2) / (tf + 1.2 * ((1.0 - 0.75) + 0.75 * length / average))
total += self.idf_table.idf_for(term) * sat
```

`df` is the **corpus-wide** df, deliberately. The region decides *which* documents
are returned; the term weights only order the regions that already matched. Using
a per-region df would make the ordering depend on a collection of regions that
changes with the query.

---

## 8. What the evaluation says

| model | P@1 | P@5 | R@10 | MRR | MAP | queries with no hit |
|---|---|---|---|---|---|---|
| BM25 | 1.000 | 0.617 | 0.958 | 1.000 | 0.925 | 0 |
| non-overlapped | 0.833 | 0.833 | 0.667 | 0.833 | 0.667 | 2 |
| overlapped | 0.750 | 0.750 | 0.542 | 0.750 | 0.542 | **3** |

**The list models score worse, and that is the honest result.** The judgements in
`qrels.txt` are about **papers**, and the list models only return a paper when a
single **region** holds every query term. That is precisely the coverage they
trade away.

Read the columns honestly and they are not the same story:

- **P@5 is *higher* for both list models** (0.750 / 0.833) than for BM25
  (0.617). Narrowing to regions does filter the tail.
- **R@10 is much worse** (0.542 / 0.667 vs 0.958). When the model cannot find the
  paper at all, recall collapses.
- The `no hit` column — **3** queries for overlapped — is invisible in MRR and is
  the single number that exposes the problem.

A judgement set that scored **regions** instead of papers would be the fair test
for these models. That would need region-level qrels, and it is the obvious next
step for this FYP.

---

## 9. What this step deliberately does **not** do

- **It does not merge across hierarchies.** An overlapped answer drawn from
  paragraphs and one drawn from sentences are different answers, so the
  hierarchy is a parameter and never mixed.
- **It does not rank papers.** It ranks *regions*. Step 7 converts to paper-level
  rankings for the evaluation, and that conversion is a stated modelling
  choice, not a detail.
- **It does not implement `M subtract M'` in the query language.** The function
  exists and is tested; no operator currently exposes it.
- **It does not claim the coverage is free.** Extra regions cost precision, and
  the table in §6 shows the two models swapping places depending on the query.