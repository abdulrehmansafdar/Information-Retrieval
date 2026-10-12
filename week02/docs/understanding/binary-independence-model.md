# Step 3 — Probability-Based Ranking: BIM and BM25

Guide to `ranking.py`: probability-based ranking, the binary indicator that
limits BIM, and why BM25 is the same formula at a different setting of one
parameter.

---

## 1. The modelling assumption

The **Binary Independence Model** assumes that whether term *t* appears in a
paper does not depend on whether any other term appears in it.

Under that assumption the evidence from each term **multiplies**, so in log
space it **adds**:

```
score(q, d) = Σ over t in q of  idf(t) · 1[t ∈ d]
```

`1[t ∈ d]` is the binary indicator: 1 if the term appears at all, 0 otherwise.
Ten occurrences count exactly as much as one. That is the model's defining
limitation, and it is what BM25 fixes.

| Symbol | On this corpus | Computed by | Line |
|---|---|---|---|
| `idf(t)` | `2.1282` for `field` | `IdfTable._idf_value()` | `ranking.py:101` |
| `N` | `20` | `number_of_documents` | `ranking.py:94` |
| `df(t)` | `2` | `len(self.index.lookup(term))` | `ranking.py:97` |
| `1[t ∈ d]` | `1` or `0` | `if tf <= 0:` | `ranking.py:234` |
| the sum | `total += idf` | `score_paper()` | `ranking.py:248` |

**Worked example** — `idf("field")`

```
N   = 20                          20 papers
df  = 2                           'field' is in p04 and p19
idf = log( 1 + (20 − 2 + 0.5) / (2 + 0.5) )
    = log(1 + 7.4)
    = log(8.4)
    = 2.128232
```

**Verify it yourself**

```powershell
python ranking.py
python -c "import structured_index as s, ranking as r; print(r.IdfTable(s.build_index()).idf_for('field'))"
# 2.1282319810855375
```

---

## 2. Three IDF variants, and why the choice matters

| Variant | Formula | Behaviour |
|---|---|---|
| `plain` (week01) | `log(N / df)` | exactly **zero** for a term in every paper |
| `robertson` | `log((N−df+0.5)/(df+0.5))` | goes **negative** once `df > N/2` |
| `bm25` (this week) | `log(1 + (N−df+0.5)/(df+0.5))` | **always positive** |

Measured on this corpus:

| term | df | plain | robertson | bm25 |
|---|---|---|---|---|
| `retriev` | 9 | +0.7985 | +0.1911 | +0.7932 |
| `queri` | 13 | +0.4308 | **−0.5878** | +0.4418 |
| `document` | 12 | +0.5108 | **−0.3857** | +0.5188 |
| `field` | 2 | +2.3026 | +2.0015 | +2.1282 |
| `deep learn` | 1 | +2.9957 | +2.5649 | +2.6391 |

**17 of 966 terms** have a negative Robertson idf here. A negative contribution
is evidence *against* a paper, which is defensible but very hard to explain, and
its worst property is that **a score can fall when you add a query term**.

This week uses the `bm25` variant: every matched term adds, never subtracts.

`plain` is still here for the week01 baseline in `engine.py`, so that "BM25
beats week01" is a comparison against what week01 actually computed rather than
against a strawman.

---

## 3. What the binary indicator costs

```
tf("field", p19) = 15        the highest term frequency in this corpus
```

| model | tf = 1 | tf = 15 | gain from repeating |
|---|---|---|---|
| BIM | 2.1282 | **2.1282** | **1.00×** |
| BM25 | 2.1839 | 4.3503 | 1.99× |

BIM cannot tell a paper that is *about* structured fields from one that merely
mentions them fifteen times. That is the whole gap between step 3's two models.

**Verify it yourself**

```powershell
python -c "import structured_index as s, ranking as r; i=s.build_index(); b=r.Bm25Model(i); print(b.term_score('field',1,161), b.term_score('field',15,161))"
```

---

## 4. BM25 and its two parameters

```
score(q, d) = Σ over t in q of  idf(t) · tf·(k₁+1) / (tf + k₁·K(d))

K(d) = (1 − b) + b · |d| / avgdl
```

| Symbol | Meaning | Computed by | Line |
|---|---|---|---|
| `tf` | occurrences in *d* | `term_frequency_in()` | `ranking.py:333` |
| `saturation` | `tf(k₁+1)/(tf + k₁·K(d))` | `saturation()` | `ranking.py:308` |
| `K(d)` | length norm | `length_norm()` | `ranking.py:308` |
| `idf(t)` | from step 5 | `idf_for(term)` | `ranking.py:336` |
| contribution | `idf × sat` | `score_paper()` | `ranking.py:338` |
| the sum | | `total += contribution` | `ranking.py:339` |

### k₁ — saturation

How fast an extra occurrence stops helping. The whole factor is one line,
`ranking.py:308`:

```python
return (tf * (self.k1 + 1.0)) / (tf + self.k1 * self.length_norm(length))
```

Measured on a paper of average length:

| tf | k₁=0 | k₁=0.2 | k₁=0.6 | **k₁=1.2** | k₁=2 | k₁=8 |
|---|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0005 | 1.0011 | **1.0017** | 1.0020 | 1.0027 |
| 2 | 1.0000 | 1.0912 | 1.2316 | **1.3766** | 1.5023 | 1.8044 |
| 3 | 1.0000 | 1.1252 | 1.3340 | **1.5728** | 1.8022 | 2.4600 |
| 5 | 1.0000 | 1.1540 | 1.4290 | **1.7752** | 2.1447 | 3.4681 |
| 10 | 1.0000 | 1.1765 | 1.5097 | **1.9649** | 2.5013 | 5.0068 |
| 20 | 1.0000 | 1.1882 | 1.5535 | **2.0758** | 2.7280 | 6.4342 |

At k₁ = 1.2, ten occurrences are worth **1.96×** one, not 10×. That is the
saturation, and it is what stops a long document from winning by repeating one
word.

**Read down any column and the shape is always the same:** large gains early,
diminishing returns forever after.

### b — length normalisation

How much a long paper is penalised. `ranking.py:308`:

```python
return (1.0 - self.b) + self.b * (length / self.average_document_length)
```

| \|d\| | b = 0 | b = 0.75 | b = 1 |
|---|---|---|---|
| 144 | 1.0000 | 0.8790 | 0.8387 |
| 171 | 1.0000 | 0.9969 | 0.9959 |
| 197 | 1.0000 | 1.1105 | 1.1474 |

At **b = 0 every K(d) is exactly 1.0000** and paper length has no effect on the
score at all.

This corpus runs 144–197 terms, a ratio of 1.37×, so the effect is real but
mild. It would dominate on a corpus of mixed article lengths.

---

## 5. The bridge — BIM **is** BM25 at k₁ = 0

Substitute `k₁ = 0` into the saturation factor:

```
(tf · (0 + 1)) / (tf + 0 · K)  =  tf / tf  =  1      for every tf ≥ 1
```

That is the binary indicator, exactly. This is not a loose analogy — it is an
identity, and it is measured rather than asserted:

```
max |saturation(tf, |d|) − 1| at k1 = 0, tf ≥ 1   =  0.0
max |BIM − BM25(k1 = 0)| over all 20 papers        =  0.0
```

`tests/test_ranking.py::TestTheBridge` asserts it on every paper and every
query term, and the Streamlit UI says so out loud when the k₁ slider hits 0.

The practical consequence: **k₁ is a continuous dial from a pure binary model to
a linear frequency model**, and BIM is simply one end of it.

```
k1 = 0  →  binary indicator  →  BIM
k1 = 1.2 → saturated          →  BM25, the default
k1 → ∞  →  tf / K(d)          →  linear in tf
```

---

## 6. The whole formula, one paper, arithmetic shown

Query `structured field search`, paper **p19**, k₁ = 1.2, b = 0.75.

```
query terms after preprocessing: 'structur field', 'search'
|p19| = 161     avgdl = 171.7
K(d)  = (1 − 0.75) + 0.75 × 161/171.7 = 0.9533
```

| term | tf | idf | saturation | = idf × sat |
|---|---|---|---|---|
| `structur field` | 1 | 2.6391 | 1.0262 | 2.7081 |
| `search` | 2 | 1.1727 | 1.3995 | 1.6413 |

```
score = 2.7081 + 1.6413 = 4.3494
```

Three words became **two** terms: preprocessing already knows that
"structured field" is a single index entry, so it scores once.

`TermContribution` (`ranking.py:155`) carries exactly these four numbers per
term, which is why the Streamlit Pipeline tab can print the derivation rather
than just the total.

---

## 7. Do the parameters change the answer?

Query `structured field search`, paragraphs:

| rank | BIM | BM25 | moved |
|---|---|---|---|
| 1 | p19 | p19 | — |
| 2 | p03 | p18 | 6→2 |
| 3 | p11 | p12 | 4→3 |
| 4 | p12 | p15 | 5→4 |
| 5 | p15 | p11 | 3→5 |
| 6 | p18 | p03 | 2→6 |

Five of six positions change. BIM ties four papers at exactly `1.1727` and
breaks the tie alphabetically; BM25 separates them because p18 and p03 mention
one of the terms more than once.

That is the practical argument for saturation: **the binary indicator produces
ties, and ties are where ranking stops working.**

---

## 8. What this step deliberately does **not** do

- **It does not know about fields.** A term in p19's title and a term in its
  results are the same term. Step 4 fixes that.
- **It does not know about positions.** Ten occurrences spread across a paper
  score the same as ten in one sentence. Step 5 and step 6 fix that.
- **It does not measure quality.** Every number above is a property of the
  *model*. Whether the ranking is any good is step 7's question.
- **It does not tune k₁ or b.** They are set to the literature defaults
  (1.2, 0.75) and exposed as sliders. Tuning them against the 12 judged queries
  would be fitting the corpus and reporting a number it has not earned.