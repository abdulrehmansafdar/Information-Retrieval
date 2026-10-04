# Step 7 — Vector Space Model

Guide to `vsm.py`: how documents and queries become vectors, and why the obvious
score function is not good enough.

---

## 1. The core idea

Everything up to step 6 produced numbers about **single documents**. The vector space
model turns those numbers into something that can answer a question none of them could:

> Which document is *about* this query?

A document becomes a vector of TF-IDF weights, one coordinate per vocabulary term. A
query becomes a vector in the same space. The score is the dot product:

```
score(q, d) = Σ over terms t of  w(t,q) × w(t,d)
```

A document that lacks a query term simply gets no contribution from it — no error, no
crash, just a smaller score.

---

## 2. A completely different model from Boolean

| | Boolean (step 4) | Vector space (step 7) |
|---|---|---|
| Query is | a logic formula | a bag of weights |
| Operators | AND, OR, NOT, parentheses | none |
| "Must contain" | expressible with AND | **not expressible** |
| Result | a set | a graded list |
| Partial match | impossible | the whole point |
| Ranking | none | inherent |

A VSM query cannot express "this term is essential". Every document gets a number and
the numbers decide the order. That is a genuine loss of expressiveness, and step 8's
Boolean-versus-ranked comparison quantifies it.

---

## 3. Building the query vector

The query goes through **the same preprocessing as the documents**, so it meets the
index in the same form:

```python
build_query_vector("machine learning healthcare", idf_model)
# {'healthcar': 1.8971, 'machin learn': 0.7985}
```

Note `machin learn` — one term, not two. The phrase term from step 2 is a single index
entry, so it must be one query term too.

### Should query terms be weighted by idf?

| Option | Query vector for the above | Rationale |
|---|---|---|
| `weight_by_idf=True` (default) | `healthcar` 1.8971, `machin learn` 0.7985 | reuse the document idf, so a term appearing in every document contributes little |
| `weight_by_idf=False` | both terms weight 1 | binary: every query term counts the same |

Both are defensible and both appear in the literature. The default reuses the document
IDF because it keeps the query and documents in the same weighting scheme. The `bool`
alternative is exposed as a parameter so the difference can be measured rather than
argued about.

### Unknown terms are dropped, not fatal

`build_query_vector("blockchain")` returns an **empty** vector and records
`unknown_terms = ['blockchain']`. A term absent from the index cannot match any
document, so it contributes nothing. `rank_documents` then returns an empty list —
a clean miss, the same behaviour step 3's `lookup()` was designed for.

---

## 4. Worked scoring example

Query `machine learning healthcare`:

| # | doc | title | len | dot product |
|---|---|---|---|---|
| 1 | doc04 | Machine Learning in Healthcare | 85 | 9.111 |
| 2 | doc03 | Artificial Intelligence in Healthcare | 86 | 7.836 |
| 3 | doc09 | Artificial Intelligence in Medical Imaging | 75 | 3.599 |
| 4 | doc02 | Machine Learning | 86 | 1.913 |
| 5 | doc15 | Machine Learning for Cybersecurity | 80 | 1.913 |

**10 of 20 documents score above zero.** A VSM query returns a graded list, not a set.

The two leaders are exactly right: both are about machine learning *in healthcare*.
doc09 ranks third because it is partly about healthcare, and doc02 fourth because it is
about machine learning only. The scoring is doing something meaningful.

---

## 5. The flaw: the dot product ignores length

The dot product has no notion of vector magnitude. A document that carries more total
weight scores higher whether or not the query is what it is about.

The cleanest demonstration in this corpus uses the term `abl` (df=2, idf=2.3026):

| doc | len | tf-idf | \|d\| | dot product | cosine |
|---|---|---|---|---|---|
| doc01 | 83 | 4.605 | 16.676 | **10.604** | 0.2762 |
| doc12 | 79 | 4.605 | 22.189 | **10.604** | 0.2075 |

Both documents contain `abl` once, so both have identical TF-IDF weight and the dot
product **ties at exactly 10.604**. There is no way to rank them. Step 8 breaks the tie.

### An important correction

An earlier version of this file measured how often dot product and cosine disagree by
sorting on **vector magnitude**. That produced **233 of 608**, and it was wrong twice over:

1. The sort direction was backwards — cosine rewards *smaller* magnitude, not larger.
2. Magnitude alone is not a valid proxy. Cosine is proportional to `w(t,d) / |d|`, so
   magnitude only determines the order when every candidate document has the **same**
   term frequency. With mixed tf it does not.

Replacing the proxy with the actual cosine formula gives the correct figure:

```
122 of 608 terms (20%) rank differently under cosine than under the dot product.
```

`cosine.py` computes this authoritatively; `vsm.py` inlines the same formula to measure
it, and a test cross-checks that the two implementations agree for all 608 terms. The
duplication exists only because `cosine.py` imports `vsm.py`, so a reverse import would
be circular.

---

## 6. What the dot product *does* get right

Worth saying, because the flaw above is easy to overstate. The dot product already
handles the case that raw TF got wrong. From step 6:

| document | query | dot product | why |
|---|---|---|---|
| doc12 (Robotics) | `robot` | 19.00 | rare-ish term, high idf |
| doc13 (Autonomous Robots) | `robot` | 13.33 | same term, larger vector, so lower |

Term **weight** dominates. The length effect only becomes decisive when two documents
have the *same* weight for the query — exactly the `abl` tie above.

---

## 7. API

```python
from vsm import build_query_vector, document_vector, dot_product, magnitude, score_documents

qv = build_query_vector("medical imaging", idf_model)
qv.terms        # {'medic imag': 2.9957}
qv.norm()       # 2.9957 - Euclidean length of the query vector

dv = document_vector("doc09", model)   # {term: tfidf}
magnitude(dv)                          # sqrt(sum of squared weights)
dot_product(qv.terms, dv)              # the VSM score

results = score_documents(qv, index, model, top=10)
results[0].matched_terms   # [('medic imag', 10.604)] - term and its contribution
```

`matched_terms` is worth keeping: it makes every score explainable, which is the
difference between a number a user must trust and a number a user can check.

---

## 8. Verification

| # | Check | Result |
|---|---|---|
| 1 | `magnitude({3,4}) == 5`; dot product of `{3,4}·{1,2,7} == 11`; missing coordinates count as zero; commutative | PASS |
| 2 | Phrase kept as one query term; unknown term dropped and reported; binary mode gives weight 1 | PASS |
| 3 | Every dot-product score matches a hand calculation | PASS |
| 4 | Cosine equals `(a·b)/(|a||b|)` to 1e-12; self-similarity exactly 1.0; symmetric; empty vector → 0.0 without dividing by zero | PASS |
| 5 | Dot product ties doc01/doc12 at 10.6038; cosine separates them 0.2762 vs 0.2075 | PASS |
| 6 | Exactly **122** of 608 terms re-rank under cosine | PASS |
| 7 | Measured top-3 for three queries, and result counts 10 / 2 / 4 / 0 | PASS |
| 8 | Results ordered, ranks numbered 1..n, deterministic, `top=` respected | PASS |
| 9 | Boolean `AND healthcare` = {doc03, doc04}; both models rank those two first; Boolean result ⊆ ranked result | PASS |
| 10 | Earlier steps untouched: 20 docs, 608 terms, 1,401 postings, 1,624 terms | PASS |

---

## 9. Honest limitations

- **The dot product cannot express a required term.** `robot AND NOT autonomous` has no
  VSM equivalent. Hybrid search (Boolean to filter, ranking to order) is the standard
  answer and is demonstrated in `cosine.py`.
- **The score has no upper bound and no fixed scale.** A document's dot product depends
  on its own length, so scores are only comparable *after* normalisation. Step 8 fixes
  this; until then, do not quote these numbers as "relevance".
- **Query terms are treated independently.** There is no query-side proximity, so
  `"artificial intelligence"` and `"intelligence artificial"` produce identical vectors.
- **Repeated query terms are counted.** `weight_by_idf` with a doubled term doubles its
  weight, which is usually not what a user means but is at least predictable.
- **Scores depend on the corpus.** idf values come from 20 documents, so they are
  noisier than idf values from a large collection.