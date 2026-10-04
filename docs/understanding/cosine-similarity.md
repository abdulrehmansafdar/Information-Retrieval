# Step 8 — Cosine Similarity and Ranked Results

Guide to `cosine.py` and `search.py`: the final required feature. This is where the
engine stops returning a set and starts returning an ordered answer.

---

## 1. The core idea

Step 7 left a specific, demonstrable problem: the dot product ties documents it cannot
tell apart. Cosine fixes it by dividing by both magnitudes.

```
cos(q, d) = (q · d) / (|q| × |d|)
```

The result is the **cosine of the angle** between two vectors:

| Score | Meaning |
|---|---|
| 1.0 | same direction — every weighted term in *d* also appears in *q* |
| 0.0 | nothing in common |
| 0.28 | partial agreement |

Dividing by `|d|` is what removes the length effect. A long document no longer wins by
being long; it wins by being **concentrated** on the query.

---

## 2. The tie is broken

Query `abl`, the same example that defeated the dot product:

| doc | len | tf-idf | \|d\| | dot | **cosine** |
|---|---|---|---|---|---|
| doc01 | 83 | 4.605 | 16.676 | 10.604 | **0.2762** |
| doc12 | 79 | 4.605 | 22.189 | 10.604 | **0.2075** |

Step 7: **tie at 10.604**, no ranking possible.
Step 8: doc01 wins, 0.2762 against 0.2075.

### It is not about length

This is the part worth getting right. **doc12 is the shorter document (79 against 83)
and still loses.** Its weight is spread more thinly across repeated terms, giving it
the *larger* magnitude (22.19 against 16.68), so cosine penalises it.

> Cosine rewards **concentration**, not brevity.

A long, tightly-focused document can and does beat a short one — that is exactly why
doc09 (*Medical Imaging*) outranks most longer documents for the query `medical imaging`.

---

## 3. How much cosine actually changes

```
122 of 608 terms (20%) produce a different top-2 order under cosine
than under the raw dot product.
```

| query | by dot product | by cosine |
|---|---|---|
| `abil` | doc08, doc17 | **doc17**, doc08 |
| `accuraci` | doc04, doc09 | **doc20**, doc04 |
| `agricultur` | doc12, doc13 | **doc13**, doc12 |

One in five queries changes its answer. Normalising is not a cosmetic adjustment.

> **A correction worth recording.** An earlier version measured this by sorting on
> vector magnitude and reported **233 of 608**. That was wrong twice: the direction was
> backwards (cosine rewards *smaller* magnitude), and magnitude is only a valid proxy when
> every candidate document has the same term frequency, because cosine scales with
> `w(t,d)/|d|`. Recomputing with the real formula gives 122. The lesson generalises —
> a shortcut that is "obviously equivalent" to a formula is worth checking.

---

## 4. Ranked results

`rank_documents()` returns documents ordered best-first, with `rank` numbers, and drops
everything scoring 0.0 so the list is the relevant set rather than all 20.

### `medical imaging` — 2 of 20

| # | doc | title | cosine |
|---|---|---|---|
| 1 | doc09 | Artificial Intelligence in Medical Imaging | 0.2343 |
| 2 | doc08 | Computer Vision | 0.1127 |

### `neural network` — 4 of 20, and note how close they are

| # | doc | title | cosine |
|---|---|---|---|
| 1 | doc07 | Deep Learning | 0.0831 |
| 2 | doc09 | Artificial Intelligence in Medical Imaging | 0.0819 |
| 3 | doc08 | Computer Vision | 0.0788 |
| 4 | doc10 | Natural Language Processing | 0.0736 |

All four contain the phrase `neural network` with the same frequency, so their scores
differ **only** by vector magnitude. doc07 wins narrowly. A near-tie like this is a
genuinely hard ranking, and cosine is the honest way to resolve it.

### `security attack anomaly` — 6 of 20

| # | doc | title | cosine |
|---|---|---|---|
| 1 | doc14 | Cybersecurity and Artificial Intelligence | 0.3186 |
| 2 | doc15 | Machine Learning for Cybersecurity | 0.1648 |
| 3 | doc06 | Machine Learning in Finance | 0.1072 |
| 4 | doc19 | Artificial Intelligence and the Internet of Things | 0.0595 |
| 5 | doc16 | Cloud Computing and Artificial Intelligence | 0.0208 |
| 6 | doc18 | Artificial Intelligence in Software Development | 0.0203 |

The top hit is close to double the second — a strong, unambiguous result. doc06
(Finance) earns its place legitimately: it describes a security team reviewing fraud
alerts.

---

## 5. Boolean and ranked, side by side

The brief asks for both models, so here is the difference on the same question.

| model | query | hits | meaning |
|---|---|---|---|
| Boolean | `machine learning AND healthcare` | **2** | must contain both — exact, no order, no partial credit |
| Ranked | `machine learning healthcare` | **10** | graded match — everything related, best first |

| group | n | documents |
|---|---|---|
| both models | 2 | doc03, doc04 |
| ranked only | 8 | doc01, doc02, doc06, doc07, doc09, doc14, doc15, doc17 |
| Boolean only | 0 | none |

**They agree** on the obvious case: both put the two healthcare documents first.

**They differ** because Boolean returns *nothing* for a document matching only one query
term. Ranked retrieval still returns 8 such documents, with lower scores.

**Neither model wins outright:**

- Boolean cannot rank, so it cannot distinguish a good partial match from a poor one.
- Ranked retrieval cannot express "must contain", so it cannot guarantee a constraint.

A production engine uses both: **Boolean to filter, ranking to order.**

---

## 6. Properties of cosine

| Property | Value | Verified |
|---|---|---|
| All scores within [0, 1] | yes | yes |
| Self-similarity `cos(d, d)` | exactly 1.000000 | yes |
| Symmetric: `cos(a,b) == cos(b,a)` | yes | yes |
| Empty vector | 0.0, guarded against division by zero | yes |

The range `[0, 1]` holds only because TF-IDF weights are non-negative. With negative
weights cosine would fall to `-1`. Worth knowing if the weighting scheme ever changes.

---

## 7. The search interfaces — `main.py`, `app.py`, and `search.py`

The primary launcher has two explicit modes. It chooses an interface; it does not
contain retrieval logic:

```powershell
python main.py --console "medical imaging"
python main.py --web
```

`main.py --console` hands off to `console_ui.py`. `main.py --web` starts Streamlit's
`app.py`. Both call the same backend in `engine.py`.

The older `search.py` remains available as a secondary, standalone terminal demo:

```
python search.py                        interactive
python search.py "medical imaging"      one query, then exit
python search.py --top 5 "neural network"
```

| Command | Does |
|---|---|
| `<query>` | ranked search — TF-IDF query vector, cosine similarity, best first |
| `boolean: A AND B` | Boolean retrieval, exact set, no ranking |
| `both: <query>` | ranked results with the AND-ed equivalent alongside |
| `top N <query>` | limit the result list |
| `terms docNN` | a document's highest-weight terms with tf, idf and TF-IDF |
| `stats` | corpus summary |
| `help`, `quit` | command list / exit |

### A bug worth recording

`both: machine learning healthcare` originally split the raw words on spaces and built
`machine AND learning AND healthcare` — which returns **zero** documents, because
"machine learning" is a *phrase term* and `machine` alone exists in only three documents.

The fix builds the Boolean form from the **preprocessed** query terms:

```
'machine learning healthcare'  -> terms ['healthcar', 'machin learn']
boolean: 'healthcar AND machin learn'  gives 2
```

The Boolean form must be derived from the same preprocessing as the ranked form,
otherwise the two models are not answering the same question. This is the third time
phrase handling has caused a real bug in this project — steps 4, 6 and 8 all hit it.

---

## 8. API

```python
from cosine import cosine_similarity, rank_documents

cosine_similarity(qv.terms, dv)            # 0.0 .. 1.0
results = rank_documents("medical imaging", index, model, top=5)

results[0].rank            # 1
results[0].doc_id          # 'doc09'
results[0].score           # 0.2343
results[0].matched_terms   # [('medic imag', 10.604)] - explains the score
```

`matched_terms` is what makes a ranked result auditable: every score can be traced back
to the terms that produced it.

---

## 9. Verification

| # | Check | Result |
|---|---|---|
| 1 | Dot product and magnitude correct by hand; missing coordinates zero; commutative | PASS |
| 2 | Query vector keeps phrases; unknown terms dropped and reported; binary mode | PASS |
| 3 | Dot-product scores match hand calculation | PASS |
| 4 | Cosine equals the formula to 1e-12; self-similarity exactly 1.0; symmetric; empty → 0.0 | PASS |
| 5 | Dot product ties doc01/doc12 at 10.6038; cosine separates them 0.2762 vs 0.2075 | PASS |
| 6 | Exactly 122 of 608 terms re-rank; `vsm.py`'s inline formula agrees with `cosine.py` for all 608 | PASS |
| 7 | Measured top-3 for three queries; counts 10 / 2 / 4 / 0 | PASS |
| 8 | Ordered, ranked 1..n, deterministic, `top=` respected | PASS |
| 9 | Boolean ⊆ ranked; both rank the two healthcare documents first | PASS |
| 10 | Earlier steps untouched: 20 docs, 608 terms, 1,401 postings, 1,624 terms | PASS |

Check 6's second half exists because `vsm.py` inlines the cosine formula to avoid a
circular import. Without the cross-check, that duplicated line could silently drift.

---

## 10. Honest limitations

- **A score of 0.24 is not "24% relevant".** Cosine measures angle, not probability.
  The absolute values mean little; only the **order** is meaningful.
- **Scores are corpus-dependent.** Change the 20 documents and every idf changes, so
  every score changes. Scores are not comparable across different collections.
- **Cosine ignores term order and proximity.** A document containing the query terms in
  a scattered way scores the same as one containing them in a relevant sentence.
- **The top hit is often correct for a trivial reason.** `medical imaging` matches the
  *phrase* `medic imag`, which appears in exactly two documents. That is precision, but
  it is precision inherited from step 2's phrase list, not earned by the ranking.
- **No stemming at query time is skipped** — the query *is* preprocessed, but a user
  searching `healthcare` must match the stored stem `healthcar`. An unstemmed query
  silently returns nothing.
- **20 documents is a very small evaluation set.** No precision/recall figure from this
  corpus would be meaningful, and none is claimed.
