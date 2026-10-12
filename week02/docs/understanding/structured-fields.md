# Step 4 — Searching Structured Fields

Guide to `fields.py`: searching one named field instead of the whole paper, and
the three designs that are routinely confused with each other.

The FYP example for this week is *"search within research papers by title,
abstract, methodology, and results"*, so the corpus is 20 papers with four named
fields and this step makes each one searchable.

---

## 1. The three designs

| Design | What it does | Discards |
|---|---|---|
| **restrict** | score **only** the named field | everything outside the field |
| **filter** | rank the **whole paper**, then throw away non-matching fields | nothing — but the order reflects text you discarded |
| **weight** (BM25F) | every field scores, weighted by importance | nothing at all |

The pair that matters:

> **restrict and filter return the same PAPERS in a different ORDER.**

If you only check which papers came back, the two designs look identical.

**Worked example** — query `title:search`, paragraph order irrelevant:

| rank | restrict | filter |
|---|---|---|
| 1 | p12 | **p18** |
| 2 | p19 | p12 |
| 3 | p03 | p19 |
| 4 | p15 | p15 |
| 5 | p18 | p03 |

Same five papers. p18 goes from **rank 5 to rank 1** because its *body* is strong
even though its title match is not the strongest.

---

## 2. Why they differ — and it is worth a paragraph in a viva

`restrict` computes the score **entirely inside the field**. Text the user did
not ask about cannot influence the order, because it is never read.

`filter` computes the score over the **whole paper** and applies the field test
**afterwards**. The score is real and the set is right, but the ranking was
decided by text the user has now been told was excluded. p18's body outranks its
title, and that ranking survives into the answer.

An interface that says *"searched in the title"* is making a claim. Only one of
the two designs backs that claim.

---

## 3. BM25F

```
score_BM25F(q, d) = Σ over fields f of  weight_f · score_BM25(q, f)

title 3.0 | abstract 2.0 | methodology 1.5 | results 1.0
```

| Symbol | Meaning | Computed by | Line |
|---|---|---|---|
| `weight_f` | per-field importance | `DEFAULT_WEIGHTS` | `fields.py:50` |
| `score_BM25(q,f)` | BM25 inside one field | `_bm25()` | `fields.py:427` |
| the sum | `total += weight * part` | `weighted()` | `fields.py:657` |
| which fields matched | `"+".join(contributing)` | `weighted()` | `fields.py:562` |

**Call chain**

```
fields.py:657    total += weight * part
 └─ part ← fields.py:442   stats.idf_for(name, term) * sat
      ├─ idf  ← fields.py:376   log(1 + (N_f − df_f + .5)/(df_f + .5))
      │        N_f  ← fields.py:356   holders  (papers that HAVE this field)
      │        df_f ← fields.py:365   per-field document frequency
      └─ sat  ← fields.py:441   tf(k1+1)/(tf + k1*kn)
               tf  ← fields.py:430   counts from tokens inside the region
               kn  ← fields.py:438   kn = (1−b) + b·norm
```

---

## 4. Per-field statistics, and why they are not optional

| field | papers | average length | default weight |
|---|---|---|---|
| title | 20 | **3.45** | 3.0 |
| abstract | 20 | 34.85 | 2.0 |
| methodology | 20 | 66.20 | 1.5 |
| results | 20 | 67.20 | 1.0 |

A title is **19× shorter** than a results section. Two consequences:

**Field idf.** A term in 5 of 20 *titles* is rarer than a term in 5 of 20
*papers*, because a title is a much smaller collection.

```
df_title(search) = 5 of 20 titles
idf_title        = log(1 + (20−5+0.5)/(5+0.5)) = log(3.818) = 1.3398
idf_paper        = log(1 + (20−5+0.5)/(5+0.5)) with N=20 too …
```

Both happen to be 1.3398 here because both collections have 20 members — the
difference shows up on a term that is common in titles but rare in bodies. The
mechanism matters, not this particular number.

**Per-field length normalisation.** Each field is divided by **its own** average,
`fields.py:394`:

```python
return length / average if average else 1.0
```

Without it, a results match scores higher purely by being long, and a weight of
3.0 on the title does almost nothing. Measured effect on
`structured field search`: raw scores run **1.5–1.7×** the normalised ones, and
the multiplier is nearly constant — so on *this* corpus it mostly rescales
rather than reorders. `fields.py:899` searches for a query where it does reorder
and reports honestly when there is not one.

---

## 5. Cross-field AND — what week01 cannot express

```
title:search AND results:precision
```

means: the **title** must contain *search* **and** the **results** must contain
*precision*. week01's Boolean engine sees two document sets and intersects them,
so it returns papers containing both words **anywhere**. It cannot express the
constraint, because week01's documents have no named fields.

Measured:

```
restrict  "title:search AND results:precision"  →  1 paper:  p19
filter    "title:search AND results:precision"  →  1 paper:  p19
bm25f     "title:search AND results:precision"  →  10 papers (nothing discarded)
```

The AND is an **intersection**: a paper must satisfy *every* restriction. Scoring
each restriction separately and returning both sets would be an OR, and would
silently answer a different question.

---

## 6. The bug this step is built around

```
title:proximity AND results:precision
```

`proximity` is **not a term in this corpus** — preprocessing folded it into the
phrases `proximity operator` and `proximity window`, so the vocabulary contains
`proxim oper`, not `proxim`.

The first version of `restrict()` dropped any restriction whose terms were all
unknown, and answered with **every paper whose results mention precision**. That
is not a smaller answer. It is a **different answer to a different query** — the
title constraint was silently deleted.

The rule now:

> A restriction that cannot be satisfied yields **zero hits and an explanation**.

```
no paper contains proxim; the title restriction cannot be satisfied
```

`tests/test_fields.py::TestUnsatisfiableConstraints` asserts it for all three
designs.

---

## 7. Query syntax: OR, brackets, and refusing the rest

A query is a **union of AND-groups**. `title:a AND results:b` is one group of two
restrictions; `title:a OR results:b` is two groups of one.

`AND` binds tighter than `OR`, and brackets override both — the SQL convention,
and the only one a user can predict without reading the source:

```
title:a OR results:b AND abstract:c
  →  (title:a)  OR  (results:b AND abstract:c)

(title:a OR results:b) AND abstract:c
  →  (title:a AND abstract:c)  OR  (results:b AND abstract:c)
```

Brackets are **honoured**. The first version of this parser split on `AND`/`OR`
with a regex and never looked at brackets at all, so `(A OR B) AND C` was read as
`A OR (B AND C)` and returned a different, confidently wrong answer set. The law
now holds, and `TestBrackets::test_the_grouped_and_flat_forms_agree_on_papers`
checks it at the level a user sees the results:

```
(title:search OR abstract:rank) AND abstract:result
  →  p03, p15, p18, p19, p20
title:search AND abstract:result OR abstract:rank AND abstract:result
  →  p03, p15, p18, p19, p20      (identical, as it must be)
```

`AND` is applied as a **cross product** of the two sides' groups, which is exactly
"every restriction from both sides must hold".

Everything else is refused, because each of these used to parse into something
plausible and answer a question nobody asked:

| query | result |
|---|---|
| `title:search` | 5 hits |
| `title:search AND results:precision` | 1 hit, intersected |
| `title:proximity AND results:precision` | 0 hits, term not in vocabulary |
| `keywords:search` | `FieldQueryError` — not a field of this corpus |
| `title:search NOT results:x` | `FieldQueryError` — NOT is not implemented |
| `(title:a OR title:b` | `FieldQueryError` — unmatched bracket |
| `:search` | `FieldQueryError` — a colon with no field name in front |
| `title:` | `FieldQueryError` — nothing to look for |
| `title:search AND` | `FieldQueryError` — dangling connector |
| `"   "` | 0 hits, no crash |

The field regex matches **any** `word:` prefix, not just the four real names. A
restricted regex would make `keywords:foo` a clean miss, which looks like an
answer rather than a typo.

Two rules are worth stating because they surprise people:

- **Everything after the colon belongs to that field**, up to the next connector.
  `title:inverted index` is one restriction whose body is the phrase
  `invert index`. This is what makes the phrase reachable inside a title at all;
  the cost is that a bare word after a restriction is a second term of the same
  restriction, not a separate all-fields constraint.
- **`NOT` is refused, not ignored.** Negation needs a different answer set, and
  silently dropping an operator is precisely how `OR` went wrong — `OR` is an
  nltk stop word, so a split done after preprocessing turned a user's `OR` into
  an `AND`.

---

## 8. Which design to use

| | use it when |
|---|---|
| **restrict** | the user means *this field*: "papers whose title says X" |
| **filter** | you need whole-paper ranking but a field-restricted result set |
| **bm25f** | "about X anywhere, but prefer the ones whose **title** says X" |

Real systems offer the first and the third, and label them honestly. The
evaluation shows BM25F reaching the same MRR as plain BM25 (0.958) with no
queries lost, which is what you would hope from a model that never discards
anything.

---

## 9. What this step deliberately does **not** do

- **It does not tune the weights.** `3.0 / 2.0 / 1.5 / 1.0` come from the
  conventional ordering (a title is a deliberate summary), not from the judged
  queries. Fitting them against qrels and reporting the result on those same
  qrels would be circular.
- **It does not normalise across fields by default in `filter`.** That is the
  *point* of `filter` — it uses whole-paper statistics so the comparison with
  `restrict` is meaningful.
- **It does not handle a corpus without named fields.** Fields are the whole
  subject. p19 of this corpus is literally about metadata quality.
- **It does not return regions within a field.** `fields.py` returns the field
  that matched; step 5 narrows further, to paragraphs and sentences.