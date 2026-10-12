# Step 6 — The Proximal Node Model

Guide to `proximity.py`: Navarro and Baeza-Yates, TOIS 1997. A paper is not a
bag of words and not a flat block of text either — it is a **tree**, and this
step is the query language that walks it.

---

## 1. The tree

```
paper  (p01)
  section        title | abstract | methodology | results
    paragraph    one blank-line block inside a field
      sentence   one sentence inside a paragraph
```

| level | node kind | regions | how it is built |
|---|---|---|---|
| paper | `paper` | 20 | the whole document |
| section | `section` | 80 | one per `##` heading — the four named fields |
| paragraph | `paragraph` | 120 | one per blank-line block |
| sentence | `sentence` | 299 | one per sentence |

Step 5 uses **one level at a time**. This step **relates levels to each other**,
which is what lets a query say *a paragraph inside the methodology, not anywhere*.

---

## 2. The operators

| Operator | Meaning | Computed by | Line |
|---|---|---|---|
| `P in Q` | P nodes inside some Q node | `_op_in()` | `proximity.py:280` |
| `P beginin Q` | only P's **start** inside Q | `_op_beginin()` | `proximity.py:295` |
| `P child Q` | direct children only | `_op_child()` | `proximity.py:306` |
| `P parent(k) Q` | P is parent of ≥ k Q nodes | `_op_parent()` | `proximity.py:372` |
| `P with(k) T` | P contains ≥ k occurrences of T | `_op_with()` | `proximity.py:354` |
| `P before(k) T (C)` | P ends ≤ k terms before T, inside C | `_op_near()` | `proximity.py:394` |
| context `(C)` | both operands inside one C node | `_in_context()` | `proximity.py:426` |

**A constructor** is a kind, optionally narrowed to a field: `paragraph`,
`methodology` (= `section:methodology`), `paragraph:methodology`.

**`looks_like_constructor()`** (`proximity.py:434`) decides whether the
right-hand side is a node type or a term. It tests **membership, not shape**, so
the answer depends on this corpus: `methodology` is a field, `retrieval` is a
term. A word that is both would be misread, and `section:retrieval` is the way to
disambiguate.

---

## 3. The bottom-up step — why the model scales

This is the single most important idea in the step.

```
occurrence position  →  innermost node of kind P containing it
```

| Step | Does | Line |
|---|---|---|
| read the postings list | `positions_of()` | `structured_index.py:131` |
| find the smallest enclosing node | `innermost_component()` | `proximity.py:162` |
| → | `enclosing()` | `structured_index.py:166` |
| count per node | `counts[node.region_id] += 1` | `proximity.py:368` |
| keep nodes with ≥ k | `if count >= k` | `proximity.py:370` |

**Measured** for `section with(2) retrieval`:

| | |
|---|---|
| occurrences of `retriev` | **28**, in 9 papers |
| sections in the corpus | **80** |
| top-down: nodes × occurrences | 80 × 28 = **2,240** containment tests |
| bottom-up: walk the postings | **28** upward steps |
| **speed-up** | **80×** |

Both paths return the same five nodes. The upward step is
`innermost_component()`, and it is the entire performance argument.

The reason one step suffices: within one hierarchy the nodes **tile** the paper,
so every occurrence belongs to exactly one node of that kind. There is no search
to do — only a lookup.

---

## 4. `P with(k) T` — counts occurrences, not documents

```powershell
python main.py --console "prox: section with(2) retrieval"
```

| node | range | occurrences of `retriev` |
|---|---|---|
| `p10:results` | `[83,144)` | 2 |
| `p12:abstract` | `[3,33)` | 4 |
| `p13:abstract` | `[3,34)` | 9 |
| `p13:methodology` | `[34,98)` | 9 |
| `p13:results` | `[98,165)` | 9 |

`k` is monotone: `with(3)` is always a subset of `with(1)`, and
`tests/test_proximity.py` asserts it.

**Terms are preprocessed like documents.** `proximity.py:330` runs the query term
through `preprocess_query`, so `retrieval` reaches the index as the stem
`retriev` and `proximity window` as the phrase `proxim window`. Without this, a
proximal query silently matches nothing for every inflected form — which looks
like a broken operator rather than a missing stemmer.

---

## 5. `P before(k) T` — the exact / broader dial

```
sentence before(k) index (paragraph)
```

means: **sentences that end at most k terms before an occurrence of `index`,
inside the same paragraph.**

Note the node does **not** have to contain the term. A reading that required the
term inside the sentence would be a much weaker operator — a sentence *leading
into* the word, not a sentence *containing* it. `tests/test_proximity.py` asserts
that some occurrence lies at or after the node's end.

**Measured**, monotonically increasing by construction:

| k | nodes | papers |
|---|---|---|
| 0 | 2 | 2 |
| 2 | 3 | 3 |
| 5 | 6 | 5 |
| 10 | 9 | 6 |
| 25 | 17 | 6 |
| 60 | 33 | 6 |

Distance is measured in **index terms**, so the number means the same thing in a
4-term title and a 70-term results section. Measuring in characters would make it
depend on word length, which is not what a reader means by "nearby".

`k = 0` is exact adjacency. Every larger value relaxes it one term at a time —
the same trade as step 5, expressed as a single number.

---

## 6. `in` versus `beginin`

| operator | requirement |
|---|---|
| `paragraph in results` | the **whole** paragraph is inside the results section |
| `paragraph beginin results` | only its **start** is inside |

They differ whenever a node **straddles** a boundary.

On this corpus they return the **same 40 nodes**, because the parser cuts fields
at a blank line before a heading, so no paragraph ever starts in one section and
ends in another.

That is a property of **this corpus's parser**, not of the operators. The
difference appears the moment a section break falls mid-paragraph — which is
exactly the case XML full-text search has to handle. Reporting them as equal here
would be overclaiming; `tests/test_proximity.py` asserts the equality *and* names
the reason.

---

## 7. Two guarantees that fall out of the tree

**A node is inside itself.** `section in section` returns every section. This is
not a special case, it is what containment means, and it means `P in Q` can never
return nothing for a `Q` that exists.

**An answer is one hierarchy.** `sentence before(10) index (paragraph)` returns
**sentences** only, never a mixture of sentences and paragraphs. The UI is
therefore told to print no score bar at all for this mode — see
[`engine.py` `_proximal`](engine.py), which leaves `score = 0.0` deliberately, so
no front end can sort by it and call the result a ranking.

---

## 8. This model returns a set, not a ranking

There is no relevance score attached to *"this sentence is a methodology
paragraph."* The result is document-ordered and unordered in relevance.

The console says so:

```
9 node(s) - a SET, not a ranking: this model answers 'which', not 'in what order'
```

For the paper-level evaluation, `engine.evaluation_models()` builds one ranking
from it by asking `section with(1) <first query term>`. That is a stated choice
about how to score a set model against document judgements, not a property of
the model, and it is why its MRR (0.764) is not comparable with BM25's 1.000
without that framing.

---

## 9. Syntax errors are messages

```
paragraph in methodology        → 40 nodes in 20 papers
paragraph child methodology     → 40 nodes in 20 papers
section parent(2) paragraph    → 40 nodes in 20 papers
paragraph methodology           → ProximitySyntaxError: expected one of …
paragraph child retrieval       → ProximitySyntaxError: 'child' needs a node
                                  type on the right, not the term 'retriev'
in methodology                  → ProximitySyntaxError: nothing before the operator
section with(1) nosuchterm      → 0 nodes, step explains the term is unknown
section in chapter              → 0 nodes; 'chapter' is neither a kind nor a
                                   field, so it is treated as a TERM
```

The last row is deliberate. Guessing that the user meant a node kind would hide a
mistake behind a plausible answer. The result carries a step saying
`'chapter' is not in the vocabulary`.

The third row is a **bug this documentation caught**. `paragraph child retrieval`
has a term on the right, so it takes the content path — and the content path only
implements `with`, `before` and `after`. Anything that is not `before` fell
through to `after`, so a malformed query silently returned an `after` answer: a
plausible answer to a query nobody wrote. All four structural operators now reject
a term on the right, and `tests/test_proximity.py` pins it.

---

## 10. What this step deliberately does **not** do

- **It does not merge nodes from two hierarchies in one answer.** The literature
  allows it; this implementation does not, so results stay a clean subset of one
  hierarchy.
- **It does not score nodes.** Section 8.
- **It does not index the structure separately from the content.** The two
  hierarchies share the same token stream; a real implementation would index
  content and structure separately to make §3's cost argument hold at scale.
- **It does not implement `M subtract M'` here.** That operator lives in
  `lists.py`, unused by this language.
- **It does not claim the operators compose into a complete algebra.** Only the
  seven above are implemented. `in`, `beginin`, `child` and `parent` take a
  constructor on the right and **raise** if given a term, rather than guessing.