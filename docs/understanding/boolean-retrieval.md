# Step 4 — Boolean Retrieval (AND, OR, NOT)

Guide to `boolean_search.py`: how the three operators work, why the query is parsed
before preprocessing, and what the empty results actually mean.

---

## 1. The core idea

Boolean retrieval is the simplest retrieval model. A query is an expression over
**sets** of documents, and the answer is a **set**. Nothing is scored or ranked.

| Query | Set operation | Meaning |
|---|---|---|
| `A AND B` | `A & B` — intersection | documents matching **every** term |
| `A OR B` | `A \| B` — union | documents matching **any** term |
| `A AND NOT B` | `A - B` — difference | `A` without `B` |

The inverted index already stores each term as a set of document ids. That is why the
three operators are Python's own set operations — no new data structure is needed:

```python
Term → documents_containing(index, term)   # already built in step 3
Not  → all_doc_ids - evaluate(operand)
And  → evaluate(left) & evaluate(right)
Or   → evaluate(left) | evaluate(right)
```

---

## 2. Why the query is parsed BEFORE preprocessing

This is the single most important design decision in the step, and it was forced by a
real measurement.

`preprocess_query()` returns unigrams first and phrases **last**, so word order is lost:

```python
preprocess_query('Machine Learning OR Robotics')
# ['or', 'robot', 'machin learn']     ← the OR now comes FIRST
```

Parsing that stream would treat `or` as the left operand and produce nonsense.
Parentheses are destroyed too, because `tokenize()` deletes every non-alphanumeric
character:

```python
preprocess_query('(a OR b) AND c')
# ['a', 'or', 'b', 'and', 'c']        ← both sets of parentheses gone
```

So the pipeline is deliberately split in two:

```
raw query text
   ↓  1. SCAN raw text        parentheses and operators found here
   ↓  2. PARSE to an AST       structure fixed, still no preprocessing
   ↓  3. EVALUATE the AST      each term chunk preprocessed lazily, one at a time
set of document ids
```

Only stage 3 calls `preprocess_query`, and only on a single chunk of user words.

---

## 3. The three stages

### Stage 1 — scan the raw text

```python
TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
```

Three token kinds: `(`, `)`, and word runs. **Consecutive words are merged into one
chunk**, which is what makes `artificial intelligence` a single operand rather than two
unlinked words.

| Raw query | Tokens |
|---|---|
| `artificial intelligence OR robotics` | `word('artificial intelligence')`, `op('OR')`, `word('robotics')` |
| `machine learning AND (healthcare OR finance)` | `word`, `op(AND)`, `(`, `word`, `op(OR)`, `word`, `)` |

### Stage 2 — recursive-descent parse to an AST

```
expr   := factor (OR factor)*           ← lowest precedence
factor := NOT factor | '(' expr ')' | TERM
```

Precedence: **NOT binds tightest, then AND, then OR.** Four frozen dataclasses:

```python
Term(text)          # raw chunk, preprocessed at evaluation time
Not(operand)
And(left, right)
Or(left, right)
```

`And` and `Or` are binary and left-associative, so a chain left-folds:

```
a AND b AND c   →   And(And(a, b), c)
```

Parsing to an AST **before** evaluating is what lets the demo print the parse tree,
which is worth far more in a write-up than evaluating during the parse.

### Stage 3 — evaluate

```python
Term → documents_containing(index, term)   # [] if unknown, never an error
Not  → {all doc ids} - evaluate(operand)
And  → left & right
Or   → left | right
```

---

## 4. Precedence, demonstrated

The difference is not academic — these two queries differ by 8 documents:

```
artificial intelligence OR robotics AND security          12 documents
  parsed as  (artificial intelligence OR (robotics AND security))

(artificial intelligence OR robotics) AND security         4 documents
  parsed as  ((artificial intelligence OR robotics) AND security)
```

Read the first one left to right and it looks like "artificial intelligence, or
robotics that mention security". It is not — `AND` binds tighter, so it is
"artificial intelligence, or (robotics that mention security)".

Verified in the test suite: the unparenthesised form equals
`artificial intelligence OR (robotics AND security)` exactly, and differs from the
parenthesised form.

---

## 5. Your seven queries, measured

| Query | Hits | Documents |
|---|---|---|
| `machine learning AND healthcare` | 2 | doc03, doc04 |
| `artificial intelligence OR robotics` | 13 | doc01, 02, 03, 05, 08, 09, 12, 13, 14, 16, 18, 19, 20 |
| `machine learning AND NOT finance` | 7 | doc02, 03, 04, 07, 14, 15, 17 |
| `healthcare AND diagnosis` | 2 | doc03, doc09 |
| `computer vision OR medical imaging` | 3 | doc01, doc08, doc09 |
| `security AND machine learning` | 3 | doc06, doc14, doc15 |
| `NLP AND education` | 2 | doc10, doc11 |

`doc06` (Machine Learning in Finance) legitimately appears under `security AND machine
learning` because it describes a security team reviewing fraud alerts.

---

## 6. The empty result that is actually correct

Your original planned query was `machine AND healthcare`. **It returns zero documents.**

```
machine AND healthcare   →   0 documents
```

This is not a bug. Step 2 turned *machine learning* into the single phrase term
`machin learn` and consumed both words, so:

| Term | Documents | Why |
|---|---|---|
| `machin` | doc12, doc13, doc20 | the only places "machine" means a *physical* machine — robots and vehicles |
| `machin learn` | 9 documents | includes doc03 and doc04 |
| `healthcar` | doc03, doc04, doc09 | — |

`{doc12, doc13, doc20} ∩ {doc03, doc04, doc09} = ∅`

A phrase term and its constituent words are **separate index entries with no automatic
relationship**. A document only appears under `machin` if it says "machine" *without*
"learning" next to it.

The fix is to write what you meant:

```
machine learning AND healthcare   →   2 documents: doc03, doc04
```

This is worth keeping in the write-up as **precision loss from phrase extraction**.
It is a real retrieval effect, not a defect, and it is the sort of thing an examiner
likes to see explained rather than hidden.

---

## 7. Edge cases, all verified

### NOT on a term that does not exist

```
machine learning AND NOT blockchain   →   9 documents
```

`blockchain` is unknown, so its document set is empty, and `A - ∅ = A`. **NOT can never
remove anything for a term the corpus does not contain.** Correct, and worth stating
plainly — a user who assumes otherwise will be confused.

### An unknown term on the left of AND

```
blockchain AND machine learning   →   0 documents
```

`∅ & anything = ∅`. No error is raised. This is the behaviour step 3's
`lookup()` returning `[]` was specifically designed for.

### Double negation

```
NOT NOT machine learning   →   9 documents
machine learning            →   9 documents      identical
```

### Every syntax error is a clean message

| Bad query | Message |
|---|---|
| `machine healthcare` | `'machine healthcare' became 2 separate terms ['machin', 'healthcar']. Separate them with AND or OR.` |
| `machine AND` | `Query ended unexpectedly.` |
| `AND machine` | `'AND' is missing a right-hand operand.` |
| `machine AND (healthcare` | `Unbalanced parenthesis: missing ')'.` |
| `machine healthcare)` | `Unexpected ')'.` |
| `machine learning OR OR robotics` | `'OR' is missing a right-hand operand.` |
| `machine learning NOT finance` | `Unexpected 'NOT' between two terms. Write 'A NOT B' only after AND or OR, or use one of 'A AND B', 'A OR B', 'A AND NOT B'.` |
| `NOT` | `Query ended unexpectedly.` |
| `''` (empty) | `Query is empty.` |
| `()` | `Unbalanced parenthesis: unexpected ')'.` |

All raise `BooleanSyntaxError`, never a raw Python traceback.

> The `machine learning NOT finance` message was originally
> *"'NOT' is missing a right-hand operand."* — technically true but actively
> misleading, because the real problem is a **missing infix operator**. A test caught it.
> `A NOT B` is invalid syntax (it is not `A AND NOT B`), so the rejection is correct;
> only the explanation was wrong.

---

## 8. Why operators must be explicit

`machine healthcare` is a **syntax error**, not an implicit AND. A silent implicit AND
would hide the most likely cause of an empty result set — the user forgetting an
operator — behind a plausible-looking zero.

Chunks are also rejected if they preprocess to more than one term:

```python
preprocess_query('medical imaging diagnosis')   # ['diagnosi', 'medic imag']
```

Two terms appeared from one chunk with no operator between them, so the parser reports
it by name rather than silently choosing an interpretation.

---

## 9. API

```python
from boolean_search import parse_query, evaluate, search, BooleanSyntaxError

tree = parse_query("machine learning AND NOT finance")
# And(left=Term(text='machine learning'), right=Not(operand=Term(text='finance')))

documents = search(index, "security AND machine learning")
# [Document(doc_id='doc06', ...), Document(doc_id='doc14', ...), Document(doc_id='doc15', ...)]

try:
    search(index, "machine healthcare")
except BooleanSyntaxError as error:
    print(error)
```

| Function | Purpose |
|---|---|
| `scan(query)` | raw text → tokens (public for testing) |
| `parse_query(query)` | raw text → AST |
| `evaluate(node, index)` | AST → `set[str]` of doc ids |
| `search(index, query)` | the whole pipeline → `list[Document]` sorted by id |
| `format_ast(node)` | readable one-line form of a parsed query |
| `explain(node, index)` | set sizes at each step, e.g. `9 & 3 = 2 docs` |

`search()` returns documents **sorted by `doc_id`**, so output is byte-identical on
every run — important for a demo or a printed assignment.

---

## 10. Verification

`python boolean_search.py` prints six sections. All of the following pass:

| # | Check | Result |
|---|---|---|
| 1 | All 10 measured query results, including the empty one | PASS |
| 2 | `a OR b AND c` equals `a OR (b AND c)` and differs from `(a OR b) AND c` (12 vs 4) | PASS |
| 3 | `evaluate()` equals hand-computed set arithmetic for 6 operand pairs | PASS |
| 4 | `NOT NOT x` equals `x` | PASS |
| 5 | Parentheses genuinely change the result; nested parens produce `And(Or, Or)`; NOT binds tighter than AND | PASS |
| 6 | 13 malformed queries all raise `BooleanSyntaxError` with a message, none leaking another exception type | PASS |
| 7 | Unknown terms return empty sets, never exceptions; `NOT unknown` returns all 20 | PASS |
| 8 | Repeat runs byte-identical and sorted | PASS |
| 9 | Earlier steps untouched — 20 docs, 1,624 terms, 608 vocab, 1,401 postings | PASS |

Checks 2, 4 and 5 matter most: they test the **parser**, not just the set maths. Correct
set operations over a wrongly-parsed query would still produce wrong answers.

---

## 11. Honest limitations

- **`OR` is not commutative in result size.** `A OR B` returning 13 of 20 documents is
  correct but useless as a search. Boolean retrieval has **no ranking**, so there is no
  "best" document to show. This is exactly the motivation for steps 5–9.
- **No stemming-aware operators.** The query goes through the same preprocessing as the
  documents, which is why `NLP AND education` works, but a user cannot search a word
  the stemmer mangled into something unrecognisable.
- **Phrase terms cannot be combined with their own words** in the way a user expects
  (`machine learning OR machine` behaves surprisingly — see §6).
- **The parser is hand-written**, not generated. For a two-operator language that is
  smaller and clearer than a parser generator.
- **No query result caching.** Each call re-parses and re-walks the AST. Irrelevant at
  20 documents; a production engine would compile queries once.

---

## 12. What this step deliberately does **not** do

| Feature | Step |
|---|---|
| TF — term frequency analysis | 5 |
| DF and **IDF** | 5 |
| **TF-IDF** weighting | 6 |
| Document length normalisation | 6 |
| Vector Space Model | 8 |
| **Cosine similarity**, ranked results | 9 |

Boolean retrieval returns an **unordered set**, which is the correct behaviour for the
model. Adding a score would not be Boolean retrieval — it would be the vector space
model, and pretending otherwise would misrepresent both.