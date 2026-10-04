# Step 3 — Inverted Index

Guide to `index.py`: what an inverted index is, why it replaces reading every document, and how to prove this one is correct.

---

## 1. The core idea

A corpus is naturally stored **document → terms**:

```
doc01 → [artifici, intellig, branch, build, softwar, ...]
doc09 → [medic, imag, neural, network, segment, ...]
```

That direction is useless for search. To answer *"which documents mention healthcare?"*
you would have to open all 20 documents and read every term in each one.

An **inverted index** flips the direction — **term → documents**:

```
healthcar     → [doc03, doc04, doc09]
autonom       → [doc13]
machin learn  → [doc01, doc02, doc03, doc04, doc06, doc07, doc14, doc15, doc17]
```

Now the question is a **single dictionary lookup**. You never open a document.

### The vocabulary

| Term | Meaning |
|---|---|
| **term** | a word or phrase after preprocessing — `healthcar`, `machin learn` |
| **vocabulary** | the set of every distinct term in the whole corpus |
| **posting** | one entry: "term *t* occurs in document *d*, *tf* times" |
| **postings list** | all postings for one term — every document containing it |
| **document frequency (df)** | length of a postings list: how many documents contain the term |
| **hapax** | a term occurring in exactly one document (from *hapax legomenon*, "occurring only once") |

---

## 2. Data structures

```python
@dataclass(frozen=True)
class Posting:
    doc_id: str
    tf: int                      # occurrences inside that document

@dataclass
class InvertedIndex:
    documents: list[Document]              # metadata + terms
    postings: dict[str, list[Posting]]     # term → postings
    vocabulary: list[str]                  # sorted term list
    doc_lengths: dict[str, int]            # |d| per document
```

`Posting` is `frozen`, so a posting cannot be modified after it is created — the
index is effectively read-only once built.

### Two guarantees

- **Postings are sorted by `doc_id`** — output is byte-identical on every run.
  Unsorted `set`s shuffle between runs, which makes a demo impossible to present.
- **`vocabulary` is sorted alphabetically** — easy to scan and to binary-search later.

### Why the index also keeps `documents`

The index is keyed by *term*, so it **cannot** efficiently answer *"what terms are in
doc09?"* — that would mean scanning all 608 keys. Keeping the document list alongside
gives both directions:

| Question | Answered by |
|---|---|
| Which documents contain term X? | `index.postings[X]` |
| What terms are in doc09? How long is it? | `index.documents`, `index.doc_lengths` |

Later steps need both: Boolean retrieval queries the index, then result display needs titles.

---

## 3. How it is built

```python
for document in documents:
    for term, tf in sorted(Counter(document.terms).items()):
        postings.setdefault(term, []).append(Posting(document.doc_id, tf))
```

Three things happen in those two lines:

1. **`Counter(document.terms)`** counts occurrences — this is where `tf` comes from.
2. **`sorted(...)`** guarantees the postings list is ordered by doc_id.
3. **`setdefault(term, [])`** appends, creating the list on first sight of the term.

Note the loop runs over `Counter` items, so each term appears **once per document** —
a document can never appear twice in one postings list.

### Why `tf` is stored even though TF is "step 4"

Building the index unavoidably counts occurrences. Reading them off the index later is
free, whereas recomputing them would mean re-walking every document.

Division of labour:

| Step | Role |
|---|---|
| **3 (this step)** | *records* tf as a by-product, never uses it |
| **4 (TF)** | *analyses* tf — per-document tables, top terms, and why raw TF favours long documents |

---

## 4. The numbers, and an honest correction

```
documents            : 20
vocabulary size      : 608
total postings       : 1401
avg postings / term  : 2.30
postings per term    : min 1 | median 1 | max 17
```

1,401 postings for 608 terms, because a term can point at up to 17 documents.

### 1,624 → 1,401: what is really going on

```
term occurrences in the documents : 1624
postings stored in the index      : 1401
repeats folded into the tf field  :  223
```

`data` occurs **29 times** but in only **17 documents**. The postings list needs 17 slots,
not 29, because each slot carries its own count:

```
data → doc01:2  doc02:2  doc03:2  doc04:2  doc05:1  doc06:3  ... (17 entries)
       tf values sum to 29
```

**This is only a 14% reduction (1,624 → 1,401), and that is the honest result.**
Compression is *not* the reason to build an inverted index. This corpus repeats words
too little for saving space to matter — a real collection of long documents compresses
far more, but even then the win is modest.

**The real benefit is lookup speed.** Answering *"which documents contain X"* becomes one
dictionary read instead of opening and scanning every document. At 20 documents that is
invisible; at 20 million documents it is the difference between instant and impossible.

### df is not collection frequency

A mistake worth avoiding, made while writing the tests:

| Term | Occurrences | Documents (df) |
|---|---|---|
| `medic imag` | 3 | **2** |

`medical imaging` appears 3 times across **2 documents** (doc08 and doc09). Document
frequency counts **documents**, never occurrences. IDF in step 5 is built on df, so
confusing the two would corrupt every weight.

---

## 5. Postings-length distribution — the raw material for IDF

```
  1 document  (hapax)   309 terms  #################
  2-3 documents         195 terms  ###########
  4-5 documents          56 terms  ###
  6-10 documents         39 terms  ##
  11-20 documents         9 terms  #
```

**51% of the vocabulary (309 of 608) occurs in exactly one document.** The median
postings list has length **1**.

This table is the strongest argument for IDF:

| Postings length | Can it identify a document? |
|---|---|
| 17 (`data`) | No — true of most documents |
| 9 (`learn`) | Barely |
| 1 (`autonom`) | Yes, perfectly — it *is* doc13 |

A term in 17 documents cannot tell documents apart. A term in 1 document names its
document outright. **IDF is the formula that converts this table into a number.**

### The gradient already exists in your data

```
  data                 17 docs
  system               17 docs
  use                  17 docs
  model                16 docs
  softwar              15 docs
  inform               13 docs
  applic               12 docs
  artifici intellig    12 docs
  comput               11 docs
  larg                 10 docs
```

Top of that list = almost no discriminating power. Bottom = maximal precision.
This is why `data` and `system` will get near-zero IDF, and `autonom` (1 document) will
get the maximum.

---

## 6. Worked lookups — what happens when a user types a word

```
'machin learn'      -> 9 docs   doc01, doc02, doc03, doc04, doc06, doc07, doc14, doc15, doc17
'machin'            -> 3 docs   doc12, doc13, doc20
'learn'             -> 9 docs   doc01, doc02, doc05, doc07, doc09, doc14, doc15, doc16, doc18
'healthcar'         -> 3 docs   doc03, doc04, doc09
'autonom'           -> 1 doc    doc13
'blockchain'        -> []       not in the corpus
```

Three things to notice:

**Phrases and unigrams are separate entries.** `machin learn`, `machin` and `learn` all
exist independently, so you can search the concept *or* the individual word. Note that
`machin` appears in only 3 documents while `machin learn` appears in 9 — the phrase was
matched even where the bare word was consumed by it during preprocessing.

**An unknown term returns `[]`, never an error.** This is a hard requirement, not
politeness:

```
machine AND NOT blockchain
```

must treat `blockchain` as "appears in no documents" and return every machine document.
If `lookup()` raised, step 7 would crash on any query containing an unknown word.

**Unstemmed input is also a miss:**

```
lookup(index, "healthcare") == []      # correct: the index stores 'healthcar'
```

This is not a bug — it is the visible consequence of step 2. Any query path **must**
run through `preprocess_query()` or it will silently find nothing.

---

## 7. Reverse view — doc09

```
5. REVERSE VIEW - repeated terms in doc09
  Artificial Intelligence in Medical Imaging
  length 75, 70 distinct, 5 occur more than once:
    imag(2)  medic imag(2)  model(2)  organ(2)  scan(2)
```

Same information read the other way. Only the **repeated** terms are printed, because
that is the part the postings list cannot give you — the index stores `tf` per document,
but a listing produced from the index still shows each term once.

Included to make the point from §2 concrete: the index alone cannot produce doc09's full
term list efficiently, so the document list is kept too.

---

## 8. Public API

```python
from index import build_index, lookup, documents_containing, document_frequency

index = build_index()                    # loads + preprocesses + indexes in one call

lookup(index, "healthcar")               # [Posting(doc_id='doc03', tf=3), ...]
documents_containing(index, "healthcar") # {'doc03', 'doc04', 'doc09'}
document_frequency(index, "healthcar")   # 3
lookup(index, "blockchain")              # []  — safe miss
```

| Function | Purpose | Used in |
|---|---|---|
| `build_index()` | single entry point | every step from 7 on |
| `lookup()` | postings for one term | step 7 (Boolean) |
| `documents_containing()` | set of doc ids — ready for set operations | step 7 (AND/OR/NOT) |
| `document_frequency()` | df — the basis of IDF | step 5 |

`documents_containing()` returning a **set** is deliberate: Boolean AND is set
intersection, OR is union, NOT is difference. Building the retrieval step on top of
Python sets is the whole reason this helper exists.

---

## 9. How to run and verify

```powershell
pip install -r requirements.txt
python index.py          # 5 summary sections
python index.py --full   # all 608 posting lists, ~699 lines
```

Console output is rendered with [`rich`](https://github.com/Textualize/rich):
`Console` for rules and layout, `Table` for every data listing, `Panel` for the
grouped doc09 summary, and `Text` for styled bars.

The default output is deliberately short. Explanations live in this file, not in the
console — printing them twice helps neither the reader nor the code. `--full` exists
because a marker may ask to see the complete index, and 608 posting lists is too much
noise to print by default (309 of them are a single document id).

### A note on characters

Your Windows console is **cp437** and Python's stdout is **cp1252**, neither of which
can represent `█` or `·`. Hard-coding them prints `?`. So `index.py` picks its bar
characters from the detected encoding:

```python
_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"
```

Piped output or a legacy console gets `#` / `.`; a UTF-8 terminal gets proper block
characters.

To be precise: the source files **do** contain those Unicode characters, but only
inside the `if _UNICODE` branch — ten in total across the project (`█`, `░`, `·` and
`→`). They are never written to a console that cannot represent them. An earlier version
of this guide claimed the files were pure ASCII; that was wrong, and the distinction
matters if the project is marked on a machine with a different code page.

### Verification results

All checks pass on this corpus:

| # | Check | Result |
|---|---|---|
| 1 | Vocabulary size = 608, postings = 1,401, documents = 20 | PASS |
| 2 | **Invariant A** — `sum(tf)` = 1,624 = total term occurrences | PASS |
| 3 | **Invariant B** — per document, summed tf = document length | PASS |
| 4 | Every `tf >= 1`; no duplicate `(term, doc)` pair; both sorts correct | PASS |
| 5 | Probe df values match: `data` 17, `model` 16, `artifici intellig` 12, `machin learn` 9, `healthcar` 3, `autonom` 1 | PASS |
| 6 | `lookup("blockchain")` = `[]`; unstemmed `healthcare` also = `[]` | PASS |
| 7 | Every phrase term has postings; phrase df ≤ unigram df | PASS |

**Invariants A and B are the important ones.** Together they prove the index neither
loses nor invents information:

- **A** — every single term occurrence in the corpus is accounted for in exactly one posting.
- **B** — for any document, the tf values pointing at it add up to its true length.

If either breaks during a later refactor, the index is corrupt.

---

## 10. What this step deliberately does **not** do

| Feature | Step |
|---|---|
| TF analysis and tables | 4 |
| DF table and **IDF** | 5 |
| **TF-IDF** weighting | 6 |
| **Boolean retrieval** (AND / OR / NOT) | 7 |
| Vector Space Model | 8 |
| Cosine similarity, ranked results | 9 |

This step builds a **lookup structure** and nothing more. It records counts because the
structure requires them, but it never uses them to score, rank, filter or return anything.