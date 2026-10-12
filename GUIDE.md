# GUIDE — running this project, and checking by hand that it is right

This repository holds two independent Information Retrieval modules.

| | what it is | where |
|---|---|---|
| **Week 1** | preprocessing, inverted index, boolean AND, TF-IDF, cosine / VSM, with a Streamlit UI | `week01/` |
| **Week 2** | positional index, BIM, BM25, BM25F, overlapped / non-overlapped lists, the proximal node model, and an effectiveness evaluation, on its own 20-paper corpus | `week02/` |

Week 2 is **self-contained**. It does not import week 1. Week 1 is kept as the
baseline it grew out of, and week 2 re-implements its boolean and TF-IDF models
so the evaluation can compare them on equal terms.

---

## 1. Setup

You need Python 3.11 or newer; this was built and tested on 3.13. Everything
runs from a virtual environment.

```powershell
git clone git@github.com:abdulrehmansafdar/Information-Retrieval.git
cd Information-Retrieval
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependencies are `nltk`, `rich`, `streamlit` and `pytest` — see `requirements.txt`.
On macOS or Linux the interpreter path is `.venv/bin/python` instead.

**No separate corpus download is needed.** Week 2's 20 papers are plain text
files in `week02/data/papers/p01.txt` … `p20.txt`, and week 1 ships its own
corpus in the same way. The first run takes a second or two while NLTK builds
its tokeniser.

---

## 2. Running it

All commands are run from the repository root.

### Week 2 — console

```powershell
.\.venv\Scripts\python.exe week02\main.py --console                    # interactive
.\.venv\Scripts\python.exe week02\main.py --console --stats           # corpus summary
.\.venv\Scripts\python.exe week02\main.py --console --eval            # effectiveness table
.\.venv\Scripts\python.exe week02\main.py --console "proximity window"  # one search
```

Inside the interactive console:

| type | effect |
|---|---|
| any words | search with the current mode |
| `bm25:` `bim:` `tfidf:` | force a ranking model |
| `bool: <query>` | force exact AND |
| `field: <query>` | force field-restricted search |
| `prox: <query>` | force a proximal node operator |
| `both: <query>` | overlapped and non-overlapped, side by side |
| `top N <query>` | limit the result list |
| `paper <p04>` | every region of one paper |
| `stats` / `modes` / `eval` | corpus summary / mode list / effectiveness |
| `help` / `quit` | the command list / exit |

The nine modes are `boolean`, `tfidf`, `bim`, `bm25`, `fields`, `bm25f`,
`overlapped`, `non-overlapped`, `proximal`.

### Week 2 — web UI

```powershell
.\.venv\Scripts\python.exe week02\main.py --web
```

Opens six tabs: **Search**, **Fields & Lists**, **Proximity**, **Evaluation**,
**Corpus**, **Pipeline**.

### Week 1

```powershell
.\.venv\Scripts\python.exe week01\main.py --console
.\.venv\Scripts\python.exe -m streamlit run week01\app.py
```

### The individual modules

Each week 2 module runs standalone and prints its own worked example. This is
the fastest way to see one model at full size rather than as a search result.

```powershell
cd week02
..\.venv\Scripts\python.exe preprocess.py        # tokenising, stemming, phrases
..\.venv\Scripts\python.exe structured_index.py  # positional index, regions
..\.venv\Scripts\python.exe ranking.py           # BIM, BM25, the k1=0 bridge
..\.venv\Scripts\python.exe fields.py            # restrict, filter, BM25F
..\.venv\Scripts\python.exe lists.py             # overlap / non-overlap
..\.venv\Scripts\python.exe proximity.py         # proximal node model
..\.venv\Scripts\python.exe evaluate.py          # the full effectiveness run
..\.venv\Scripts\python.exe engine.py            # the shared search API
```

`cd` back to the root before running the commands in the next section.

---

## 3. Verifying it automatically

Run all four from the repository root. Every one must pass.

```powershell
.\.venv\Scripts\python.exe week02\verify.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe week02\check_doc_lines.py
.\.venv\Scripts\python.exe week02\check_doc_claims.py
```

What each one covers:

| command | checks | expected last line |
|---|---|---|
| `week02\verify.py` | corpus shape, index invariants, every mode answers, error paths, BM25 ≡ BIM, evaluation sanity, **and both doc checkers** | `all checks passed` |
| `-m pytest` | 266 tests across 7 files | `266 passed` |
| `week02\check_doc_lines.py` | 74 `file.py:NN` references cited in the docs, each pointing at code **and that line containing what the page claims it contains** | `every reference resolves to code, and every claim matches that line` |
| `week02\check_doc_claims.py` | re-derives every number quoted in the docs from the corpus | `every number in the docs is still true` |

Plus the UI, which cannot be checked from `verify.py`:

```powershell
cd week02
..\.venv\Scripts\python.exe apptest_smoke.py      # every tab, mode and error path
cd ..
```

Expected: `ALL PASSED`. It drives the real Streamlit app through `AppTest`,
covering all six tabs, all nine modes, the `k1 = 0` switch, and six malformed
queries that must be refused rather than crash.

The two doc checkers exist because this repository's documentation was wrong
twice, for different reasons — once a line number drifted after an edit, and
once a figure had been written down without ever being measured. `verify.py`
calls both, so `all checks passed` already implies they agree.

**This guide is checked too.** `week02/tests/test_guide.py` asserts every "you
must see" in section 4, re-derives every figure quoted here, verifies that each
command it prints names a file that exists, and checks that the test count in the
table above matches what pytest actually collected. It found three wrong numbers
in the first draft of this page.

---

## 4. Verifying it by hand

This is the part that automation cannot do for you: **deciding whether the
answers are actually correct.** Every check below gives you a command, the
exact output that is correct, and where to look to convince yourself.

Where a check mentions a paper, you can open `week02/data/papers/pXX.txt` and
read it. That is the ground truth — if the engine disagrees with the file, the
engine is wrong.

### 4.1 The corpus is what the header claims

```powershell
.\.venv\Scripts\python.exe week02\main.py --console --stats
```

| you must see | why it matters |
|---|---|
| `20 papers` | one file per paper, `p01` … `p20` |
| `966 terms` | vocabulary size after stemming and phrase folding |
| `2,134 postings` | term → paper pairs |
| `3,434` stored positions | posting lists are *positional*, not just term → paper |
| `80 sections` / `120 paragraphs` / `299 sentences` | the region hierarchy the list and node models need |
| `144 to 197 terms (ratio 1.37x, average 171.7)` | papers are similar in length, so length normalisation has little to do — worth knowing before reading anything about BM25's `b` |
| `idf range 0.2412 to 2.6391` | a term in every paper sits near zero and cannot separate them |

**Read it yourself:** the most frequent term is `two` in 16 of 20 papers. Open
any two papers; both say "two". That is why its idf is the smallest in the
corpus.

### 4.2 BM25 and BIM are the same model at `k1 = 0`

This is the central identity of the module: BM25 with the saturation knob
turned to zero *is* the Binary Independence Model, because `idf · tf/(tf+0)` is
just `idf`.

```powershell
.\.venv\Scripts\python.exe week02\main.py --console --eval
```

| you must see | why it matters |
|---|---|
| `bm25 … MRR 1.000` | BM25 finds the right paper first for all 12 judged queries |
| `bim … MRR 0.917` | the simpler model is slightly worse |
| `bm25 … nohit 0` | BM25 never fails to return a judged-relevant paper |

Then, in the web UI's **Search** tab, set the **k1** slider to **0**. The app
tells you the model has become the Binary Independence Model, and the result
list must be identical to switching the mode to `bim`. If they differ, the
bridge is broken.

### 4.3 The list models find something different from each other

Overlapped requires every query term inside **one** region; non-overlapped
allows them in **different** regions. So non-overlapped must find a superset,
and it must be a strict one.

```powershell
.\.venv\Scripts\python.exe week02\main.py --console "both: document index positions"
```

| you must see | why it matters |
|---|---|
| `OVERLAPPED … 1 regions in 1 papers`, top hit `p06:results:p2`, terms `3/3` | all three words sit in one paragraph of p06 |
| `NON-OVERLAPPED … 4 regions in 2 papers` | the same three words spread over four regions in two papers |
| `non-overlapped only → p11` | p11 is only findable when you relax the constraint |
| `overlapped only → none` | if overlapped ever found something non-overlapped missed, one of them is wrong |
| `coverage 4 regions across 2 papers` | the trade is stated plainly: precision vs coverage |

**Read it yourself:** open `p06.txt` and find the results paragraph. It should
contain *document*, *index* and *positions* together. Open `p11.txt` and check
they are split across its abstract and its results.

### 4.4 A strict phrase is found; a loose phrase is not

```powershell
.\.venv\Scripts\python.exe week02\main.py --console "both: ranking web search"
```

Overlapped must return **0 regions** — no single region in this corpus contains
all three words. Non-overlapped must still return papers. If overlapped returns
a hit, the strictness check has been lost.

### 4.5 Cross-field AND means "in both fields"

```powershell
.\.venv\Scripts\python.exe week02\main.py --console "field: title:search AND results:precision"
```

| you must see | why it matters |
|---|---|
| exactly **1** result, `p19`, `title+results`, terms `2/2` | the title must contain *search* **and** the results must contain *precision* |
| `1 results in 1 of 20 papers` | |

This is the constraint week 1 cannot express — its documents have no named
fields, so it can only say "contains both words somewhere".

**Read it yourself:** open `p19.txt`. *Metadata Quality and Structured Field
Search* is its title, and its results section discusses precision. Then confirm
that no other paper has both of those words in those two places.

### 4.6 Brackets and OR mean what they look like

```powershell
.\.venv\Scripts\python.exe week02\main.py --console "field: (title:search OR abstract:rank) AND abstract:result"
.\.venv\Scripts\python.exe week02\main.py --console "field: title:search AND abstract:result OR abstract:rank AND abstract:result"
```

**Both must return the same five papers: `p03, p15, p18, p19, p20`.**

That is the algebra: `(A or B) and C` is the same as `(A and C) or (B and C)`.
The two commands differ only in punctuation, so any difference between them is
a parser bug. This is worth doing by hand precisely because it is so easy to
get wrong — an earlier version ignored brackets completely and answered the
second question while displaying the first.

### 4.7 A query answers less than it appears to — and says so

This is the part worth spending time on. A system that returns nothing is
ambiguous: the word may be misspelled, folded into a phrase, not in the corpus,
or meaningless. Each of these must be explained on screen.

| type this | you must see | why |
|---|---|---|
| `the` | `none of the words in 'the' were searched - they are all stop words` | **must not return 20 results.** Boolean search requires every term to be present, so a query with zero terms is trivially satisfied by every paper. 20/20 here is a bug, not a match. |
| `zzzznotaword` | `not in the vocabulary, ignored: zzzznotaword` | an unknown word must never be silently treated as a match |
| `proximity` | `'proxim' only occurs inside 'proxim oper' or 'proxim window'` | the word exists, but only inside two phrases, so it is not a term by itself |
| `title:neural` **in bm25 mode** | `this query has a field prefix, but mode 'bm25' ranks whole papers` | otherwise the engine searched for the ordinary word *title* and pretended it was a restriction |
| `machine learning` | `1/2` on the matched column | `machin` and `learn` are real terms but no paper has both, so 6 papers match one of the two — not a perfect match |

The `1/2` case is the subtle one. An earlier version divided *searched terms*
by *typed words*, so a legitimate two-word phrase like `inverted index` — which
is correctly folded into **one** term — reported `1/2`, looking like a partial
miss. A dropped word and a folded word are different things and are now counted
differently.

### 4.8 A malformed query is refused, not guessed at

Each of these used to parse into something plausible and answer a question you
did not ask.

```powershell
.\.venv\Scripts\python.exe week02\main.py --console "field: title:"
.\.venv\Scripts\python.exe week02\main.py --console "field: :search"
.\.venv\Scripts\python.exe week02\main.py --console "field: title:search AND"
.\.venv\Scripts\python.exe week02\main.py --console "field: (title:a OR title:b"
.\.venv\Scripts\python.exe week02\main.py --console "field: keywords:search"
.\.venv\Scripts\python.exe week02\main.py --console "field: title:search NOT results:x"
```

Every one must print an explanatory message and **no results**. Specifically:

- `title:` → *nothing to look for*
- `:search` → *no field name in front of it*
- `title:search AND` → *nothing to join to*
- `(title:a OR title:b` → *unmatched*
- `keywords:search` → *not a field of this corpus*, and the message must list `title`, `abstract`, `methodology`, `results`
- `NOT` → *not supported*. Negation needs a different answer set, and quietly
  dropping an operator is how `OR` went wrong: `OR` is an NLTK stop word, so a
  parser that split after preprocessing turned your `OR` into an `AND`.

### 4.9 The web UI is the same engine, not a second one

```powershell
.\.venv\Scripts\python.exe week02\main.py --web
```

1. **Search** — run `document index positions`; compare against the console
   result for the same query and mode. The scores, the order and the count must
   be identical. The console and the app both call `engine.search()`, so if they
   ever disagree you have found a bug in one of the two front ends.
2. **Fields & Lists** — `title:search` under *Fields*, and `document index
   positions` under *Lists*. Same rule.
3. **Proximity** — `section with(2) retrieval`. The tab must say the result is
   a **set, not a ranking**. This model has no scoring function, so any ordering
   displayed would imply a comparison the model never made. Then type
   `paragraph methodology`, which is missing its constructor, and confirm a
   readable error rather than a crash.
4. **Evaluation** — the same table as `--eval`, with BM25's MRR at 1.000.
5. **Corpus** — the same figures as section 4.1.
6. **Pipeline** — the raw-text → tokens → stems → phrases → stop words → terms
   trace for a query you choose. Pick something with a phrase in it and confirm
   the phrase folding happens at the stage the diagram says it does.

### 4.10 The evaluation is honest about what it is

```powershell
.\.venv\Scripts\python.exe week02\main.py --console --eval
```

| column | what it means |
|---|---|
| `P@1` | fraction of queries whose **first** result is relevant |
| `P@5` | precision across the top 5 |
| `R@10` | fraction of the relevant set found in the top 10 |
| `MRR` | mean of 1 / rank of the first relevant result |
| `MAP` | mean average precision over the whole ranked list |
| `nohit` | **queries where the model never returned a judged-relevant paper at all** |

Read `nohit` before MRR. It is the column that exposes a model which is perfect
on the easy queries and useless on the hard ones, and it is invisible in MRR.
Here `overlap` has an MRR of 0.750 *and* returns no relevant paper at all for
**three** of the twelve queries.

The relevance judgements are in `week02/data/relevance/qrels.txt` — 12 queries,
18 positives and **61** negatives. The negatives are the hard part of a qrels
file and the part most often left out: without them, a system that returns one
paper and happens to be right scores 1.0, and recall and MAP are undefined.
**Open this file.** Every number in the table is a direct function of it. If you
disagree with a judgement, the number is wrong no matter what the code says.

---

## 5. Reading the documentation

Seven guides, in `week02/docs/understanding/`. Start with `formulas.md`: it is
the index, and it traces every formula from the symbol to the function to the
line that computes it.

| guide | covers |
|---|---|
| `formulas.md` | every formula, and where each part of it is computed |
| `binary-independence-model.md` | BIM, the probability ranking model |
| `bm25.md` | BM25, length normalisation, the `k1 = 0` bridge |
| `structured-fields.md` | restrict / filter / BM25F, and the query parser |
| `overlapped-non-overlapped-lists.md` | the two list models |
| `proximal-nodes.md` | the proximal node model and its operators |
| `evaluation.md` | P@k, MRR, MAP, and what the numbers do not prove |

Week 1 has the same layout under `week01/docs/understanding/`.

**The docs are executable.** `check_doc_lines.py` asserts that each cited line
actually contains what the page claims it contains, and `check_doc_claims.py`
re-derives every quoted number from the corpus. If you change code and a doc
stops matching, those scripts fail rather than letting the page quietly go
wrong.

---

## 6. What these numbers do not prove

Worth being able to say in a viva.

- **20 papers is a demonstration, not a benchmark.** With 20 documents, `idf`
  is coarse, `R@10` is nearly always saturated, and a single lucky hit moves
  MRR by 0.08. Nothing here supports a claim about production quality.
- **12 queries cannot separate the top four models.** `tfidf`, `bm25` and
  `bm25f` are within one relevant paper of each other. The gap that is real is
  between those three and the list models, and it is visible in `nohit`, not
  only in MRR.
- **The judgements are mine**, made by reading the corpus. They are not a
  standard collection. `qrels.txt` is the input to every number, so it is also
  the first place to look if a result seems wrong.
- **The corpus is synthetic.** The 20 papers were written for this module so
  that each model has something distinctive to find. Real collections have more
  term-frequency skew, much longer documents, and no guarantee that any two
  papers differ in an interesting way.
- **BM25's length normalisation does very little here** (1.37× length ratio).
  On `structured field search` the raw scores run **1.35–1.66×** the normalised
  ones, which is close to a constant rescale rather than a reordering.
  `fields.py` searches for a query where it does change the order and reports
  honestly when there is not one.

---

## 7. If something fails

| symptom | cause |
|---|---|
| `ModuleNotFoundError: nltk` / `streamlit` / `rich` | the virtual environment is not active, or `pip install -r requirements.txt` was not run |
| pytest collects 0 tests | run from the repository root; `pytest.ini` sets `testpaths = week02/tests` |
| `failed: all checks passed` | read the lines above it — `verify.py` names each failure |
| `check_doc_lines.py` reports a bad reference | a line moved after an edit. Update the number **and** its entry in `CLAIMS` in that script |
| `check_doc_claims.py` reports a claim is false | either the code changed the behaviour, or the documentation was wrong. Work out which before editing either |
| the app will not start | `.\.venv\Scripts\python.exe -m streamlit run week02\app.py` bypasses `main.py` and shows the real traceback |

**Never "fix" a failing check by editing the expected value to match the code.**
That inverts the point of the check. Decide which one is wrong first. Every one
of these numbers was wrong at least once, and both of the ways it happened —
a line that moved, and a figure never measured — are exactly what these scripts
exist to catch.