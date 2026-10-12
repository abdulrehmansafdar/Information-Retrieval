# Step 2 — Preprocessing

Guide to `preprocess.py`: what each stage does, why it exists, and how to prove it worked.

---

## 1. The core idea

Raw text is a **string**. A search engine can only look things up in a **dictionary of terms**.
Preprocessing is the factory that converts one into the other.

```
"Artificial Intelligence is changing Healthcare!"
        ↓ 1. lowercase
"artificial intelligence is changing healthcare!"
        ↓ 2. tokenize (split on punctuation)
['artificial', 'intelligence', 'is', 'changing', 'healthcare']
        ↓ 3. remove punctuation   (commas, dots, quotes → gone)
        ↓ 4. stemming             (cut words to their root)
['artifici', 'intellig', 'is', 'chang', 'healthcar']
        ↓ 5. phrase extraction
['artifici', 'intellig', 'chang', 'healthcar', 'artifici intellig']
        ↓ 6. remove stop words
['artifici', 'intellig', 'chang', 'healthcar', 'artifici intellig']
```

A computer cannot match the string `"Healthcare"` against the string `"healthcare!"`.
After preprocessing both sides of the comparison are in the *same* normalised form,
which is the only reason matching is possible at all.

---

## 2. The pipeline in code

Implemented in `preprocess.py`:

| # | Stage | Function |
|---|-------|----------|
| 1 | Lowercase + tokenize + strip punctuation | `tokenize()` |
| 2 | Stemming | `stem_token()` / `stem_tokens()` |
| 3 | Phrase extraction | `extract_phrases()` |
| 4 | Stop word removal | inside `preprocess_text()` |
| 5 | Whole pipeline | `preprocess_text()` |
| 6 | Query version (keeps stop words) | `preprocess_query()` |
| 7 | Load + preprocess corpus | `load_documents()`, `build_documents()` |

Entry points for later steps:

```python
from preprocess import build_documents, preprocess_query

documents = build_documents()
documents[0].doc_id    # 'doc01'
documents[0].title     # 'Artificial Intelligence'
documents[0].terms     # ['artifici', 'intellig', 'branch', 'build', ...]

preprocess_query("machine AND NOT finance")
# ['machin', 'and', 'not', 'financ']
```

`build_documents()` loads **and** preprocesses, so the inverted index step never
touches raw text again. Nothing is written to disk — every step re-derives terms
from the source files, so there is no stale cache to invalidate.

---

## 3. Each stage, and why it exists

### 3.1 Lowercase

So `Healthcare`, `healthcare` and `HEALTHCARE` are **one** term.

Without it they occupy three separate index entries, and a query for one spelling
silently misses the other two. This is the cheapest, highest-impact fix in the whole pipeline.

### 3.2 Tokenization

Split the text into individual units. `nltk.word_tokenize.wordpunct_tokenize()`
splits on anything that is not a letter or digit, so punctuation arrives as its own
token and can be deleted in the next stage.

### 3.3 Punctuation removal

Commas, full stops, semicolons, quotes, brackets and hyphens are **not terms**.
They carry no meaning for retrieval, and leaving them in pollutes the vocabulary
with thousands of junk entries.

Implementation detail: every character outside `a-z0-9` is deleted. This also glues
contractions, so `isn't` becomes `isnt` instead of the two fragments `isn` + `t`.

### 3.4 Stemming

English expresses one idea in many surface forms:

| Written as | Stem |
|---|---|
| model, models, modelling, modelled | `model` |
| learn, learns, learning, learned | `learn` |
| image, images, imaging | `imag` |
| application, applications | `applic` |
| diagnos, diagnosis, diagnoses | `diagnos` |

Without stemming, a user searching **learning** would find nothing in a document
that says **learns**. Stemming collapses every variant into one term, so a single
match is enough.

*Snowball* is the algorithm used (the same family as Porter). It is a set of suffix
rules derived from English morphology. It only cuts word **endings**, never
beginnings — which is why results look chopped: `applic`, `diagnos`.

### 3.5 Stop word removal

Words like `the`, `is`, `of`, `and`, `that` appear in nearly every document.
They are useless for telling documents apart and would dominate any frequency count.

- nltk's English list: **198** words
- Plus academic filler added in `EXTRA_STOP_WORDS`: **26** words
  (`also, often, usually, still, thus, however, within, upon, whether, rather,
  said, almost, whenever, along, must, may, might, shall, one, every, many,
  much, like, non, go, even`)
- **Total: 224**

**Deliberately NOT removed:** `system`, `model`, `data`, `software`, `information`,
`network`, `security`, `computer`, `algorithm`.

These are the exact terms the whole assignment is built on. A stop-word list that
ate them would leave nothing to demonstrate. This is why a curated list beats a
generic one.

### 3.6 Phrase extraction

`machine learning` is **one concept**, not two. So when those two adjacent words are
found, the code emits an extra term `machin learn` **and keeps** the single words
`machin` and `learn`. Both forms go into the index.

Measured result:

| Term | Occurrences | Documents |
|---|---|---|
| `machin learn` | 17 | 9 |
| `machin` (alone) | 4 | 3 |
| `learn` (alone) | 12 | 9 |
| `artifici intellig` | 21 | 12 |
| `comput vision` | 4 | 3 |
| `neural network` | 4 | 4 |
| `anomali detect` | 2 | 2 |
| `medic imag` | 3 | 2 |
| `token` | 5 | — |

You can therefore search the **concept** or the **individual word**. This is what
makes the Boolean query `machine AND healthcare` still work while phrase search
stays available.

Phrases are matched **after** stemming, so the stored form is consistent with the
stemmed single words. That is why you see `machin learn` and not `machine learning` —
a query for `machine learning` is stemmed the same way and still matches.

`extract_phrases()` tries the **longest** match first, so `large language model`
wins over `language model`. 21 phrases are configured in the `PHRASES` tuple.

---

## 4. Two pipelines, not one

```python
preprocess_text("Machine Learning in Healthcare")
# → ['machin', 'learn', 'healthcar']          stop words REMOVED

preprocess_query("machine AND NOT finance")
# → ['machin', 'and', 'not', 'financ']        stop words KEPT
```

**Documents** drop stop words. **Queries** must not, because `NOT` has to survive.
Remove it and `machine AND NOT finance` silently becomes `machine AND finance`,
which returns exactly the wrong documents. This is why `preprocess_query()` is a
separate function — the Boolean step (step 7) depends on this behaviour.

---

## 5. Two design decisions worth defending

### 5.1 The `PROTECTED` set

Snowball sometimes over-chops or merges unrelated words:

| Word | Naive stem | Problem |
|---|---|---|
| `its` | `it` | merged with the pronoun *it* |
| `business` | `busi` | same stem as *busy* |
| `series` | `seri` | meaningless fragment |

These are exempted in `PROTECTED`. Only words with a genuine collision risk are
listed — protecting everything would defeat the purpose of stemming.

### 5.2 Order of stages, and a bug it caused

Stage order is not arbitrary:

```
tokenize → stem → extract phrases → remove stop words
```

Phrases are found **before** stop words are removed so that true original adjacency
survives. If stop words went first, `"state of the art"` could become the false
bigram `"state art"`.

The stop-word check tests **both** the surface form and the stem:

```python
if drop_stop_words and (original in stop_words or term in stop_words):
    continue
```

This was not theoretical. Testing showed the token `like` surviving in the output.
Cause: the corpus contains **"likely"** (doc20). `likely` is not a stop word, so a
surface-form-only check let it through — but its stem `like` **is** a stop word, and
the stem is what got stored. Fixed by checking both forms.

---

## 6. Results on this corpus

| Metric | Value |
|---|---|
| Documents | 20 |
| Words before preprocessing | ~141 per document |
| Terms per document | 71–90 |
| Total terms | 1,624 |
| Unique terms (vocabulary) | 608 |
| Terms appearing in only 1 document | 309 |
| Stop words leaked into output | 0 |

### Stemming worked

These surface forms no longer exist anywhere in the corpus — every occurrence was
merged into its stem:

| Surface form | Merged into | Total in corpus |
|---|---|---|
| `models` | `model` | 28 |
| `learning` | `learn` | 12 |
| `applications` | `applic` | 13 |
| `images` | `imag` | 8 |
| `networks` | `network` | 9 |
| `learning`, `models`, `images`, `applications`, `networks` | — | **0 each** |

### Document frequency spread — what step 5 (IDF) needs

| Term | df | Expected IDF behaviour |
|---|---|---|
| `data` | 17 | lowest |
| `system` | 17 | lowest |
| `model` | 16 | lowest |
| `softwar` | 15 | low |
| `artifici intellig` | 12 | low-medium |
| `machin learn` | 9 | medium |
| `secur` | 6 | medium |
| `educ` | 4 | high |
| `neural network` | 4 | high |
| `comput vision` | 3 | high |
| `healthcar` | 3 | high |
| `anomali detect` | 2 | highest |
| `transform` | 2 | highest |
| `autonom` | 1 | maximum |
| `iot` | 1 | maximum |

This is exactly the gradient needed to demonstrate IDF: a term in every document is
worthless for discrimination, a term in one document is gold.

---

## 7. How to verify it yourself

```powershell
python preprocess.py
```

`preprocess.py` prints seven sections using [`rich`](https://github.com/Textualize/rich):

1. **Pipeline walkthrough** — one sentence from doc03 traced through all seven stages
2. **All documents** — 20-row table: terms, distinct count, repeated-term count
3. **Corpus statistics**
4. **Top 20 terms** by collection frequency
5. **Stemming probe** — surface form → stem, with an honesty check
6. **Phrase terms found**, with occurrence and document counts
7. **Sample query processing** (showing that `NOT` survives)

An earlier version printed the complete term list of all 20 documents — about 110 of
the 135 output lines, and a wall of stems that demonstrated nothing. It was replaced by
a single traced example plus a summary table.

### Section 1 is the one worth reading

It picks a sentence containing punctuation and applies the **real** pipeline functions
in order, so each stage visibly differs from the one above it:

```
3 tokenized             ... medical data , including laboratory ...
4 punctuation removed   ... medical data including laboratory ...
5 stemmed               ... medic data, includ laboratori result ...
6 phrases found         medic data   patient record   machin learn
7 stop words removed    hospit clinic collect larg amount includ ...
```

Stage 3 → 4 shows punctuation being deleted. Stage 5 shows stemming. Stage 6 shows
**three** phrases extracted from a single sentence.

Because the walkthrough calls `tokenize()`, `stem_tokens()`, `extract_phrases()` and
`preprocess_text()` rather than re-implementing them, the demo cannot drift out of sync
with the real pipeline.

### Section 5 is honest about what it proves

```
surface form | became   | was in raw text | now in corpus | note
machine      | machin   |       yes       |             4 | merged
diagnosed    | diagnos  |        no       |             0 | never in corpus
```

This table used to label every probe word "merged away". That was **wrong on two rows**:
`diagnosed` and `neurons` were never in the corpus, so a count of 0 proved nothing about
stemming. The `was in raw text` column now separates the two cases:

- `merged` — the word occurs in the raw documents and its surface form is now gone.
  That is real evidence the stemmer worked.
- `never in corpus` — the word was never there; 0 means nothing.

---

## 7a. A note on console characters

Your Windows console is **cp437** and Python's stdout is **cp1252**. Neither can
represent `█` or `·`, so hard-coding them prints `?`. Both scripts pick their bar
characters from the detected encoding:

```python
_UNICODE = "utf" in (CONSOLE.options.encoding or "").lower()
BAR_ON, BAR_OFF = ("█", "░") if _UNICODE else ("#", ".")
SEP = "·" if _UNICODE else "|"
```

Piped output or a legacy console gets `#` / `.`; a UTF-8 terminal gets proper block
characters.

To be precise: the source files **do** contain those Unicode characters, but only inside
the `if _UNICODE` branch — ten in total across the project (`█`, `░`, `·` and `→`). They
are never written to a console that cannot represent them. An earlier version of this
guide claimed the files were pure ASCII; that was wrong.

---

## 8. Known limitations (good viva material)

**The stemmer is not perfect.** `application` → `applic` but `applied` → `appli`,
so those two do **not** merge — a genuine weakness of the Porter/Snowball family.
The proper fix is lemmatisation with a dictionary (`nltk` `WordNetLemmatizer`),
which maps words to real dictionary forms and needs an extra download:

```powershell
python -m nltk.downloader wordnet
```

The stemmer was kept because it is dependency-light, needs no extra data, and the
inconsistency does not affect any of the planned demonstration queries.

**Other honest limitations:**

- `wordpunct_tokenize` splits on every non-alphanumeric character, so `state-of-the-art`
  becomes four separate tokens and is never recognised as one phrase.
- Phrase detection is a fixed list of 21 phrases, not general n-gram extraction.
  It is predictable and easy to explain, which is the right trade-off here.
- `pd.read_csv`-style scaling is not a concern at this corpus size (1,624 terms),
  but document length normalisation becomes necessary once documents vary a lot in
  length. Relevant at the TF-IDF step.

---

## 9. Setup notes

```powershell
pip install nltk
python -m nltk.downloader stopwords
```

Only `stopwords` is required. `wordpunct_tokenize` and `SnowballStemmer` are pure
code and need no downloaded data — `punkt_tab` and `wordnet` are not used.

If the stop-word corpus is missing, `preprocess.py` prints a warning and falls back
to a small built-in list so the code still runs on another machine.

---

## 10. What preprocessing deliberately does **not** do

Out of scope for this step — each belongs to a later step:

| Feature | Step |
|---|---|
| Inverted index / postings lists | 3 |
| TF — term frequency | 4 |
| DF and IDF | 5 |
| TF-IDF weighting | 6 |
| Boolean retrieval (AND / OR / NOT) | 7 |
| Vector Space Model | 8 |
| Cosine similarity, ranked results | 9 |

Preprocessing only decides **what counts as a term**. It does not count them,
weight them, or search them.