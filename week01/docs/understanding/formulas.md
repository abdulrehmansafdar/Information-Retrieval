# Week 01 — Every Formula and Where It Is Computed

One page that answers a single question: **for each model in this week, what is the
formula, and which line of code turns a raw `.txt` file into that number?**

The seven guides in this folder explain *why* each model is the way it is. This page
answers *where*, so a viva question like "how is the cosine actually computed?" has a
file and a line number, not a guess.

Every number quoted below was measured by running the code, not read off a slide.

---

## 0. The master table

| Step | Model | Main formula | Computed by | Line |
|---|---|---|---|---|
| 2 | Preprocessing | `raw → terms` | `preprocess_text()` | `preprocess.py:199` |
| 3 | Inverted index | `postings[t] = [(d, tf)]` | `build_index()` | `index.py:55` |
| 4 | Boolean | `A AND B`, `A OR B`, `A − B` | `evaluate()` | `boolean_search.py:227` |
| 5 | IDF | `idf(t) = log(N / df(t))` | `build_idf_model()` | `idf.py:94` |
| 6 | TF-IDF | `tfidf(t,d) = tf(t,d) × idf(t)` | `TfidfModel.term_weight()` | `tfidf.py:68` |
| 7 | Vector space | `score(q,d) = Σ w(t,q)·w(t,d)` | `dot_product()` | `vsm.py:59` |
| 7 | Vector space | `‖v‖ = √(Σ w²)` | `magnitude()` | `vsm.py:54` |
| 8 | Cosine | `cos(q,d) = (q·d) / (‖q‖·‖d‖)` | `cosine_similarity()` | `cosine.py:52` |
| 8 | Ranked results | sort by score, assign rank | `rank_documents()` | `cosine.py:75` |

---

## 1. Step 2 — Preprocessing

There is no arithmetic here, so "the formula" is the pipeline itself:

```
raw text
  → lowercase          → tokens          → stems
  → phrase terms       → drop stop words → final terms
```

**Every stage, and the line that runs it**

| Stage | Does | Computed by | Line |
|---|---|---|---|
| 1 Lowercase | `"Machine"` → `"machine"` | `tokenize()` | `preprocess.py:143` |
| 2 Tokenize | split on punctuation | `tokenize()` | `preprocess.py:145` |
| 3 Punctuation removal | keep only letters/digits | `tokenize()` | `preprocess.py:146` |
| 4 Stemming | `"networks"` → `"network"` | `stem_token()` | `preprocess.py:152` |
| 5 Phrase extraction | `"machine learning"` → one term | `extract_phrases()` | `preprocess.py:163` |
| 6 Stop word removal | drop `the`, `of`, … | `preprocess_text()` | `preprocess.py:211` |
| 7 Assemble | unigrams + phrases | `preprocess_text()` | `preprocess.py:214` |

**Call chain** — one document, from file on disk to a term list
```
preprocess.py:214   return unigrams + phrases
 ├─ unigrams ← preprocess.py:206-213   loop over zipped tokens/stems/keep
 │   ├─ stemmed ← preprocess.py:153     _stemmer.stem(token)      SnowballStemmer("english")
 │   ├─ keep    ← preprocess.py:166-179 extract_phrases()          longest match wins
 │   └─ stops   ← preprocess.py:206     get_stop_words()          nltk + academic extras
 └─ phrases ← preprocess.py:167          _phrase_lookup[token tuple]
                ← preprocess.py:131-138  _build_phrase_lookup()   21 phrases, pre-stemmed
```

The document's term list is title **plus** body, at `preprocess.py:250`:

```python
document.terms = preprocess_text(document.title) + preprocess_text(document.raw_text)
```

**Worked example**

```
"Hospitals and clinics collect medical data"
 → lowercase    hospitals and clinics collect medical data
 → tokens       hospitals and clinics collect medical data
 → stems        hospit and clinic collect medic data
 → stop words   "and" is a stop word, dropped
 → final terms  hospit  clinic  collect  medic  data
```

**Verify it yourself**

```powershell
python -c "import preprocess; preprocess._build_phrase_lookup(); print(preprocess.preprocess_text('Machine Learning in Healthcare'))"
# ['healthcar', 'machin learn']      <- unigrams first, phrases LAST
```

⚠️ **Read this before reusing the output.** `preprocess_text()` returns
`unigrams + phrases` — phrases are **appended last**, so list order does not match
word order in the document. That is harmless for TF-IDF, which only counts. It is
**fatal** for phrase or proximity work, which needs real positions. Week 02 solves it
with a separate positional pipeline (`week02/preprocess.py`).

---

## 2. Step 3 — Inverted index

```
postings[t] = [ (doc, tf), (doc, tf), … ]
```

**Symbol table**

| Symbol | Meaning | Computed by | Line |
|---|---|---|---|
| `postings[t]` | documents containing term *t* | `build_index()` | `index.py:60` |
| `tf(t,d)` | occurrences of *t* in *d* | `Counter(document.terms)` | `index.py:64` |
| `(d, tf)` pair | one posting | `Posting(document.doc_id, tf)` | `index.py:65` |
| `N` | number of documents | `number_of_documents` | `index.py:47` |
| `vocabulary` | every indexed term | `sorted(postings)` | `index.py:70` |
| `doc_lengths[d]` | terms in document *d* | `{d.doc_id: len(d.terms)}` | `index.py:71` |

**Call chain**

```
index.py:65     postings.setdefault(term, []).append(Posting(document.doc_id, tf))
 ├─ term ← index.py:64      sorted(Counter(document.terms).items())
 │                        ← preprocess.py:250   preprocess_text(title) + preprocess_text(raw_text)
 └─ tf   ← index.py:64      the count Counter produced, free by-product of one pass
```

Both `tf` and `df` come out of this single pass. `df` needs no work at all — it is the
length of the list this line just built:

```python
document_frequency = {term: len(index.postings[term]) for term in index.vocabulary}
```

**Worked example**

```
doc03 contains "clinic" twice, doc04 contains it once
 → index.py:64   Counter(...) gives {"clinic": 2, "data": 2, "use": 4, …}  for doc03
 → index.py:65   postings["clinic"] = [Posting("doc03", 2), Posting("doc04", 1)]
 → later         len(postings["clinic"]) = 2   ← that is df("clinic")
```

Note the asymmetry: `tf` differs between the two documents (2 vs 1) but `df` is a single
number for the whole corpus (2). Mixing them up is the single most common error in IR
code, and `inverted-index.md` §4 spends a page on it.

**Verify it yourself**

```powershell
python -c "import index; i=index.build_index(); print(i.postings['clinic'])"
# [Posting(doc_id='doc03', tf=2), Posting(doc_id='doc04', tf=1)]
```

---

## 3. Step 4 — Boolean retrieval

No scoring, so the "formula" is a set algebra over the postings lists from step 3:

| Query | Set operation | Computed by | Line |
|---|---|---|---|
| `A AND B` | intersection `A & B` | `evaluate()` on an `And` node | `boolean_search.py:241` |
| `A OR B` | union `A \| B` | `evaluate()` on an `Or` node | `boolean_search.py:243` |
| `A NOT B` | difference `A − B` | `evaluate()` on a `Not` node | `boolean_search.py:239` |
| one term | `{d : t ∈ d}` | `documents_containing()` | `boolean_search.py:236` → `index.py:84` |

**Call chain**

```
boolean_search.py:247   search(index, query)
 ├─ parse   ← boolean_search.py:217  parse_query()
 │             ← boolean_search.py:118  scan()      parentheses + operators, raw text
 │             ← boolean_search.py:162  Parser.parse() → parse_or → parse_and → parse_factor
 └─ evaluate ← boolean_search.py:227   evaluate(node, index)
               Not  → line 239   {all doc_ids} − evaluate(operand)
               And  → line 241   evaluate(left) & evaluate(right)
               Or   → line 243   evaluate(left) | evaluate(right)
               Term → line 236   documents_containing(index, term)
                                  ← index.py:86  {posting.doc_id for posting in lookup(index, term)}
                                    ← index.py:75  lookup() -> index.postings.get(term, [])
```

`NOT` at line 239 is the only operator that needs the full document set, because it must
subtract from *everything*. That set comes from `index.documents`, not from the postings.

**Worked example**

```
query:  machine learning AND healthcare
scan    → ['machine learning'] [AND] ['healthcare']
parse   → And(Term('machine learning'), Term('healthcare'))
evaluate:
   Term('machine learning') → ['doc01','doc02','doc03','doc04','doc06','doc07','doc14','doc15','doc17']
   Term('healthcare')       → ['doc03','doc04','doc09']
   And                      → ['doc03','doc04']                      ← the answer
```

**Verify it yourself**

```powershell
python -c "import index; from boolean_search import explain, parse_query; print(explain(parse_query('machine learning AND healthcare'), index.build_index()))"
# 9 & 3 = 2 docs
```

`explain()` (`boolean_search.py:267`) prints the set sizes and the result count, which is
the fastest way to check that the parser built the tree you expected rather than a
different one.

---

## 4. Step 5 — IDF

```
idf(t) = log( N / df(t) )              N = 20
```

**Symbol table**

| Symbol | On this corpus | Computed by | Line |
|---|---|---|---|
| `idf(t)` | `2.3026` for `clinic` | `IdfModel.idf_for()` | `idf.py:64` |
| `idf(t)`, in-corpus | dictionary lookup | `return self.idf[term]` | `idf.py:69` |
| `idf(t)`, unknown term | `2.9957` = `log(N/1)` | fallback | `idf.py:70` |
| `N` | `20` | `index.number_of_documents` | `idf.py:90` → `index.py:47` |
| `df(t)` | `2` for `clinic` | `len(index.postings[term])` | `idf.py:92` |
| `log(·)` | `2.3026` | `math.log` | `idf.py:94` |
| `tf(t,d)` | `2` in doc03 | `Counter(document.terms)` | `idf.py:46` (via `index.py:64`) |

The entire model is one line, `idf.py:94`:

```python
idf = {term: math.log(n / df) for term, df in document_frequency.items()}
```

**Call chain** — every hop from a raw `.txt` file to the number 2.3026

```
idf.py:94            math.log(n / df)
 ├─ n   ← idf.py:90   index.number_of_documents
 │       ← index.py:47   len(self.documents)                  20
 └─ df  ← idf.py:92   len(index.postings[term])                2
         ← index.py:65   postings.setdefault(term, []).append(Posting(...))
           ← index.py:64   Counter(document.terms)
             ← preprocess.py:250   preprocess_text(title) + preprocess_text(raw_text)
```

**Worked example** — `idf("clinic")`

```
N   = 20                         20 documents in the corpus
df  = 2                          clinic is in doc03 and doc04 only
idf = log(20 / 2)
    = log(10)
    = 2.302585092994046
```

Contrast with a useless term, `idf("use")`:

```
df  = 17                         use appears in 17 of 20 documents
idf = log(20 / 17) = 0.16251892949777494        ← 14× smaller
```

**Verify it yourself**

```powershell
python -c "import index, idf; print(idf.build_idf_model(index.build_index()).idf_for('clinic'))"
# 2.302585092994046
```

**The unknown-term rule at `idf.py:70`.** A term outside the corpus is given the weight
of a hapax (`log(N/1)`, the maximum available) rather than raising `KeyError`. This is
safe because such a term matches no document, so the weight can never reach a score.

---

## 5. Step 6 — TF-IDF

```
tfidf(t, d) = tf(t, d) × idf(t)
```

**Symbol table**

| Symbol | Meaning | Computed by | Line |
|---|---|---|---|
| `tf(t,d)` | count in *d* | `document.terms.count(term)` | `tfidf.py:84` |
| `tf(t,d)`, scaled | `float(tf)`, or `1+log(tf)` if sublinear | `term_weight()` | `tfidf.py:73` / `tfidf.py:71` |
| `idf(t)` | from step 5 | `idf_model.idf_for(term)` | `tfidf.py:74` → `idf.py:64` |
| the product | one weight | `return scaled * …` | `tfidf.py:74` |
| lookup for one pair | 0.0 if absent | `weight()` | `tfidf.py:49` |
| whole document vector | `{term: weight}` | `document_terms()` | `tfidf.py:53` |
| sum of a document's weights | **not** a score — grows with length | `document_total()` | `tfidf.py:60` |

The formula itself, isolated in `term_weight()` (`tfidf.py:68`):

```python
def term_weight(self, term: str, tf: int) -> float:
    if self.sublinear:
        scaled = 1.0 + math.log(tf) if tf > 0 else 0.0     # line 71
    else:
        scaled = float(tf)                                  # line 73
    return scaled * self.idf_model.idf_for(term)            # line 74
```

**Call chain** — every `(document, term)` weight in the corpus

```
tfidf.py:74    scaled * idf_model.idf_for(term)
 ├─ scaled ← tfidf.py:73    float(tf)
 │            ← tfidf.py:84    document.terms.count(term)         2
 │              ← preprocess.py:250   preprocess_text(title + raw_text)
 └─ idf    ← idf.py:64      IdfModel.idf_for(term)
              ← idf.py:94    math.log(n / df)                    2.3026
              ← idf.py:92    len(index.postings[term])           df = 2
```

`build_tfidf_model()` (`tfidf.py:77`) loops every document × every distinct term, which
is why there are exactly as many weights as there are postings: **1,401**.

**Worked example** — doc03, the term that shows why IDF exists

```
tfidf("clinic", doc03) = 2 × 2.302585  = 4.605170     ← the most useful term in doc03
tfidf("use",    doc03) = 4 × 0.162519  = 0.650076     ← the most frequent term in doc03
```

By raw tf, `use` wins 4 to 2. By TF-IDF, `clinic` wins by **7×**. That reversal is the
whole justification for step 6.

**Verify it yourself**

```powershell
python -c "import index, idf, tfidf; i=index.build_index(); t=tfidf.build_tfidf_model(i, idf.build_idf_model(i)); print(t.weight('doc03','clinic'), t.weight('doc03','use'))"
# 4.605170185988092 0.6500757179910998
```

---

## 6. Step 7 — Vector space model

Two formulas, two functions.

```
score(q, d) = Σ over terms t of  w(t,q) × w(t,d)          ← dot_product()
‖v‖         = √( Σ over terms t of w(t,v)² )              ← magnitude()
```

**Symbol table — the dot product**

| Symbol | Meaning | Computed by | Line |
|---|---|---|---|
| `score(q,d)` | `9.1110` for q=`machine learning healthcare`, d=doc04 | `dot_product()` | `vsm.py:63` |
| `w(t,q)` | query weight for *t* | `build_query_vector()` | `vsm.py:102` |
| `w(t,d)` | document weight for *t* | `document_vector()` | `vsm.py:49` → `tfidf.py:53` |
| `q·d` | sum over **shared** terms | `sum(weight * b.get(term, 0.0) …)` | `vsm.py:63` |
| drop score 0 | a document with no shared term is not returned | `if score <= 0: continue` | `vsm.py:127` |
| sort best first | `key=(-score, doc_id)` | `score_documents()` | `vsm.py:139` |

**Symbol table — the magnitude**

| Symbol | Meaning | Computed by | Line |
|---|---|---|---|
| `‖v‖` | `18.0907` for doc04 | `magnitude()` | `vsm.py:56` |
| `√` | `math.sqrt` | `magnitude()` | `vsm.py:56` |
| `Σ w²` | sum of squared weights | `sum(weight * weight …)` | `vsm.py:56` |
| `‖q‖` | `2.0583` | `QueryVector.norm()` | `vsm.py:74` |

Both functions iterate the **shorter** vector and look the other up with
`.get(term, 0.0)` (`vsm.py:61-63`). A term the document does not contain contributes
nothing — that is why a partial match still scores, and why there is no penalty term.

**Call chain** — the query weights

```
vsm.py:102    terms[term] = terms.get(term, 0.0) + count * weight
 ├─ count  ← vsm.py:94      Counter(found)
 └─ weight ← vsm.py:101     idf_model.idf[term] if weight_by_idf else 1.0
              ← idf.py:69     self.idf[term]                        2.3026
              where found ← vsm.py:93  preprocess_query(query)
```

**Worked example** — query `machine learning healthcare` against doc04

```
q = {healthcar: 1.8971, machin learn: 0.7985}
note: "machine", "learning", "healthcare" → the phrase term "machin learn" plus
      "healthcar". Three words become TWO terms, because preprocessing already knows
      that "machine learning" is one index entry.

|q| = √(1.8971² + 0.7985²) = √(3.5990 + 0.6376) = √4.2366 = 2.0583
q·d = 1.8971×w(healthcar,doc04) + 0.7985×w(machin learn,doc04)
    = 9.1110
```

**Verify it yourself**

```powershell
python -c "import index, idf, tfidf, vsm; i=index.build_index(); t=tfidf.build_tfidf_model(i, idf.build_idf_model(i)); q=vsm.build_query_vector('machine learning healthcare', t.idf_model); print(q.terms); print(vsm.dot_product(q.terms, vsm.document_vector('doc04', t)))"
```

---

## 7. Step 8 — Cosine similarity

```
cos(q, d) = ( q · d ) / ( ‖q‖ × ‖d‖ )
```

**Symbol table**

| Symbol | Meaning | Computed by | Line |
|---|---|---|---|
| `cos(q,d)` | `0.2447` for doc04 | `cosine_similarity()` | `cosine.py:62` |
| `q·d` | `9.1110` (from step 7) | `dot_product()` | `cosine.py:62` → `vsm.py:59` |
| `‖q‖` | `2.0583` | `magnitude()` | `cosine.py:59` → `vsm.py:54` |
| `‖d‖` | `18.0907` for doc04 | `magnitude()` | `cosine.py:59` → `vsm.py:54` |
| divide-by-zero guard | returns `0.0` | `if length_a == 0.0 …` | `cosine.py:60` |
| rank assignment | 1-based, best first | `rank_documents()` | `cosine.py:101` |

The entire formula is `cosine.py:62`:

```python
return dot_product(a, b) / (length_a * length_b)
```

**Call chain**

```
cosine.py:62    dot_product(a, b) / (length_a * length_b)
 ├─ dot_product  ← cosine.py:62 → vsm.py:63   Σ w(t,q)×w(t,d)
 ├─ length_a     ← cosine.py:59 → vsm.py:56   √(Σ w(t,q)²)
 └─ length_b     ← cosine.py:59 → vsm.py:56   √(Σ w(t,d)²)
                   where w(t,d) ← cosine.py:89 → vsm.py:49 → tfidf.py:53
                                    → tfidf.py:85 → tfidf.py:74 → tfidf.py:84, idf.py:64
```

`rank_documents()` (`cosine.py:75`) loops every document, computes one cosine, drops
the zeros (`cosine.py:91`), sorts (`cosine.py:100`) and numbers the survivors
(`cosine.py:101`).

**Worked example** — the tie cosine breaks

Query term `abl`, which doc01 and doc12 both contain exactly twice:

| | len | tfidf(`abl`) | ‖d‖ | **dot** | **cos** |
|---|---|---|---|---|---|
| doc01 | 83 | 4.6052 | 16.676 | **10.6038** | **0.2762** |
| doc12 | 79 | 4.6052 | 22.189 | **10.6038** | **0.2075** |

```
cos(q,doc01) = 10.6038 / (2.3026 × 16.676) = 10.6038 / 38.395 = 0.2762
cos(q,doc12) = 10.6038 / (2.3026 × 22.189) = 10.6038 / 51.090 = 0.2075
```

The dot product cannot separate them — they are tied to four decimal places. Cosine
ranks doc01 first. Note doc12 is the **shorter** document and still loses: its weight is
spread across more repeated terms, giving it the larger magnitude. The lesson is that
cosine rewards **concentration**, not brevity.

**Verify it yourself**

```powershell
python -c "import index, idf, tfidf, vsm, cosine; i=index.build_index(); t=tfidf.build_tfidf_model(i, idf.build_idf_model(i)); print([(r.doc_id, round(r.score,4)) for r in cosine.rank_documents('abl', i, t)])"
# [('doc01', 0.2762), ('doc12', 0.2075)]
```

---

## 8. Where the user actually sees all of this

| Surface | Entry point | Line |
|---|---|---|
| Command line | `python main.py --console "query"` | `main.py:35` |
| Web UI | `python main.py --web` → `app.py` | `main.py:40` |
| Shared engine | `search(query, top, mode)` | `engine.py:136` |
| Ranked path | `_search_ranked()` | `engine.py:176` |
| Boolean path | `_search_boolean()` | `engine.py:159` |
| One-pair similarity | `similarity(query, doc_id)` | `engine.py:211` |
| Corpus numbers for the UI | `corpus_stats()` | `engine.py:236` |

`engine.py` holds no retrieval logic of its own — it only decides *which* model runs and
wraps the result in a `SearchResponse`. That is why `app.py` and `console_ui.py` can
never disagree about an answer: they both call `engine.search()`.

---

## 9. Maintenance note

Line numbers above were verified against commit `80b502c`. **They are exact as of that
commit.** If you edit a file named in a call chain, re-check it:

```powershell
Select-String -Path week01\idf.py -Pattern "math.log"
```

The function names in the second column are stable; the line numbers are a convenience.
When they disagree after an edit, the function name is the source of truth.