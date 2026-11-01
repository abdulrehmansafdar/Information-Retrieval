# 2. LSI and SVD

**In one line:** split the term-document matrix into concepts, keep only the strongest k, and compare the query with documents by concept instead of by word.

## Step 1: the matrix A

Rows are terms, columns are documents. Slide 11's example is built by `slide_matrix` at `slide_examples.py:43`:

| | D1 | D2 | D3 |
|---|---|---|---|
| **t1** | 1 | 1 | 0 |
| **t2** | 1 | 0 | 1 |
| **t3** | 0 | 1 | 1 |

For the clinical articles, A is **1,128 terms by 30 articles**. The class is at `term_matrix.py:31`.

## Step 2: the SVD

A ≈ U Σ_k V_k^T

| matrix | size | meaning | one row or column says |
|---|---|---|---|
| A | terms × documents | the term-document matrix | what is in the documents |
| U | terms × k | the term-concept matrix | how much a term belongs to a concept |
| Σ_k | k × k | the singular values | how strong a concept is |
| V_k^T | k × documents | the document-concept matrix | how much a document belongs to a concept |

`numpy.linalg.svd` computes the decomposition at `lsi_model.py:50`. The cut to k is at `lsi_model.py:66`. Finding singular values by hand is a separate numerical topic, so a library does that one step. Everything after it is here to read.

Slide 11's matrix has singular values **2, 1, 1**. The second and third tie, so at k = 2 either of those concepts could be kept, or any mix of them. Its U and V at k = 2 show how SVD works, but they are not the only valid answer.

The matrices are kept out of the web page. To print them, run `--matrix slide` and `--svd slide` on the console (`slide`, `space` or `corpus`).

## Step 3: place the query

The lecture stops at the SVD and does not say how a query enters the concept space. This module uses:

- query: q_k = U_k^T q (`query_vector` at `lsi_model.py:89`)
- document: its column of Σ_k V_k^T

The reason for this choice: both are the same operation, because U_k^T A = Σ_k V_k^T. A query and a document are projected onto the concept axes in the same way, and Σ keeps its slide meaning of concept strength.

Another common textbook form is q_k = Σ_k^-1 U_k^T q, compared with the columns of V_k^T. On this corpus the two gave almost the same top results. This form has one property that can be tested: at full k it gives the same order as GVSM, which a test checks.

## Step 4: rank with the same cosine

`rank` at `lsi_model.py:93` uses the same `cosine_similarity` as GVSM. Only the vectors differ: k numbers, not 1,128. A negative score means the document points away from the query, so it is not returned.

## Choosing k

k = 15 keeps **67.3%** of A's squared weight (`energy_kept` at `lsi_model.py:82`). The default is set at `semantic_search.py:26`, from this measurement on the 20 judged queries:

| k | mean average precision |
|---|---|
| GVSM, for comparison | 0.679 |
| 2 | 0.369 |
| 5 | 0.646 |
| **15** | **0.758** |
| 25 | 0.740 |
| 30, every concept | 0.679 |

- Too few concepts blur different topics together.
- Keeping every concept gives back GVSM's order and removes the benefit.
- From about 14 to 24 the score holds steady, so 15 is a safe choice inside that range. It is not the single highest number: k = 12 scored 0.781, but k = 10 lost the "talking therapy" query that k = 15 keeps, so a peak that narrow is not one to rely on.

## Reading a concept

SVD gives concepts no names. `concepts` at `lsi_model.py:109` lists the terms that load on each one. On this corpus they are not always tidy: concept 1 mixes stone, iron and joint. Because the 30 articles cover quite separate topics, the first concepts tend to contrast topics rather than name one, as the signs of their weights show. The slide's neat "Space Exploration" theme appears clearly only in a small example, which is `space_exploration` at `slide_examples.py:54`.
