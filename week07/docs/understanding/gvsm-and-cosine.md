# 1. GVSM and cosine similarity

**In one line:** a document and a query are both arrows in a space with one direction per term, and the smaller the angle between the arrows, the more relevant the document.

## The slide's example

Query: "fruits vegetables". Document D1 mentions fruits 5 times and vegetables 3 times.

| | fruits | vegetables |
|---|---|---|
| **D1** | 5 | 3 |
| **Q** | 1 | 1 |

So D1 = (5, 3) and Q = (1, 1). In the code, `query_vector` is at `term_matrix.py:69` and `document_vector` at `term_matrix.py:65`.

## The formula, step by step

Cosine Similarity(Q, D) = (Q · D) / (||Q|| × ||D||)

| step | formula | the slide's numbers | code |
|---|---|---|---|
| 1. Dot product | multiply matching positions, add them up | (1×5) + (1×3) = **8** | `vector_cosine.py:26` |
| 2. Lengths | square root of the sum of the squares | ‖Q‖ = √2, ‖D‖ = √34 | `vector_cosine.py:36` |
| 3. Divide | the dot product over the two lengths multiplied | 8 / 8.246 = **0.97** | `vector_cosine.py:44` |

No library computes the cosine. Each step is a plain loop, so the code reads like the formula. `cosine_steps` at `vector_cosine.py:98` keeps every number, and that is what the console and each result's "why" panel print.

## Why the graph is exact

The web page draws each cosine as an angle: θ = arccos(cosine). Vectors with 1,128 dimensions can still be drawn truly, because any two vectors lie on one flat plane. So the angle between Q and D on the screen is the real angle between them.

Two things in the graph have no meaning. Arrow length is not to scale, since cosine ignores length. And on the graph of all results, the gap between two articles' arrows is not their real angle: each arrow shares a plane with Q, not with the other articles.

## What the score means

- **1** means the same direction: highly relevant.
- **0** means no relation: no term in common.
- **-1** means the opposite direction.

The slide states the range as 0 to 1 and also lists -1. Both are right, for different vectors. Weights in GVSM are never negative, so GVSM scores stay between 0 and 1. LSI coordinates can be negative, so LSI scores can go down to -1.

## The model

`GvsmModel` at `gvsm_model.py:22` follows section 2 of the lecture: each dimension is a term, and documents are ranked by cosine to the query (`rank` at `gvsm_model.py:34`).

The VSM against GVSM table on slide 9 says the weighting is flexible, so `term_matrix.py:28` offers three:

| weighting | a value in the matrix | where it comes from |
|---|---|---|
| `count` | how many times the term occurs | slide 6 |
| `binary` | 1 if present, 0 if absent | slide 11 |
| `tfidf` (default) | count × log10(N / df) | slide 15 |

In the slide 16 notation, a document is d_j = Σ w_ij g_i. Here w_ij is the weight in the matrix, and g_i is the axis of term i.

## What this module does not do

The textbook GVSM (Wong and colleagues, 1985) also estimates how terms relate to each other and bends the axes to match. The slides and notes do not describe that, so it is not built here: this module follows the method in the lecture. In this module, terms get related through LSI.

Because of this, GVSM here cannot reach an article that shares no word with the query. That limit is what LSI removes.
