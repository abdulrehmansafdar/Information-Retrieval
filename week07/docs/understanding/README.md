# Week 7: reading guide

Read the documents in this order. Each one builds on the one before it.

1. `../../../corpus/README.md`: the data. What an article is and how it is stored.
2. `gvsm-and-cosine.md`: documents and queries as vectors, and the cosine formula step by step.
3. `lsi-and-svd.md`: the matrix A, its SVD, and what k does.
4. `semantic-matching.md`: how semantic matching improves retrieval, and where it does not.

Then read the code in this order. Each file says what it does in its first lines.

1. `clinical_docs.py`: loads the articles and turns words into stems.
2. `term_matrix.py`: the term-document matrix A and the query vector.
3. `vector_cosine.py`: the cosine formula, written out as plain loops.
4. `gvsm_model.py`: ranks articles by cosine in term space.
5. `lsi_model.py`: the SVD, the cut to k concepts, and ranking in concept space.
6. `semantic_search.py`: the engine. `search`, `compare` and `steps` are the main functions.
7. `slide_examples.py`: the lecture's examples, run through the same code.
8. `semantic_eval.py`: the measured comparison of the two models.
9. `console_ui.py` and `app.py`: presentation only. They call the engine and show its results.

Then check your understanding by running:

```powershell
.\.venv\Scripts\python.exe week07\main.py --console --query "wheezing" --compare
```

GVSM should find only Asthma. LSI should also find Chronic Obstructive Pulmonary Disease at 0.874, marked "no shared word".
