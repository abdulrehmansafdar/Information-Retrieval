# 4. The extended Boolean model

**In one line:** the set of articles follows exact Boolean logic, and the order inside that set follows a p-norm score.

## Part 1: which articles are in

A word's set contains the articles where its membership is above **alpha** (the default, 0, means "mentions it at all"). Then:

- OR is the union of the sets.
- AND is the overlap of the sets.
- NOT removes the articles in the set.

So `(A OR B) AND NOT C` is the same as (A union B) minus C.

**Real check:** A = diabetes: 11 articles. B = hypertension: 7 articles. Their union has 12. Removing C = kidney leaves **m02, m08, m17, m18, m22**. The query `((diabetes) OR (hypertension)) AND NOT (kidney)` returns the same five.

In the code: `crisp_set` is at `fuzzy_search.py:185` and `set_view` at `fuzzy_search.py:191`.

## Part 2: the order, from the p-norm

The operators use the p-norm, from `fuzzy_logic.py:45` and `fuzzy_logic.py:53`. The exponent p sets how strict they are:

| p | behaviour |
|---|---|
| 1 | an average: a strong word can make up for a weak one |
| 2 | the default |
| larger | stricter, closer to min and max |
| infinity | exactly min for AND and max for OR (the fuzzy model) |

Examples with p = 2:

- AND of 0.8 and 0.6 gives 0.68, where fuzzy gives 0.6.
- OR of 0.8 and 0.6 gives 0.71, where fuzzy gives 0.8.
- `hypertension treatment` scores **0.65** for m01 in extended mode, against 0.50 in fuzzy mode.

## Part 3: exclusion is hard

`hypertension AND NOT diabetes` in extended mode returns **m01 only**. In fuzzy mode it returns 7 articles, because fuzzy NOT only lowers a score and does not remove the article.

## Compared on one query

`asthma OR COPD` returns 2 articles in every mode. The top score is 1.00 in Boolean and fuzzy mode, and 0.71 in extended mode. The articles are the same. Only the scores differ.

## Remember

- The set follows Boolean logic. The order follows the p-norm.
- p = 1 is an average, infinity is min and max, and 2 is the default.
