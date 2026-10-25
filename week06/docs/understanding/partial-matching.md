# 3. Partial matching

**In one line:** a query word can reach an article word that is spelled a little differently.

## Step 1: stems

Each word is lower-cased, stop words are dropped, and the word is cut down to a stem (`tokenize` at `articles.py:64`). For example, "treatments" becomes the same stem as "treatment".

## Step 2: trigrams and the Dice score

A stem is split into three-letter pieces, with a space at each end. The similarity of two stems is the number of pieces they share, doubled and divided by the total pieces of both. Identical stems score 1.

In the code: `trigrams` is at `membership.py:48` and `dice` at `membership.py:54`.

## Example: a typo

The query `hypertenion` (a letter missing) matches the stem `hypertens` with similarity **0.70**.

- Fuzzy mode finds **7 articles**, and Hypertension is first at 0.70.
- Boolean mode finds **0 articles**, because the typo is not an exact word.

## Step 3: the threshold

Only matches at or above the threshold `tau` count. The default is 0.6. At `tau = 0.95`, the typo above gets **no match at all**. The code is `expansions` at `membership.py:101`.

## Step 4: the score of a matched word

The score is the similarity multiplied by the membership of the matched stem. The code is `best_match` at `membership.py:119`.

For the typo above in Hypertension: 0.70 × 1.00 = 0.70.

## More examples

- "anemia" matches "anaemia" with similarity **0.62**.
- "diabetis" matches "diabet" with similarity **0.77**.

## A known mistake

"heart failure" finds Type 2 Diabetes (m06) with a score of 0.60. The word "failure" becomes the stem "failur", and the diabetes article says the pancreas "gradually fails", which becomes the stem "fail". The two are close enough to pass the threshold. A lower threshold would make more of these mistakes. Fixing them needs a list of medical synonyms, which is the research gap this module points to.
