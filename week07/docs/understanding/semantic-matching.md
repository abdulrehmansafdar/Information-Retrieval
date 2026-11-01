# 3. How semantic matching improves retrieval

**In one line:** a word-matching model cannot see an article that uses different words for the same thing, and LSI can, because words that keep appearing together share a concept.

## The idea, in the slide's example

Query: "space exploration". Six short documents, written for this demo using the slide's words:

| document | text | GVSM | LSI, k = 2 |
|---|---|---|---|
| D1 | space exploration mission | 0.82 | 1.00 |
| D2 | NASA space mission rocket | 0.35 | 1.00 |
| D3 | NASA rocket moon landing | 0.00 | 1.00 |
| D4 | astronaut programme moon rocket | 0.00 | 1.00 |
| D5 | fruits vegetables market | 0.00 | 0.00 |
| D6 | fresh fruits vegetables diet | 0.00 | 0.00 |

D3 and D4 never say "space" or "exploration", so GVSM gives them 0. D2 links the words: it has "space" along with "NASA" and "rocket". SVD merges these into one concept, so LSI reaches D3 and D4, and the food documents stay out.

Note that all four space documents score 1.00. With two concepts, LSI knows which topic a document is on but cannot rank inside that topic. This is the cost of a small k.

## On the clinical articles

Query: `wheezing`. Only the Asthma article contains the word.

| | GVSM | LSI, k = 15 |
|---|---|---|
| Asthma (m09) | 0.212 | 0.986 |
| COPD (m10), no shared word | not found | **0.874** |

COPD is the right second answer, and it does not contain "wheezing". LSI finds it because the two articles share a lot of other vocabulary, such as inhalers and airways.

## Measured

A claim needs a test, so `semantic_eval.py:76` runs 20 plain-language queries, two per specialty, written before any result was seen. An article is relevant when it has the query's specialty, and the specialty is never indexed, so neither model can see the answer.

| | GVSM | LSI, k = 15 |
|---|---|---|
| Mean average precision | 0.679 | **0.758** |
| Recall at R | 0.688 | **0.724** |
| Relevant articles with no shared word | 0 | 3 |

Query by query: **10 improve, 0 get worse, 10 stay the same**. The biggest gain is "mosquito bite" (0.40 to 0.93).

## Where it goes wrong

Say these out loud, because they are the honest part of the result.

- **LSI also brings wrong articles.** After Asthma and COPD, `wheezing` lists Migraine at 0.369 and Coronary Artery Disease at 0.339. They share no word and are not relevant. Reaching further finds more of the wrong articles as well as the right ones, which is why GVSM returns 1 article and LSI 13.
- **The answer key is coarse.** Of the 3 relevant articles with no shared word, only COPD for "breathing difficulty" is a clear success. The other two are Typhoid Fever and COVID-19 for "mosquito bite". They count because they are in the Infectious Disease specialty, yet neither is spread by mosquitoes.
- **k was chosen on the same queries it is scored on.** That flatters the 0.758. The direction of the result is believable, and k = 14 to 24 all beat GVSM, but its exact size is not.
- **The scores of the two models are not comparable.** LSI scores run much higher because there are far fewer dimensions. A 0.87 in LSI does not mean more than a 0.21 in GVSM. Compare the order, not the numbers.
- **It cannot use a word that is not in the collection.** `sadness` stems to `sad`, which appears in no article, so the query has no vector and both models refuse it. LSI relates words already in A; it does not know English.
- **This is a small test.** 30 short hand-written articles and 20 queries, with no clinical review. These articles are teaching material, not clinical guidance.
