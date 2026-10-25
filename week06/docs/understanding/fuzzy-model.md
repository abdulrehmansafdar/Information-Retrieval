# 1. The fuzzy model

**In one line:** every article gets a score from 0 to 1 for each word, and AND, OR and NOT combine those scores.

## Why

Boolean search only says yes or no. A clinical article can be useful even when it does not contain every word in the query. The fuzzy model gives each article a score, so useful articles are ranked, not dropped.

## The corpus

The shared corpus holds 30 articles, 120 sections (four per article) and 10 specialties. The examples below use it.

## The three operators

| query | score |
|---|---|
| `A AND B` | the smaller of the two scores |
| `A OR B` | the larger of the two scores |
| `NOT A` | 1 minus the score |

In the code: `conjunction` is at `fuzzy_logic.py:45`, `disjunction` at `fuzzy_logic.py:53`, and `evaluate` walks the query tree at `fuzzy_logic.py:69`.

## Example

Query: `hypertension treatment`, which means `hypertension AND treatment`.

In Hypertension (m01), "hypertension" scores 1.00 and "treatment" scores 0.50. AND takes the smaller, so the score is 0.50.

The query returns **4 articles of 30**. m01 and m02 both score 0.50.

The lecture's example also works: min(0.8, 0.6) = 0.6.

## Boolean is a special case

If every score is either 0 or 1, the three operators give exactly the answers of Boolean search. The tests check every combination.

## Compared with Boolean

| | Boolean | Fuzzy |
|---|---|---|
| Result | in or out | a score from 0 to 1 |
| Typos | not found | found, with partial matching |
| Order | none | by score |

## Remember

- A score is a membership value between 0 and 1.
- AND = smaller, OR = larger, NOT = 1 minus the score.
