# 2. Membership functions

**In one line:** a membership function turns how often a word appears in an article into a number from 0 to 1.

## Step 1: the weight

A word's weight is its count in the article body, plus 3 for each time it appears in the title. A title counts more because it says what the article is about. The constant is `TITLE_WEIGHT = 3` at `membership.py:42`.

## Step 2: the function

The code is at `membership.py:87`. There are three functions:

| function | rule |
|---|---|
| binary | 1 if the word is present, otherwise 0 |
| linear | the weight divided by the largest weight of that word |
| log | log(1 + weight) divided by log(1 + largest weight) |

The default is **log**. The "largest weight" is taken across all articles, so the article that uses the word most scores 1.

## Example: "hypertension"

| article | binary | linear | log |
|---|---|---|---|
| m01 Hypertension | 1.00 | 1.00 | 1.00 |
| m02 Coronary Artery Disease | 1.00 | 0.40 | 0.61 |
| m03 Heart Failure | 1.00 | 0.20 | 0.39 |

- **Binary** says m02 and m03 are equal, because both mention the word.
- **Linear** says m02 is twice as strong as m03 (0.40 against 0.20).
- **Log** sits between them: 0.61 for m02 and 0.39 for m03.

## Remember

- Log is never lower than linear.
- An article that does not use a word scores 0 under every function.
