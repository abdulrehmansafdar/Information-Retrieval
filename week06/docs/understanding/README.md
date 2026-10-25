# Week 6: reading guide

Read the documents in this order. Each one builds on the one before it.

1. `../../../corpus/README.md`: the data. What an article is and how it is stored.
2. `fuzzy-model.md`: the main idea. Every article gets a score from 0 to 1, and AND, OR and NOT combine the scores.
3. `membership-functions.md`: where the scores come from.
4. `partial-matching.md`: how a typo still reaches the right article.
5. `extended-boolean.md`: the second model. Exact include and exclude sets, with the order set by a p-norm.

Then read the code in this order. Each file says what it does in its first lines.

1. `articles.py`: loads the articles and turns words into stems.
2. `membership.py`: weights, membership values and partial matching.
3. `query_parser.py`: turns a query like `A AND NOT B` into a tree.
4. `fuzzy_logic.py`: the AND, OR and NOT rules.
5. `fuzzy_search.py`: the engine. `search`, `compare` and `set_view` are the main functions.
6. `console_ui.py` and `app.py`: presentation only. They call the engine and show its results.

Then check your understanding by running:

```powershell
.\.venv\Scripts\python.exe week06\main.py --console --query "hypertenion" --explain
```

You should see 7 articles, with Hypertension first at 0.70.
