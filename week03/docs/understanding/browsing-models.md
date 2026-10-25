# Week 03 — Browsing Models and Where They Are Implemented

The assignment: **design and implement browsing/navigation architecture for a
selected FYP application.** The application here is the paper repository the
other two modules already use, and the architecture is the three browsing models
from the lecture — flat, structured-guided and hypertext — built on one shared
engine.

This page is the index. It answers one question for every part of the module:
**which file and which line does it?** The other two pages go deeper:

| page | covers |
|---|---|
| `taxonomy.md` | how the category hierarchy is derived from the papers |
| `navigation.md` | breadcrumbs, the navigation history, and H = (N, L) |

---

## 0. The requirements, and where each one lives

| requirement | the code that does it |
|---|---|
| 1 Home page | `browsing.py:270` `BrowseEngine.home()`, and the Home tab of `app.py` |
| 2 Category hierarchy | `taxonomy.py:236` `build_tree()`, read through `browsing.py:254` `children()` |
| 3 Document list | `browsing.py:261` `documents_in()` |
| 4 Search + browse | `browsing.py:284` `search()` |
| 5 Breadcrumb navigation | `taxonomy.py:194` `breadcrumb()` |
| 6 Related document links | `hypertext.py:83` `related_scores()`, read through `hypertext.py:203` `related()` |
| 7 Document preview | `browsing.py:343` `preview()` |
| filters | `browsing.py:61` `Filters`, applied by `browsing.py:394` `_passes()` |
| navigation history | `browsing.py:155` `NavigationState`, `browsing.py:168` `visit()` |
| internal links | the internal-link pass inside `hypertext.py:234` `build_graph()` |
| H = (N, L) | `hypertext.py:129` `HypertextGraph`, summarised by `hypertext.py:215` `graph_stats()` |

Everything is reached through `browsing.py:408` `get_engine()`, which builds the
collection, the tree and the graph once and caches them.

---

## 1. Flat browsing

> Explore documents without a deep structure — move document by document, or read
> one document from end to end.

The flat model is the collection with no category in the way. In code that is
`documents_in("root")`, which returns all twenty documents in id order, and the
History tab, which shows the trail of documents the user has actually opened.

**Its weakness, stated the way the lecture does it.** Flat browsing in a large
collection loses context. The lecture's own measured example sums the time spent
on each document:

    Browsing Time = D1 + D2 + D3 = 30 + 10 + 45 = 85 minutes

Nothing in this module measures reading time, but the same idea shows up as
`NavigationState` recording where the user has been, so `recent()` can answer
"what was I just looking at?" without the user holding it in their head.

## 2. Structured guided browsing

> Organise documents into a hierarchy, and follow it level by level.

The hierarchy is the tree from `taxonomy.py`, walked by `children()` and
`documents_in()`, with `breadcrumb()` producing the trail. The lecture writes the
model as

    StructuredGuidedBrowsing(D, H, UI) = N

with D the documents, H the hierarchy, UI the interface elements that expose it,
and N the path the user actually takes. Here D is the collection, H is the
taxonomy tree, UI is the Browse tab (or the console `tree`/`cd` commands), and N
is whatever `NavigationState` records — including the fact that a user can skip
levels, exactly as they skip straight to a paper's Results.

The hierarchy is the same object the hypertext model walks: the `hierarchy`
links in `hypertext.py` ARE this tree, so the two models are never out of step.

## 3. Hypertext browsing

> Non-linear navigation through nodes and links: H = (N, L).

The lecture's example is a set of nodes and a set of links:

    N = {Introduction, Chapter 1, Chapter 2, References, Conclusion}
    L = {Introduction → Chapter 1, Chapter 1 → References, ...}

This module builds the same pair over the collection, with five kinds of link
(see `navigation.md` for the full table). The two operations a browser needs on a
graph are here: `out_links()` to see what a node points at, and
`hypertext.py:180` `shortest_path()` to answer "how do I get from here to there?".

---

## 4. How one page is built

Every page is assembled from the same three steps, so the two front ends cannot
disagree:

1. **Where am I** — `breadcrumb(node_id)` gives the trail.
2. **What is here** — `children()` for a category, `documents_in()` for a list,
   `preview()` for a document.
3. **Where can I go** — `out_links()` / `related()` for a document, `children()`
   for a category.

The console (`console_ui.py`) and the browser (`app.py`) both walk those steps and
render the same data; only the widgets differ.

---

## 5. What this does not prove

- **Twenty documents is a demonstration, not a benchmark.** A category tree over
  twenty papers is small enough to check by eye; a real portal has tens of
  thousands and its hierarchy is a design problem this page does not solve.
- **The categories are derived by keyword rules**, which is why they can be
  checked, but a rule set is only as good as the corpus it was written against.
  `taxonomy.md` lists the rules, and one test asserts no document falls through
  them — a corpus with a topic the rules have never seen would land in the
  fallback, which is the failure mode to watch for.
- **"Related" means lexically related.** It is a TF-IDF cosine over the title and
  abstract, so two papers that use the same words are linked even if they argue
  against each other, and a paper whose topic is phrased in fresh words has weak
  links. A production system would blend in citations and co-readership.
- **The shortest path is the fewest hops, not the best explanation.** BFS returns
  the path with the fewest moves; a user might prefer a longer path whose hops
  make more sense. The module shows the moves, not a story.
