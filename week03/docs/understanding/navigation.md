# Week 03 — Breadcrumbs, History, and the Hypertext Graph

The three things this page explains are the three things a user feels rather than
sees: where they are (breadcrumb), where they have been (history), and where they
can go (the graph).

---

## 1. Breadcrumbs

A breadcrumb is the root-to-node path of the taxonomy, and it is produced by
`taxonomy.py:194` `breadcrumb()`, which walks `parent_id` from the node up to the
root and reverses the list. Because it is built from the tree and not stored on
each page, it can never disagree with the hierarchy: move a category and every
breadcrumb below it changes with it.

    p04  →  Repository  /  Information Retrieval  /  Structured & Proximity Search

The browser shows this on the Browse, Document and Sidebar areas of `app.py`, and
the console prints it before every listing.

## 2. Navigation history

History is the one piece of per-user state, so it lives outside the shared engine
in `browsing.py:155` `NavigationState`. It is a list of node ids plus a cursor:

    history = [root, root/ir, doc/p04]
    position = 2                          # we are standing on doc/p04

`browsing.py:168` `visit()` appends and advances — but first it **truncates
everything after the cursor**. That single line is what makes `forward` mean
anything: go back two steps, open a new node, and the redone future is gone, the
way it is in every browser. `back()` and `forward()` move the cursor; `recent()`
returns the most-recent-first unique nodes for the home page.

The lecture's opportunistic browsing — start with a goal, pause for something
useful on the way — is exactly what this stack records: the goal is where you are,
and the pauses are the entries behind you.

## 3. The hypertext graph, H = (N, L)

`hypertext.py:234` `build_graph()` assembles the graph, and `hypertext.py:215`
`graph_stats()` reports it. It has **98 nodes** and **261 links**:

| nodes | count |
|---|---|
| repository | 1 |
| categories (6 domains + 11 subtopics) | 17 |
| documents | 20 |
| sections (3 per document) | 60 |

| link kind | count | one link means |
|---|---|---|
| hierarchy | 37 | repository → domain → subtopic → document |
| internal | 100 | document → its sections, and section → next section |
| related | 60 | document → document, by content |
| shared-venue | 6 | two documents in the same venue |
| shared-year | 58 | two documents in the same year |

The hierarchy links are the same edges the structured-guided model walks, so the
tree and the graph are one object, not two that can drift.

### 3.1 Where "related" comes from

`hypertext.py:83` `related_scores()` scores every document against every other
with a TF-IDF cosine over the **title and abstract**:

    score(a, b) = Σ_t  w_a(t) · w_b(t)
    w_d(t)      = (1 + log tf) · log( (N + 1) / (df + 1) )

and keeps the top three per document, requiring a positive score. That is
`20 × 3 = 60` related links, and every document has at least one. The score is a
lexical one — see `browsing-models.md` §5 for what it does not mean.

### 3.2 Shortest navigation paths

`hypertext.py:180` `shortest_path()` runs a breadth-first search over the
undirected view of the links: a browser can follow a link and it can go back up
one, so every edge is walkable both ways. BFS is enough because every link is one
move, so the fewest-moves path is the shortest one.

A worked path, `p01 → p18`:

    doc/p01  -[related]->  doc/p03  -[related]->  doc/p18

Two moves: the medical-imaging paper is linked to the learning-to-rank paper,
which is linked to the distributed-search paper. Every hop is a real link that
appears in the out-links table, which is what makes the path auditable rather than
invented.

### 3.3 The same links, rendered as prose

The Hypertext section draws `H = (N, L)` as a table. The Document preview draws
the same edges as underlined words inside the text: a word becomes a link when it
occurs in this paper and in one of its related papers, and is distinctive enough
to mean something (its idf is above 1). Clicking one opens `?doc=p04`, which the
app reads on load and records in the navigation history. So the graph and the
prose are two views of one structure, not two systems.

The links are **derived, not authored**: the corpus contains no citations, so a
link means "these two papers share a distinctive term", not "this paper cites
that one". That distinction is worth stating plainly in a viva.

## 4. What this does not prove

- **Breadcrumbs describe the category, not the visit.** They show where a document
  lives, not how the user got there. The history is the record of the visit; the
  breadcrumb is the map.
- **"Shortest" is fewest hops, not clearest route.** BFS ignores how meaningful a
  hop is. A path through a shared year can be shorter than one through related
  content and read worse.
- **The graph is small enough to draw.** 98 nodes is a demonstration; a graph a
  person could still follow by hand. A real catalogue would need clustering and
  pruning before its graph meant anything.
