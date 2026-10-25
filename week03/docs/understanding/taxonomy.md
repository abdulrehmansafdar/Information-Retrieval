# Week 03 — How the Category Hierarchy Is Derived

The browser needs a shape, and the shape should not be invented by hand if it can
be derived from the papers and then checked. This page states the rules, shows
the assignment they produce, and says what would make it wrong.

The rules live in `taxonomy.py:84` `RULES`, and `taxonomy.py:115` `classify()`
applies them: a document is tested against its **title** first, then its
**abstract**, and the first rule whose keyword occurs wins. Order is precedence.

---

## 1. The rules, in precedence order

| # | domain | subtopic | keywords |
|---|---|---|---|
| 1 | Applied AI | Healthcare | medical, clinical, patient, healthcare |
| 2 | Human-Computer Interaction | Interfaces & Facets | interface, faceted, usability, user experience |
| 3 | Evaluation & Measurement | Metrics & Judging | evaluating, evaluation, metrics, judged, assessor |
| 4 | Systems & Scalability | Distributed Retrieval | distributed, web scale, shard, latency |
| 5 | Information Retrieval | Query Processing | query expansion, boolean retrieval, query processing |
| 6 | Machine Learning & NLP | Representation & Graphs | graph, recommendation, embedding |
| 7 | Machine Learning & NLP | Neural Retrieval | neural, language model, cross-encoder, deep learning, transformer, code search |
| 8 | Information Retrieval | Structured & Proximity Search | proximity, structured field, structured text, metadata quality, passage retrieval |
| 9 | Information Retrieval | Indexing & Compression | index, compression, phrase quer |
| 10 | Information Retrieval | Ranking & Scoring | learning to rank, term weighting, ranking, scoring |
| 11 | Information Retrieval | Collection Organisation | clustering, document collection, collection organisation |

**Why the order matters.** *Deep Learning for Medical Image Segmentation* (p01)
contains both an ML word and an Applied word. Rule 1 tests `medical` before rule 7
tests `deep learning`, so the paper is Applied AI — which is the honest answer for
a paper about segmenting medical images. Swap the two rules and p01 changes
category; the order is the whole classification, not a detail.

## 2. The assignment it produces

The tree is `Repository → Domain → Subtopic → Document`. Every document's trail
has the shape `Repository / <domain> / <subtopic>`, e.g.

    p04  →  Repository / Information Retrieval / Structured & Proximity Search

The five subtopics of Information Retrieval and how the twenty papers fall:

| domain | subtopic | papers |
|---|---|---|
| Information Retrieval | Ranking & Scoring | p03, p16 |
| Information Retrieval | Indexing & Compression | p06, p11 |
| Information Retrieval | Structured & Proximity Search | p04, p17, p19 |
| Information Retrieval | Query Processing | p05, p10 |
| Information Retrieval | Collection Organisation | p13 |
| Machine Learning & NLP | Neural Retrieval | p02, p12, p14 |
| Machine Learning & NLP | Representation & Graphs | p08 |
| Applied AI | Healthcare | p01, p07 |
| Systems & Scalability | Distributed Retrieval | p18 |
| Evaluation & Measurement | Metrics & Judging | p09 |
| Human-Computer Interaction | Interfaces & Facets | p15, p20 |

Domain totals, largest first: Information Retrieval **10**, Machine Learning & NLP
**4**, Applied AI **2**, Human-Computer Interaction **2**, Systems & Scalability
**1**, Evaluation & Measurement **1** — twenty documents in **6** domains across
**11** subtopics.

## 3. The rules are checked, not trusted

Deriving the tree buys the ability to test it. `week03/verify.py` and
`tests/test_taxonomy.py` assert:

- every one of the twenty documents gets a domain and a subtopic;
- **no document falls through to the fallback**, `("general", "general")`. A
  fallback that is never empty is the signal that the rules no longer match the
  corpus;
- no category is empty — a domain or subtopic appears only if it holds a paper;
- each domain's total equals the sum of its subtopics;
- every document hangs off exactly one subtopic, so the subtopics partition the
  collection with no overlap and no gap.

## 4. What this does not prove

- **The keywords are a heuristic.** They are chosen to be defensible, not
  learned. A different reader would group these papers differently, and the point
  is not that this grouping is the only one — it is that *a* grouping is stated,
  applied consistently, and re-checked from the corpus every run.
- **A title carries the decision.** Because titles are tested first, a paper whose
  abstract is more representative of another domain is still classified by its
  title. That is a deliberate rule (titles are the author's own summary), but it
  is a rule, not a law.
- **The facets are metadata, not categories.** Year and venue are filters, not
  levels of the tree. A user can filter `year=2021` inside any category; the
  hierarchy itself never changes.
