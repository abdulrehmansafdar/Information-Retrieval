"""Step 3 - the browsing engine, and the navigation state a browser sits on.

Every front end calls this module and nothing else, exactly as week02's
console_ui.py and app.py both call engine.py. The terminal and the browser
therefore cannot disagree about what a category contains, what a breadcrumb
reads, or what a search returned. It prints nothing and imports no UI framework.

The four moves a browser makes, and the function for each:

    list / walk a level     documents_in(node_id), children(node_id)
    see the trail           breadcrumb(node_id), paths.of(doc_id)
    search                  search(query, filters)
    open a document         preview(doc_id)         (preview + links)

The three browsing models of the lecture are all served from here:

    Flat         documents_in("root") - the whole collection as one list
    Structured   the taxonomy tree + breadcrumb + filters
    Hypertext    hypertext.py's H = (N, L), reached through preview()/related()

NavigationState is plain Python, kept out of the engine so the console and the
tests can drive it directly; Streamlit keeps one instance in session_state.

Run:  python browsing.py
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from collection import Document, corpus_stats, get_documents, tokenize
from hypertext import HypertextGraph, document_node, get_graph
from taxonomy import Category, Taxonomy, get_taxonomy

# The three models, described once so both front ends and the docs agree.
BROWSING_MODELS = (
    {"name": "Flat browsing",
     "idea": "Move document by document, or read one document from end to end.",
     "best_for": "simple reading and scanning a plain list.",
     "here": "documents_in('root') and the History tab"},
    {"name": "Structured guided browsing",
     "idea": "Follow a hierarchy one level at a time, with a breadcrumb back.",
     "best_for": "portals, repositories and folders.",
     "here": "taxonomy.py plus breadcrumb() and the Browse tab"},
    {"name": "Hypertext browsing",
     "idea": "Jump between nodes by following links: H = (N, L).",
     "best_for": "web pages, citations and documentation.",
     "here": "hypertext.py, related() and the Hypertext tab"},
)


# ---------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Filters:
    """The facet selection shared by the Browse and Search tabs."""

    domain: str = ""        # a domain node_id, e.g. 'root/ir'
    subtopic: str = ""      # a subtopic node_id, e.g. 'root/ir/ranking'
    year: int = 0           # 0 means any year
    venue: str = ""         # '' means any venue

    @property
    def active(self) -> bool:
        return bool(self.domain or self.subtopic or self.year or self.venue)

    def describe(self) -> str:
        parts = []
        if self.domain:
            parts.append(f"domain={self.domain}")
        if self.subtopic:
            parts.append(f"subtopic={self.subtopic}")
        if self.year:
            parts.append(f"year={self.year}")
        if self.venue:
            parts.append(f"venue={self.venue}")
        return ", ".join(parts) or "no filter"


@dataclass(frozen=True)
class DocSummary:
    """One row of a document list."""

    doc_id: str
    title: str
    year: int
    venue: str
    category: str        # subtopic name
    path: str            # 'Domain / Subtopic'


@dataclass(frozen=True)
class SearchHit:
    rank: int
    doc_id: str
    title: str
    score: float
    category: str
    path: str
    matched: int


@dataclass(frozen=True)
class SearchResponse:
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    error: str | None = None
    filters: Filters = field(default_factory=Filters)
    considered: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def count(self) -> int:
        return len(self.hits)


@dataclass(frozen=True)
class Preview:
    """Everything the Document page shows for one document."""

    doc_id: str
    title: str
    year: int
    venue: str
    category: str
    trails: list[Category]
    sections: list[tuple[str, str]]
    related: list[tuple[str, str, float]]      # (doc_id, title, score)
    peers_venue: list[tuple[str, str]]
    peers_year: list[tuple[str, str]]
    internal: list[str]                        # section names

    @property
    def path(self) -> str:
        return " / ".join(node.name for node in self.trails)


# ---------------------------------------------------------------------------
# navigation state
# ---------------------------------------------------------------------------


@dataclass
class NavigationState:
    """The back/forward stack and the recently-viewed list.

    Kept separate from the engine because it is per-user, per-session state while
    the engine is shared and read-only. The truncation on `visit` is what makes
    `forward` mean anything: moving after going back discards the redone future,
    the way every browser does it.
    """

    history: list[str] = field(default_factory=list)
    position: int = -1
    limit: int = 50

    def visit(self, node_id: str) -> str:
        if self.history and self.position >= 0 and \
                self.history[self.position] == node_id:
            return node_id
        del self.history[self.position + 1:]
        self.history.append(node_id)
        if len(self.history) > self.limit:
            self.history = self.history[-self.limit:]
        self.position = len(self.history) - 1
        return node_id

    @property
    def current(self) -> str:
        if 0 <= self.position < len(self.history):
            return self.history[self.position]
        return ""

    def can_go_back(self) -> bool:
        return self.position > 0

    def can_go_forward(self) -> bool:
        return 0 <= self.position < len(self.history) - 1

    def back(self) -> str:
        if self.can_go_back():
            self.position -= 1
        return self.current

    def forward(self) -> str:
        if self.can_go_forward():
            self.position += 1
        return self.current

    def recent(self, n: int = 5) -> list[str]:
        """Most-recent-first unique nodes, for the home page."""
        seen: list[str] = []
        for node_id in reversed(self.history[:self.position + 1]):
            if node_id not in seen:
                seen.append(node_id)
            if len(seen) >= n:
                break
        return seen

    def clear(self) -> None:
        self.history.clear()
        self.position = -1

    def breadcrumb_nodes(self, n: int = 12) -> list[str]:
        """The trail up to and including the current node."""
        return list(self.history[:self.position + 1])[-n:]


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


@dataclass
class BrowseEngine:
    """The loaded collection, its tree, its graph, and the one index over it."""

    documents: tuple[Document, ...]
    taxonomy: Taxonomy
    graph: HypertextGraph
    by_id: dict[str, Document]
    idf: dict[str, float]
    norm: dict[str, float]

    # --- labels ------------------------------------------------------------

    def label(self, node_id: str) -> str:
        """A readable name for any node a browser can stand on."""
        if node_id in self.taxonomy:
            return self.taxonomy.find(node_id).name
        if node_id in self.graph:
            return self.graph.find(node_id).label
        return node_id

    def summary(self, document: Document) -> DocSummary:
        leaf = self.taxonomy.category_of(document.doc_id)
        domain = self.taxonomy.find(leaf.parent_id)
        return DocSummary(document.doc_id, document.title, document.year,
                          document.venue, leaf.name, f"{domain.name} / {leaf.name}")

    # --- the tree ----------------------------------------------------------

    def children(self, node_id: str = "root") -> list[Category]:
        node = self.taxonomy.find(node_id)
        return [self.taxonomy.find(c) for c in node.children]

    def breadcrumb(self, node_id: str) -> list[Category]:
        return self.taxonomy.breadcrumb(node_id)

    def documents_in(self, node_id: str = "root",
                     filters: Filters | None = None) -> list[DocSummary]:
        """The documents under a node, with the facet filters applied."""
        filters = filters or Filters()
        ids = self.taxonomy.documents_in(node_id)
        out = [self.summary(self.by_id[i]) for i in ids
               if _passes(self.by_id[i], filters, self.taxonomy)]
        return sorted(out, key=lambda s: s.doc_id)

    def home(self, state: NavigationState | None = None) -> dict:
        stats = corpus_stats()
        domains = [{"node_id": d.node_id, "name": d.name, "count": d.count}
                   for d in self.taxonomy.domains()]
        recent = []
        if state is not None:
            recent = [{"node_id": n, "label": self.label(n)} for n in
                      state.recent(6)]
        return {"stats": stats, "domains": domains, "recent": recent,
                "models": list(BROWSING_MODELS),
                "document_count": len(self.documents)}

    # --- search ------------------------------------------------------------

    def search(self, query: str, filters: Filters | None = None,
               top: int | None = None) -> SearchResponse:
        """TF-IDF cosine over the whole text. Data only; prints nothing."""
        filters = filters or Filters()
        query = query.strip()
        if not query:
            return SearchResponse(query=query, filters=filters,
                                  error="Type something to search for.")
        terms = tokenize(query)
        if not terms:
            return SearchResponse(
                query=query, filters=filters,
                error=f"none of the words in {query!r} were searched - "
                      f"they are all stop words.")

        candidates = [d for d in self.documents
                      if _passes(d, filters, self.taxonomy)]
        unknown = sorted({t for t in terms if t not in self.idf})

        query_weights = {t: self.idf.get(t, 0.0) for t in set(terms)}
        qnorm = math.sqrt(sum(w * w for w in query_weights.values())) or 1.0

        scored: list[tuple[float, Document, int]] = []
        for document in candidates:
            vector = Counter(document.terms)
            dot = 0.0
            matched = 0
            for term, qw in query_weights.items():
                if term in vector:
                    matched += 1
                    dot += qw * self.idf.get(term, 0.0) * (1.0 + math.log(vector[term]))
            if dot <= 0.0:
                continue
            score = dot / (qnorm * (self.norm[document.doc_id] or 1.0))
            scored.append((score, document, matched))

        scored.sort(key=lambda row: (-row[0], row[1].doc_id))
        hits = []
        for position, (score, document, matched) in enumerate(scored, 1):
            leaf = self.taxonomy.category_of(document.doc_id)
            domain = self.taxonomy.find(leaf.parent_id)
            hits.append(SearchHit(position, document.doc_id, document.title,
                                  score, leaf.name,
                                  f"{domain.name} / {leaf.name}", matched))
        limited = hits[:top] if top else hits
        return SearchResponse(query=query, hits=limited, terms=sorted(set(terms)),
                              unknown=unknown, filters=filters,
                              considered=len(candidates))

    def filters_available(self) -> dict:
        facets = self.taxonomy.facets()
        facets["domain_nodes"] = {d.name: d.node_id
                                  for d in self.taxonomy.domains()}
        facets["subtopic_nodes"] = {n.name: n.node_id
                                    for n in self.taxonomy.leaves()}
        return facets

    # --- one document ------------------------------------------------------

    def preview(self, doc_id: str, k: int = 3) -> Preview:
        document = self.by_id[doc_id]
        leaf = self.taxonomy.category_of(doc_id)
        trails = self.taxonomy.breadcrumb(leaf.node_id)
        related = [(other, self.by_id[other].title, score)
                   for other, score in self.graph.related_docs(doc_id, k)]
        peers_venue = self._peers(doc_id, "shared-venue")
        peers_year = self._peers(doc_id, "shared-year")
        return Preview(
            doc_id=doc_id, title=document.title, year=document.year,
            venue=document.venue, category=leaf.name, trails=trails,
            sections=[(s.name, s.text) for s in document.sections],
            related=related, peers_venue=peers_venue, peers_year=peers_year,
            internal=list(document.section_names),
        )

    def _peers(self, doc_id: str, kind: str) -> list[tuple[str, str]]:
        links = self.graph.out_links(document_node(doc_id), kind)
        return [(l.target.split("/", 1)[1], self.by_id[l.target.split("/", 1)[1]].title)
                for l in links]

    def related(self, doc_id: str, k: int = 3) -> list[tuple[str, str, float]]:
        return [(other, self.by_id[other].title, score)
                for other, score in self.graph.related_docs(doc_id, k)]

    # --- paths -------------------------------------------------------------

    def path_between(self, start: str, goal: str) -> list[str]:
        """Node ids of a shortest navigation path, inclusive of both ends."""
        return self.graph.shortest_path(_as_node(start), _as_node(goal))

    def describe_path(self, path: list[str]) -> list[tuple[str, str, str]]:
        """(source label, link kind, target label) for each hop of a path."""
        hops = []
        for source, target in zip(path, path[1:]):
            kinds = {l.kind for l in self.graph.out_links(source)
                     if l.target == target}
            kinds |= {l.kind for l in self.graph.out_links(target)
                      if l.target == source}
            hops.append((self.label(source), ", ".join(sorted(kinds)) or "?",
                         self.label(target)))
        return hops


def _as_node(value: str) -> str:
    """Accept a document id ('p04'), a doc/section node, or a category node."""
    if value.startswith("doc/") or value.startswith("root"):
        return value
    return document_node(value)


def _passes(document: Document, filters: Filters, taxonomy: Taxonomy) -> bool:
    if filters.domain and not taxonomy.leaf_of(document.doc_id).startswith(
            filters.domain + "/"):
        return False
    if filters.subtopic and taxonomy.leaf_of(document.doc_id) != filters.subtopic:
        return False
    if filters.year and document.year != filters.year:
        return False
    if filters.venue and document.venue != filters.venue:
        return False
    return True


@lru_cache(maxsize=1)
def get_engine() -> BrowseEngine:
    """Build everything once. Repeated calls return the identical object."""
    documents = get_documents()
    idf: dict[str, float] = {}
    df: Counter = Counter()
    for document in documents:
        df.update(set(document.terms))
    total = len(documents)
    for term, count in df.items():
        idf[term] = math.log((total + 1.0) / (count + 1.0))

    norm: dict[str, float] = {}
    for document in documents:
        vector = Counter(document.terms)
        norm[document.doc_id] = math.sqrt(sum(
            (idf[t] * (1.0 + math.log(tf))) ** 2 for t, tf in vector.items())) or 1.0

    return BrowseEngine(
        documents=documents,
        taxonomy=get_taxonomy(),
        graph=get_graph(),
        by_id={d.doc_id: d for d in documents},
        idf=idf,
        norm=norm,
    )


def main() -> None:
    engine = get_engine()
    state = NavigationState()

    home = engine.home(state)
    print(f"home: {home['document_count']} documents in "
          f"{len(home['domains'])} domains")
    for domain in home["domains"]:
        print(f"  {domain['name']:30s} {domain['count']:>2}   {domain['node_id']}")

    print("\nbreadcrumb of p04:")
    print("  " + " / ".join(n.name for n in
                            engine.taxonomy.breadcrumb(
                                engine.taxonomy.leaf_of("p04"))))

    print("\nsearch 'proximity window':")
    response = engine.search("proximity window", top=3)
    for hit in response.hits:
        print(f"  {hit.rank}. {hit.doc_id}  {hit.score:.3f}  {hit.title}")

    print("\nsearch filtered to year=2021:")
    filtered = engine.search("ranking", Filters(year=2021), top=3)
    print(f"  {filtered.count} hits of {filtered.considered} candidates "
          f"({filtered.filters.describe()})")

    print("\npreview p17:")
    preview = engine.preview("p17")
    print(f"  {preview.path}")
    print(f"  sections: {', '.join(name for name, _ in preview.sections)}")
    print(f"  related:  " + ", ".join(f"{d} ({s:.2f})"
                                      for d, _, s in preview.related))

    print("\npath p01 -> p18:")
    for label, kind, target in engine.describe_path(
            engine.path_between("p01", "p18")):
        print(f"  {label:28s} -[{kind}]-> {target}")


if __name__ == "__main__":
    main()
