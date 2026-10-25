"""Step 2 - the hypertext model, H = (N, L).

The lecture gives the hypertext model as a graph:

    H = (N, L)

    N = {Introduction, Chapter 1, Chapter 2, References, Conclusion}
    L = {Introduction -> Chapter 1, Chapter 1 -> References, ...}

This module builds that graph over the collection. Every node the browser can
stand on is an N, and every move the browser can make is an L:

    N   repository, categories (domain/subtopic), documents, and each document's
        sections (Abstract, Methodology, Results)
    L   five kinds of link, each with a direction and a reason:

        hierarchy      repository -> domain -> subtopic -> document
        internal       document  -> its sections, section -> next section
        related        document  -> document, by cosine over title + abstract
        shared-venue   document  -> document, same publication venue
        shared-year    document  -> document, same year

This is the same graph the structured-guided browsing model walks; the hierarchy
links ARE the category tree. Keeping one graph instead of three is what lets the
browser ask a question like "how do I get from this paper to that one?" and get
an answer made of the same moves the user sees on screen.

Run:  python hypertext.py
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache

from collection import Document, get_documents
from taxonomy import get_taxonomy

RELATED_TOP_K = 3

LINK_KINDS = ("hierarchy", "internal", "related", "shared-venue", "shared-year")


@dataclass(frozen=True)
class Node:
    """One N. Documents and sections carry the doc_id they belong to."""

    node_id: str
    kind: str            # 'repository' | 'category' | 'document' | 'section'
    label: str
    doc_id: str = ""
    section: str = ""


@dataclass(frozen=True)
class Link:
    """One L: a move from one node to another, and why it is possible."""

    source: str
    target: str
    kind: str
    weight: float = 0.0

    def __str__(self) -> str:
        return f"{self.source} -[{self.kind}]-> {self.target}"


def document_node(doc_id: str) -> str:
    return f"doc/{doc_id}"


def section_node(doc_id: str, section: str) -> str:
    return f"doc/{doc_id}/{section.strip().lower().replace(' ', '-')}"


# ---------------------------------------------------------------------------
# content links: related documents
# ---------------------------------------------------------------------------


def related_scores(documents: list[Document], k: int = RELATED_TOP_K
                   ) -> dict[str, list[tuple[str, float]]]:
    """Top-k documents per document, by TF-IDF cosine over title + abstract.

    Ties break on doc_id so the graph is byte-identical between runs, which is
    what makes graph_stats() a number the docs can quote.
    """
    ids = [d.doc_id for d in documents]
    counts = {d.doc_id: Counter(d.topic_terms) for d in documents}

    df: Counter = Counter()
    for doc in documents:
        df.update(set(doc.topic_terms))
    total = len(documents)

    def idf(term: str) -> float:
        return math.log((total + 1.0) / (df[term] + 1.0))

    weights: dict[str, dict[str, float]] = {}
    for doc in documents:
        vector = {term: (1.0 + math.log(tf)) * idf(term)
                  for term, tf in counts[doc.doc_id].items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        weights[doc.doc_id] = {term: v / norm for term, v in vector.items()}

    out: dict[str, list[tuple[str, float]]] = {}
    for a in ids:
        sims: list[tuple[float, str]] = []
        for b in ids:
            if a == b:
                continue
            va, vb = weights[a], weights[b]
            if len(va) > len(vb):
                va, vb = vb, va
            dot = sum(w * vb.get(term, 0.0) for term, w in va.items())
            sims.append((round(dot, 6), b))
        sims.sort(key=lambda pair: (-pair[0], pair[1]))
        out[a] = [(b, s) for s, b in sims[:k] if s > 0.0]
    return out


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


class HypertextGraph:
    """H = (N, L), with the lookups a browser and the docs both need."""

    def __init__(self, nodes: dict[str, Node], links: list[Link]) -> None:
        self.nodes = nodes
        self.links = links
        self._out: dict[str, list[Link]] = defaultdict(list)
        self._in: dict[str, list[Link]] = defaultdict(list)
        for link in links:
            self._out[link.source].append(link)
            self._in[link.target].append(link)
        self._neighbor_cache: dict[str, set[str]] | None = None

    # --- navigation --------------------------------------------------------

    def find(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            raise KeyError(f"no such node: {node_id!r}")
        return self.nodes[node_id]

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    def out_links(self, node_id: str, kind: str | None = None) -> list[Link]:
        links = self._out.get(node_id, [])
        return [l for l in links if kind is None or l.kind == kind]

    def in_links(self, node_id: str, kind: str | None = None) -> list[Link]:
        links = self._in.get(node_id, [])
        return [l for l in links if kind is None or l.kind == kind]

    def neighbors(self, node_id: str) -> list[str]:
        """Every node reachable in one move, either direction, sorted.

        The undirected view of the directed links, which is what a path finder
        needs: a browser can follow a link and it can also go back up one.
        """
        return sorted(self._neighbors().get(node_id, ()))

    def _neighbors(self) -> dict[str, set[str]]:
        if self._neighbor_cache is None:
            table: dict[str, set[str]] = defaultdict(set)
            for link in self.links:
                table[link.source].add(link.target)
                table[link.target].add(link.source)
            self._neighbor_cache = dict(table)
        return self._neighbor_cache

    def degree(self, node_id: str) -> int:
        return len(self.neighbors(node_id))

    def shortest_path(self, start: str, goal: str) -> list[str]:
        """Breadth-first path between two nodes, or [] when none exists.

        BFS, not Dijkstra: every link here is one move, so the fewest-moves path
        is the shortest one and a plain queue finds it.
        """
        if start not in self.nodes or goal not in self.nodes:
            raise KeyError(f"unknown node in path {start!r} -> {goal!r}")
        if start == goal:
            return [start]
        seen = {start}
        queue: deque[list[str]] = deque([[start]])
        while queue:
            path = queue.popleft()
            for step in self.neighbors(path[-1]):
                if step in seen:
                    continue
                if step == goal:
                    return path + [step]
                seen.add(step)
                queue.append(path + [step])
        return []

    def related(self, doc_id: str, k: int = RELATED_TOP_K) -> list[Link]:
        node = document_node(doc_id)
        links = self.out_links(node, "related")
        return sorted(links, key=lambda l: (-l.weight, l.target))[:k]

    def related_docs(self, doc_id: str, k: int = RELATED_TOP_K
                     ) -> list[tuple[str, float]]:
        return [(link.target.split("/", 1)[1], link.weight)
                for link in self.related(doc_id, k)]

    # --- description -------------------------------------------------------

    def graph_stats(self) -> dict:
        by_kind = Counter(link.kind for link in self.links)
        kinds = Counter(node.kind for node in self.nodes.values())
        return {
            "nodes": len(self.nodes),
            "links": len(self.links),
            "repository": kinds["repository"],
            "categories": kinds["category"],
            "documents": kinds["document"],
            "sections": kinds["section"],
            "by_kind": {kind: by_kind.get(kind, 0) for kind in LINK_KINDS},
        }

    def __repr__(self) -> str:
        stats = self.graph_stats()
        return (f"HypertextGraph(|N|={stats['nodes']}, |L|={stats['links']}, "
                f"docs={stats['documents']}, sections={stats['sections']})")


def build_graph(documents: list[Document] | None = None) -> HypertextGraph:
    """Build H = (N, L) over the taxonomy and the collection."""
    documents = documents if documents is not None else list(get_documents())
    taxonomy = get_taxonomy()
    by_id = {d.doc_id: d for d in documents}

    nodes: dict[str, Node] = {}
    links: list[Link] = []

    def add_node(node: Node) -> None:
        nodes[node.node_id] = node

    def add_link(source: str, target: str, kind: str, weight: float = 0.0) -> None:
        links.append(Link(source, target, kind, weight))

    # 1. categories: repository -> domain -> subtopic
    add_node(Node("root", "repository", "Repository"))
    for category in taxonomy.walk():
        if category.node_id == "root":
            continue
        kind = "category"
        add_node(Node(category.node_id, kind, category.name))
        add_link(category.parent_id, category.node_id, "hierarchy")

    # 2. documents hang off their subtopic
    for document in documents:
        node_id = document_node(document.doc_id)
        add_node(Node(node_id, "document", document.title, doc_id=document.doc_id))
        add_link(taxonomy.leaf_of(document.doc_id), node_id, "hierarchy")

    # 3. internal links: a document and its sections, sections in reading order
    for document in documents:
        names = document.section_names
        for position, name in enumerate(names):
            node_id = section_node(document.doc_id, name)
            add_node(Node(node_id, "section", name, doc_id=document.doc_id,
                          section=name))
            add_link(document_node(document.doc_id), node_id, "internal")
            if position + 1 < len(names):
                add_link(node_id, section_node(document.doc_id, names[position + 1]),
                         "internal")

    # 4. related documents, from content
    for doc_id, pairs in related_scores(documents).items():
        for other, score in pairs:
            add_link(document_node(doc_id), document_node(other), "related",
                     score)

    # 5. shared venue and shared year
    by_venue: dict[str, list[str]] = defaultdict(list)
    by_year: dict[int, list[str]] = defaultdict(list)
    for document in documents:
        by_venue[document.venue].append(document.doc_id)
        by_year[document.year].append(document.doc_id)

    def pair_up(groups: dict, kind: str) -> None:
        for members in groups.values():
            members = sorted(members)
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    add_link(document_node(a), document_node(b), kind)
                    add_link(document_node(b), document_node(a), kind)

    pair_up(by_venue, "shared-venue")
    pair_up(by_year, "shared-year")

    links.sort(key=lambda l: (l.source, l.kind, l.target, -l.weight))
    return HypertextGraph(nodes, links)


@lru_cache(maxsize=1)
def get_graph() -> HypertextGraph:
    return build_graph()


def main() -> None:
    graph = get_graph()
    stats = graph.graph_stats()
    print(f"H = (N, L)   |N| = {stats['nodes']}   |L| = {stats['links']}")
    print(f"  nodes: repository {stats['repository']}, categories "
          f"{stats['categories']}, documents {stats['documents']}, sections "
          f"{stats['sections']}")
    print("  links: " + ", ".join(f"{k} {v}" for k, v in stats["by_kind"].items()))

    print("\nrelated documents (top 3 by cosine over title + abstract):")
    for doc in get_documents():
        pairs = graph.related_docs(doc.doc_id)
        shown = ", ".join(f"{other} ({score:.2f})" for other, score in pairs)
        print(f"  {doc.doc_id}  {doc.title[:44]:44s}  {shown}")

    print("\nshortest navigation path p01 -> p18:")
    path = graph.shortest_path(document_node("p01"), document_node("p18"))
    for source, target in zip(path, path[1:]):
        kinds = {l.kind for l in graph.out_links(source)
                 if l.target == target}
        also = {l.kind for l in graph.out_links(target) if l.target == source}
        via = ", ".join(sorted(kinds | also)) or "?"
        print(f"  {source:22s} -> {target:22s} via {via}")


if __name__ == "__main__":
    main()
