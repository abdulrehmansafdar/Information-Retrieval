"""Step 1 - the category hierarchy, derived from the papers themselves.

A browser needs a shape to walk. The lecture's FYP example is

    FYP Repository -> AI Projects -> Healthcare AI -> Diabetes Prediction
                   -> Proposal / Dataset / Code / Demo

so this module builds the same three-level shape over the paper collection:

    Repository -> Domain -> Subtopic -> Document

    Repository
      Information Retrieval
        Ranking & Scoring          p03, p16
        Indexing & Compression     p06, p11
        ...

The categories are DERIVED, not hand-labelled. `RULES` is an ordered list of
keyword groups; a document is matched against the TITLE first, then the
ABSTRACT, and the first rule whose keyword occurs wins. Order is precedence:
`medical` is tested before `learning`, so a paper about medical image
segmentation is Applied AI rather than generic machine learning.

Because the rules decide the tree, the tree can be checked: the assignment is
re-derived from the corpus in tests and in check_doc_claims.py, and a document
that falls through every rule lands in an explicit 'General' bucket that a test
asserts is empty. A category system that silently swallows an unmatched
document is worse than one that admits it has no rule for it.

Run:  python taxonomy.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from collection import Document, get_documents

# Display order of the domains. The first is the heart of the collection.
DOMAINS: tuple[tuple[str, str], ...] = (
    ("ir", "Information Retrieval"),
    ("ml", "Machine Learning & NLP"),
    ("applied", "Applied AI"),
    ("systems", "Systems & Scalability"),
    ("evaluation", "Evaluation & Measurement"),
    ("hci", "Human-Computer Interaction"),
)

DOMAIN_NAMES = dict(DOMAINS)

# The subtopics each domain may contain, in display order.
SUBTOPICS: dict[str, tuple[tuple[str, str], ...]] = {
    "ir": (
        ("ranking", "Ranking & Scoring"),
        ("indexing", "Indexing & Compression"),
        ("structured", "Structured & Proximity Search"),
        ("query", "Query Processing"),
        ("collections", "Collection Organisation"),
    ),
    "ml": (
        ("neural", "Neural Retrieval"),
        ("graphs", "Representation & Graphs"),
    ),
    "applied": (("healthcare", "Healthcare"),),
    "systems": (("distributed", "Distributed Retrieval"),),
    "evaluation": (("metrics", "Metrics & Judging"),),
    "hci": (("interfaces", "Interfaces & Facets"),),
}


@dataclass(frozen=True)
class Rule:
    """One keyword group: 'a title containing any of these is this category'."""

    domain: str
    subtopic: str
    keywords: tuple[str, ...]


# Ordered by precedence. A rule is tested against the title first; only if no
# rule matches the title is the abstract tested, again in this order.
RULES: tuple[Rule, ...] = (
    Rule("applied", "healthcare",
         ("medical", "clinical", "patient", "healthcare")),
    Rule("hci", "interfaces",
         ("interface", "faceted", "usability", "user experience")),
    Rule("evaluation", "metrics",
         ("evaluating", "evaluation", "metrics", "judged", "assessor")),
    Rule("systems", "distributed",
         ("distributed", "web scale", "shard", "latency")),
    Rule("ir", "query",
         ("query expansion", "boolean retrieval", "query processing")),
    Rule("ml", "graphs",
         ("graph", "recommendation", "embedding")),
    Rule("ml", "neural",
         ("neural", "language model", "cross-encoder", "deep learning",
          "transformer", "code search")),
    Rule("ir", "structured",
         ("proximity", "structured field", "structured text", "metadata quality",
          "passage retrieval")),
    Rule("ir", "indexing",
         ("index", "compression", "phrase quer")),
    Rule("ir", "ranking",
         ("learning to rank", "term weighting", "ranking", "scoring")),
    Rule("ir", "collections",
         ("clustering", "document collection", "collection organisation")),
)

FALLBACK = ("general", "general")
FALLBACK_DOMAIN_NAME = "General"


def classify(document: Document) -> tuple[str, str]:
    """The (domain, subtopic) a document belongs to. Title beats abstract."""
    title = document.title.lower()
    abstract = document.abstract.lower()
    for haystack in (title, abstract):
        for rule in RULES:
            if any(keyword in haystack for keyword in rule.keywords):
                return rule.domain, rule.subtopic
    return FALLBACK


@dataclass(frozen=True)
class Category:
    """One node of the hierarchy. Documents hang off the subtopic nodes."""

    node_id: str
    name: str
    kind: str            # 'root' | 'domain' | 'subtopic'
    parent_id: str
    children: tuple[str, ...]
    doc_ids: tuple[str, ...]
    depth: int

    @property
    def is_leaf(self) -> bool:
        return self.kind == "subtopic"

    @property
    def count(self) -> int:
        return len(self.doc_ids)


class Taxonomy:
    """The built category tree plus the lookups a browser needs."""

    def __init__(self, nodes: dict[str, Category], root_id: str,
                 leaf_of: dict[str, str]) -> None:
        self.nodes = nodes
        self.root_id = root_id
        self._leaf_of = leaf_of

    # --- lookups -----------------------------------------------------------

    def find(self, node_id: str) -> Category:
        if node_id not in self.nodes:
            raise KeyError(f"no such category: {node_id!r}; "
                           f"try one of {', '.join(sorted(self.nodes)[:6])} ...")
        return self.nodes[node_id]

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    @property
    def root(self) -> Category:
        return self.nodes[self.root_id]

    def domains(self) -> list[Category]:
        return [self.nodes[c] for c in self.root.children]

    def subtopics(self, domain_id: str) -> list[Category]:
        return [self.nodes[c] for c in self.find(domain_id).children]

    def leaves(self) -> list[Category]:
        """Every subtopic node, in tree order."""
        return [n for n in self.walk() if n.is_leaf]

    def walk(self) -> list[Category]:
        """Depth-first, children in the order they were built."""
        out: list[Category] = []

        def visit(node_id: str) -> None:
            node = self.nodes[node_id]
            out.append(node)
            for child in node.children:
                visit(child)

        visit(self.root_id)
        return out

    def breadcrumb(self, node_id: str) -> list[Category]:
        """Root-to-node path, inclusive. The trail a browser shows."""
        trail: list[Category] = []
        while node_id:
            node = self.find(node_id)
            trail.append(node)
            node_id = node.parent_id
        return list(reversed(trail))

    def documents_in(self, node_id: str) -> list[str]:
        """Every document under a node, sorted by id."""
        return list(self.find(node_id).doc_ids)

    def category_of(self, doc_id: str) -> Category:
        """The subtopic node a document hangs from."""
        try:
            return self.nodes[self._leaf_of[doc_id]]
        except KeyError:
            raise KeyError(f"no document {doc_id!r} in the collection") from None

    def leaf_of(self, doc_id: str) -> str:
        return self._leaf_of[doc_id]

    # --- facets ------------------------------------------------------------

    def facets(self) -> dict[str, dict[str, int]]:
        """Filter values and how many documents each covers."""
        documents = {d.doc_id: d for d in get_documents()}
        years: dict[str, int] = defaultdict(int)
        venues: dict[str, int] = defaultdict(int)
        domains: dict[str, int] = {n.name: n.count for n in self.domains()}
        for document in documents.values():
            years[str(document.year)] += 1
            venues[document.venue] += 1
        return {
            "domain": dict(sorted(domains.items())),
            "subtopic": {n.name: n.count for n in self.leaves()},
            "year": dict(sorted(years.items())),
            "venue": dict(sorted(venues.items())),
        }


def build_tree(documents: list[Document] | None = None) -> Taxonomy:
    """Derive the whole hierarchy from the documents' topics."""
    documents = documents if documents is not None else list(get_documents())

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for document in documents:
        grouped[classify(document)].append(document.doc_id)

    nodes: dict[str, Category] = {}
    children: dict[str, list[str]] = defaultdict(list)

    def add(node_id: str, name: str, kind: str, parent: str, depth: int,
            doc_ids: list[str]) -> None:
        nodes[node_id] = Category(node_id, name, kind, parent,
                                  (), tuple(sorted(doc_ids)), depth)
        if parent:
            children[parent].append(node_id)

    add("root", "Repository", "root", "",
        0, [d.doc_id for d in documents])

    # Domains in DOMAINS order, then any domain a rule created that is not
    # listed there (so a new rule cannot produce an invisible category).
    order = [d for d, _ in DOMAINS]
    present_domains = {domain for domain, _ in grouped}
    extra = sorted(present_domains - set(order) - {FALLBACK[0]})
    for domain in order + extra:
        if domain not in present_domains:
            continue
        name = DOMAIN_NAMES.get(domain, domain.title())
        domain_id = f"root/{domain}"
        domain_docs: list[str] = []
        for subtopic, subtopic_name in _subtopic_order(domain):
            doc_ids = grouped.get((domain, subtopic))
            if not doc_ids:
                continue
            node_id = f"{domain_id}/{subtopic}"
            add(node_id, subtopic_name, "subtopic", domain_id, 2, doc_ids)
            domain_docs.extend(doc_ids)
        add(domain_id, name, "domain", "root", 1, domain_docs)

    if FALLBACK[0] in present_domains:
        fallback_docs = grouped[(FALLBACK[0], FALLBACK[1])]
        add("root/general", FALLBACK_DOMAIN_NAME, "domain", "root", 1,
            fallback_docs)
        add("root/general/general", "Uncategorised", "subtopic",
            "root/general", 2, fallback_docs)

    # Freeze each node's child list.
    nodes = {nid: Category(node.node_id, node.name, node.kind, node.parent_id,
                           tuple(children.get(nid, ())), node.doc_ids,
                           node.depth)
             for nid, node in nodes.items()}

    leaf_of = {doc_id: f"root/{domain}/{subtopic}"
               for (domain, subtopic), doc_ids in grouped.items()
               for doc_id in doc_ids}
    return Taxonomy(nodes, "root", leaf_of)


def _subtopic_order(domain: str) -> tuple[tuple[str, str], ...]:
    """Defined subtopics first, in display order, then any the rules invented."""
    defined = SUBTOPICS.get(domain, ())
    known = {sid for sid, _ in defined}
    invented = sorted({rule.subtopic for rule in RULES
                       if rule.domain == domain and rule.subtopic not in known})
    return defined + tuple((sid, sid.title()) for sid in invented)


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    return build_tree()


def facets() -> dict[str, dict[str, int]]:
    return get_taxonomy().facets()


def main() -> None:
    tree = get_taxonomy()
    root = tree.root
    print(f"{root.name}: {root.count} documents in "
          f"{len(tree.domains())} domains\n")
    for domain in tree.domains():
        print(f"  {domain.name}  ({domain.count})")
        for subtopic in tree.subtopics(domain.node_id):
            ids = ", ".join(subtopic.doc_ids)
            print(f"      {subtopic.name:32s} {subtopic.count:>2}  {ids}")
    print()
    print("filter facets:")
    for facet, values in facets().items():
        shown = list(values.items())[:8]
        print(f"  {facet:9s} {', '.join(f'{k}={v}' for k, v in shown)}")


if __name__ == "__main__":
    main()
