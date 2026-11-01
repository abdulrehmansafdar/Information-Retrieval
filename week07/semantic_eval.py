"""Measuring the requirement "explain how semantic matching improves retrieval".

A claim that LSI is better needs a test the model cannot influence. Each query below is
written in plain words, and an article counts as relevant when its Specialty label
matches. The specialty is never indexed (see clinical_docs.py), so the answer key is
independent of both models.

Two queries per specialty were written BEFORE any result was seen, and none is removed
afterwards, so the numbers include the queries LSI does badly on. Two limits remain:
the set is small and the specialty is a coarse answer key.

    average precision   how high up the relevant articles appear (1.0 is perfect)
    recall at R         of the R relevant articles, how many are in the top R
    no_word_hits        relevant articles in the top R that share no word with the query

Run:  python semantic_eval.py
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_search import Settings, get_engine

JUDGED_QUERIES: tuple[tuple[str, str], ...] = (
    ("heart problems", "Cardiology"),
    ("chest pain and blood vessels", "Cardiology"),
    ("hormone problems", "Endocrinology"),
    ("blood sugar and weight", "Endocrinology"),
    ("breathing difficulty", "Respiratory"),
    ("lung disease and cough", "Respiratory"),
    ("mosquito bite", "Infectious Disease"),
    ("germs spreading between people", "Infectious Disease"),
    ("brain and nerve disorders", "Neurology"),
    ("headache and seizures", "Neurology"),
    ("stomach and digestion", "Gastroenterology"),
    ("liver damage", "Gastroenterology"),
    ("kidney and urine problems", "Nephrology"),
    ("filtering waste from blood", "Nephrology"),
    ("bones and joints pain", "Musculoskeletal"),
    ("stiff swollen joints", "Musculoskeletal"),
    ("low mood and sadness", "Mental Health"),
    ("talking therapy", "Mental Health"),
    ("tired from low red blood cells", "Haematology"),
    ("lack of iron", "Haematology"),
)


@dataclass(frozen=True)
class QueryResult:
    query: str
    specialty: str
    average_precision: float
    recall_at_r: float
    no_word_hits: tuple[str, ...]


@dataclass(frozen=True)
class EvalSummary:
    label: str
    mean_average_precision: float
    mean_recall_at_r: float
    no_word_hits: int
    per_query: tuple[QueryResult, ...]


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    found, total = 0, 0.0
    for position, doc in enumerate(ranked, start=1):
        if doc in relevant:
            found += 1
            total += found / position
    return total / len(relevant) if relevant else 0.0


def evaluate(settings: Settings) -> EvalSummary:
    """Run every judged query under one model and one set of settings."""
    engine = get_engine()
    rows = []
    for query, specialty in JUDGED_QUERIES:
        relevant = {a.doc_id for a in engine.articles if a.specialty == specialty}
        hits = engine.search(query, Settings(model=settings.model,
                                             weighting=settings.weighting,
                                             k=settings.k, top=len(engine.articles))).hits
        ranked = [h.doc_id for h in hits]
        top_r = hits[:len(relevant)]
        rows.append(QueryResult(
            query=query, specialty=specialty,
            average_precision=average_precision(ranked, relevant),
            recall_at_r=sum(h.doc_id in relevant for h in top_r) / len(relevant),
            no_word_hits=tuple(h.doc_id for h in top_r
                               if h.doc_id in relevant and not h.shared_terms),
        ))
    label = settings.model.upper() + (f" k={settings.k}" if settings.model == "lsi" else "")
    return EvalSummary(
        label=label,
        mean_average_precision=sum(r.average_precision for r in rows) / len(rows),
        mean_recall_at_r=sum(r.recall_at_r for r in rows) / len(rows),
        no_word_hits=sum(len(r.no_word_hits) for r in rows),
        per_query=tuple(rows),
    )


def main() -> None:
    print(f"{'model':12s} {'MAP':>6s} {'recall@R':>9s} {'no-word hits':>13s}")
    runs = [Settings(model="gvsm")] + [Settings(model="lsi", k=k)
                                       for k in (2, 5, 10, 15, 20, 25, 30)]
    for settings in runs:
        s = evaluate(settings)
        print(f"{s.label:12s} {s.mean_average_precision:6.3f} "
              f"{s.mean_recall_at_r:9.3f} {s.no_word_hits:13d}")


if __name__ == "__main__":
    main()
