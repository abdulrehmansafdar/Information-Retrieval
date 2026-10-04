"""Full-engine pipeline walkthrough data for the Streamlit Pipeline tab.

Computes, with the REAL preprocessing functions and the REAL engine, everything
needed to display the two-act pipeline as plain text and tables:

    Act 1  Preprocessing   raw -> lower -> tokens -> clean -> stems
                           -> phrases -> terms
    Act 2  Retrieval       terms -> inverted index -> weights -> query
                           -> ranked results

This module only computes and returns data. The Streamlit app renders it.
"""

from __future__ import annotations

from nltk.tokenize import wordpunct_tokenize
from preprocess import (
    _build_phrase_lookup,
    extract_phrases,
    get_stop_words,
    preprocess_query,
    preprocess_text,
    stem_tokens,
)

from cosine import rank_documents
from engine import get_engine

STAGE_LABELS_ACT1 = [
    "Raw", "Lowercase", "Tokens", "Clean", "Stems", "Phrases", "Terms",
]

STAGE_CAPTIONS_ACT1 = [
    "the sentence before any step",
    "one spelling, so the matcher never misses a capital",
    "the tokenizer splits on punctuation, which survives as its own token",
    "only letters and digits remain",
    "Snowball cuts endings, so variants merge into one stem",
    "adjacent phrase-list words become ONE term",
    "stop words are dropped. final terms",
]

STAGE_CAPTIONS_ACT2 = [
    "each term resolves to its postings: which documents contain it, and how often",
    "idf gives each term its value; tf-idf is the weight inside a document",
    "the terms become a query, matched against those postings",
    "documents that share a term are scored by cosine and ranked",
]


def build_stage_model(sentence: str) -> dict:
    """Compute every stage's data with the REAL pipeline functions and engine."""
    _build_phrase_lookup()

    lower = sentence.lower()
    lowered_pieces = wordpunct_tokenize(lower)
    original_pieces = wordpunct_tokenize(sentence)

    cleaned = [p for p in lowered_pieces if any(c.isalnum() for c in p)]
    stems = stem_tokens(cleaned)
    phrases, keep = extract_phrases(stems)
    stops = get_stop_words()
    final = preprocess_text(sentence)

    chips = []
    clean_index = -1
    for position in range(len(lowered_pieces)):
        piece = lowered_pieces[position]
        is_punct = not any(c.isalnum() for c in piece)
        if not is_punct:
            clean_index += 1
        chips.append({
            "orig": original_pieces[position] if position < len(original_pieces)
                    else piece,
            "lower": piece,
            "punct": is_punct,
            "ci": clean_index if not is_punct else -1,
        })

    phrase_ranges: list[dict] = []
    i = 0
    phrase_index = 0
    while i < len(stems):
        if keep[i] is False:
            start, j = i, i
            while j + 1 < len(stems) and keep[j + 1] is False:
                j += 1
            text = phrases[phrase_index] if phrase_index < len(phrases) else "*"
            phrase_ranges.append({"start": start, "end": j, "text": text})
            phrase_index += 1
            i = j + 1
        else:
            i += 1

    act1 = {
        "raw": sentence,
        "lower": lower,
        "tokens": [c["lower"] for c in chips],
        "punct_positions": [i for i, c in enumerate(chips) if c["punct"]],
        "clean": cleaned,
        "stems": stems,
        "phrases": phrases,
        "phrase_ranges": phrase_ranges,
        "stops_used": sorted({t for t in cleaned if t in stops}),
        "final": final,
    }

    model = {"act1": act1}
    model.update(_retrieval_act(final))
    return model


def _retrieval_act(final_terms: list[str]) -> dict:
    """Act 2 structured data: inverted index, weights, query, ranked results."""
    engine = get_engine()
    index = engine.index
    idf_model = engine.idf_model
    tfidf_model = engine.tfidf_model

    unique = list(dict.fromkeys(final_terms))

    postings_rows = []
    weight_rows = []
    for term in unique:
        postings = index.postings.get(term, [])
        postings_rows.append({
            "term": term,
            "df": len(postings),
            "postings (doc : tf)":
                ", ".join(f"{p.doc_id}:{p.tf}" for p in postings) or "-",
        })
        idf = idf_model.idf_for(term)
        weight_rows.append({
            "term": term,
            "df": len(postings),
            "idf": round(idf, 4),
            "highest tf-idf": round(
                max((p.tf * idf for p in postings), default=0.0), 4),
        })

    query_text = " ".join(final_terms)
    query_terms = preprocess_query(query_text) if query_text else []
    all_ranked = rank_documents(query_text, index, tfidf_model) if query_text else []
    ranked_rows = [
        {"rank": r.rank, "doc": r.doc_id, "title": r.title,
         "cosine": round(r.score, 4)}
        for r in all_ranked[:5]
    ]

    return {
        "postings_rows": postings_rows,
        "weight_rows": weight_rows,
        "query_terms": query_terms,
        "scored_count": len(all_ranked),
        "ranked_rows": ranked_rows,
        "number_of_documents": index.number_of_documents,
    }