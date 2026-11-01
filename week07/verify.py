"""Final verification for week07. Run from the repository root:

    .venv\\Scripts\\python.exe week07\\verify.py

or from inside the folder:

    python verify.py

Covers the shared corpus, the term-document matrix, the cosine formula against the
slide's numbers, the SVD's mathematical properties, the two models, the measured
comparison, and both documentation checkers. The same shape as week06's verify.py.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

import check_doc_claims
import check_doc_lines
from clinical_docs import CORPUS_DIR, tokenize
from gvsm_model import GvsmModel
from semantic_eval import evaluate
from semantic_search import MODELS, Settings, get_engine
from slide_examples import fruits_and_vegetables, slide_matrix, space_exploration
from term_matrix import WEIGHTINGS

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark} {name}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    engine = get_engine()

    print("== shared corpus ==")
    check("30 articles", len(engine.articles) == 30, str(len(engine.articles)))
    check("corpus is at the repository root", CORPUS_DIR.is_dir())
    check("week07 keeps no copy of the data",
          not (pathlib.Path(__file__).resolve().parent / "data").exists())

    print("\n== term-document matrix ==")
    matrix = engine.matrix()
    check("rows are terms and columns are documents",
          matrix.shape == (len(matrix.terms), len(engine.articles)))
    check("slide 11's matrix", slide_matrix().values.tolist() == [[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    check("slide 6's document vector", fruits_and_vegetables().document_vector("D1") == [5, 3])
    check("slide 6's query vector",
          fruits_and_vegetables().query_vector(tokenize("fruits vegetables")) == [1, 1])

    print("\n== cosine similarity (slides 7 and 8) ==")
    steps = fruits_and_vegetables().steps(tokenize("fruits vegetables"), "D1")
    check("the dot product is 8", steps.dot == 8)
    check("the denominator is 8.24", round(steps.denominator, 2) in (8.24, 8.25))
    check("the cosine is 0.97", round(steps.score, 2) == 0.97)

    print("\n== SVD ==")
    for weighting in WEIGHTINGS:
        factors = engine.factors(weighting)
        full = engine.lsi(weighting, factors.rank)
        check(f"{weighting}: full rank rebuilds A",
              np.allclose(full.reconstruct(), engine.matrix(weighting).values))
        check(f"{weighting}: U has orthonormal columns",
              np.allclose(factors.u.T @ factors.u, np.eye(factors.u.shape[1])))
        check(f"{weighting}: singular values are sorted", (np.diff(factors.sigma) <= 0).all())
    errors = [np.linalg.norm(matrix.values - engine.lsi(k=k).reconstruct()) for k in (2, 15, 30)]
    check("more concepts rebuild A more closely", errors == sorted(errors, reverse=True))

    print("\n== the two models ==")
    scores_ok, order_ok, shared_ok = True, True, True
    for model in MODELS:
        for query in ("wheezing", "chest pain", "kidney", "blood sugar and weight"):
            hits = engine.search(query, Settings(model=model, top=30)).hits
            scores_ok &= all(0 < h.score <= 1 for h in hits)
            keys = [(-h.score, h.doc_id) for h in hits]
            order_ok &= keys == sorted(keys)
            if model == "gvsm":
                shared_ok &= all(h.shared_terms for h in hits)
    check("every returned score is above 0 and at most 1", scores_ok)
    check("results are ranked best first", order_ok)
    check("GVSM only returns articles with a shared word", shared_ok)

    gvsm = GvsmModel(matrix)
    full = engine.lsi(k=30)
    check("full-rank LSI orders articles exactly as GVSM",
          all([d for d, _ in full.rank(t)] == [d for d, _ in gvsm.rank(t)]
              for t in (["wheez"], ["chest", "pain"], ["blood", "sugar"])))

    copd = next(h for h in engine.search("wheezing").hits if h.doc_id == "m10")
    check("LSI reaches COPD for 'wheezing' without the word", not copd.shared_terms)
    space = space_exploration()
    check("the slide's example: GVSM misses D3 and D4",
          space["D3"]["gvsm"] == 0 and space["D4"]["gvsm"] == 0)
    check("the slide's example: LSI reaches D3 and D4, and not the food documents",
          space["D3"]["lsi"] > 0.9 and space["D4"]["lsi"] > 0.9 and space["D5"]["lsi"] == 0)

    print("\n== the error paths ==")
    check("a query of stop words is refused", not engine.search("the of").ok)
    check("a word outside the collection is refused", not engine.search("zzzqqq").ok)
    check("a k outside the rank is refused", not engine.search("wheezing", Settings(k=99)).ok)

    print("\n== measured comparison ==")
    check("LSI at the default k beats GVSM",
          evaluate(Settings("lsi")).mean_average_precision
          > evaluate(Settings("gvsm")).mean_average_precision)

    print("\n== documentation ==")
    check("doc line references", check_doc_lines.main() == 0)
    check("doc numbers", check_doc_claims.main() == 0)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
