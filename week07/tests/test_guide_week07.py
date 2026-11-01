"""The week 7 section of GUIDE.md must stay true.

week02's tests/test_guide.py owns the repository-wide pytest count. This file owns the
week 7 claims: that the guide describes the module, and points only at files that exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "GUIDE.md"
WEEK07 = REPO / "week07"


@pytest.fixture(scope="module")
def guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


class TestItCoversWeek07:
    @pytest.mark.parametrize("word", ["week07", "GVSM", "LSI", "SVD", "cosine"])
    def test_the_guide_names_it(self, guide, word):
        assert word in guide


class TestItsFileReferencesResolve:
    def test_every_week07_file_it_names_exists(self, guide):
        named = set(re.findall(r"week07[\\/][\w./\\-]+\.py", guide))
        assert named, "GUIDE.md names no week07 files"
        dead = [n for n in named if not (REPO / n.replace("\\", "/")).exists()]
        assert dead == [], f"GUIDE.md names week07 files that do not exist: {dead}"


class TestTheModuleIsPresent:
    @pytest.mark.parametrize("name", [
        "clinical_docs.py", "term_matrix.py", "vector_cosine.py", "gvsm_model.py",
        "lsi_model.py", "semantic_search.py", "semantic_eval.py", "slide_examples.py",
        "console_ui.py", "app.py", "main.py", "verify.py", "apptest_smoke.py",
        "check_doc_lines.py", "check_doc_claims.py",
        "docs/understanding/README.md", "docs/understanding/gvsm-and-cosine.md",
        "docs/understanding/lsi-and-svd.md", "docs/understanding/semantic-matching.md",
    ])
    def test_the_file_exists(self, name):
        assert (WEEK07 / name).exists(), f"week07/{name} is missing"
