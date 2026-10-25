"""The week 6 section of GUIDE.md must stay true.

week02's tests/test_guide.py owns the repository-wide pytest count. This file owns
the week 6 claims: that the guide describes the module, names the shared corpus,
and points only at files that exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "GUIDE.md"
WEEK06 = REPO / "week06"


@pytest.fixture(scope="module")
def guide() -> str:
    assert GUIDE.exists(), "GUIDE.md is missing from the repository root"
    return GUIDE.read_text(encoding="utf-8")


class TestItCoversWeek06:
    def test_the_guide_mentions_week06(self, guide):
        assert "week06" in guide

    def test_it_names_the_shared_corpus(self, guide):
        assert "corpus/" in guide

    def test_it_names_the_three_models(self, guide):
        lowered = guide.lower()
        for model in ("boolean", "fuzzy", "extended boolean"):
            assert model in lowered, f"GUIDE.md never mentions {model}"

    def test_it_names_the_membership_functions(self, guide):
        lowered = guide.lower()
        for function in ("binary", "linear", "log"):
            assert function in lowered, f"GUIDE.md never mentions {function}"


class TestItsFileReferencesResolve:
    def test_every_week06_file_it_names_exists(self, guide):
        named = set(re.findall(r"week06[\\/][\w./\\-]+\.py", guide))
        assert named, "GUIDE.md names no week06 files"
        dead = [n for n in named if not (REPO / n.replace("\\", "/")).exists()]
        assert dead == [], f"GUIDE.md names week06 files that do not exist: {dead}"

    def test_it_names_the_verification_commands(self, guide):
        for command in ("verify.py", "check_doc_lines.py",
                        "check_doc_claims.py", "apptest_smoke.py"):
            assert command in guide, f"GUIDE.md never names {command}"


class TestTheModuleIsPresent:
    def test_the_expected_files_exist(self):
        for name in ("articles.py", "membership.py", "query_parser.py",
                     "fuzzy_logic.py", "fuzzy_search.py", "console_ui.py",
                     "app.py", "main.py", "verify.py",
                     "docs/understanding/fuzzy-model.md",
                     "docs/understanding/membership-functions.md",
                     "docs/understanding/partial-matching.md",
                     "docs/understanding/extended-boolean.md"):
            assert (WEEK06 / name).exists(), f"week06/{name} is missing"

    def test_the_shared_corpus_is_present(self):
        assert (REPO / "corpus" / "README.md").exists()
        assert len(list((REPO / "corpus" / "articles").glob("m*.txt"))) == 30
