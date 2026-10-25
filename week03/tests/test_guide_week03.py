"""The Week 3 section of GUIDE.md must stay true.

week02's tests/test_guide.py owns the repository-wide pytest count. This file
owns the Week 3 claims: that the guide describes the module, names the seven
requirements the assignment asks for, and points only at files that exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "GUIDE.md"
WEEK03 = REPO / "week03"

REQUIREMENTS = ("home page", "category hierarchy", "document list",
                "search + browse", "breadcrumb", "related document",
                "document preview")


@pytest.fixture(scope="module")
def guide() -> str:
    assert GUIDE.exists(), "GUIDE.md is missing from the repository root"
    return GUIDE.read_text(encoding="utf-8")


class TestItCoversWeek03:
    def test_the_guide_mentions_week03(self, guide):
        assert "week03" in guide

    def test_it_names_the_week_three_module(self, guide):
        assert "Browsing" in guide or "browsing" in guide

    def test_it_names_all_seven_requirements(self, guide):
        lowered = guide.lower()
        for requirement in REQUIREMENTS:
            assert requirement in lowered, f"GUIDE.md never mentions {requirement}"

    def test_it_names_the_three_browsing_models(self, guide):
        lowered = guide.lower()
        for model in ("flat", "structured", "hypertext"):
            assert model in lowered, f"GUIDE.md never mentions {model} browsing"


class TestItsFileReferencesResolve:
    def test_every_week03_file_it_names_exists(self, guide):
        named = set(re.findall(r"week03[\\/][\w./\\-]+\.py", guide))
        assert named, "GUIDE.md names no week03 files"
        dead = [n for n in named if not (REPO / n.replace("\\", "/")).exists()]
        assert dead == [], f"GUIDE.md names week03 files that do not exist: {dead}"

    def test_it_names_the_verification_commands(self, guide):
        for command in ("verify.py", "check_doc_lines.py",
                        "check_doc_claims.py", "apptest_smoke.py"):
            assert command in guide, f"GUIDE.md never names {command}"


class TestTheModuleIsPresent:
    def test_the_expected_files_exist(self):
        for name in ("collection.py", "taxonomy.py", "hypertext.py",
                     "browsing.py", "console_ui.py", "app.py", "main.py",
                     "verify.py", "docs/understanding/browsing-models.md"):
            assert (WEEK03 / name).exists(), f"week03/{name} is missing"
