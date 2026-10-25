"""Shared fixtures.

week06 uses flat imports (`import membership`), not a package, so the folder has to
be on sys.path before any module in it can be imported. pytest is run from the
repository root:

    .venv\\Scripts\\python.exe -m pytest

Every module week06 adds has a unique name (articles, membership, query_parser,
fuzzy_logic, fuzzy_search), so running next to week02 and week03 in one pytest
process cannot shadow their modules. The index and engine are built once per session
and treated as read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEEK06 = Path(__file__).resolve().parent.parent
if str(WEEK06) not in sys.path:
    sys.path.insert(0, str(WEEK06))

import pytest


@pytest.fixture(scope="session")
def articles():
    from articles import get_articles
    return get_articles()


@pytest.fixture(scope="session")
def engine():
    """The one built search engine every test shares."""
    from fuzzy_search import get_engine
    return get_engine()


@pytest.fixture(scope="session")
def index(engine):
    return engine.index
