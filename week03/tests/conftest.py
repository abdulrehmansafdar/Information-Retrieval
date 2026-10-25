"""Shared fixtures.

week03 uses flat imports (`import browsing as browse`), not a package, so the
folder has to be on sys.path before any module in it can be imported. pytest is
run from the repository root:

    .venv\\Scripts\\python.exe -m pytest

Every module week03 adds is uniquely named (collection, taxonomy, hypertext,
browsing) so that running next to week02 in one pytest process cannot shadow
week02's engine.py / preprocess.py. The engine, tree and graph are built once
per session and treated as read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEEK03 = Path(__file__).resolve().parent.parent
if str(WEEK03) not in sys.path:
    sys.path.insert(0, str(WEEK03))

import pytest


@pytest.fixture(scope="session")
def documents():
    from collection import get_documents
    return get_documents()


@pytest.fixture(scope="session")
def engine():
    """The one built browsing engine every test shares."""
    import browsing as browse
    return browse.get_engine()


@pytest.fixture(scope="session")
def taxonomy(engine):
    return engine.taxonomy


@pytest.fixture(scope="session")
def graph(engine):
    return engine.graph


@pytest.fixture()
def state():
    from browsing import NavigationState
    return NavigationState()
