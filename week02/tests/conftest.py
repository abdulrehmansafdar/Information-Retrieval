"""Shared fixtures.

week02 uses flat imports (`from ranking import Bm25Model`), not a package, so
the folder itself has to be on sys.path before any module in it can be
imported. pytest is normally run from the repository root:

    .venv\\Scripts\\python.exe -m pytest

The engine is built ONCE for the whole session. Building the corpus, the
positional index, the IDF tables and the per-field statistics takes about half
a second, and rebuilding that for every test function would dominate the run.
The models are treated as read-only, so sharing them is safe; the one exception
is `fresh_index`, for tests that deliberately need their own copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEEK02 = Path(__file__).resolve().parent.parent
if str(WEEK02) not in sys.path:
    sys.path.insert(0, str(WEEK02))

import pytest


@pytest.fixture(scope="session")
def index():
    from structured_index import build_index
    return build_index()


@pytest.fixture(scope="session")
def engine():
    """The one built engine every test shares."""
    import engine as ir
    return ir.get_engine()


@pytest.fixture(scope="session")
def idf_table(index):
    from ranking import IdfTable
    return IdfTable(index, "bm25")


@pytest.fixture()
def bm25(index):
    from ranking import Bm25Model
    return Bm25Model(index)


@pytest.fixture()
def bim(index):
    from ranking import BinaryIndependenceModel
    return BinaryIndependenceModel(index)


@pytest.fixture()
def list_models(index):
    from lists import ListModels
    return ListModels(index)


@pytest.fixture()
def proximal(index):
    from proximity import ProximalNodeModel
    return ProximalNodeModel(index)


@pytest.fixture()
def fields(index):
    from fields import FieldSearch
    return FieldSearch(index)


@pytest.fixture(scope="session")
def judgements(engine):
    return engine.judgements