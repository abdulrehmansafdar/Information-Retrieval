"""Shared fixtures.

week07 uses flat imports (`import lsi_model`), not a package, so the folder has to be
on sys.path before any module in it can be imported. pytest is run from the repository
root:

    .venv\\Scripts\\python.exe -m pytest

Every module the tests import has a unique name (clinical_docs, term_matrix,
vector_cosine, gvsm_model, lsi_model, semantic_search, semantic_eval, slide_examples),
so running next to the other weeks in one pytest process cannot shadow their modules.
The engine is built once per session and treated as read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEEK07 = Path(__file__).resolve().parent.parent
if str(WEEK07) not in sys.path:
    sys.path.insert(0, str(WEEK07))

import pytest


@pytest.fixture(scope="session")
def semantic_engine():
    """The one built search engine every test shares."""
    from semantic_search import get_engine
    return get_engine()
