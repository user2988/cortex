"""
Stub out heavy optional dependencies before any test module imports them.
This lets the unit tests run without installing prophet, xgboost, etc.
"""
import sys
from unittest.mock import MagicMock

_STUBS = [
    "prophet",
    "plotly",
    "plotly.graph_objects",
    "streamlit",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
