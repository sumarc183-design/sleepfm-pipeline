import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Sur Mac, XGBoost et torch chargent chacun leur OpenMP : XGBoost plante ou se
# bloque dès que torch a calculé dans le même processus (ce que font d'autres
# tests). Ces tests tournent sur Linux (CI) ; run_baseline retire XGBoost sur Mac.
XGBOOST_UNSAFE = sys.platform == "darwin"


def pytest_configure(config):
    config.addinivalue_line("markers", "xgboost: entraîne XGBoost (sauté sur Mac, conflit OpenMP avec torch)")


def pytest_collection_modifyitems(config, items):
    if not XGBOOST_UNSAFE:
        return
    skip = pytest.mark.skip(reason="XGBoost sur Mac : conflit OpenMP avec torch (testé en CI Linux)")
    for item in items:
        if "xgboost" in item.keywords:
            item.add_marker(skip)
