import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def settlements_path() -> Path:
    files = sorted(RAW_DIR.glob("settlements_*.csv"))
    if not files:
        pytest.skip("no settlements_*.csv in data/raw (run scripts/fetch_di1.py)")
    return files[-1]


@pytest.fixture(scope="session")
def settlements_fixture(settlements_path):
    from yieldcurve.bootstrap import load_settlements

    return load_settlements(settlements_path)
