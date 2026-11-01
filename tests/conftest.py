import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def toy_market():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=900)
    n = 12
    r = rng.normal(0, 0.015, (len(idx), n))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"S{i}" for i in range(n)])
    volume = pd.DataFrame(rng.integers(1_000, 10_000, (len(idx), n)).astype(float), index=idx, columns=close.columns)
    mkt = close.mean(axis=1)
    return close, volume, mkt
