from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def load(market: str = "SPY"):
    """Daily adjusted closes and volumes for 86 US large caps plus the market ETF, 2004 to 2025.

    The universe is today's large caps that already traded in 2005, so it carries
    survivorship bias: none of the names went bankrupt in the sample.
    """
    close = pd.read_csv(DATA / "close.csv.gz", index_col="date", parse_dates=True)
    volume = pd.read_csv(DATA / "volume.csv.gz", index_col="date", parse_dates=True)
    mkt = close.pop(market)
    volume = volume.drop(columns=[market])
    return close, volume, mkt
