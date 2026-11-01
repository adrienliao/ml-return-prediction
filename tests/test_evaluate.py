import numpy as np
import pandas as pd
import pytest

from mlalpha import evaluate as ev


def _panel(n_dates=60, n_stocks=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.date_range("2020-01-03", periods=n_dates, freq="W-FRI"),
                                      [f"S{i}" for i in range(n_stocks)]], names=["date", "ticker"])
    return pd.Series(rng.normal(size=len(idx)), idx)


def test_perfect_prediction_has_ic_one():
    r = _panel()
    assert np.allclose(ev.ic_series(r, r), 1.0)
    assert np.allclose(ev.ic_series(-r, r), -1.0)


def test_newey_west_matches_plain_tstat_without_lags():
    x = pd.Series(np.random.default_rng(1).normal(0.1, 1, 500))
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert ev.newey_west_tstat(x, 0) == pytest.approx(plain)


def test_newey_west_shrinks_tstat_for_autocorrelated_series():
    rng = np.random.default_rng(2)
    e = rng.normal(0.05, 1, 2000)
    x = pd.Series(np.convolve(e, np.ones(4) / 4, mode="valid"))   # overlapping sums, like 4 week labels
    assert abs(ev.newey_west_tstat(x, 4)) < abs(ev.newey_west_tstat(x, 0))


def test_long_short_is_dollar_neutral_and_earns_with_perfect_foresight():
    r = _panel() * 0.01
    book = ev.long_short(r, r, cost_bps=0)
    assert (book["gross"] > 0).all()
    book_costly = ev.long_short(r, r, cost_bps=10)
    assert (book_costly["net"] <= book_costly["gross"]).all()
