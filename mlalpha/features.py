"""Turn daily prices into a (date, ticker) panel of features and targets.

Every feature at date t only uses data up to the close of t. Targets are
forward returns starting at the close of t. Samples are taken once a week
(last trading day of each week).

Features are ranked cross sectionally and mapped to [-0.5, 0.5] on each date.
This removes market wide level shifts, handles outliers, and makes the
problem stationary enough for a model trained on 2008 to say something about 2020.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1w", "ret_1m", "ret_3m", "ret_6m", "mom_12_1",
    "vol_1m", "vol_3m", "beta_6m", "idio_vol_3m",
    "dist_52w_high", "max_ret_1m", "volume_trend", "amihud_illiq",
]


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True) - 0.5


def daily_features(close: pd.DataFrame, volume: pd.DataFrame, mkt: pd.Series) -> dict[str, pd.DataFrame]:
    r = close.pct_change(fill_method=None)
    m = mkt.pct_change(fill_method=None)
    f = {
        "ret_1w": close.pct_change(5, fill_method=None),
        "ret_1m": close.pct_change(21, fill_method=None),
        "ret_3m": close.pct_change(63, fill_method=None),
        "ret_6m": close.pct_change(126, fill_method=None),
        "mom_12_1": close.shift(21) / close.shift(252) - 1,
        "vol_1m": r.rolling(21).std(),
        "vol_3m": r.rolling(63).std(),
        "dist_52w_high": close / close.rolling(252).max() - 1,
        "max_ret_1m": r.rolling(21).max(),
        "volume_trend": np.log(volume.rolling(21).mean() / volume.rolling(252).mean()),
        "amihud_illiq": (r.abs() / (close * volume)).rolling(63).mean(),
    }
    cov = r.rolling(126).cov(m)
    beta = cov.div(m.rolling(126).var(), axis=0)
    f["beta_6m"] = beta
    resid = r - beta.shift(1).mul(m, axis=0)
    f["idio_vol_3m"] = resid.rolling(63).std()
    return f


def weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("W")).last().values)


def build_panel(close, volume, mkt, horizon: int = 21, hold: int = 5) -> pd.DataFrame:
    """Rows are (date, ticker). Columns are ranked features plus:

    target       ranked forward `horizon` day return, what the models learn
    fwd_ret      raw forward `horizon` day return in excess of the cross sectional mean
    hold_ret     raw forward `hold` day return, used to trade the signal weekly
    """
    feats = daily_features(close, volume, mkt)
    dates = weekly_dates(close.index)
    fwd = close.shift(-horizon) / close - 1
    hold_ret = close.shift(-hold) / close - 1

    cols = {name: _rank(df.loc[dates]).stack(future_stack=True) for name, df in feats.items()}
    fwd_w = fwd.loc[dates]
    cols["target"] = _rank(fwd_w).stack(future_stack=True)
    cols["fwd_ret"] = fwd_w.sub(fwd_w.mean(axis=1), axis=0).stack(future_stack=True)
    cols["hold_ret"] = hold_ret.loc[dates].stack(future_stack=True)
    panel = pd.DataFrame(cols)
    panel.index.names = ["date", "ticker"]
    # drop warm up period and the last weeks whose targets are not known yet
    return panel.dropna(subset=FEATURES + ["target"])
