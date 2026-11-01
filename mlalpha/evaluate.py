"""Out of sample predictions and the statistics that judge them."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import clone

from .features import FEATURES


def oos_predictions(panel: pd.DataFrame, model, splitter, features=FEATURES) -> pd.Series:
    preds = pd.Series(np.nan, index=panel.index)
    X, y = panel[features].values, panel["target"].values
    for train, test in splitter.split(panel.index):
        m = clone(model).fit(X[train], y[train])
        preds.iloc[test] = m.predict(X[test])
    return preds.dropna()


def ic_series(pred: pd.Series, realised: pd.Series) -> pd.Series:
    """Spearman rank correlation between prediction and realised return, date by date."""
    df = pd.DataFrame({"p": pred, "r": realised}).dropna()
    return df.groupby(level="date").apply(lambda g: g["p"].rank().corr(g["r"].rank()))


def newey_west_tstat(x: pd.Series, lags: int) -> float:
    """t-stat of the mean, robust to the autocorrelation created by overlapping labels."""
    x = np.asarray(x.dropna(), dtype=float)
    n = len(x)
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(var / n))


def long_short(pred: pd.Series, hold_ret: pd.Series, quantile: float = 0.2, cost_bps: float = 5.0) -> pd.DataFrame:
    """Equal weight long top quintile, short bottom quintile, rebalanced weekly.

    Costs are charged on the change in weights, at `cost_bps` per unit traded.
    """
    df = pd.DataFrame({"p": pred, "r": hold_ret}).dropna()

    def weights(g):
        q = g["p"].rank(pct=True)
        long, short = q > 1 - quantile, q <= quantile
        return long / long.sum() - short / short.sum()

    w = df.groupby(level="date", group_keys=False).apply(weights).unstack().fillna(0.0)
    r = df["r"].unstack().reindex_like(w).fillna(0.0)
    gross = (w * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover})


def summary(ic: pd.Series, ls: pd.DataFrame | None = None, nw_lags: int = 4) -> dict:
    out = {
        "mean_ic": float(ic.mean()),
        "ic_ir": float(ic.mean() / ic.std() * math.sqrt(52)),
        "ic_tstat_nw": newey_west_tstat(ic, nw_lags),
        "hit_rate": float((ic > 0).mean()),
    }
    if ls is not None:
        for k in ("gross", "net"):
            out[f"sharpe_{k}"] = float(ls[k].mean() / ls[k].std() * math.sqrt(52))
        out["ann_turnover"] = float(ls["turnover"].mean() * 52)
    return out
