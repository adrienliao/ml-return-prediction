"""Cross validation schemes for panels whose labels overlap in time.

The target at date t is the return from t to t + horizon. With weekly samples
and a 21 day horizon, consecutive labels share three weeks of returns. A plain
shuffled K-fold therefore puts in the training set labels that almost contain
the test labels, and flexible models learn to exploit it. Purging removes
those training samples (Lopez de Prado, Advances in Financial Machine Learning, ch. 7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


def _dates(panel_index: pd.MultiIndex) -> np.ndarray:
    return panel_index.get_level_values("date").values


@dataclass
class WalkForward:
    """Expanding window, refit once per test block, purged before each test block."""
    test_weeks: int = 52
    min_train_weeks: int = 260
    purge_weeks: int = 5

    def split(self, index: pd.MultiIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        d = _dates(index)
        uniq = np.unique(d)
        for start in range(self.min_train_weeks, len(uniq), self.test_weeks):
            test_dates = uniq[start:start + self.test_weeks]
            train_end = uniq[max(start - self.purge_weeks, 0)]
            yield np.where(d < train_end)[0], np.where(np.isin(d, test_dates))[0]


@dataclass
class PurgedKFold:
    """K contiguous blocks of dates. Training drops `purge_weeks` on each side of the test block."""
    n_splits: int = 5
    purge_weeks: int = 5

    def split(self, index):
        d = _dates(index)
        uniq = np.unique(d)
        for block in np.array_split(np.arange(len(uniq)), self.n_splits):
            lo, hi = block[0], block[-1]
            test_dates = uniq[lo:hi + 1]
            banned = uniq[max(lo - self.purge_weeks, 0):min(hi + self.purge_weeks + 1, len(uniq))]
            yield np.where(~np.isin(d, banned))[0], np.where(np.isin(d, test_dates))[0]


@dataclass
class ShuffledKFold:
    """What a generic ML tutorial would do. Here only to measure how much it leaks."""
    n_splits: int = 5
    seed: int = 0

    def split(self, index):
        idx = np.random.default_rng(self.seed).permutation(len(index))
        for test in np.array_split(idx, self.n_splits):
            yield np.setdiff1d(idx, test), np.sort(test)
