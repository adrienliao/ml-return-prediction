import numpy as np
import pandas as pd
import pytest

from mlalpha import cv


@pytest.fixture
def index():
    dates = pd.date_range("2010-01-01", periods=400, freq="W-FRI")
    return pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "ticker"])


def test_walk_forward_trains_only_on_the_past_with_a_gap(index):
    d = index.get_level_values("date").values
    uniq = np.unique(d)
    for train, test in cv.WalkForward(test_weeks=52, min_train_weeks=200, purge_weeks=5).split(index):
        gap = np.searchsorted(uniq, d[test].min()) - np.searchsorted(uniq, d[train].max())
        assert gap > 5


def test_walk_forward_covers_everything_after_warmup(index):
    tests = np.concatenate([t for _, t in cv.WalkForward(52, 200, 5).split(index)])
    assert len(np.unique(tests)) == len(index) - 200 * 3


def test_purged_kfold_leaves_a_gap_on_both_sides(index):
    d = index.get_level_values("date").values
    uniq = np.unique(d)
    for train, test in cv.PurgedKFold(5, purge_weeks=4).split(index):
        lo, hi = np.searchsorted(uniq, d[test].min()), np.searchsorted(uniq, d[test].max())
        pos = np.searchsorted(uniq, d[train])
        assert not np.any((pos >= lo - 4) & (pos <= hi + 4))


def test_folds_are_disjoint(index):
    for splitter in (cv.PurgedKFold(), cv.ShuffledKFold(), cv.WalkForward(52, 200, 5)):
        for train, test in splitter.split(index):
            assert np.intersect1d(train, test).size == 0
