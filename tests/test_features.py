import numpy as np
import pandas as pd

from mlalpha import features


def test_panel_shape_and_ranges(toy_market):
    panel = features.build_panel(*toy_market)
    assert set(features.FEATURES) <= set(panel.columns)
    ranked = panel[features.FEATURES + ["target"]]
    assert ranked.max().max() <= 0.5 and ranked.min().min() >= -0.5
    assert panel.index.names == ["date", "ticker"]


def test_features_do_not_look_ahead(toy_market):
    close, volume, mkt = toy_market
    cut = 700
    base = features.build_panel(close, volume, mkt)
    c2, v2 = close.copy(), volume.copy()
    rng = np.random.default_rng(1)
    c2.iloc[cut:] *= rng.uniform(0.7, 1.3, c2.iloc[cut:].shape)
    v2.iloc[cut:] *= 5
    tampered = features.build_panel(c2, v2, c2.mean(axis=1).where(c2.index >= close.index[cut], mkt))
    early = base.index.get_level_values("date") < close.index[cut - 30]
    pd.testing.assert_frame_equal(base.loc[early, features.FEATURES],
                                  tampered.loc[tampered.index.isin(base.index[early]), features.FEATURES])


def test_target_is_forward_return(toy_market):
    close, volume, mkt = toy_market
    panel = features.build_panel(close, volume, mkt, horizon=21)
    d, t = panel.index[0]
    i = close.index.get_loc(d)
    raw = close[t].iloc[i + 21] / close[t].iloc[i] - 1
    cs_mean = (close.iloc[i + 21] / close.iloc[i] - 1).mean()
    assert np.isclose(panel.loc[(d, t), "fwd_ret"], raw - cs_mean)


def test_weekly_dates_are_last_day_of_week():
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    w = features.weekly_dates(idx)
    assert all(d.dayofweek == 4 or d == idx[-1] for d in w)
