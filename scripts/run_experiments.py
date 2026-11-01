"""Run every experiment in the README and regenerate results/ and docs/.

    python scripts/run_experiments.py      # about 5 minutes on a laptop
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlalpha import cv, data, evaluate as ev, features, models  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
BLUE, ORANGE, AQUA, GREY, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#6250d6"
COLORS = {"gbm": ORANGE, "ridge": BLUE, "mlp": VIOLET, "momentum_12_1": GREY}
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY, "legend.frameon": False,
})
DOCS, RES = ROOT / "docs", ROOT / "results"
SMOOTH_HALFLIFE = 4   # weeks; 2, 4 and 8 all give net Sharpe between 0.41 and 0.50
COST_BPS = 5.0


def smooth(pred: pd.Series, halflife: float) -> pd.Series:
    return pred.unstack().ewm(halflife=halflife).mean().stack()


def main():
    close, volume, mkt = data.load()
    panel = features.build_panel(close, volume, mkt)
    print(f"panel: {len(panel):,} rows, {panel.index.get_level_values('date').nunique()} weeks, "
          f"{panel.index.get_level_values('ticker').nunique()} stocks")
    results, preds = {}, {}

    # 1. walk-forward out of sample, models and baselines
    wf = cv.WalkForward()
    for name, model in models.make_models().items():
        preds[name] = ev.oos_predictions(panel, model, wf)
    oos_index = preds["gbm"].index
    for name, (col, sign) in models.BASELINES.items():
        preds[name] = sign * panel.loc[oos_index, col]
    preds["gbm_smoothed"] = smooth(preds["gbm"], SMOOTH_HALFLIFE)

    ics, books = {}, {}
    for name, p in preds.items():
        ics[name] = ev.ic_series(p, panel["fwd_ret"])
        books[name] = ev.long_short(p, panel["hold_ret"], cost_bps=COST_BPS)
        results[name] = ev.summary(ics[name], books[name])
        print(f"{name:15s} " + "  ".join(f"{k} {v:.3f}" for k, v in results[name].items()))

    # 2. how much does the validation scheme flatter the model?
    leak = {}
    for scheme, splitter in [("shuffled k-fold", cv.ShuffledKFold()), ("purged k-fold", cv.PurgedKFold()),
                             ("walk-forward", wf)]:
        for name in ("ridge", "gbm"):
            p = preds[name] if scheme == "walk-forward" else ev.oos_predictions(panel, models.make_models()[name], splitter)
            leak.setdefault(scheme, {})[name] = float(ev.ic_series(p, panel["fwd_ret"]).mean())
    results["validation_leakage"] = leak
    print("leakage:", leak)

    # 3. permutation importance on the last fold: shuffle one feature within each date
    imp = permutation_importance(panel, wf)
    results["permutation_importance"] = imp

    RES.mkdir(exist_ok=True)
    (RES / "results.json").write_text(json.dumps(results, indent=1))
    figures(ics, books, leak, imp)


def permutation_importance(panel, wf, n_repeats=5, seed=0):
    *_, (train, test) = list(wf.split(panel.index))
    X = panel[features.FEATURES]
    m = clone(models.make_models()["gbm"]).fit(X.iloc[train].values, panel["target"].iloc[train].values)
    test_df = panel.iloc[test]
    base = ev.ic_series(pd.Series(m.predict(test_df[features.FEATURES].values), test_df.index), test_df["fwd_ret"]).mean()
    rng = np.random.default_rng(seed)
    out = {}
    for f in features.FEATURES:
        drops = []
        for _ in range(n_repeats):
            shuffled = test_df[features.FEATURES].copy()
            shuffled[f] = shuffled[f].groupby(level="date").transform(lambda s: rng.permutation(s.values))
            ic = ev.ic_series(pd.Series(m.predict(shuffled.values), test_df.index), test_df["fwd_ret"]).mean()
            drops.append(base - ic)
        out[f] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def figures(ics, books, leak, imp):
    DOCS.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    schemes = list(leak)
    x = np.arange(len(schemes))
    for i, (name, c) in enumerate([("ridge", BLUE), ("gbm", ORANGE)]):
        vals = [leak[s][name] for s in schemes]
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=c, label=name)
        ax.bar_label(bars, fmt="%.3f", padding=2, color=GREY, fontsize=9)
    ax.set_xticks(x, schemes)
    ax.set_ylabel("mean information coefficient")
    ax.set_title("Shuffled CV makes gradient boosting look 4x better than it is", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "validation_leakage.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("gbm", "ridge", "mlp", "momentum_12_1"):
        ic = ics[name]
        ax.plot(ic.cumsum(), color=COLORS[name], lw=1.8, label=f"{name}  (mean IC {ic.mean():.3f})")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("cumulative IC")
    ax.set_title("Walk-forward out of sample, 2010 to 2025", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "cumulative_ic.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label, c, ls in [("gbm_smoothed", f"GBM, smoothed (halflife {SMOOTH_HALFLIFE} weeks)", ORANGE, "-"),
                               ("gbm", "GBM, raw weekly signal", ORANGE, ":"),
                               ("momentum_12_1", "12-1 momentum", GREY, "--")]:
        net = books[name]["net"]
        sr = net.mean() / net.std() * np.sqrt(52)
        ax.plot((1 + net).cumprod(), color=c, ls=ls, lw=1.8, label=f"{label}  (Sharpe {sr:.2f})")
    ax.set_ylabel("growth of 1, net of costs")
    ax.set_title(f"Quintile long-short, weekly rebalance, {COST_BPS:g} bp per unit traded", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "long_short.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(imp)[::-1]
    ax.barh(names, [imp[n] for n in names], color=BLUE)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("drop in mean IC when the feature is shuffled")
    ax.set_title("What the gradient boosting model relies on (last fold)", loc="left")
    fig.tight_layout()
    fig.savefig(DOCS / "feature_importance.png", dpi=160)


if __name__ == "__main__":
    main()
