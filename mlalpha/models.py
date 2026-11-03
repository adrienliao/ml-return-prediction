from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def make_models(seed: int = 0) -> dict:
    return {
        "ridge": Ridge(alpha=10.0),
        "gbm": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=500,
            l2_regularization=1.0, random_state=seed,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3, max_iter=60,
            early_stopping=True, validation_fraction=0.1, random_state=seed,
        ),
    }


# single factor baselines: no fitting, the sign is fixed by the literature
BASELINES = {
    "momentum_12_1": ("mom_12_1", +1.0),   # Jegadeesh and Titman (1993)
    "reversal_1m": ("ret_1m", -1.0),       # Jegadeesh (1990)
    "low_vol": ("vol_3m", -1.0),           # Ang, Hodrick, Xing, Zhang (2006)
}
