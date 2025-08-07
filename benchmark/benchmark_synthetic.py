import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from codetiming import Timer
from hydra_zen import store, zen
from numpy.typing import NDArray
from shap_select import shap_select
from xgboost import XGBRegressor

from flash_select.flash_select import FEATURE_NAME, T_VALUE, flash_select

log = logging.getLogger(__name__)


def get_model(n: int) -> XGBRegressor:
    rng = np.random.default_rng(42)
    m = 10 * n
    X_train = rng.normal(size=(m, n))
    y_train = rng.normal(size=m)

    model = XGBRegressor(
        n_estimators=10 * n,
        max_depth=10,
        max_leaves=2**10,
        colsample_bytree=0.1,
    )
    model.fit(X_train, y_train)

    num_used = len(model.get_booster().get_score().keys())
    log.info(f"XGBRegressor uses {num_used}/{n} features")

    return model


def shap_select_wrapper(
    tree_model: Any,
    X: NDArray,
    y: NDArray,
    features: list[str],
    threshold: float = 0.05,
) -> pl.DataFrame:
    X_df = pd.DataFrame(X, columns=features)
    y_df = pd.Series(y, name="target")
    df = shap_select(tree_model, X_df, y_df, task="regression", threshold=threshold, alpha=0.0)
    df = pl.from_pandas(df, nan_to_null=False).sort([T_VALUE, FEATURE_NAME], descending=[True, False])
    return df


def benchmark(n: int) -> None:
    tree_model = get_model(n)

    rng = np.random.default_rng(314159)
    m = 10 * n
    X = rng.normal(size=(m, n))
    y = rng.normal(size=m)
    features = [f"feature_{i}" for i in range(n)]

    t = Timer(initial_text="Running flash_select")
    with t:
        df_flash = flash_select(tree_model, X, y, features)
    t_flash_numpy = t.last

    t = Timer(initial_text="Running shap_select")
    with t:
        df_shap = shap_select_wrapper(tree_model, X, y, features)
    t_shap_select = t.last

    speedup = t_shap_select / t_flash_numpy

    log.info(f"flash_select: {t_flash_numpy:8.3f} s")
    log.info(f"shap_select:  {t_shap_select:8.3f} s")
    log.info(f"speedup:      {speedup:8.3f}x")
    log.info(df_flash)
    log.info(df_shap)


def main() -> None:
    store(benchmark, name="config")
    store.add_to_hydra_store()
    zen(benchmark).hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
