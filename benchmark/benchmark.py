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


def get_model(n: int) -> XGBRegressor:
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(n, n))
    y_train = rng.normal(size=n)

    model = XGBRegressor(
        n_estimators=10 * n,
        max_depth=10,
        max_leaves=2**10,
        colsample_bytree=0.2,
    )
    model.fit(X_train, y_train)

    num_used = len(model.get_booster().get_score().keys())

    assert num_used == n, f"num_used = {num_used} < n = {n}"

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

    t = Timer(initial_text="Running flash_select with numpy")
    with t:
        df_flash_numpy = flash_select(tree_model, X, y, features)
    t_flash_numpy = t.last

    t = Timer(initial_text="Running flash_select with torch cpu")
    with t:
        df_flash_torch_cpu = flash_select(tree_model, X, y, features, backend="torch", device="cpu")
    t_flash_torch_cpu = t.last

    t = Timer(initial_text="Running flash_select with torch gpu")
    with t:
        df_flash_torch_cuda = flash_select(tree_model, X, y, features, backend="torch", device="cuda")
    t_flash_torch_cuda = t.last

    t = Timer(initial_text="Running shap_select")
    with t:
        if n < 100:
            df_shap_select = shap_select_wrapper(tree_model, X, y, features)
        else:
            df_shap_select = pl.DataFrame()
    t_shap_select = t.last

    print(f"flash_select with numpy:     {t_flash_numpy:8.3f} s")
    print(f"flash_select with torch cpu: {t_flash_torch_cpu:8.3f} s")
    print(f"flash_select with torch gpu: {t_flash_torch_cuda:8.3f} s")
    print(f"shap_select:                 {t_shap_select:8.3f} s")


def main() -> None:
    store(benchmark, name="config")
    store.add_to_hydra_store()
    zen(benchmark).hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
