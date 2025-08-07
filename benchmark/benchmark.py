import logging
from time import time
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray
from shap_select import shap_select
from tqdm import tqdm
from xgboost import XGBRegressor

from flash_select.flash_select import FEATURE_NAME, SELECTED, T_VALUE, flash_select

log = logging.getLogger(__name__)
rng = np.random.default_rng(42)


def get_y(X: NDArray) -> NDArray:
    m, n = X.shape
    w = np.zeros(n)
    w[: (n // 2)] = 1
    y = np.sum(X * w, axis=1) + rng.normal(size=m, scale=10)
    return y


def get_model(m: int, n: int) -> XGBRegressor:
    X_train = rng.normal(size=(m, n))
    y_train = get_y(X_train)

    model = XGBRegressor(
        n_estimators=10 * n,
        learning_rate=0.1,
        max_depth=4,
        max_leaves=2**4,
        colsample_bytree=0.1,
        colsample_bylevel=0.1,
        colsample_bynode=0.1,
    )
    model.fit(X_train, y_train)

    return model


def shap_select_regression(
    tree_model: Any,
    X: NDArray,
    y: NDArray,
    features: list[str],
    threshold: float = 0.05,
    alpha: float = 1e-6,
) -> pl.DataFrame:
    X_df = pd.DataFrame(X, columns=features)
    y_df = pd.Series(y, name="target")
    df = shap_select(tree_model, X_df, y_df, task="regression", threshold=threshold, alpha=alpha)
    df = pl.from_pandas(df, nan_to_null=False).sort([T_VALUE, FEATURE_NAME], descending=[True, False])
    return df


def benchmark(m: int, n: int, alpha: float) -> None:
    tree_model = get_model(m, n)

    X = rng.normal(size=(m, n))
    y = get_y(X)
    features = [f"feature_{i}" for i in range(n)]

    t0 = time()
    df_flash = flash_select(tree_model, X, y, features)
    t_flash = time() - t0

    t0 = time()
    df_shap = shap_select_regression(tree_model, X, y, features, alpha=alpha)
    t_shap = time() - t0

    speedup = t_shap / t_flash
    n_xgboost = len(tree_model.get_booster().get_score().keys())
    n_flash = (df_flash[SELECTED] == 1).sum()
    equal_selected = df_flash[SELECTED].equals(df_shap[SELECTED])

    df = pl.DataFrame(
        {
            "m": m,
            "n": n,
            "time flash": t_flash,
            "time shap": t_shap,
            "speedup": speedup,
            "n_xgboost": n_xgboost,
            "n_flash": n_flash,
            "equal_selected": equal_selected,
        }
    )

    return df


def main() -> None:
    dfs = []

    for n in tqdm(range(50, 500 + 1, 50)):
        m = 100 * n
        df = benchmark(m, n, alpha=1e-6)
        dfs.append(df)

    df = pl.concat(dfs)
    print(df)


if __name__ == "__main__":
    main()
