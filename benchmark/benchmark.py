from time import time
from typing import Any

import click
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shap_select import shap_select
from xgboost import XGBRegressor

from flash_select.flash_select import FEATURE_NAME, SELECTED, T_VALUE, flash_select

rng = np.random.default_rng(42)
# TODO: see if this should be kept
M_KAGGLE = 284807
M_KAGGLE_TRAIN = int(0.6 * M_KAGGLE)
M_KAGGLE_VAL = int(0.2 * M_KAGGLE)
N_KAGGLE = 30


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
        n_estimators=100,
        verbosity=0,
        seed=42,
        nthread=1,
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
) -> pd.DataFrame:
    X_df = pd.DataFrame(X, columns=features)
    y_df = pd.Series(y, name="target")
    df = shap_select(tree_model, X_df, y_df, task="regression", threshold=threshold, alpha=alpha)
    df = df.sort_values([T_VALUE, FEATURE_NAME], ascending=[False, False])
    return df


def benchmark(m_train: int, m_val: int, n: int, alpha: float = 1e-6) -> pd.Series:
    print(f"Fitting xgboost model with {m_train} samples and {n} features")
    tree_model = get_model(m_train, n)

    print(f"Creating validation set with {m_val} samples and {n} features")
    X = rng.normal(size=(m_val, n))
    y = get_y(X)
    features = [f"feature_{i}" for i in range(n)]

    print("Running flash_select...")
    t0 = time()
    df_flash = flash_select(tree_model, X, y, features)
    t_flash = time() - t0
    print(f"flash_select took {t_flash} seconds")

    print("Running shap_select...")
    t0 = time()
    df_shap = shap_select_regression(tree_model, X, y, features, alpha=alpha)
    t_shap = time() - t0
    print(f"shap_select took {t_shap} seconds")

    speedup = t_shap / t_flash
    print(f"Speedup: {speedup}")

    equal_selected = df_flash[SELECTED].equals(df_shap[SELECTED])
    print(f"Same set of selected features? {'yes' if equal_selected else 'no'}")

    df = pd.Series(
        {
            "m_val": m_val,
            "n": n,
            "time flash": t_flash,
            "time shap": t_shap,
            "speedup": speedup,
            "equal_selected": equal_selected,
        }
    )

    return df


@click.command()
@click.option("--m_train", default=M_KAGGLE_TRAIN, help="Number of training samples")
@click.option("--m_val", default=M_KAGGLE_VAL, help="Number of validation samples")
@click.option("--n", default=N_KAGGLE, help="Number of features")
@click.option("--alpha", default=1e-6, help="Alpha parameter for shap_select")
def main(m_train: int, m_val: int, n: int, alpha: float) -> None:
    print("Running benchmark with parameters:")
    print(f"* m_train: {m_train}")
    print(f"* m_val: {m_val}")
    print(f"* n: {n}")
    print(f"* alpha: {alpha:.2e}")

    result = benchmark(m_train, m_val, n, alpha)
    print(result)


if __name__ == "__main__":
    main()
