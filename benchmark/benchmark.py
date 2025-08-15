from time import time
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shap_select import shap_select
from xgboost import XGBRegressor

from flash_select.flash_select import FEATURE_NAME, SELECTED, T_VALUE, flash_select

rng = np.random.default_rng(42)


def get_y(X: NDArray) -> NDArray:
    m, n = X.shape
    w = np.zeros(n)
    w[: (n // 2)] = 1
    y = np.sum(X * w, axis=1) + rng.normal(size=m, scale=10)
    return y


def get_model(m: int, n: int, n_estimators: int = 100) -> XGBRegressor:
    X_train = rng.normal(size=(m, n))
    y_train = get_y(X_train)

    model = XGBRegressor(n_estimators=n_estimators)
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


def benchmark(
    m_train: int,
    m_val: int,
    n: int,
    n_estimators: int = 100,
    alpha: float = 1e-6,
    plot_results: bool = False,
) -> None:
    print(f"Fitting xgboost model with {m_train} samples, {n} features, and {n_estimators} trees")
    tree_model = get_model(m_train, n, n_estimators=n_estimators)

    feature_scores = tree_model.get_booster().get_score()
    num_unused_features = n - len(feature_scores)
    print(f"* Number of unused features: {num_unused_features}")

    print(f"Creating validation set with {m_val} samples and {n} features")
    X = rng.normal(size=(m_val, n))
    y = get_y(X)
    features = [f"feature_{i}" for i in range(n)]

    print("Running flash_select...")
    t0 = time()
    df_flash = flash_select(tree_model, X, y, features)
    t_flash = time() - t0
    print(f"* flash_select took {t_flash} seconds")

    print("Running shap_select...")
    t0 = time()
    df_shap = shap_select_regression(tree_model, X, y, features, alpha=alpha)
    t_shap = time() - t0
    print(f"* shap_select took {t_shap} seconds")

    speedup = t_shap / t_flash
    print(f"* Speedup: {speedup}")

    equal_selected = df_flash[SELECTED].equals(df_shap[SELECTED])
    print(f"* Same set of selected features? {'yes' if equal_selected else 'no'}")

    print(df_flash)
    print(df_shap)

    if plot_results:
        plot(m_val, n, t_flash, t_shap)


def plot(m_val: int, n: int, time_flash: float, time_shap: float) -> None:
    _, ax = plt.subplots(figsize=(10, 3))

    methods = ["shap-select", "flash-select"]
    times = [time_shap, time_flash]

    bars = ax.barh(methods, times, color="#a155e7", height=0.5)

    for b, t in zip(bars, times, strict=True):
        ax.text(
            t + max(times) * 0.01,
            b.get_y() + b.get_height() / 2,
            f"{t:.2f}s",
            va="center",
            ha="left",
            fontweight="bold",
        )

    ax.set_title(
        f"Time to run on a dataset with {m_val} examples and {n} features", fontweight="bold", fontsize=14, pad=20
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_xlim(0, max(times) * 1.1)

    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    ax.set_ylabel("")

    plt.tight_layout()

    filename = f"logs/benchmark_{m_val}_{n}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {filename}")

    plt.close()


@click.command()
@click.option("--m_train", default=1000, help="Number of training samples")
@click.option("--m_val", default=1000, help="Number of validation samples")
@click.option("--n", default=10, help="Number of features")
@click.option("--n_estimators", default=100, help="Number of estimators for xgboost")
@click.option("--alpha", default=1e-6, help="Alpha parameter for shap_select")
@click.option("--plot_results", is_flag=True, help="Plot the results")
def main(m_train: int, m_val: int, n: int, n_estimators: int, alpha: float, plot_results: bool) -> None:
    print("Running benchmark with parameters:")
    print(f"* m_train: {m_train}")
    print(f"* m_val: {m_val}")
    print(f"* n: {n}")
    print(f"* n_estimators: {n_estimators}")
    print(f"* alpha: {alpha:.2e}")

    benchmark(m_train, m_val, n, n_estimators, alpha, plot_results)


if __name__ == "__main__":
    main()
