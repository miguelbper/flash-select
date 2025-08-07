import logging
from typing import Any

import numpy as np
import openml
import pandas as pd
import polars as pl
from codetiming import Timer
from numpy.typing import NDArray
from shap_select import shap_select
from xgboost import XGBRegressor

from flash_select.flash_select import FEATURE_NAME, SELECTED, T_VALUE, flash_select

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def shap_select_(
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


def get_model(X: NDArray, y: NDArray) -> XGBRegressor:
    model = XGBRegressor(n_estimators=100, verbosity=0, seed=42, nthread=1)
    model.fit(X, y)

    n = X.shape[1]
    n_used = len(model.get_booster().get_score())
    log.info(f"Model used {n_used}/{n} features")

    return model


def benchmark_task(task_id):
    log.info(f"Downloading task {task_id}")
    task = openml.tasks.get_task(task_id)

    log.info("Downloading dataset")
    dataset = task.get_dataset()

    log.info("Downloading data")
    X, y, *_ = dataset.get_data(target=task.target_name)

    X = pl.from_pandas(X)
    y = pl.from_pandas(y)

    categorical_cols = X.select(pl.col(pl.Categorical)).columns
    X = X.to_dummies(columns=categorical_cols)

    features = X.columns
    m, n = X.shape
    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)

    splits = task.get_train_test_split_indices()

    train_split_idxs = splits[0]
    X_train = X[train_split_idxs]
    y_train = y[train_split_idxs]
    m_train = len(X_train)

    test_split_idxs = splits[1]
    X_test = X[test_split_idxs]
    m_test = len(X_test)

    log.info(f"Dataset: {dataset.name}, Shape: {m}x{n}, Train size: {m_train}, Test size: {m_test}")

    log.info("Training model")
    model = get_model(X_train, y_train)

    t_flash = Timer(initial_text="Running flash-select")
    with t_flash:
        df_flash = flash_select(model, X_train, y_train, features)
    time_flash = t_flash.last

    # t_shap = Timer(initial_text="Running shap-select with alpha=0.0")
    # with t_shap:
    #     df_shap = shap_select_(model, X_train, y_train, features, alpha=0.0)
    # time_shap = t_shap.last

    t_shap_alpha = Timer(initial_text="Running shap-select with alpha=1e-6")
    with t_shap_alpha:
        df_shap_alpha = shap_select_(model, X_train, y_train, features, alpha=1e-6)
    time_shap_alpha = t_shap_alpha.last

    # speedup_shap = time_shap / time_flash
    speedup_shap_alpha = time_shap_alpha / time_flash
    # same_features_shap = df_flash[SELECTED].equals(df_shap[SELECTED])
    same_features_shap_alpha = df_flash[SELECTED].equals(df_shap_alpha[SELECTED])

    df_time = pl.DataFrame(
        {
            "dataset": dataset.name,
            "m": m,
            "m_train": m_train,
            "m_test": m_test,
            "n": n,
            "time_flash": time_flash,
            # "time_shap": time_shap,
            "time_shap": time_shap_alpha,
            # "speedup_shap": speedup_shap,
            "speedup": speedup_shap_alpha,
            # "same_features_shap": same_features_shap,
            "same_output": same_features_shap_alpha,
        }
    )

    return df_time


def benchmark_suite(suite_id):
    log.info(f"Downloading suite {suite_id}")
    suite = openml.study.get_suite(suite_id)

    dfs = []

    for task_id in suite.tasks:
        log.info(f"Benchmarking task {task_id}")
        df_time = benchmark_task(task_id)
        dfs.append(df_time)

    df = pl.concat(dfs)
    print(df)
    df.write_csv("benchmark.csv")


def main():
    benchmark_suite(353)


if __name__ == "__main__":
    main()
