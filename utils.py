import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ==== Models Functions ====


# Function to train and evaluate a model using TimeSeriesSplit
def train_and_evaluate_model(
    model,
    X,
    y,
    date,
    n_splits=5,
    training_years=3,
    pca_warmup=False,
    target_transform=None,
):
    tscv = TimeSeriesSplit(
        n_splits=n_splits, max_train_size=365 * training_years, test_size=365
    )

    # If target_transform is provided, apply it to y and remember inverse
    inverse_transform = None
    if target_transform:
        y = target_transform(y)
        if target_transform == np.log:
            inverse_transform = np.exp
        elif target_transform == np.log1p:
            inverse_transform = np.expm1

    results = {
        "metrics": {"rmse": [], "r2": []},
        "train": {"X": [], "y": [], "date": []},
        "test": {"X": [], "y": [], "y_pred": [], "date": []},
    }
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        date_train, date_val = date.iloc[train_index], date.iloc[val_index]

        # Fit PCA on training slice only and transform if enabled (avoids leakage).
        if pca_warmup:
            pca = PCA()
            pca.fit(X_train)
            X_train_fold = pca.transform(X_train)
            X_val_fold = pca.transform(X_val)
        else:
            X_train_fold = X_train
            X_val_fold = X_val

        model.fit(X_train_fold, y_train)
        val_pred = model.predict(X_val_fold)

        # Invert transform if needed
        if inverse_transform is not None:
            val_pred = inverse_transform(val_pred)
            y_val = inverse_transform(y_val)

        mse = np.mean((y_val - val_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, val_pred)

        results["metrics"]["rmse"].append(rmse)
        results["metrics"]["r2"].append(r2)
        results["train"]["X"].append(X_train)
        results["train"]["y"].append(y_train)
        results["train"]["date"].append(date_train)
        results["test"]["X"].append(X_val)
        results["test"]["y"].append(y_val)
        results["test"]["date"].append(date_val)
        results["test"]["y_pred"].append(val_pred)
    return results


# ==== Plotting Functions ====


# Plot the prediction vs. true demand for every fold
def plot_pred_vs_true(predictions, title):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    for fold in range(len(predictions["y"])):
        y_true = predictions["y"][fold]
        y_pred = predictions["y_pred"][fold]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10, label=f"Fold {fold + 1}")
    limits = [0, 300]
    ax.plot(limits, limits, color="red", linestyle="--")

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("True Demand")
    ax.set_ylabel("Predicted Demand")
    ax.set_title(title)
    ax.legend()
    plt.show()
    plt.close(fig)
    return None


# Plot the errors for each fold as scatter points instead of bars
def plot_errors(metrics, title):
    x = np.arange(1, len(metrics["rmse"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    ax[0].scatter(x, metrics["rmse"], color="C1", s=50, alpha=0.8)
    ax[0].plot(x, metrics["rmse"], color="C1", alpha=0.4, linestyle="--")
    ax[0].set_xticks(x)
    ax[0].set_title("RMSE per Fold")
    ax[0].set_xlabel("Fold")
    ax[0].set_ylabel("RMSE")

    ax[1].scatter(x, metrics["r2"], color="C2", s=50, alpha=0.8)
    ax[1].plot(x, metrics["r2"], color="C2", alpha=0.4, linestyle="--")
    ax[1].set_xticks(x)
    ax[1].set_ylim(0, 1)
    ax[1].set_title("R² per Fold")
    ax[1].set_xlabel("Fold")
    ax[1].set_ylabel("R²")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close(fig)
    return None


def print_train_test_sizes(results):
    # Print the number of training points vs test points for each fold
    for name, res in results.items():
        print(f"=== {name} ===")
        for fold in range(len(res["train"]["X"])):
            n_train = res["train"]["X"][fold].shape[0]
            n_test = res["test"]["X"][fold].shape[0]
            date_range_training = f"{res['train']['date'][fold].min()} to {res['train']['date'][fold].max()}"
            date_range_testing = f"{res['test']['date'][fold].min()} to {res['test']['date'][fold].max()}"
            print(
                f"Fold {fold + 1}: Train points = {n_train}, Test points = {n_test}, Ratio = {n_test / n_train:.2%}"
            )
            print(
                f"\tTraining dates: {date_range_training}, Testing dates: {date_range_testing}"
            )
    return None
