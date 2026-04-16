"""
Model training and hyperparameter tuning.

Tuning strategy (efficient single-phase):
  RandomizedSearchCV over a parameter grid.
  20 iterations × 3-fold CV = 60 model fits — fast and sufficient for this
  dataset. Covers a wide region of the hyperparameter space cheaply.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)

from california_housing.config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def train_baseline(X_train, y_train, random_state: int = RANDOM_STATE):
    """
    Train a default Random Forest — establishes the performance floor.
    n_estimators=100 is sklearn's default and sufficient for a baseline.
    oob_score gives a free internal validation estimate.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)
    print(f"  Baseline OOB R²: {model.oob_score_:.4f}")
    return model


def cross_validate_model(model, X_train, y_train, cv: int = 3) -> dict:
    """
    Cross-validate a model on the training set.
    Returns mean and std across folds for R², MAE, and RMSE.
    """
    r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    mae_scores = -cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
    )
    rmse_scores = np.sqrt(
        -cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
        )
    )
    return {
        "cv_r2_mean" : r2_scores.mean(),
        "cv_r2_std"  : r2_scores.std(),
        "cv_mae_mean": mae_scores.mean(),
        "cv_rmse_mean": rmse_scores.mean(),
    }


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_random_search(
    X_train, y_train,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = RANDOM_STATE,
):
    """
    RandomizedSearchCV over a wide parameter grid.

    n_iter=20, cv=3 → 60 total model fits; fast yet effective.
    """
    param_distributions = {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [None, 10, 15, 20, 25],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "max_features"     : ["sqrt", "log2"],
    }

    base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rs = RandomizedSearchCV(
        base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    rs.fit(X_train, y_train)

    print(f"  Best CV R²   : {rs.best_score_:.4f}")
    print(f"  Best params  : {rs.best_params_}")
    return rs


def tune_grid_search(X_train, y_train, best_params: dict, random_state: int = RANDOM_STATE):
    """
    Focused GridSearchCV ±1 step around best_params from a prior random search.
    Use when you want to fine-tune after tune_random_search.
    """
    n_est = best_params["n_estimators"]
    n_est_grid = sorted({max(50, n_est - 50), n_est, n_est + 50})

    md = best_params.get("max_depth")
    md_grid = [None, 25, 30] if md is None else sorted({max(3, md - 5), md, md + 5})

    mss = best_params["min_samples_split"]
    mss_grid = sorted({max(2, mss - 3), mss, mss + 3})

    msl = best_params["min_samples_leaf"]
    msl_grid = sorted({max(1, msl - 1), msl, msl + 1})

    param_grid = {
        "n_estimators"     : n_est_grid,
        "max_depth"        : md_grid,
        "min_samples_split": mss_grid,
        "min_samples_leaf" : msl_grid,
        "max_features"     : [best_params["max_features"]],
    }

    base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    gs = GridSearchCV(base, param_grid=param_grid, cv=3, scoring="r2",
                      n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    print(f"  Best CV R² (grid search): {gs.best_score_:.4f}")
    print(f"  Best params: {gs.best_params_}")
    return gs


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)
