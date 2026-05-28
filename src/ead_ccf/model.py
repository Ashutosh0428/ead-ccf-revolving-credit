"""XGBoost CCF model training — mean regression and P90 quantile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from ead_ccf.data import FEATURES

MONOTONE_CONSTRAINTS = {
    "current_utilisation": 1,
    "payment_ratio_6m": -1,
}


def _build_monotone_tuple(features: list[str]) -> tuple[int, ...]:
    """Convert feature-name constraint dict to positional tuple for XGBoost."""
    return tuple(MONOTONE_CONSTRAINTS.get(f, 0) for f in features)


def train_ccf_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    val_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train mean and P90 quantile XGBoost regressors.

    Returns dict with keys: ``ccf_mean_model``, ``ccf_p90_model``,
    ``X_val``, ``y_val``, ``eval_results``.
    """
    if feature_names is None:
        feature_names = FEATURES

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed
    )

    monotone = _build_monotone_tuple(feature_names)

    # --- Mean CCF model (squared-error) ---
    ccf_mean = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        monotone_constraints=monotone,
        reg_lambda=2.0,
        random_state=seed,
    )
    ccf_mean.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # --- P90 quantile model (upper-bound estimate) ---
    ccf_p90 = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.90,
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        random_state=seed,
    )
    ccf_p90.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return {
        "ccf_mean_model": ccf_mean,
        "ccf_p90_model": ccf_p90,
        "X_val": X_val,
        "y_val": y_val,
    }


def save_models(models: dict[str, Any], output_dir: Path) -> None:
    """Persist trained models to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["ccf_mean_model"], output_dir / "ccf_mean_model.joblib")
    joblib.dump(models["ccf_p90_model"], output_dir / "ccf_p90_model.joblib")


def load_models(model_dir: Path) -> dict[str, xgb.XGBRegressor]:
    """Load persisted models."""
    return {
        "ccf_mean_model": joblib.load(model_dir / "ccf_mean_model.joblib"),
        "ccf_p90_model": joblib.load(model_dir / "ccf_p90_model.joblib"),
    }
