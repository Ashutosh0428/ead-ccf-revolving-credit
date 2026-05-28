"""Tests from LLD section 5 — CCF model invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ead_ccf.data import FEATURES, generate_synthetic_accounts
from ead_ccf.features import prepare_features
from ead_ccf.model import train_ccf_models
from ead_ccf.predict import predict_ead


@pytest.fixture(scope="module")
def trained_models():
    """Train models once for all tests in this module."""
    df = generate_synthetic_accounts(n=5_000, seed=99)
    X = prepare_features(df).values
    y = df["realised_ccf"].values
    result = train_ccf_models(X, y, feature_names=FEATURES)
    return result


@pytest.fixture()
def models(trained_models):
    return {
        "ccf_mean_model": trained_models["ccf_mean_model"],
        "ccf_p90_model": trained_models["ccf_p90_model"],
    }


class TestCCFInvariants:
    """LLD §5 — required test cases."""

    def test_fully_utilised_ead_equals_drawn(self, models):
        """When undrawn=0, EAD must equal drawn balance regardless of CCF."""
        account = pd.DataFrame(
            {
                "drawn": [50_000.0],
                "undrawn": [0.0],
                "limit": [50_000.0],
                "current_utilisation": [1.0],
                "utilisation_trend_3m": [0.0],
                "utilisation_trend_6m": [0.0],
                "payment_ratio_6m": [0.5],
                "months_on_book": [36.0],
                "recent_inquiries": [1.0],
                "unemployment_rate": [5.5],
            }
        )
        result = predict_ead(account, **models)
        np.testing.assert_allclose(result["ead_mean"].values, [50_000.0], atol=1.0)
        np.testing.assert_allclose(result["ead_p90"].values, [50_000.0], atol=1.0)

    def test_inactive_account_low_ccf(self, models):
        """Inactive account with zero utilisation trend → CCF < 0.1."""
        account = pd.DataFrame(
            {
                "drawn": [0.0],
                "undrawn": [50_000.0],
                "limit": [50_000.0],
                "current_utilisation": [0.0],
                "utilisation_trend_3m": [0.0],
                "utilisation_trend_6m": [0.0],
                "payment_ratio_6m": [0.9],
                "months_on_book": [60.0],
                "recent_inquiries": [0.0],
                "unemployment_rate": [4.0],
            }
        )
        result = predict_ead(account, **models)
        assert result["ccf_mean"].values[0] < 0.1

    def test_quantile_bound_gte_point_estimate(self, models):
        """P90 quantile bound ≥ mean estimate for all inputs."""
        df = generate_synthetic_accounts(n=500, seed=77)
        result = predict_ead(df, **models)
        violations = (result["ccf_p90"] < result["ccf_mean"] - 1e-6).sum()
        assert violations == 0, f"{violations} rows where P90 < mean"


class TestEvaluationMetrics:
    """PRD §4 — success metric targets."""

    def test_mae_below_threshold(self, trained_models):
        y_val = trained_models["y_val"]
        ccf_pred = np.clip(
            trained_models["ccf_mean_model"].predict(trained_models["X_val"]), 0, 1
        )
        mae = np.mean(np.abs(y_val - ccf_pred))
        assert mae < 0.15, f"MAE {mae:.4f} exceeds 0.15 threshold"

    def test_bias_within_bounds(self, trained_models):
        y_val = trained_models["y_val"]
        ccf_pred = np.clip(
            trained_models["ccf_mean_model"].predict(trained_models["X_val"]), 0, 1
        )
        bias = np.mean(ccf_pred - y_val)
        assert abs(bias) < 0.03, f"Bias {bias:+.4f} exceeds ±3pp"

    def test_coverage_in_range(self, trained_models):
        y_val = trained_models["y_val"]
        ccf_p90 = np.clip(
            trained_models["ccf_p90_model"].predict(trained_models["X_val"]), 0, 1
        )
        coverage = np.mean(y_val <= ccf_p90)
        assert 0.85 <= coverage <= 0.95, f"Coverage {coverage:.1%} outside [85%, 95%]"
