"""EAD prediction using trained CCF models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ead_ccf.data import FEATURES

REGULATORY_CCF_FLOOR = 0.40


def predict_ead(
    account: pd.DataFrame,
    ccf_mean_model: xgb.XGBRegressor,
    ccf_p90_model: xgb.XGBRegressor,
) -> pd.DataFrame:
    """Predict EAD for one or more accounts.

    Returns DataFrame with columns:
    - ccf_mean, ccf_p90
    - ead_mean, ead_p90
    - ead_regulatory (using Basel III floor CCF=0.40)
    """
    X = account[FEATURES].values
    drawn = account["drawn"].values
    undrawn = account["undrawn"].values

    ccf_mean = np.clip(ccf_mean_model.predict(X), 0.0, 1.0)
    ccf_p90 = np.clip(ccf_p90_model.predict(X), 0.0, 1.0)
    # Enforce quantile ordering: P90 must be ≥ mean estimate
    ccf_p90 = np.maximum(ccf_p90, ccf_mean)

    return pd.DataFrame(
        {
            "ccf_mean": ccf_mean,
            "ccf_p90": ccf_p90,
            "ead_mean": drawn + ccf_mean * undrawn,
            "ead_p90": drawn + ccf_p90 * undrawn,
            "ead_regulatory": drawn + REGULATORY_CCF_FLOOR * undrawn,
        },
        index=account.index,
    )
