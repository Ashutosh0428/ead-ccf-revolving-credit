"""Feature engineering utilities for CCF model."""

from __future__ import annotations

import pandas as pd

from ead_ccf.data import FEATURES


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and validate model features from raw account data."""
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURES].copy()
