"""Synthetic data generation and target construction for CCF modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "current_utilisation",
    "utilisation_trend_3m",
    "utilisation_trend_6m",
    "payment_ratio_6m",
    "months_on_book",
    "recent_inquiries",
    "unemployment_rate",
]


def generate_synthetic_accounts(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic revolving-credit account snapshots with default events.

    Each row is one account at a month-end observation date.  Accounts that
    subsequently default have an ``ead`` column representing the balance at
    the moment of default.
    """
    rng = np.random.default_rng(seed)

    limit = rng.uniform(5_000, 100_000, size=n)
    current_utilisation = rng.beta(2, 5, size=n)
    drawn = current_utilisation * limit
    undrawn = limit - drawn

    utilisation_trend_3m = rng.normal(0.0, 0.08, size=n)
    utilisation_trend_6m = rng.normal(0.0, 0.10, size=n)
    payment_ratio_6m = rng.beta(5, 2, size=n)
    months_on_book = rng.integers(6, 120, size=n).astype(float)
    recent_inquiries = rng.poisson(1.5, size=n).astype(float)
    unemployment_rate = rng.normal(5.5, 1.2, size=n).clip(2.0, 12.0)

    # Simulate realised CCF — higher for high-utilisation / low-payment accounts
    base_ccf = (
        0.15
        + 0.35 * current_utilisation
        + 0.20 * np.clip(utilisation_trend_6m, 0, None)
        - 0.25 * payment_ratio_6m
        + 0.02 * unemployment_rate
        + rng.normal(0, 0.08, size=n)
    )
    realised_ccf = np.clip(base_ccf, 0.0, 1.0)
    ead = drawn + realised_ccf * undrawn

    return pd.DataFrame(
        {
            "account_id": np.arange(n),
            "limit": limit,
            "drawn": drawn,
            "undrawn": undrawn,
            "current_utilisation": current_utilisation,
            "utilisation_trend_3m": utilisation_trend_3m,
            "utilisation_trend_6m": utilisation_trend_6m,
            "payment_ratio_6m": payment_ratio_6m,
            "months_on_book": months_on_book,
            "recent_inquiries": recent_inquiries,
            "unemployment_rate": unemployment_rate,
            "ead": ead,
            "realised_ccf": realised_ccf,
        }
    )


def compute_realised_ccf(snapshot: pd.DataFrame, defaults: pd.DataFrame) -> pd.Series:
    """Compute realised CCF from snapshot and default-event tables.

    Formula from LLD: ``(EAD - drawn) / max(undrawn, 1)`` clipped to [0, 1].
    """
    merged = snapshot.merge(
        defaults[["account_id", "ead", "default_date"]],
        on="account_id",
        how="inner",
    )
    merged["undrawn"] = merged["limit"] - merged["drawn"]
    merged["realised_ccf"] = (
        (merged["ead"] - merged["drawn"]) / merged["undrawn"].clip(lower=1)
    ).clip(0, 1)
    return merged["realised_ccf"]
