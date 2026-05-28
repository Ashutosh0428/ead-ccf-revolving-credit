#!/usr/bin/env python3
"""End-to-end training pipeline: generate data → train → evaluate → save."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ead_ccf.data import FEATURES, generate_synthetic_accounts
from ead_ccf.evaluate import evaluate_ccf
from ead_ccf.features import prepare_features
from ead_ccf.model import save_models, train_ccf_models
from ead_ccf.predict import predict_ead


def main() -> None:
    output_dir = Path("artifacts")

    print("Generating synthetic data …")
    df = generate_synthetic_accounts(n=10_000)
    X = prepare_features(df).values
    y = df["realised_ccf"].values

    print("Training CCF models …")
    result = train_ccf_models(X, y, feature_names=FEATURES)

    print("Evaluating on validation set …")
    ccf_mean_pred = np.clip(result["ccf_mean_model"].predict(result["X_val"]), 0, 1)
    ccf_p90_pred = np.clip(result["ccf_p90_model"].predict(result["X_val"]), 0, 1)

    report = evaluate_ccf(result["y_val"], ccf_mean_pred, ccf_p90_pred)
    print(report.summary())

    print("\nPredicting EAD for sample accounts …")
    sample = df.head(5)
    ead_results = predict_ead(sample, result["ccf_mean_model"], result["ccf_p90_model"])
    print(ead_results.to_string())

    print(f"\nSaving models to {output_dir}/ …")
    save_models(result, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
