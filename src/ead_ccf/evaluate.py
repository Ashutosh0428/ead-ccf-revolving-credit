"""Model evaluation metrics and regulatory comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ead_ccf.predict import REGULATORY_CCF_FLOOR


@dataclass
class EvaluationReport:
    mae: float
    bias: float
    coverage_90: float
    regulatory_floor: float
    pct_below_floor: float

    def summary(self) -> str:
        return (
            f"MAE:              {self.mae:.4f}\n"
            f"Bias:             {self.bias:+.4f}\n"
            f"90% CI coverage:  {self.coverage_90:.1%}\n"
            f"Regulatory floor: {self.regulatory_floor}\n"
            f"% below floor:    {self.pct_below_floor:.1%}"
        )


def evaluate_ccf(
    y_true: np.ndarray,
    ccf_mean_pred: np.ndarray,
    ccf_p90_pred: np.ndarray,
) -> EvaluationReport:
    """Compute PRD success metrics for CCF predictions."""
    mae = float(np.mean(np.abs(y_true - ccf_mean_pred)))
    bias = float(np.mean(ccf_mean_pred - y_true))
    coverage_90 = float(np.mean(y_true <= ccf_p90_pred))
    pct_below_floor = float(np.mean(ccf_mean_pred < REGULATORY_CCF_FLOOR))

    return EvaluationReport(
        mae=mae,
        bias=bias,
        coverage_90=coverage_90,
        regulatory_floor=REGULATORY_CCF_FLOOR,
        pct_below_floor=pct_below_floor,
    )
