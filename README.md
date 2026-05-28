# EAD with Credit Conversion Factor for Revolving Credit

Predicts **Exposure at Default (EAD)** for revolving credit products using a **Credit Conversion Factor (CCF)** model built with XGBoost.

## Overview

For revolving credit (credit cards, lines of credit, overdrafts), the bank's exposure at default is typically larger than the current outstanding balance because borrowers tend to draw down available credit as they approach default. The CCF estimates how much of the undrawn limit will become drawn by the time of default.

### Key Components

- **Mean CCF Model** — XGBoost regression with monotonic constraints on utilisation and payment ratio
- **P90 Upper Bound** — Quantile regression (90th percentile) for conservative EAD estimation
- **Regulatory Comparison** — Benchmarks against the Basel III floor of CCF = 0.40

### Features

| Feature | Description |
|---------|-------------|
| `current_utilisation` | Drawn / Limit |
| `utilisation_trend_3m` | 3-month utilisation delta |
| `utilisation_trend_6m` | 6-month utilisation delta |
| `payment_ratio_6m` | Payment-to-balance ratio over 6 months |
| `months_on_book` | Account age in months |
| `recent_inquiries` | Number of recent credit inquiries |
| `unemployment_rate` | Macro indicator at scoring date |

## Setup

```bash
pip install -e ".[dev]"
```

## Train

```bash
python scripts/train.py
```

## Test

```bash
pytest tests/ -v
```

## Target Definition

```
realised_CCF = clip((EAD - drawn) / max(undrawn, 1), 0, 1)
```

## Success Metrics (PRD)

| Metric | Target |
|--------|--------|
| MAE | < 0.15 |
| Bias | ±3 pp |
| 90% CI Coverage | 85–95% |
