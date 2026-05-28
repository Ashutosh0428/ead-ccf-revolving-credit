# EAD CCF Model — Internship Guide

A deep-dive walkthrough of how this project works, written for someone new to credit risk modelling.

---

## Table of Contents

1. [The Business Problem — Why Does This Exist?](#1-the-business-problem)
2. [Core Concepts You Need to Know](#2-core-concepts)
3. [Project Structure — What Lives Where](#3-project-structure)
4. [The Data Layer — `data.py`](#4-the-data-layer)
5. [Feature Engineering — `features.py`](#5-feature-engineering)
6. [The CCF Formula — The Heart of Everything](#6-the-ccf-formula)
7. [Model Training — `model.py`](#7-model-training)
8. [Prediction — `predict.py`](#8-prediction)
9. [Evaluation — `evaluate.py`](#9-evaluation)
10. [The Training Pipeline — `scripts/train.py`](#10-the-training-pipeline)
11. [Tests — `tests/test_model.py`](#11-tests)
12. [How to Run Everything](#12-how-to-run)
13. [Key Design Decisions Explained](#13-design-decisions)
14. [Glossary](#14-glossary)

---

## 1. The Business Problem

### What is revolving credit?

Think of a credit card. You have a **limit** (say $10,000) and a **balance** (say $3,000 currently drawn). The remaining $7,000 is **undrawn** — you could spend it anytime.

Other examples: lines of credit, overdraft facilities, home equity lines.

### Why do banks care about EAD?

When a borrower **defaults** (stops paying), the bank loses money. But how much? Not just the current $3,000 balance — research shows borrowers tend to **draw down more credit** as they slide toward default. Maybe they max out cards, tap their line of credit, etc.

So at the moment of default, the balance might be $6,500 instead of $3,000. That $6,500 is the **Exposure at Default (EAD)**.

### What is CCF?

The **Credit Conversion Factor** answers: "What fraction of the currently undrawn amount will become drawn by default?"

```
CCF = (Balance at Default - Balance Today) / Undrawn Amount Today
```

In our example:
```
CCF = ($6,500 - $3,000) / $7,000 = 0.50
```

This means 50% of the available credit got used before default.

### Why build a model?

Banks are **required** by regulation (Basel III) to estimate EAD for capital adequacy calculations. They have two choices:

1. **Regulatory floor**: Use CCF = 0.40 for everyone (simple but wasteful — one-size-fits-all)
2. **Internal model**: Build a model that predicts CCF per account (more precise, less capital held unnecessarily)

This project builds option 2.

### Who uses this?

| Team | How They Use CCF |
|------|-----------------|
| **Capital Planning** | Calculate how much capital the bank must hold (Basel III requirement) |
| **Relationship Managers** | Decide whether to approve a credit limit increase |
| **Finance / Provisioning** | Feed into IFRS 9 Expected Credit Loss (ECL) calculations |

---

## 2. Core Concepts

### EAD Formula

```
EAD = Drawn + CCF × Undrawn
```

If someone has drawn $3,000 of a $10,000 limit, and our model predicts CCF = 0.30:

```
EAD = $3,000 + 0.30 × $7,000 = $5,100
```

The bank should plan for a $5,100 exposure, not just $3,000.

### Why Two Models?

We train **two** models, not one:

1. **Mean model** — predicts the average/expected CCF. Used for "best estimate" EAD.
2. **P90 model** — predicts the 90th percentile CCF. This is the conservative upper bound.

Think of it like weather: "The temperature will likely be 25°C (mean), but there's a 10% chance it could be 32°C (P90)." Banks use the conservative number for capital buffers.

### What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is a machine learning algorithm that builds many small decision trees sequentially, where each tree tries to correct the mistakes of the previous ones. It is the industry standard for tabular/structured data problems like credit risk.

Why XGBoost for this problem:
- Handles non-linear relationships (CCF doesn't increase linearly with utilisation)
- Supports **monotonic constraints** (we can enforce domain knowledge — see below)
- Supports **quantile regression** (for our P90 model)
- Fast training, good out-of-the-box performance

---

## 3. Project Structure

```
ead-ccf-model/
├── src/ead_ccf/           # The Python package
│   ├── __init__.py        # Package marker + version
│   ├── data.py            # Data generation + target formula
│   ├── features.py        # Feature selection/validation
│   ├── model.py           # XGBoost training (mean + quantile)
│   ├── predict.py         # EAD prediction from trained models
│   └── evaluate.py        # Metrics + regulatory comparison
├── scripts/
│   └── train.py           # End-to-end pipeline script
├── tests/
│   └── test_model.py      # Automated test suite
├── pyproject.toml         # Project config + dependencies
└── docs/
    └── INTERNSHIP_GUIDE.md  # This file
```

**Data flows through the modules like this:**

```
data.py → features.py → model.py → predict.py → evaluate.py
  │            │             │           │             │
  │            │             │           │             └─ Are predictions good?
  │            │             │           └─ Compute EAD from CCF
  │            │             └─ Train XGBoost (mean + P90)
  │            └─ Select & validate 7 features
  └─ Generate synthetic accounts with CCF targets
```

---

## 4. The Data Layer

**File: `src/ead_ccf/data.py`**

### Why synthetic data?

Real bank data is confidential. This project generates **realistic synthetic data** that mimics the statistical properties of real revolving credit portfolios. In production, you would replace `generate_synthetic_accounts()` with a database query.

### How accounts are generated

```python
limit = rng.uniform(5_000, 100_000, size=n)           # Credit limits: $5K–$100K
current_utilisation = rng.beta(2, 5, size=n)           # Beta(2,5) → right-skewed, mostly low
drawn = current_utilisation * limit                     # Balance = utilisation × limit
undrawn = limit - drawn                                 # What's left to draw
```

**Why Beta(2,5) for utilisation?** Most credit card holders use a small fraction of their limit. A Beta(2,5) distribution peaks around 0.2–0.3, which matches real-world patterns. Very few accounts are maxed out.

### The CCF simulation formula

```python
base_ccf = (
    0.15                                          # Base drawdown rate
    + 0.35 * current_utilisation                  # High users draw more
    + 0.20 * clip(utilisation_trend_6m, 0, None)  # Rising trend = danger
    - 0.25 * payment_ratio_6m                     # Good payers draw less
    + 0.02 * unemployment_rate                    # Macro stress increases CCF
    + noise                                       # Random variation
)
```

Each coefficient represents a real credit risk relationship:
- **+0.35 × utilisation**: Borrowers already using a lot of credit tend to use even more before defaulting
- **+0.20 × rising trend**: An increasing utilisation trend signals financial stress
- **−0.25 × payment ratio**: Borrowers making payments regularly are less likely to max out
- **+0.02 × unemployment**: Economic stress pushes everyone's CCF higher

### The `compute_realised_ccf` function

This function implements the **LLD formula** for computing CCF from actual default data:

```python
realised_ccf = ((ead - drawn) / max(undrawn, 1)).clip(0, 1)
```

Key details:
- `max(undrawn, 1)` prevents division by zero when account is fully drawn
- `.clip(0, 1)` ensures CCF stays in [0, 1] — it is a fraction, not a raw number
- Clipping also handles edge cases where borrowers *reduced* their balance before default (negative numerator → clips to 0)

---

## 5. Feature Engineering

**File: `src/ead_ccf/features.py`**

This module is intentionally simple — it selects and validates the 7 model features:

| # | Feature | What It Measures | Range |
|---|---------|-----------------|-------|
| 1 | `current_utilisation` | Drawn / Limit | 0.0–1.0 |
| 2 | `utilisation_trend_3m` | 3-month change in utilisation | ~ -0.3 to +0.3 |
| 3 | `utilisation_trend_6m` | 6-month change in utilisation | ~ -0.4 to +0.4 |
| 4 | `payment_ratio_6m` | Payments / Balance over 6 months | 0.0–1.0 |
| 5 | `months_on_book` | Account age in months | 6–120 |
| 6 | `recent_inquiries` | Credit inquiries (new credit seeking) | 0–10+ |
| 7 | `unemployment_rate` | National unemployment at scoring date | 2.0–12.0% |

**Why these specific features?**

Features 1–4 are **behavioural** — they describe how the borrower is using their credit. These are the strongest predictors because borrower actions directly drive CCF.

Feature 5 is **vintage** — newer accounts behave differently from mature ones.

Feature 6 is a **risk signal** — people seeking new credit may be in financial distress.

Feature 7 is **macroeconomic** — during recessions, CCFs increase across the board.

### The validation step

```python
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURES].copy()
```

This is a **defensive check** — it ensures the input data contains all required columns before training. The `.copy()` prevents accidental modification of the original DataFrame.

---

## 6. The CCF Formula

The central formula for this entire project:

```
realised_CCF = clip((EAD - drawn_today) / max(undrawn_today, 1), 0, 1)
```

Let's trace through three examples:

### Example A: Typical account
```
Limit: $50,000 | Drawn: $15,000 | Undrawn: $35,000 | EAD: $28,000

CCF = ($28,000 - $15,000) / $35,000 = 0.371

Interpretation: 37.1% of the undrawn credit was drawn before default.
```

### Example B: Fully utilised account
```
Limit: $50,000 | Drawn: $50,000 | Undrawn: $0 | EAD: $50,000

CCF = ($50,000 - $50,000) / max($0, 1) = 0.0

Interpretation: No undrawn credit exists, so CCF is 0. EAD = Drawn.
This is why undrawn is clipped to min=1 — prevents division by zero.
```

### Example C: Borrower who paid down before default
```
Limit: $50,000 | Drawn: $30,000 | Undrawn: $20,000 | EAD: $25,000

CCF = ($25,000 - $30,000) / $20,000 = -0.25 → clipped to 0.0

Interpretation: Borrower actually reduced their balance. Negative CCF
gets clipped to 0 because CCF represents additional drawdown only.
```

---

## 7. Model Training

**File: `src/ead_ccf/model.py`**

### Monotonic constraints — encoding domain knowledge

This is one of the most important concepts in the project:

```python
MONOTONE_CONSTRAINTS = {
    "current_utilisation": 1,     # +1 = monotonically increasing
    "payment_ratio_6m": -1,       # -1 = monotonically decreasing
}
```

**What does this mean?**

- `current_utilisation: 1` — The model is **forced** to predict higher CCF for higher utilisation, all else equal. It can never learn "high utilisation → lower CCF" even if noise in the data suggests it.

- `payment_ratio_6m: -1` — The model is **forced** to predict lower CCF for better payment behaviour. A borrower making regular payments must always get a lower CCF than an identical borrower not paying.

**Why is this critical?**

Without constraints, XGBoost might find spurious patterns in noisy data (overfitting). In credit risk, regulators and auditors expect the model to behave in economically intuitive ways. A model that says "paying your bills makes you riskier" would be rejected in any model validation review.

### The `_build_monotone_tuple` function

XGBoost requires monotonic constraints as a **positional tuple**, not a dictionary. If features are `[A, B, C, D, E, F, G]` and only `A` (+1) and `D` (-1) are constrained:

```python
# Result: (1, 0, 0, -1, 0, 0, 0)
#          A  B  C   D  E  F  G
```

`0` means "no constraint" — the model can learn any relationship for that feature.

### Mean model configuration

```python
ccf_mean = xgb.XGBRegressor(
    objective="reg:squarederror",   # Minimise MSE (mean squared error)
    n_estimators=600,               # Build up to 600 trees
    max_depth=5,                    # Each tree has max 5 levels (prevents overfitting)
    learning_rate=0.05,             # Small steps = more stable learning
    monotone_constraints=monotone,  # Domain knowledge enforcement
    reg_lambda=2.0,                 # L2 regularisation (prevents overfitting)
    random_state=seed,              # Reproducibility
)
```

**Why these hyperparameters?**

| Parameter | Value | Why |
|-----------|-------|-----|
| `n_estimators=600` | Max 600 trees. Early stopping usually stops earlier. More trees = more capacity. |
| `max_depth=5` | Shallow trees prevent overfitting. Deep trees memorize noise. |
| `learning_rate=0.05` | Low rate means each tree contributes a small correction. Needs more trees but generalises better. |
| `reg_lambda=2.0` | L2 penalty on leaf weights. Higher than default (1.0) because CCF data can be noisy. |

### Early stopping

```python
ccf_mean.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
```

XGBoost monitors validation error at each boosting round. When validation error stops improving, training stops automatically. This prevents overfitting — the model uses only as many trees as actually help.

### P90 quantile model

```python
ccf_p90 = xgb.XGBRegressor(
    objective="reg:quantileerror",  # Quantile loss function
    quantile_alpha=0.90,            # Target the 90th percentile
)
```

**What is quantile regression?**

Normal regression predicts the **average** (mean). Quantile regression predicts a specific **percentile**.

With `quantile_alpha=0.90`, the model learns to predict a value such that **90% of actual CCFs fall below it**. This gives a conservative upper bound.

**The loss function is asymmetric:**
- Underpredicting (actual > predicted) gets penalised 9× more than overpredicting
- This pushes predictions upward toward the 90th percentile
- The math: `loss = alpha * max(y - pred, 0) + (1 - alpha) * max(pred - y, 0)`

### Model serialisation

```python
def save_models(models, output_dir):
    joblib.dump(models["ccf_mean_model"], output_dir / "ccf_mean_model.joblib")
    joblib.dump(models["ccf_p90_model"], output_dir / "ccf_p90_model.joblib")
```

`joblib` serialises the trained XGBoost objects to disk. The `.joblib` format is more efficient than Python's `pickle` for numpy-heavy objects. Models can be loaded later for inference without retraining.

---

## 8. Prediction

**File: `src/ead_ccf/predict.py`**

### The prediction flow

```python
def predict_ead(account, ccf_mean_model, ccf_p90_model):
    X = account[FEATURES].values            # Extract feature matrix
    drawn = account["drawn"].values          # Current balance
    undrawn = account["undrawn"].values      # Available credit

    ccf_mean = clip(ccf_mean_model.predict(X), 0.0, 1.0)
    ccf_p90 = clip(ccf_p90_model.predict(X), 0.0, 1.0)
    ccf_p90 = maximum(ccf_p90, ccf_mean)    # Enforce ordering

    return DataFrame({
        "ccf_mean": ccf_mean,
        "ccf_p90": ccf_p90,
        "ead_mean": drawn + ccf_mean * undrawn,       # Best estimate
        "ead_p90": drawn + ccf_p90 * undrawn,          # Conservative
        "ead_regulatory": drawn + 0.40 * undrawn,      # Basel III floor
    })
```

### Why clip CCF to [0, 1]?

XGBoost can predict values outside the training range. A prediction of -0.05 or 1.12 is mathematically possible but meaningless — CCF is a fraction between 0% and 100%.

### Why enforce `ccf_p90 >= ccf_mean`?

The mean and P90 models are trained **independently**. They don't know about each other. In some edge cases, the P90 model might output a lower value than the mean model (called "quantile crossing").

This would be nonsensical — the conservative estimate can't be lower than the best estimate. So we enforce: `ccf_p90 = max(ccf_p90, ccf_mean)`.

### The three EAD outputs

Each prediction returns three EAD values for comparison:

```
ead_mean       = drawn + ccf_mean × undrawn     ← "Our best estimate"
ead_p90        = drawn + ccf_p90  × undrawn     ← "Conservative buffer (90% confidence)"
ead_regulatory = drawn + 0.40     × undrawn     ← "What regulators assume if no model"
```

**Example output:**
```
Account: Limit=$80,000 | Drawn=$20,000 | Undrawn=$60,000

Model predicts CCF_mean=0.18, CCF_p90=0.29

ead_mean       = $20,000 + 0.18 × $60,000 = $30,800
ead_p90        = $20,000 + 0.29 × $60,000 = $37,400
ead_regulatory = $20,000 + 0.40 × $60,000 = $44,000
```

The model saves the bank $13,200 in capital vs. the regulatory floor for this single account. Multiply across millions of accounts and the capital savings are enormous — this is why banks build internal models.

---

## 9. Evaluation

**File: `src/ead_ccf/evaluate.py`**

### What gets measured

The PRD defines three success criteria:

#### 1. MAE (Mean Absolute Error) < 0.15

```python
mae = mean(|y_true - ccf_mean_pred|)
```

On average, our prediction should be within 0.15 of the actual CCF. Since CCF ranges from 0 to 1, an MAE of 0.15 means we're typically off by 15 percentage points.

**Our result: MAE = 0.065** — well under the threshold.

#### 2. Bias within ±3 percentage points

```python
bias = mean(ccf_mean_pred - y_true)
```

Bias measures whether we **systematically** over- or under-predict. Positive bias = we overestimate (conservative). Negative = we underestimate (risky).

A biased model is dangerous:
- **Over-predicting** → bank holds too much capital (expensive)
- **Under-predicting** → bank is under-capitalised (regulatory violation)

**Our result: Bias = +0.000** — essentially zero.

#### 3. 90% CI Coverage between 85-95%

```python
coverage = mean(y_true <= ccf_p90_pred)
```

The P90 model claims "90% of actual CCFs should fall below my prediction." Coverage measures if that's actually true.

- Coverage < 85% → P90 model is not conservative enough (undercovers)
- Coverage > 95% → P90 model is too conservative (wasteful)
- 85-95% → well calibrated

**Our result: Coverage = 86.8%** — within range.

### Regulatory floor comparison

```python
pct_below_floor = mean(ccf_mean_pred < 0.40)
```

This tells us what percentage of accounts have a model CCF below the regulatory flat 0.40. If it's high (ours is 99.3%), the bank saves significant capital by using the internal model.

---

## 10. The Training Pipeline

**File: `scripts/train.py`**

This script ties everything together:

```
Step 1: Generate 10,000 synthetic accounts
         ↓
Step 2: Extract 7 features from account data
         ↓
Step 3: Train both XGBoost models (mean + P90)
         ↓
Step 4: Evaluate on held-out validation set
         ↓
Step 5: Print sample EAD predictions for 5 accounts
         ↓
Step 6: Save models to artifacts/ directory
```

The pipeline is **deterministic** — running it twice with the same seed produces identical results.

---

## 11. Tests

**File: `tests/test_model.py`**

### Test architecture

Tests use `pytest` fixtures to train models **once** and reuse across all tests:

```python
@pytest.fixture(scope="module")
def trained_models():
    # Trains on 5,000 samples — fast enough for CI
    ...
```

`scope="module"` means the fixture runs once per test file, not once per test. This saves ~10 seconds of redundant training.

### LLD Invariant Tests

These three tests verify **mathematical properties** that must always hold:

#### Test 1: Fully utilised account → EAD = drawn

```
If undrawn = 0, then EAD = drawn + CCF × 0 = drawn
```

No matter what CCF the model predicts, multiplying by zero undrawn gives zero. This is a **mathematical identity**, not a model prediction — it tests the EAD formula itself.

#### Test 2: Inactive account → low CCF

An account with:
- Zero drawn, zero utilisation trend
- High payment ratio (0.9)
- No recent inquiries
- Low unemployment

This is the "safest possible" profile. The model should predict CCF < 0.10 — very little additional drawdown expected.

#### Test 3: P90 >= mean for all inputs

Tests the quantile ordering enforcement. Generates 500 random accounts and checks that P90 prediction is never below the mean prediction for any of them.

### PRD Metric Tests

These three tests verify the success criteria from the Product Requirements Document:

- `test_mae_below_threshold` — MAE < 0.15
- `test_bias_within_bounds` — |Bias| < 0.03
- `test_coverage_in_range` — 85% ≤ Coverage ≤ 95%

If any of these fail, the model doesn't meet requirements and shouldn't be deployed.

---

## 12. How to Run

### Setup

```bash
# Clone the repo
git clone https://github.com/Ashutosh0428/ead-ccf-revolving-credit.git
cd ead-ccf-revolving-credit

# Install (creates editable install + dev dependencies)
pip install -e ".[dev]"
```

### Train the model

```bash
python scripts/train.py
```

Expected output:
```
Generating synthetic data …
Training CCF models …
Evaluating on validation set …
MAE:              0.0646
Bias:             +0.0000
90% CI coverage:  86.8%
Regulatory floor: 0.4
% below floor:    99.3%

Saving models to artifacts/ …
Done.
```

### Run tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_model.py::TestCCFInvariants::test_fully_utilised_ead_equals_drawn PASSED
tests/test_model.py::TestCCFInvariants::test_inactive_account_low_ccf PASSED
tests/test_model.py::TestCCFInvariants::test_quantile_bound_gte_point_estimate PASSED
tests/test_model.py::TestEvaluationMetrics::test_mae_below_threshold PASSED
tests/test_model.py::TestEvaluationMetrics::test_bias_within_bounds PASSED
tests/test_model.py::TestEvaluationMetrics::test_coverage_in_range PASSED
```

### Use trained models in your own code

```python
from pathlib import Path
from ead_ccf.model import load_models
from ead_ccf.predict import predict_ead

models = load_models(Path("artifacts"))
# 'account' is a DataFrame with the 7 features + drawn + undrawn columns
result = predict_ead(account, **models)
print(result[["ead_mean", "ead_p90", "ead_regulatory"]])
```

---

## 13. Key Design Decisions Explained

### Why two separate models instead of one model with two outputs?

Simplicity and maintainability. Each model has a single well-defined objective. The mean model minimises squared error; the P90 model minimises quantile loss. Combining them would require custom multi-output training.

The tradeoff: separately trained models can have "quantile crossing" (P90 < mean), which we fix with post-hoc clamping. In production, you could also use a joint quantile model or conformal prediction to avoid this.

### Why XGBoost and not a neural network?

1. **Tabular data** — XGBoost consistently beats deep learning on structured/tabular datasets
2. **Interpretability** — regulators require model explainability; XGBoost has built-in feature importance
3. **Monotonic constraints** — native support for domain knowledge encoding
4. **Small data** — neural networks need millions of rows; XGBoost works well with 10K–100K
5. **Industry standard** — credit risk teams worldwide use gradient boosting

### Why synthetic data?

Real credit data is highly confidential (PII, financial records). Synthetic data allows:
- Open-source development and sharing
- Reproducible experiments
- Controlled experiments (vary one parameter at a time)

In production, you'd replace `generate_synthetic_accounts()` with a SQL query against the bank's data warehouse.

### Why clip CCF to [0, 1]?

CCF is defined as a fraction: "how much of the undrawn amount gets drawn." Values below 0 (borrower paid down) or above 1 (borrower exceeded limit — shouldn't happen) are meaningless for modelling purposes.

### Why `reg_lambda=2.0` (higher than default)?

L2 regularisation penalises large leaf weights. CCF data is noisy (human behaviour has high variance), so stronger regularisation prevents the model from fitting to noise in individual accounts. The default of 1.0 can lead to overfitting on small-to-medium datasets.

### Why the regulatory floor comparison?

Banks must justify to regulators why their internal model is better than the simple floor. By showing that 99.3% of accounts have model CCF < 0.40, we demonstrate the floor is overly conservative and the internal model provides meaningful risk differentiation.

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **EAD** | Exposure at Default — the total amount the bank is owed when a borrower defaults |
| **CCF** | Credit Conversion Factor — fraction of undrawn credit that becomes drawn before default |
| **Drawn** | The amount currently borrowed/used from the credit line |
| **Undrawn** | The remaining available credit (Limit - Drawn) |
| **Limit** | The maximum credit amount granted to the borrower |
| **Utilisation** | Drawn / Limit — how much of the limit is currently used |
| **Basel III** | International banking regulation framework that defines capital requirements |
| **IFRS 9** | Accounting standard requiring Expected Credit Loss provisioning |
| **ECL** | Expected Credit Loss = PD × LGD × EAD (probability × loss severity × exposure) |
| **PD** | Probability of Default — likelihood the borrower will stop paying |
| **LGD** | Loss Given Default — fraction of EAD that the bank actually loses |
| **IRB** | Internal Ratings-Based approach — using internal models instead of regulatory floors |
| **Monotonic constraint** | Forces the model to maintain a directional relationship (more X → more Y) |
| **Quantile regression** | Predicts a specific percentile instead of the mean |
| **P90** | 90th percentile — the value below which 90% of observations fall |
| **MAE** | Mean Absolute Error — average prediction error magnitude |
| **Bias** | Systematic over- or under-prediction tendency |
| **Coverage** | Fraction of actual values that fall below the predicted upper bound |
| **XGBoost** | Gradient boosting library — builds an ensemble of decision trees sequentially |
| **Early stopping** | Halting training when validation performance stops improving |
| **L2 regularisation** | Penalty on large model weights to prevent overfitting (also called Ridge) |

---

*Last updated: May 2026*
