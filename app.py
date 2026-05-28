"""Streamlit dashboard for EAD CCF Model — interactive demo."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ead_ccf.data import FEATURES, generate_synthetic_accounts
from ead_ccf.evaluate import evaluate_ccf
from ead_ccf.features import prepare_features
from ead_ccf.model import train_ccf_models
from ead_ccf.predict import REGULATORY_CCF_FLOOR, predict_ead

st.set_page_config(
    page_title="EAD CCF Model — Revolving Credit",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def get_trained_models():
    """Train models once and cache across sessions."""
    df = generate_synthetic_accounts(n=10_000)
    X = prepare_features(df).values
    y = df["realised_ccf"].values
    result = train_ccf_models(X, y, feature_names=FEATURES)

    ccf_mean_pred = np.clip(result["ccf_mean_model"].predict(result["X_val"]), 0, 1)
    ccf_p90_pred = np.clip(result["ccf_p90_model"].predict(result["X_val"]), 0, 1)
    report = evaluate_ccf(result["y_val"], ccf_mean_pred, ccf_p90_pred)

    return result, df, report


result, full_df, report = get_trained_models()
ccf_mean_model = result["ccf_mean_model"]
ccf_p90_model = result["ccf_p90_model"]

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🏦 EAD with Credit Conversion Factor")
st.markdown("**Predict Exposure at Default for Revolving Credit Products**")
st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Account Prediction",
    "📊 Portfolio Analysis",
    "📈 Model Performance",
    "📖 How It Works",
])

# ── Tab 1: Single Account ───────────────────────────────────────────────────
with tab1:
    st.header("Predict CCF & EAD for a Single Account")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Account Details")
        limit = st.number_input("Credit Limit ($)", min_value=1000, max_value=500000,
                                value=50000, step=5000)
        drawn = st.number_input("Current Drawn Balance ($)", min_value=0,
                                max_value=int(limit), value=15000, step=1000)
        undrawn = limit - drawn

        st.metric("Undrawn Amount", f"${undrawn:,.0f}")

    with col2:
        st.subheader("Risk Features")
        current_utilisation = drawn / max(limit, 1)
        st.metric("Current Utilisation", f"{current_utilisation:.1%}")

        utilisation_trend_3m = st.slider("3-Month Utilisation Trend",
                                         -0.30, 0.30, 0.00, 0.01)
        utilisation_trend_6m = st.slider("6-Month Utilisation Trend",
                                         -0.40, 0.40, 0.00, 0.01)
        payment_ratio_6m = st.slider("Payment Ratio (6 months)", 0.0, 1.0, 0.70, 0.05)
        months_on_book = st.slider("Months on Book", 6, 120, 36)
        recent_inquiries = st.slider("Recent Credit Inquiries", 0, 10, 1)
        unemployment_rate = st.slider("Unemployment Rate (%)", 2.0, 12.0, 5.5, 0.1)

    if st.button("🔮 Predict EAD", type="primary", use_container_width=True):
        account = pd.DataFrame({
            "drawn": [float(drawn)],
            "undrawn": [float(undrawn)],
            "limit": [float(limit)],
            "current_utilisation": [current_utilisation],
            "utilisation_trend_3m": [utilisation_trend_3m],
            "utilisation_trend_6m": [utilisation_trend_6m],
            "payment_ratio_6m": [payment_ratio_6m],
            "months_on_book": [float(months_on_book)],
            "recent_inquiries": [float(recent_inquiries)],
            "unemployment_rate": [unemployment_rate],
        })

        pred = predict_ead(account, ccf_mean_model, ccf_p90_model)

        st.markdown("---")
        st.subheader("Results")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("CCF (Mean)", f"{pred['ccf_mean'].values[0]:.3f}",
                      help="Best estimate of Credit Conversion Factor")
        with r2:
            st.metric("CCF (P90)", f"{pred['ccf_p90'].values[0]:.3f}",
                      help="Conservative 90th percentile estimate")
        with r3:
            st.metric("Regulatory Floor", f"{REGULATORY_CCF_FLOOR:.2f}",
                      help="Basel III standard CCF = 0.40")

        st.markdown("### EAD Comparison")

        ead_data = pd.DataFrame({
            "Method": ["Model (Mean)", "Model (P90)", "Regulatory Floor"],
            "EAD ($)": [
                pred["ead_mean"].values[0],
                pred["ead_p90"].values[0],
                pred["ead_regulatory"].values[0],
            ],
        })

        e1, e2 = st.columns([2, 3])

        with e1:
            for _, row in ead_data.iterrows():
                st.metric(row["Method"], f"${row['EAD ($)']:,.0f}")

        with e2:
            st.bar_chart(ead_data.set_index("Method"), horizontal=True)

        savings = pred["ead_regulatory"].values[0] - pred["ead_mean"].values[0]
        if savings > 0:
            st.success(f"💰 **Capital savings vs regulatory floor: ${savings:,.0f}** per account")
        else:
            st.info("Model EAD exceeds regulatory floor for this account profile.")


# ── Tab 2: Portfolio Analysis ───────────────────────────────────────────────
with tab2:
    st.header("Portfolio-Level CCF Distribution")

    n_accounts = st.slider("Number of accounts to analyse", 100, 10000, 2000, 100)
    portfolio = generate_synthetic_accounts(n=n_accounts, seed=55)
    preds = predict_ead(portfolio, ccf_mean_model, ccf_p90_model)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mean CCF", f"{preds['ccf_mean'].mean():.3f}")
    with c2:
        st.metric("Median CCF", f"{preds['ccf_mean'].median():.3f}")
    with c3:
        st.metric("Mean P90 CCF", f"{preds['ccf_p90'].mean():.3f}")
    with c4:
        total_savings = (preds["ead_regulatory"] - preds["ead_mean"]).sum()
        st.metric("Total Capital Savings", f"${total_savings:,.0f}")

    st.subheader("CCF Distribution — Model vs Regulatory Floor")

    hist_data = pd.DataFrame({
        "CCF (Model Mean)": preds["ccf_mean"],
        "CCF (Model P90)": preds["ccf_p90"],
    })
    st.bar_chart(
        hist_data.melt(var_name="Type", value_name="CCF")
        .groupby(["Type", pd.cut(
            hist_data.melt(var_name="Type", value_name="CCF")["CCF"],
            bins=20
        )])
        .size()
        .unstack(level=0)
        .fillna(0),
    )

    st.subheader("EAD Comparison: Model vs Regulatory")

    comparison = pd.DataFrame({
        "Account": range(min(50, n_accounts)),
        "EAD Model": preds["ead_mean"].head(50).values,
        "EAD Regulatory": preds["ead_regulatory"].head(50).values,
    }).set_index("Account")

    st.line_chart(comparison)

    below_floor = (preds["ccf_mean"] < REGULATORY_CCF_FLOOR).mean()
    st.info(f"📊 **{below_floor:.1%}** of accounts have model CCF below "
            f"the regulatory floor of {REGULATORY_CCF_FLOOR:.0%}")


# ── Tab 3: Model Performance ───────────────────────────────────────────────
with tab3:
    st.header("Model Evaluation Metrics")

    m1, m2, m3 = st.columns(3)

    with m1:
        mae_ok = report.mae < 0.15
        st.metric("MAE", f"{report.mae:.4f}",
                  delta="PASS" if mae_ok else "FAIL",
                  delta_color="normal" if mae_ok else "inverse")
        st.caption("Target: < 0.15")

    with m2:
        bias_ok = abs(report.bias) < 0.03
        st.metric("Bias", f"{report.bias:+.4f}",
                  delta="PASS" if bias_ok else "FAIL",
                  delta_color="normal" if bias_ok else "inverse")
        st.caption("Target: ±3 pp")

    with m3:
        cov_ok = 0.85 <= report.coverage_90 <= 0.95
        st.metric("90% CI Coverage", f"{report.coverage_90:.1%}",
                  delta="PASS" if cov_ok else "FAIL",
                  delta_color="normal" if cov_ok else "inverse")
        st.caption("Target: 85–95%")

    st.markdown("---")

    st.subheader("Actual vs Predicted CCF")
    y_val = result["y_val"]
    ccf_pred = np.clip(ccf_mean_model.predict(result["X_val"]), 0, 1)

    scatter_df = pd.DataFrame({
        "Actual CCF": y_val[:500],
        "Predicted CCF": ccf_pred[:500],
    })
    st.scatter_chart(scatter_df, x="Actual CCF", y="Predicted CCF")

    st.subheader("Prediction Error Distribution")
    errors = ccf_pred - y_val
    error_df = pd.DataFrame({"Prediction Error (CCF)": errors})
    st.bar_chart(
        error_df.value_counts(bins=30, sort=False)
    )

    st.subheader("Feature Importance")
    importance = ccf_mean_model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance,
    }).sort_values("Importance", ascending=True)
    st.bar_chart(fi_df.set_index("Feature"), horizontal=True)


# ── Tab 4: How It Works ────────────────────────────────────────────────────
with tab4:
    st.header("How the CCF Model Works")

    st.markdown("""
    ### The Business Problem

    For revolving credit (credit cards, lines of credit, overdrafts), the bank's exposure
    at default is generally **larger** than the current balance because borrowers tend to
    draw down available credit as they approach default.

    ### The Formula

    ```
    CCF = (EAD - Drawn) / max(Undrawn, 1)     clipped to [0, 1]
    EAD = Drawn + CCF × Undrawn
    ```

    ### The Model

    We train two **XGBoost** regressors:

    | Model | Purpose | Objective |
    |-------|---------|-----------|
    | **Mean CCF** | Best estimate | Minimise squared error |
    | **P90 CCF** | Conservative bound | 90th percentile quantile loss |

    ### Monotonic Constraints

    The model enforces domain knowledge:
    - **Higher utilisation → higher CCF** (borrowers using more credit draw even more)
    - **Higher payment ratio → lower CCF** (good payers draw less before default)

    ### Features Used

    | Feature | Description |
    |---------|-------------|
    | `current_utilisation` | Drawn / Limit |
    | `utilisation_trend_3m` | 3-month utilisation change |
    | `utilisation_trend_6m` | 6-month utilisation change |
    | `payment_ratio_6m` | Payment-to-balance ratio |
    | `months_on_book` | Account age |
    | `recent_inquiries` | New credit seeking behaviour |
    | `unemployment_rate` | Macroeconomic stress indicator |

    ### Why This Matters

    The Basel III regulatory floor sets CCF = 0.40 for all accounts. Our model shows
    that **~99% of accounts** have a CCF well below this floor. Using an internal model
    lets the bank hold less capital while still being adequately protected.
    """)

    st.markdown("---")
    st.caption("Built by Ashutosh | EAD CCF Model for Revolving Credit")
