"""
Credit Risk Explainable AI Dashboard
=====================================
An end-to-end ML app: synthetic data -> model comparison -> SHAP
explainability -> interactive what-if risk simulator.

Run locally:    streamlit run app.py
Deploy free:    share.streamlit.io (see README.md)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import generate_credit_data
from src.model_trainer import ALL_FEATURES, train_all_models
from src.shap_utils import compute_shap_values, single_prediction_shap

st.set_page_config(
    page_title="Credit Risk Explainable AI",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data + model training (so re-runs from widget interaction are fast)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    return generate_credit_data(n_samples=5000)


@st.cache_resource
def train_models(df):
    return train_all_models(df)


df = load_data()
models = train_models(df)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("💳 Credit Risk AI")
st.sidebar.markdown(
    "An explainable machine learning dashboard for loan default risk "
    "assessment, built with scikit-learn, XGBoost and SHAP."
)
selected_model_name = st.sidebar.selectbox(
    "Active model", list(models.keys()), index=2
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** 5,000 synthetic loan applicants\n\n"
    "**Target:** Probability of default\n\n"
    "[View source on GitHub](https://github.com/YOUR_USERNAME/credit-risk-explainable-ai)"
)

active_model = models[selected_model_name]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Credit Risk Explainable AI Dashboard")
st.caption(
    "Predict loan default risk and understand *why* the model made each "
    "decision — not just what it decided."
)

tab_overview, tab_models, tab_explain, tab_simulate = st.tabs(
    ["📊 Data Overview", "🤖 Model Comparison", "🔍 Explainability", "🎛️ What-If Simulator"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Data Overview
# ---------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Applicants", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['default'].mean() * 100:.1f}%")
    col3.metric("Avg Loan Amount", f"${df['loan_amount'].mean():,.0f}")
    col4.metric("Avg Income", f"${df['annual_income'].mean():,.0f}")

    st.markdown("### Explore the data")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df, x="debt_to_income", color="default", barmode="overlay",
            nbins=40, title="Debt-to-Income Ratio by Default Status",
            labels={"default": "Defaulted"},
            color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(
            df, x="default", y="credit_utilization", color="default",
            title="Credit Utilization by Default Status",
            labels={"default": "Defaulted"},
            color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        purpose_default = df.groupby("loan_purpose")["default"].mean().sort_values(ascending=False)
        fig3 = px.bar(
            purpose_default, title="Default Rate by Loan Purpose",
            labels={"value": "Default Rate", "loan_purpose": "Purpose"},
        )
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.scatter(
            df.sample(800, random_state=1), x="annual_income", y="loan_amount",
            color="default", opacity=0.6,
            title="Loan Amount vs. Income (sample of 800)",
            color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
        )
        st.plotly_chart(fig4, use_container_width=True)

    with st.expander("View raw data sample"):
        st.dataframe(df.sample(50, random_state=1), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Model Comparison
# ---------------------------------------------------------------------------
with tab_models:
    st.markdown("### Head-to-head model performance")
    metrics_df = pd.DataFrame({name: m.metrics for name, m in models.items()}).T
    st.dataframe(
        metrics_df.style.format("{:.3f}").highlight_max(axis=0, color="#1f7a3f4d"),
        use_container_width=True,
    )

    st.markdown("### ROC Curves")
    fig = go.Figure()
    for name, m in models.items():
        fig.add_trace(go.Scatter(x=m.fpr, y=m.tpr, mode="lines", name=f"{name} (AUC={m.metrics['ROC AUC']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Why compare models instead of picking one?** In credit risk, "
        "false negatives (missed defaulters) and false positives (rejected "
        "good applicants) have very different business costs. A recall-heavy "
        "model protects the lender; a precision-heavy model protects good "
        "applicants from unfair rejection. Choosing the right model is a "
        "business decision, not just an accuracy contest."
    )

# ---------------------------------------------------------------------------
# TAB 3 — Explainability
# ---------------------------------------------------------------------------
with tab_explain:
    st.markdown(f"### Global feature importance — {selected_model_name}")
    st.caption(
        "SHAP values show how much each feature pushes predictions toward "
        "or away from default risk, across a sample of the test set."
    )

    with st.spinner("Computing SHAP values..."):
        _, shap_values, X_transformed, feature_names = compute_shap_values(
            active_model.pipeline, active_model.X_test, max_samples=250
        )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=True).tail(15)

    fig = px.bar(
        importance_df, x="mean_abs_shap", y="feature", orientation="h",
        title="Top 15 Features by Mean |SHAP value|",
        labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SHAP summary (distribution of impact)")
    summary_df = pd.DataFrame(shap_values, columns=feature_names)
    top_features = importance_df["feature"].tolist()[::-1][:8]
    melted = summary_df[top_features].melt(var_name="feature", value_name="shap_value")
    fig2 = px.strip(
        melted, x="shap_value", y="feature", orientation="h",
        title="SHAP Value Spread — Top 8 Features",
        labels={"shap_value": "SHAP value (impact on default risk)"},
    )
    fig2.update_traces(marker=dict(opacity=0.5, size=4))
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4 — What-If Simulator
# ---------------------------------------------------------------------------
with tab_simulate:
    st.markdown("### Build an applicant and see the risk assessment live")
    st.caption(
        "Adjust the sliders to simulate a loan applicant. The model "
        "re-scores instantly and SHAP shows exactly which factors drove "
        "that specific decision."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 21, 70, 35)
        annual_income = st.slider("Annual Income ($)", 12000, 250000, 55000, step=1000)
        employment_years = st.slider("Employment (years)", 0, 40, 5)
        credit_history_years = st.slider("Credit History (years)", 0, 35, 8)
    with c2:
        loan_amount = st.slider("Loan Amount ($)", 1000, 60000, 15000, step=500)
        loan_term_months = st.select_slider("Loan Term (months)", options=[12, 24, 36, 48, 60], value=36)
        existing_loans = st.slider("Existing Loans", 0, 6, 1)
        num_credit_inquiries = st.slider("Credit Inquiries (recent)", 0, 12, 2)
    with c3:
        late_payments_last_2yrs = st.slider("Late Payments (last 2 yrs)", 0, 10, 0)
        credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "home_improvement", "medical", "auto", "business", "education"],
        )

    debt_to_income = min((loan_amount / loan_term_months * 12) / annual_income, 3.0)

    applicant = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "credit_history_years": credit_history_years,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "existing_loans": existing_loans,
        "num_credit_inquiries": num_credit_inquiries,
        "late_payments_last_2yrs": late_payments_last_2yrs,
        "credit_utilization": credit_utilization,
        "debt_to_income": debt_to_income,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
    }])[ALL_FEATURES]

    proba = active_model.pipeline.predict_proba(applicant)[0, 1]

    st.markdown("---")
    r1, r2 = st.columns([1, 2])
    with r1:
        risk_label = "🔴 High Risk" if proba > 0.5 else ("🟡 Moderate Risk" if proba > 0.25 else "🟢 Low Risk")
        st.metric("Predicted Default Probability", f"{proba * 100:.1f}%")
        st.markdown(f"### {risk_label}")
        st.caption(f"Debt-to-income ratio: {debt_to_income:.2f}")

    with r2:
        shap_vals, base_value, feat_names = single_prediction_shap(active_model.pipeline, applicant)
        contrib_df = pd.DataFrame({"feature": feat_names, "impact": shap_vals}).sort_values(
            "impact", key=abs, ascending=True
        ).tail(10)
        fig = px.bar(
            contrib_df, x="impact", y="feature", orientation="h",
            title="What's driving THIS prediction",
            color="impact", color_continuous_scale=["#3B82F6", "#EF4444"],
            labels={"impact": "Impact on default risk", "feature": ""},
        )
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 Try it: push **credit utilization** near 1.0 and **late payments** "
        "up — watch risk climb. Then raise **credit history years** and drop "
        "**debt-to-income** — watch it fall. This is what makes the model "
        "trustworthy: every decision can be explained to a loan officer or "
        "an applicant, not just handed down as a number."
    )
