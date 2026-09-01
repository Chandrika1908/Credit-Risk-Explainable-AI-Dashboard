"""
Synthetic credit-risk dataset generator.

We generate data instead of pulling an external CSV so the app has
zero external dependencies at deploy time (no broken Kaggle links,
no rate-limited downloads). The relationships between features and
default risk are hand-designed to be realistic and explainable.
"""

import numpy as np
import pandas as pd


def generate_credit_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic but realistic loan-applicant dataset.

    Returns a DataFrame with applicant features and a binary
    'default' target (1 = defaulted on loan, 0 = repaid).
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 70, n_samples)
    annual_income = rng.normal(55000, 22000, n_samples).clip(12000, 250000)
    employment_years = rng.integers(0, 40, n_samples)
    credit_history_years = rng.integers(0, 35, n_samples)
    loan_amount = rng.normal(15000, 9000, n_samples).clip(1000, 60000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n_samples)
    existing_loans = rng.integers(0, 6, n_samples)
    num_credit_inquiries = rng.integers(0, 12, n_samples)
    late_payments_last_2yrs = rng.poisson(0.8, n_samples)
    credit_utilization = rng.uniform(0, 1, n_samples)
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], n_samples, p=[0.45, 0.4, 0.15])
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "medical", "auto", "business", "education"],
        n_samples,
    )

    debt_to_income = (loan_amount / loan_term_months * 12) / annual_income
    debt_to_income = debt_to_income.clip(0, 3)

    # Hand-designed risk score driving the true default probability.
    # Positive coefficients increase default risk.
    risk_score = (
        -0.035 * (annual_income / 1000)
        + 2.4 * debt_to_income
        + 0.55 * late_payments_last_2yrs
        + 1.8 * credit_utilization
        + 0.28 * existing_loans
        + 0.18 * num_credit_inquiries
        - 0.045 * credit_history_years
        - 0.02 * employment_years
        - 0.01 * age
        + rng.normal(0, 1.1, n_samples)  # noise
    )

    # Standardize the risk score, then shift the logit so the *base* default
    # rate lands near a realistic ~17% (real-world consumer credit portfolios
    # typically run 10-25%) rather than an accidental ~50/50 split.
    z = (risk_score - risk_score.mean()) / risk_score.std() - 1.55
    prob_default = 1 / (1 + np.exp(-z))
    default = (rng.uniform(0, 1, n_samples) < prob_default).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income.round(2),
            "employment_years": employment_years,
            "credit_history_years": credit_history_years,
            "loan_amount": loan_amount.round(2),
            "loan_term_months": loan_term_months,
            "existing_loans": existing_loans,
            "num_credit_inquiries": num_credit_inquiries,
            "late_payments_last_2yrs": late_payments_last_2yrs,
            "credit_utilization": credit_utilization.round(3),
            "debt_to_income": debt_to_income.round(3),
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "default": default,
        }
    )
    return df


if __name__ == "__main__":
    data = generate_credit_data()
    print(data.head())
    print("\nDefault rate:", data["default"].mean().round(3))
