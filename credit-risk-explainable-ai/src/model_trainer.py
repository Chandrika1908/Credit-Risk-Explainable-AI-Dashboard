"""
Training pipeline for the credit-risk models.

Trains and evaluates three model families so the app can show a
head-to-head comparison, not just a single black-box score.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "credit_history_years",
    "loan_amount",
    "loan_term_months",
    "existing_loans",
    "num_credit_inquiries",
    "late_payments_last_2yrs",
    "credit_utilization",
    "debt_to_income",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    metrics: dict
    fpr: np.ndarray
    tpr: np.ndarray
    X_test: pd.DataFrame
    y_test: pd.Series


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _model_zoo(random_state: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def train_all_models(df: pd.DataFrame, random_state: int = 42) -> dict:
    """Train every model in the zoo and return a dict of TrainedModel."""
    X = df[ALL_FEATURES]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    results = {}
    for name, estimator in _model_zoo(random_state).items():
        pipeline = Pipeline(
            steps=[("preprocess", _build_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, y_proba),
        }

        results[name] = TrainedModel(
            name=name,
            pipeline=pipeline,
            metrics=metrics,
            fpr=fpr,
            tpr=tpr,
            X_test=X_test,
            y_test=y_test,
        )

    return results
