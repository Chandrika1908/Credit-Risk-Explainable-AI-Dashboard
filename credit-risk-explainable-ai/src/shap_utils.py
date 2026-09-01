"""
Helpers for generating SHAP explanations from a fitted sklearn Pipeline
that has a ColumnTransformer preprocessing step followed by a tree or
linear model.
"""

import numpy as np
import pandas as pd
import shap


def get_feature_names(pipeline) -> list:
    """Recover human-readable feature names after a ColumnTransformer."""
    preprocessor = pipeline.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())


def compute_shap_values(pipeline, X: pd.DataFrame, max_samples: int = 300):
    """Compute SHAP values for a sample of X (capped for speed on free-tier CPU).

    Returns (explainer, shap_values, X_transformed_sample, feature_names).
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    sample = X.sample(n=min(max_samples, len(X)), random_state=42)
    X_transformed = preprocessor.transform(sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = get_feature_names(pipeline)

    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        # TreeExplainer on binary classifiers may return a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        background = shap.sample(X_transformed, min(100, len(X_transformed)))
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_transformed)

    return explainer, shap_values, X_transformed, feature_names


def single_prediction_shap(pipeline, single_row: pd.DataFrame):
    """SHAP values for exactly one applicant row (used by the what-if simulator)."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    X_transformed = preprocessor.transform(single_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = get_feature_names(pipeline)
    model_type = type(model).__name__

    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        base_value = explainer.expected_value

    return shap_values[0], base_value, feature_names
