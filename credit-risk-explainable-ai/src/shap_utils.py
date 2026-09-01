"""
Helpers for generating explanations from a fitted sklearn Pipeline that
has a ColumnTransformer preprocessing step followed by a tree or linear
model.

SHAP's TreeExplainer/LinearExplainer rely on a compiled C extension
(`shap._cext`). On some hosting environments (e.g. a Python version the
shap package hasn't shipped a prebuilt wheel for yet), that extension
fails to import. Rather than let the whole app crash, every explanation
function here degrades gracefully to a pure-Python fallback:

- Global importance falls back to the model's own native feature
  importance (feature_importances_ for tree models, |coefficients| for
  linear models).
- Local (single-prediction) importance falls back to an occlusion-based
  sensitivity analysis: for each feature, we swap in a "typical" value
  (median for numeric, mode for categorical) and measure how much the
  predicted probability moves. This needs no compiled dependencies at
  all, so it always works.
"""

import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    shap = None
    _SHAP_IMPORT_ERROR = exc


class ExplainabilityBackend:
    """Tags which method actually produced an explanation, so the UI can
    tell the user honestly whether they're looking at SHAP or a fallback."""

    SHAP = "shap"
    NATIVE_IMPORTANCE = "native_importance"
    OCCLUSION = "occlusion"


def get_feature_names(pipeline) -> list:
    """Recover human-readable feature names after a ColumnTransformer."""
    preprocessor = pipeline.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())


def _is_tree_model(model) -> bool:
    return type(model).__name__ in ("RandomForestClassifier", "XGBClassifier")


# ---------------------------------------------------------------------------
# Global explanation (Explainability tab)
# ---------------------------------------------------------------------------

def compute_global_importance(pipeline, X: pd.DataFrame, max_samples: int = 250):
    """Return (backend, feature_names, importances).

    When SHAP works: importances is an (n_samples, n_features) array of
    signed SHAP values, so the caller can show both magnitude and spread.
    When falling back: importances is a single (n_features,) array of
    unsigned native importances, and backend == NATIVE_IMPORTANCE.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = get_feature_names(pipeline)

    if shap is not None:
        try:
            sample = X.sample(n=min(max_samples, len(X)), random_state=42)
            X_transformed = preprocessor.transform(sample)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            if _is_tree_model(model):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_transformed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                background = shap.sample(X_transformed, min(100, len(X_transformed)))
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(X_transformed)

            return ExplainabilityBackend.SHAP, feature_names, shap_values
        except Exception:
            pass  # fall through to native importance below

    # Fallback: model's own built-in importance (unsigned, dataset-level only)
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_[0]))
    else:
        importance = np.zeros(len(feature_names))

    return ExplainabilityBackend.NATIVE_IMPORTANCE, feature_names, importance


# ---------------------------------------------------------------------------
# Local explanation (What-If Simulator tab)
# ---------------------------------------------------------------------------

def compute_local_importance(pipeline, single_row: pd.DataFrame, background_df: pd.DataFrame,
                              raw_feature_cols: list, categorical_cols: list):
    """Return (backend, feature_names, signed_impacts) for one applicant.

    Tries SHAP first (transformed/one-hot feature space). Falls back to an
    occlusion analysis in the *raw* feature space (age, income, ... — the
    same names as the What-If sliders), which is arguably more readable
    for an end user anyway.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    if shap is not None:
        try:
            X_transformed = preprocessor.transform(single_row)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            feature_names = get_feature_names(pipeline)

            if _is_tree_model(model):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_transformed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                explainer = shap.LinearExplainer(model, X_transformed)
                shap_values = explainer.shap_values(X_transformed)

            return ExplainabilityBackend.SHAP, feature_names, shap_values[0]
        except Exception:
            pass  # fall through to occlusion analysis below

    # Fallback: occlusion / sensitivity analysis in raw feature space.
    baseline_proba = pipeline.predict_proba(single_row)[0, 1]
    impacts = []
    for col in raw_feature_cols:
        modified = single_row.copy()
        if col in categorical_cols:
            replacement = background_df[col].mode().iloc[0]
        else:
            replacement = background_df[col].median()
        modified[col] = replacement
        modified_proba = pipeline.predict_proba(modified)[0, 1]
        # Positive impact = this applicant's actual value raises risk
        # relative to a "typical" applicant's value for that feature.
        impacts.append(baseline_proba - modified_proba)

    return ExplainabilityBackend.OCCLUSION, raw_feature_cols, np.array(impacts)
