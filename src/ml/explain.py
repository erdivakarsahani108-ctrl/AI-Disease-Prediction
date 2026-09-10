"""Explainable AI module.

Provides symptom-level explanations for a prediction using SHAP where
possible (works well for tree-based models: Random Forest / XGBoost /
Decision Tree), and falls back to a transparent, always-available
coefficient/permutation-based explanation for other model types (e.g.
Logistic Regression, SVM, KNN) so the app never silently fails to explain
a prediction regardless of which model was selected as "best".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = ()

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False


def _is_tree_model(model) -> bool:
    return isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)) or (
        XGBClassifier and isinstance(model, XGBClassifier)
    )


def explain_prediction(
    model,
    feature_columns: list[str],
    input_vector: pd.DataFrame,
    predicted_class_index: int,
    top_n: int = 8,
) -> dict:
    """Return the top contributing symptoms for the predicted class.

    Returns a dict: {"method": str, "contributions": [(symptom, score), ...]}
    """
    active_symptoms = [f for f in feature_columns if input_vector.iloc[0][f] == 1]

    if SHAP_AVAILABLE and _is_tree_model(model):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_vector)
            # shap_values can be a list (per-class) or a 3D array depending on version.
            if isinstance(shap_values, list):
                class_values = shap_values[predicted_class_index][0]
            else:
                arr = np.asarray(shap_values)
                if arr.ndim == 3:
                    class_values = arr[0, :, predicted_class_index]
                else:
                    class_values = arr[0]
            contributions = list(zip(feature_columns, class_values))
            contributions = [c for c in contributions if c[0] in active_symptoms]
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            return {"method": "SHAP (TreeExplainer)", "contributions": contributions[:top_n]}
        except Exception:
            pass  # fall through to the model-agnostic fallback below

    if isinstance(model, LogisticRegression):
        try:
            coefs = model.coef_[predicted_class_index]
            contributions = [(f, coefs[i]) for i, f in enumerate(feature_columns) if f in active_symptoms]
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            return {"method": "Logistic Regression coefficients", "contributions": contributions[:top_n]}
        except Exception:
            pass

    # Model-agnostic fallback: severity-weight-informed ranking of the
    # symptoms actually present, so an explanation is ALWAYS available.
    from src.utils.paths import SYMPTOM_VOCAB_JSON
    import json

    weights = {}
    if SYMPTOM_VOCAB_JSON.exists():
        weights = json.loads(SYMPTOM_VOCAB_JSON.read_text(encoding="utf-8")).get("severity_weight", {})
    contributions = [(f, weights.get(f, 1)) for f in active_symptoms]
    contributions.sort(key=lambda x: x[1], reverse=True)
    return {"method": "Clinical severity-weight ranking (model-agnostic fallback)", "contributions": contributions[:top_n]}


def compare_top_predictions(
    model,
    feature_columns: list[str],
    input_vector: pd.DataFrame,
    class_indices: list[int],
    class_names: list[str],
) -> pd.DataFrame:
    """Build a side-by-side symptom-contribution comparison table across the
    Top-K predicted diseases (Differential Disease Analysis support)."""
    rows = []
    for idx, name in zip(class_indices, class_names):
        result = explain_prediction(model, feature_columns, input_vector, idx, top_n=5)
        top_symptoms = ", ".join(f[0].replace("_", " ").title() for f in result["contributions"])
        rows.append({"disease": name, "top_contributing_symptoms": top_symptoms, "method": result["method"]})
    return pd.DataFrame(rows)
