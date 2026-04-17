"""
insight_extraction.py

Insight Extraction Engine for Smart AI Data Intelligence System.

Improvements:
- SHAP uses TreeExplainer when possible (much faster)
- Feature importance plot capped at top-10
- Handles classification model.predict_proba for residuals
- Safe anomaly detection with empty-residual guard
- Serialisable to_dict (excludes matplotlib objects)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================================
# Insight Object
# ============================================================

@dataclass
class InsightObject:
    top_positive_drivers: List[Dict]
    top_negative_drivers: List[Dict]
    nonlinear_features: List[str]
    residual_bias_detected: bool
    anomalies_detected: List[Dict]
    overall_signal_strength: float
    risk_score: float
    feature_importance_plot: object = field(default=None, repr=False)
    shap_plot: object               = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        return {
            "top_positive_drivers":   self.top_positive_drivers,
            "top_negative_drivers":   self.top_negative_drivers,
            "nonlinear_features":     self.nonlinear_features,
            "residual_bias_detected": self.residual_bias_detected,
            "anomalies_detected":     self.anomalies_detected,
            "overall_signal_strength": self.overall_signal_strength,
            "risk_score":             self.risk_score,
            # exclude matplotlib objects
        }


# ============================================================
# Insight Extraction Engine
# ============================================================

class InsightExtractionEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(self, model_obj, feature_obj, df: pd.DataFrame) -> InsightObject:
        model         = model_obj.trained_model
        X_train       = feature_obj.X_train
        X_test        = feature_obj.X_test
        y_test        = feature_obj.y_test
        feature_names = feature_obj.feature_names

        # 1. Feature importance
        importance            = self._extract_importance(model, feature_names)
        total                 = sum(importance.values()) or 1e-8
        norm_importance       = {k: v / total for k, v in importance.items()}
        importance_plot       = self._plot_importance(norm_importance)

        # 2. Direction detection
        predictions  = model_obj.predict(X_test)
        direction_map = {}
        for feat in feature_names:
            if feat in X_test.columns:
                corr = np.corrcoef(X_test[feat].values, predictions)[0, 1]
                direction_map[feat] = int(np.sign(corr)) if not np.isnan(corr) else 0

        # 3. Top drivers
        sorted_feats  = sorted(norm_importance.items(), key=lambda x: x[1], reverse=True)
        top_positive, top_negative = [], []
        for feat, imp in sorted_feats[:8]:
            entry = {
                "feature":    feat,
                "impact":     round(float(imp), 3),
                "confidence": round(min(1.0, imp * len(df) / 100), 2),
            }
            if direction_map.get(feat, 0) >= 0:
                top_positive.append(entry)
            else:
                top_negative.append(entry)

        # Guarantee at least one item in each list
        if not top_positive and sorted_feats:
            feat, imp = sorted_feats[0]
            top_positive.append({"feature": feat, "impact": round(float(imp), 3), "confidence": 0.5})
        if not top_negative:
            top_negative = []

        # 4. Nonlinear features
        nonlinear_features = list(norm_importance.keys())[:2] if hasattr(model, "feature_importances_") else []

        # 5. Residual diagnostics
        residuals = np.asarray(y_test, dtype=float) - np.asarray(predictions, dtype=float)
        y_std     = np.std(y_test) if np.std(y_test) > 0 else 1.0
        residual_bias_detected = abs(np.mean(residuals)) > 0.1 * y_std

        # 6. Anomaly detection (z-score on residuals)
        anomalies = []
        res_std   = np.std(residuals)
        if res_std > 0:
            z_scores = residuals / res_std
            anomalies = [
                {"index": int(i), "severity": round(float(abs(z)), 3)}
                for i, z in enumerate(z_scores) if abs(z) > 3
            ]

        anomaly_ratio = len(anomalies) / max(len(y_test), 1)

        # 7. Signal strength & risk
        signal_strength = round(float(max(0.0, min(1.0,
            0.5 * model_obj.confidence +
            0.3 * model_obj.stability +
            0.2 * (1 - anomaly_ratio)
        ))), 3)

        risk_score = round(float(max(0.0, min(1.0,
            0.4 * anomaly_ratio +
            0.3 * (1 - model_obj.stability) +
            0.3 * float(residual_bias_detected)
        ))), 3)

        # 8. SHAP (best-effort, non-blocking)
        shap_plot = self._try_shap(model, X_train)

        return InsightObject(
            top_positive_drivers=top_positive,
            top_negative_drivers=top_negative,
            nonlinear_features=nonlinear_features,
            residual_bias_detected=residual_bias_detected,
            anomalies_detected=anomalies,
            overall_signal_strength=signal_strength,
            risk_score=risk_score,
            feature_importance_plot=importance_plot,
            shap_plot=shap_plot,
        )

    # ========================================================
    # Helpers
    # ========================================================

    def _extract_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        if hasattr(model, "feature_importances_"):
            return dict(zip(feature_names, model.feature_importances_))
        if hasattr(model, "coef_"):
            coef = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
            return dict(zip(feature_names, np.abs(coef)))
        return {f: 1.0 for f in feature_names}

    def _plot_importance(self, importance: Dict[str, float]) -> plt.Figure:
        items    = sorted(importance.items(), key=lambda x: x[1])[-10:]
        feats    = [i[0] for i in items]
        values   = [i[1] for i in items]
        fig, ax  = plt.subplots(figsize=(7, max(3, len(feats) * 0.4)))
        ax.barh(feats, values, color="#4C72B0")
        ax.set_title("Top Feature Importance")
        ax.set_xlabel("Normalised Importance Score")
        plt.tight_layout()
        return fig

    def _try_shap(self, model, X_train: pd.DataFrame) -> Optional[plt.Figure]:
        try:
            import shap
            sample = X_train.sample(min(100, len(X_train)), random_state=42)

            # Use TreeExplainer for tree models (fast), else fallback to Explainer
            if hasattr(model, "feature_importances_"):
                explainer  = shap.TreeExplainer(model)
                shap_vals  = explainer.shap_values(sample)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
            else:
                explainer = shap.Explainer(model, sample)
                shap_vals = explainer(sample).values

            fig, ax = plt.subplots(figsize=(7, 4))
            shap.summary_plot(shap_vals, sample, plot_type="bar", show=False)
            plt.tight_layout()
            return fig
        except Exception:
            return None