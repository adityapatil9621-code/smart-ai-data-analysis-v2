"""
model_training.py

Model Training Engine for Smart AI Data Intelligence System.

Changes in this version:
- Added XGBoost (regression + classification)
- Added LightGBM (regression + classification)
- Added ElasticNet replacing Ridge (L1+L2, strictly more general)
- Added SVC with RBF kernel for classification
- Dynamic threshold ensemble: only models scoring above MIN_SCORE join
  so a weak model never dilutes the ensemble
- Weighted ensemble: each model contributes proportional to its CV mean score
- Graceful optional import — works without xgboost/lightgbm installed

Install deps: pip install xgboost lightgbm
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.impute import SimpleImputer

# ── Optional heavy deps (graceful fallback if not installed) ──────────────────
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
# ─────────────────────────────────────────────────────────────────────────────

# Minimum CV score a model must reach to join the ensemble
MIN_REGRESSION_SCORE     = 0.10   # R²
MIN_CLASSIFICATION_SCORE = 0.40   # accuracy


# ============================================================
# Model Object
# ============================================================

@dataclass
class ModelObject:
    selected_model:    str
    confidence:        float
    stability:         float
    trained_model:     object                              # best model (SHAP / importance)
    feature_names:     List[str]
    task_type:         str = "regression"
    ensemble_models:   List[Tuple] = field(default_factory=list, repr=False)  # (model, weight)
    regression_details: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "selected_model": self.selected_model,
            "confidence":     self.confidence,
            "stability":      self.stability,
            "task_type":      self.task_type,
        }

    def predict(self, X) -> np.ndarray:
        """Weighted ensemble prediction across all qualifying models."""
        if not self.ensemble_models:
            return self.trained_model.predict(X)

        total_weight = sum(w for _, w in self.ensemble_models)
        if total_weight == 0:
            return self.trained_model.predict(X)

        weighted_sum = np.zeros(len(X))
        for model, weight in self.ensemble_models:
            weighted_sum += model.predict(X) * weight

        return weighted_sum / total_weight


# ============================================================
# Model Training Engine
# ============================================================

class ModelTrainingEngine:

    def __init__(self, config: Dict = None):
        self.config       = config or {}
        self.random_state = self.config.get("random_state", 42)

    # ──────────────────────────────────────────────────────────
    # Candidate model pools
    # ──────────────────────────────────────────────────────────

    def _get_candidate_models(self, task_type: str) -> Dict:
        rs = self.random_state

        if task_type == "regression":
            models = {
                "ElasticNet":        ElasticNet(max_iter=2000),
                "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=rs, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(random_state=rs),
            }
            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=rs,
                    verbosity=0,
                    n_jobs=-1,
                )
            if LIGHTGBM_AVAILABLE:
                models["LightGBM"] = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=rs,
                    verbosity=-1,
                    n_jobs=-1,
                )
            return models

        # ── Classification ────────────────────────────────────
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=rs, n_jobs=-1),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingClassifier(random_state=rs),
            "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=rs),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=rs,
                verbosity=0,
                n_jobs=-1,
            )
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=rs,
                verbosity=-1,
                n_jobs=-1,
            )
        return models

    # ──────────────────────────────────────────────────────────
    # Main run
    # ──────────────────────────────────────────────────────────

    def run(self, feature_obj, task_type: str = "regression") -> ModelObject:
        X_train       = feature_obj.X_train
        X_test        = feature_obj.X_test
        y_train       = feature_obj.y_train
        y_test        = feature_obj.y_test
        feature_names = feature_obj.feature_names

        # ── Sanitise ──────────────────────────────────────────
        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)
        X_test  = pd.DataFrame(imputer.transform(X_test),      columns=feature_names)
        y_train = np.nan_to_num(np.asarray(y_train, dtype=float))
        y_test  = np.nan_to_num(np.asarray(y_test,  dtype=float))

        scoring   = "r2" if task_type == "regression" else "accuracy"
        min_score = MIN_REGRESSION_SCORE if task_type == "regression" else MIN_CLASSIFICATION_SCORE
        models    = self._get_candidate_models(task_type)
        results   = {}

        # ── Cross-validation ──────────────────────────────────
        for name, model in models.items():
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5, scoring=scoring, n_jobs=-1,
                )
                results[name] = {
                    "mean":  float(np.mean(cv_scores)),
                    "std":   float(np.std(cv_scores)),
                    "model": model,
                }
            except Exception as e:
                print(f"[ModelTraining] {name} CV failed: {e}")
                results[name] = {"mean": -999.0, "std": 1.0, "model": model}

        # ── Rank: rewards accuracy AND stability ──────────────
        ranked = sorted(
            results.items(),
            key=lambda x: x[1]["mean"] - x[1]["std"],
            reverse=True,
        )
        best_name, best_data = ranked[0]

        # ── Dynamic threshold ensemble ────────────────────────
        qualifying = [
            (name, data) for name, data in ranked
            if data["mean"] >= min_score
        ]
        if not qualifying:
            qualifying = [ranked[0]]   # always keep at least the best

        # ── Fit qualifying models & assign weights ────────────
        ensemble_models: List[Tuple] = []
        for name, data in qualifying:
            try:
                data["model"].fit(X_train, y_train)
                weight = max(0.0, data["mean"])
                ensemble_models.append((data["model"], weight))
            except Exception as e:
                print(f"[ModelTraining] {name} final fit failed: {e}")

        if not ensemble_models:
            raise RuntimeError("All models failed to fit. Check your dataset.")

        # ── Weighted ensemble prediction ──────────────────────
        total_w = sum(w for _, w in ensemble_models) or 1.0
        y_pred  = sum(m.predict(X_test) * w for m, w in ensemble_models) / total_w

        # ── Metrics ───────────────────────────────────────────
        if task_type == "regression":
            test_score = float(r2_score(y_test, y_pred))
        else:
            test_score = float(accuracy_score(y_test, np.round(y_pred).astype(int)))

        confidence = round(max(0.0, best_data["mean"] - best_data["std"]), 4)
        stability  = round(max(0.0, 1.0 - best_data["std"]), 4)

        # ── Label ─────────────────────────────────────────────
        contributing = [name for name, _ in qualifying]
        if len(contributing) == 1:
            ensemble_label = contributing[0]
        else:
            ensemble_label = f"Ensemble ({', '.join(contributing)})"

        # ── Linear interpretability (ElasticNet only) ─────────
        regression_details = None
        if task_type == "regression" and "ElasticNet" in best_name:
            regression_details = {
                "r2_score":     round(test_score, 4),
                "coefficients": dict(zip(feature_names, best_data["model"].coef_)),
            }

        # Log summary
        score_summary = {n: round(d["mean"], 3) for n, d in results.items()}
        print(f"[ModelTraining] CV scores:  {score_summary}")
        print(f"[ModelTraining] Ensemble ({len(contributing)} models): {contributing}")
        print(f"[ModelTraining] Test {scoring}: {round(test_score, 4)}")

        return ModelObject(
            selected_model=ensemble_label,
            confidence=confidence,
            stability=stability,
            trained_model=ensemble_models[0][0],   # best model for SHAP
            ensemble_models=ensemble_models,
            feature_names=feature_names,
            task_type=task_type,
            regression_details=regression_details,
        )