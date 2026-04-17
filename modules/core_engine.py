"""
core_engine.py

Central Orchestrator for Smart AI Data Intelligence System.

Improvements:
- task_type flows from UnderstandingObject → ModelTrainingEngine
- safe_step provides structured error messages
- System memory consolidated (no duplication)
- cleaned_df included in returned memory for downstream use
"""

import pandas as pd
from typing import Dict, Any

from data_cleaning       import DataCleaningEngine
from data_understanding  import DataUnderstandingEngine
from visual_intelligence import VisualIntelligenceEngine
from feature_engineering import FeatureEngineeringEngine
from model_training      import ModelTrainingEngine
from insight_extraction  import InsightExtractionEngine
from forecasting_engine  import ForecastEngine
from scoring_engine      import IntelligenceScoringEngine


# ============================================================
# Core Engine
# ============================================================

class SmartAIEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.cleaning_engine   = DataCleaningEngine(config)
        self.understanding_engine = DataUnderstandingEngine(config)
        self.visual_engine     = VisualIntelligenceEngine(config)
        self.feature_engine    = FeatureEngineeringEngine(config)
        self.model_engine      = ModelTrainingEngine(config)
        self.insight_engine    = InsightExtractionEngine(config)
        self.forecast_engine   = ForecastEngine(config)
        self.scoring_engine    = IntelligenceScoringEngine(config)

    # ========================================================
    # Safe Step Wrapper
    # ========================================================

    @staticmethod
    def safe_step(step_name: str, func, *args):
        try:
            return func(*args)
        except Exception as e:
            raise RuntimeError(f"[{step_name}] {type(e).__name__}: {e}") from e

    # ========================================================
    # MASTER PIPELINE
    # ========================================================

    def run_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("DataFrame is empty.")

        # 1. Cleaning
        cleaned_obj = self.safe_step("Data Cleaning", self.cleaning_engine.run, df)
        if cleaned_obj.quality_score < 0.4:
            raise ValueError(
                f"Data quality too low ({cleaned_obj.quality_score:.2f}). "
                "Please improve the dataset before analysis."
            )
        cleaned_df = cleaned_obj.cleaned_df

        # 2. Understanding
        understanding_obj = self.safe_step("Data Understanding", self.understanding_engine.run, cleaned_df)

        # 3. Visuals
        visual_obj = self.safe_step("Visual Intelligence", self.visual_engine.run, cleaned_df, understanding_obj)

        # 4. Feature Engineering
        feature_obj = self.safe_step("Feature Engineering", self.feature_engine.run, cleaned_df, understanding_obj)

        # 5. Model Training (pass task_type)
        model_obj = self.safe_step(
            "Model Training",
            self.model_engine.run,
            feature_obj,
            understanding_obj.task_type,
        )

        # 6. Insights
        insight_obj = self.safe_step("Insight Extraction", self.insight_engine.run, model_obj, feature_obj, cleaned_df)

        # 7. Forecast (time-series only)
        forecast_obj = None
        if understanding_obj.is_time_series:
            forecast_obj = self.safe_step("Forecasting", self.forecast_engine.run, cleaned_df, understanding_obj)

        # 8. Intelligence Score
        score_obj = self.safe_step(
            "Scoring",
            self.scoring_engine.run,
            cleaned_obj, model_obj, insight_obj, forecast_obj,
        )

        # 9. Build & return memory
        return self._build_memory(cleaned_obj, understanding_obj, visual_obj, model_obj, insight_obj, forecast_obj, score_obj)

    # ========================================================
    # Memory Builder
    # ========================================================

    def _build_memory(self, cleaned_obj, understanding_obj, visual_obj, model_obj, insight_obj, forecast_obj, score_obj) -> Dict[str, Any]:
        return {
            "metadata": {
                "rows":          cleaned_obj.cleaned_df.shape[0],
                "columns":       cleaned_obj.cleaned_df.shape[1],
                "quality_score": cleaned_obj.quality_score,
                "domain":        understanding_obj.domain,
            },
            "cleaned_df":           cleaned_obj.cleaned_df,
            "data_profile":         understanding_obj.to_dict(),
            "visual_intelligence":  visual_obj.to_dict(),
            "model_intelligence":   model_obj.to_dict(),
            "insight_intelligence": insight_obj.to_dict(),
            "forecast_intelligence": forecast_obj.to_dict() if forecast_obj else None,
            "intelligence_score":   score_obj.to_dict(),
            "audit_log": [
                f"[CLEANING]  Quality Score     = {cleaned_obj.quality_score}",
                f"[DOMAIN]    Detected Domain   = {understanding_obj.domain}",
                f"[TASK]      Task Type         = {understanding_obj.task_type}",
                f"[MODEL]     Selected Model    = {model_obj.selected_model}",
                f"[FORECAST]  Time-Series       = {understanding_obj.is_time_series}",
                f"[SCORE]     Intelligence Score = {score_obj.score}",
            ],
        }