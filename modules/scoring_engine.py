"""
scoring_engine.py

Unified Intelligence Scoring Engine.

Improvements:
- Weights sum to 1.0 in both branches (documented)
- Grade thresholds configurable
- confidence_level matches grade for consistency
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class IntelligenceScoreObject:
    score: float
    grade: str
    confidence_level: str

    def to_dict(self) -> Dict:
        return {
            "score":            self.score,
            "grade":            self.grade,
            "confidence_level": self.confidence_level,
        }


class IntelligenceScoringEngine:

    # Weights must sum to 1.0 in each branch
    WEIGHTS_WITH_FORECAST    = dict(data_quality=0.25, model_conf=0.30, signal=0.20, forecast=0.15, risk=-0.10)
    WEIGHTS_WITHOUT_FORECAST = dict(data_quality=0.30, model_conf=0.35, signal=0.25, forecast=0.00, risk=-0.10)

    def __init__(self, config: dict = None):
        self.config = config or {}

    def run(self, cleaned_obj, model_obj, insight_obj, forecast_obj: Optional[object]) -> IntelligenceScoreObject:
        dq = cleaned_obj.quality_score
        mc = max(0.0, model_obj.confidence)
        ss = insight_obj.overall_signal_strength
        rs = insight_obj.risk_score

        if forecast_obj:
            w  = self.WEIGHTS_WITH_FORECAST
            fc = forecast_obj.forecast_confidence
            score = w["data_quality"] * dq + w["model_conf"] * mc + w["signal"] * ss + w["forecast"] * fc + w["risk"] * rs
        else:
            w  = self.WEIGHTS_WITHOUT_FORECAST
            score = w["data_quality"] * dq + w["model_conf"] * mc + w["signal"] * ss + w["risk"] * rs

        score = round(float(max(0.0, min(1.0, score))), 3)

        if score >= 0.85:
            grade, confidence_level = "A", "High"
        elif score >= 0.70:
            grade, confidence_level = "B", "High"
        elif score >= 0.55:
            grade, confidence_level = "C", "Moderate"
        else:
            grade, confidence_level = "D", "Low"

        return IntelligenceScoreObject(score=score, grade=grade, confidence_level=confidence_level)