"""
suggestion_engine.py

Strategic Recommendation Engine.

Improvements:
- Accepts plain dicts OR dataclass instances (flexible)
- Deduplicates recommendations
- Default lists so StrategicObject never has empty required fields
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


def _get(obj: Any, key: str, default=None):
    """Works with both dict and dataclass-style objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class StrategicObject:
    growth_opportunities:      List[str] = field(default_factory=list)
    risk_mitigation_actions:   List[str] = field(default_factory=list)
    stability_recommendations: List[str] = field(default_factory=list)
    confidence_advisory:       str = ""
    priority_level:            str = "Moderate"
    human_oversight_note:      str = (
        "These recommendations are derived from historical data patterns and model-based analysis. "
        "Final strategic decisions should incorporate external factors and human expertise."
    )

    def to_dict(self) -> Dict:
        return self.__dict__


class SuggestionEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}

    def run(self, insight_obj, forecast_obj: Optional[Any], score_obj) -> StrategicObject:
        growth, risk_actions, stability = [], [], []

        signal_strength = _get(insight_obj, "overall_signal_strength", 0.5)
        risk_score      = _get(insight_obj, "risk_score", 0.0)
        pos_drivers     = _get(insight_obj, "top_positive_drivers", [])
        neg_drivers     = _get(insight_obj, "top_negative_drivers", [])

        # 1. Driver-based recommendations
        for driver in pos_drivers:
            if _get(driver, "impact", 0) > 0.2:
                growth.append(f"Leverage '{driver['feature']}' — strong positive influence detected.")

        for driver in neg_drivers:
            if _get(driver, "impact", 0) > 0.15:
                risk_actions.append(f"Review '{driver['feature']}' to reduce its negative impact.")

        # 2. Forecast-based recommendations
        if forecast_obj:
            trend      = _get(forecast_obj, "trend_direction", "Stable")
            volatility = _get(forecast_obj, "volatility_score", 0.0)

            if trend == "Upward":
                growth.append("Forecast shows an upward trend — expansion strategies may be considered.")
            elif trend == "Downward":
                risk_actions.append("Forecast indicates a downward movement — proactive risk containment advised.")

            if volatility > 0.5:
                risk_actions.append("High forecast volatility — implement monitoring mechanisms.")
            elif volatility < 0.2:
                stability.append("Forecast is stable — optimisation strategies can be explored.")

        # 3. Signal strength advisory
        if signal_strength < 0.5:
            stability.append("Moderate model signal — interpret findings with analytical caution.")
        elif signal_strength > 0.8:
            stability.append("Strong analytical signal — insights appear robust.")

        # 4. Risk advisory
        if risk_score > 0.6:
            risk_actions.append("Elevated analytical risk — further data validation is recommended.")

        # 5. Ensure non-empty lists
        if not growth:
            growth = ["No strong positive drivers identified — explore additional features or data sources."]
        if not risk_actions:
            risk_actions = ["No critical risks identified at this time."]
        if not stability:
            stability = ["Continue monitoring key metrics for emerging trends."]

        # 6. Priority & confidence
        priority        = "High" if risk_score > 0.6 else ("Medium" if signal_strength > 0.75 else "Moderate")
        conf_level      = _get(score_obj, "confidence_level", "Moderate")
        confidence_note = {
            "High":     "Overall analytical confidence is high.",
            "Moderate": "Analytical confidence is moderate — validate key findings.",
        }.get(conf_level, "Analytical confidence is limited — cautious interpretation advised.")

        return StrategicObject(
            growth_opportunities=list(dict.fromkeys(growth)),          # deduplicate
            risk_mitigation_actions=list(dict.fromkeys(risk_actions)),
            stability_recommendations=list(dict.fromkeys(stability)),
            confidence_advisory=confidence_note,
            priority_level=priority,
        )