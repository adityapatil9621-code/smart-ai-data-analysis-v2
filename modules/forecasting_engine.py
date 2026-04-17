"""
forecasting_engine.py

Tree-Based Lag Forecasting Engine for Smart AI Data Intelligence System.

Improvements:
- Configurable lags & smoothing window
- Drift-corrected trend detection (normalized slope)
- Bootstrap skips failed fits gracefully
- Returns target_column name so UI can label correctly
- to_dict() always serialisable (no numpy types)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.ensemble import GradientBoostingRegressor


# ============================================================
# Forecast Object
# ============================================================

@dataclass
class ForecastObject:
    forecast_horizon: int
    forecast_values: List[float]
    confidence_band: Dict[str, List[float]]
    trend_direction: str
    volatility_score: float
    forecast_confidence: float
    target_column: str = ""

    def to_dict(self) -> Dict:
        return {
            "forecast_horizon":  self.forecast_horizon,
            "forecast_values":   [round(float(v), 4) for v in self.forecast_values],
            "confidence_band": {
                "lower": [round(float(v), 4) for v in self.confidence_band["lower"]],
                "upper": [round(float(v), 4) for v in self.confidence_band["upper"]],
            },
            "trend_direction":    self.trend_direction,
            "volatility_score":   round(float(self.volatility_score), 4),
            "forecast_confidence": round(float(self.forecast_confidence), 4),
            "target_column":      self.target_column,
        }


# ============================================================
# Forecast Engine
# ============================================================

class ForecastEngine:

    def __init__(self, config: dict = None):
        self.config               = config or {}
        self.forecast_horizon     = self.config.get("forecast_horizon", 6)
        self.bootstrap_iterations = self.config.get("bootstrap_iterations", 20)
        self.random_state         = self.config.get("random_state", 42)
        self.lags                 = self.config.get("lags", [1, 2, 3, 6])
        self.smoothing_window     = self.config.get("smoothing_window", 3)

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(self, df: pd.DataFrame, understanding_obj=None) -> Optional[ForecastObject]:
        try:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                return None

            # Sort by time if available
            if understanding_obj and understanding_obj.time_column:
                df = df.sort_values(by=understanding_obj.time_column)

            target_col = (
                understanding_obj.target_column
                if understanding_obj and understanding_obj.target_column
                else numeric_cols[-1]
            )

            raw_series = df[target_col].dropna().reset_index(drop=True).values
            if len(raw_series) < max(self.lags) + 10:
                return None

            # Apply rolling smoothing
            series = pd.Series(raw_series).rolling(window=self.smoothing_window, min_periods=1).mean().values

            # Train model & forecast
            lag_df  = self._create_lag_features(series)
            X, y    = lag_df.drop(columns=["target"]), lag_df["target"]
            model   = GradientBoostingRegressor(random_state=self.random_state)
            model.fit(X, y)

            forecast_values = self._recursive_forecast(model, series, self.forecast_horizon)
            lower, upper    = self._bootstrap_confidence(series, self.forecast_horizon)

            # Trend direction (normalized slope)
            slope            = np.polyfit(range(len(series)), series, 1)[0]
            norm_slope       = slope / (np.mean(np.abs(series)) + 1e-8)
            if norm_slope > 0.01:
                trend = "Upward"
            elif norm_slope < -0.01:
                trend = "Downward"
            else:
                trend = "Stable"

            volatility         = float(np.std(np.diff(series)) / (np.mean(np.abs(series)) + 1e-8))
            forecast_confidence = float(max(0.2, min(0.95, 1.0 - volatility)))

            return ForecastObject(
                forecast_horizon=self.forecast_horizon,
                forecast_values=forecast_values,
                confidence_band={"lower": lower.tolist(), "upper": upper.tolist()},
                trend_direction=trend,
                volatility_score=volatility,
                forecast_confidence=forecast_confidence,
                target_column=target_col,
            )

        except Exception as e:
            print(f"[ForecastEngine] Warning: {e}")
            return None

    # ========================================================
    # Lag Feature Creation
    # ========================================================

    def _create_lag_features(self, series: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"target": series})
        for lag in self.lags:
            df[f"lag_{lag}"] = df["target"].shift(lag)
        df.dropna(inplace=True)
        return df

    # ========================================================
    # Recursive Forecast
    # ========================================================

    def _recursive_forecast(self, model, series: np.ndarray, horizon: int) -> List[float]:
        history  = list(series[-max(self.lags):])
        forecast = []
        for _ in range(horizon):
            features = pd.DataFrame(
                [[history[-lag] for lag in self.lags]],
                columns=[f"lag_{lag}" for lag in self.lags],
            )
            pred = float(model.predict(features)[0])
            forecast.append(pred)
            history.append(pred)
        return forecast

    # ========================================================
    # Bootstrap Confidence Band
    # ========================================================

    def _bootstrap_confidence(self, series: np.ndarray, horizon: int):
        forecasts = []
        for _ in range(self.bootstrap_iterations):
            try:
                sample  = np.random.choice(series, size=len(series), replace=True)
                lag_df  = self._create_lag_features(sample)
                Xb, yb  = lag_df.drop(columns=["target"]), lag_df["target"]
                m       = GradientBoostingRegressor()
                m.fit(Xb, yb)
                forecasts.append(self._recursive_forecast(m, sample, horizon))
            except Exception:
                continue

        if not forecasts:
            zeros = np.zeros(horizon)
            return zeros, zeros

        arr   = np.array(forecasts)
        return np.percentile(arr, 5, axis=0), np.percentile(arr, 95, axis=0)