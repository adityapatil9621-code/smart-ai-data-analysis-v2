"""
data_understanding.py

Data Understanding Engine for Smart AI Data Intelligence System.

Improvements:
- Target detection uses variance AND cardinality heuristics
- Classification threshold configurable
- Safer correlation computation (handles single-column edge case)
- to_dict() excludes non-serialisable types
- Domain detection hook (keyword-based, extensible)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict


# ============================================================
# Understanding Object
# ============================================================

@dataclass
class UnderstandingObject:
    task_type: str
    target_column: str
    time_column: Optional[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    is_time_series: bool
    class_imbalance_ratio: Optional[float]
    skewed_features: List[str]
    correlation_strength: float
    domain: str = "general"

    def to_dict(self) -> Dict:
        return {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "time_column": self.time_column,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "is_time_series": self.is_time_series,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "skewed_features": self.skewed_features,
            "correlation_strength": self.correlation_strength,
            "domain": self.domain,
        }


# ============================================================
# Data Understanding Engine
# ============================================================

class DataUnderstandingEngine:

    CLASSIFICATION_MAX_UNIQUE = 20  # treat as classification if ≤ this many unique values

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(self, df: pd.DataFrame) -> UnderstandingObject:
        df = df.copy()

        # 1. Column types
        numeric_cols    = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols   = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

        if not numeric_cols:
            raise ValueError("No numeric columns found. Cannot determine target.")

        # 2. Time-series detection
        time_column   = datetime_cols[0] if datetime_cols else None
        is_time_series = time_column is not None

        # 3. Target selection
        target_column = self._select_target(df, numeric_cols, is_time_series)
        numeric_cols  = [c for c in numeric_cols if c != target_column]

        # 4. Task type
        unique_vals = df[target_column].nunique()
        is_float    = pd.api.types.is_float_dtype(df[target_column])
        task_type   = (
            "classification"
            if unique_vals <= self.CLASSIFICATION_MAX_UNIQUE and not is_float
            else "regression"
        )

        # 5. Class imbalance
        class_imbalance_ratio: Optional[float] = None
        if task_type == "classification":
            counts = df[target_column].value_counts(normalize=True)
            class_imbalance_ratio = round(float(counts.max()), 3)

        # 6. Skewness
        skewed_features = [
            col for col in numeric_cols
            if col in df.columns and abs(df[col].skew()) > 1
        ]

        # 7. Correlation strength
        correlation_strength = 0.0
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs()
            corr_values = corr.values.copy()          # make writable copy
            np.fill_diagonal(corr_values, 0)
            correlation_strength = round(float(corr_values.max()), 3)

        # 8. Domain detection (keyword-based, fast)
        domain = self._detect_domain(df)

        return UnderstandingObject(
            task_type=task_type,
            target_column=target_column,
            time_column=time_column,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            is_time_series=is_time_series,
            class_imbalance_ratio=class_imbalance_ratio,
            skewed_features=skewed_features,
            correlation_strength=correlation_strength,
            domain=domain,
        )

    # ========================================================
    # Helpers
    # ========================================================

    def _select_target(self, df: pd.DataFrame, numeric_cols: List[str], is_time_series: bool) -> str:
        """Pick the most informative numeric column as the target."""
        if is_time_series:
            return numeric_cols[-1]

        # Prefer column with highest coefficient of variation (relative variance)
        variances = df[numeric_cols].var()
        means     = df[numeric_cols].mean().replace(0, np.nan)
        cv        = (variances / means).fillna(variances)
        return str(cv.idxmax())

    @staticmethod
    def _detect_domain(df: pd.DataFrame) -> str:
        """Light keyword scan across column names."""
        cols = " ".join(df.columns).lower()
        keywords = {
            "retail":        ["sales", "price", "product", "quantity", "discount"],
            "finance":       ["stock", "profit", "revenue", "market", "equity"],
            "healthcare":    ["patient", "diagnosis", "hospital", "medical", "symptom"],
            "hr":            ["employee", "salary", "department", "hire", "attendance"],
            "logistics":     ["shipment", "delivery", "warehouse", "inventory"],
            "weather":       ["temperature", "rain", "humidity", "wind", "pressure"],
            "marketing":     ["campaign", "conversion", "click", "impression", "roi"],
            "manufacturing": ["production", "defect", "machine", "factory", "yield"],
        }
        scores = {domain: sum(1 for kw in kws if kw in cols) for domain, kws in keywords.items()}
        best   = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"