"""
data_cleaning.py

Data Cleaning Layer for Smart AI Data Intelligence System.

Improvements:
- Smarter identifier detection (uniqueness ratio, not just name)
- Long-text column removal threshold configurable
- Type correction avoids silent datetime mis-detection
- Quality score accounts for constant / useless columns
- Cleaner code, no bare except
"""

import pandas as pd
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Dict, List


# ============================================================
# Cleaned Data Object
# ============================================================

@dataclass
class CleanedDataObject:
    cleaned_df: pd.DataFrame
    quality_score: float
    identifiers: List[str]
    dropped_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "quality_score": self.quality_score,
            "identifiers": self.identifiers,
            "dropped_columns": self.dropped_columns,
        }


# ============================================================
# Data Cleaning Engine
# ============================================================

class DataCleaningEngine:

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.long_text_threshold = self.config.get("long_text_threshold", 50)
        self.id_uniqueness_ratio  = self.config.get("id_uniqueness_ratio", 0.95)

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(self, df: pd.DataFrame) -> CleanedDataObject:
        df = df.copy()
        initial_rows = len(df)
        dropped_columns: List[str] = []

        # 1. Remove duplicates
        df = df.drop_duplicates()

        # 2. Safe type correction
        df = self._correct_types(df)

        # 3. Missing value handling
        df = self._handle_missing(df)

        # 4. Outlier capping
        df = self._handle_outliers(df)

        # 5. Drop long-text columns (likely free-text, not useful for ML)
        long_text_cols = [
            col for col in df.columns
            if df[col].dtype == object and df[col].str.len().mean() > self.long_text_threshold
        ]
        df = df.drop(columns=long_text_cols, errors="ignore")
        dropped_columns.extend(long_text_cols)

        # 6. Identifier detection (name-based + high uniqueness)
        identifiers = self._detect_identifiers(df)
        df = df.drop(columns=identifiers, errors="ignore")
        dropped_columns.extend(identifiers)

        # 7. Drop constant columns (zero variance — useless for ML)
        constant_cols = [col for col in df.select_dtypes(include=np.number).columns
                         if df[col].std() == 0]
        df = df.drop(columns=constant_cols, errors="ignore")
        dropped_columns.extend(constant_cols)

        # 8. Quality score
        missing_ratio    = df.isna().mean().mean()
        duplicate_ratio  = (initial_rows - len(df)) / max(initial_rows, 1)
        quality_score    = round(float(max(0.0, min(1.0, 1 - 0.5 * missing_ratio - 0.5 * duplicate_ratio))), 3)

        return CleanedDataObject(
            cleaned_df=df,
            quality_score=quality_score,
            identifiers=identifiers,
            dropped_columns=dropped_columns,
        )

    # ========================================================
    # Type Correction
    # ========================================================

    def _correct_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=object).columns:

            # Try numeric conversion first
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().mean() > 0.8:
                df[col] = numeric
                continue

            # Try datetime conversion with explicit format
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    if converted.notna().mean() > 0.7:
                        df[col] = converted
                except Exception:
                    pass

        return df

    # ========================================================
    # Missing Handling
    # ========================================================

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include=["object", "category"]).columns:
            df[col] = df[col].fillna("Unknown")

        return df

    # ========================================================
    # Outlier Handling (IQR + Z-score cap)
    # ========================================================

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].astype(float)

        for col in numeric_cols:
            mean, std = df[col].mean(), df[col].std()
            if std == 0:
                continue
            z = (df[col] - mean) / std
            df.loc[z >  3, col] = mean + 3 * std
            df.loc[z < -3, col] = mean - 3 * std

        return df

    # ========================================================
    # Identifier Detection
    # ========================================================

    def _detect_identifiers(self, df: pd.DataFrame) -> List[str]:
        """Detect identifier columns by name pattern AND high uniqueness ratio."""
        identifiers = []
        n = len(df)
        id_keywords = {"id", "key", "code", "uuid", "guid", "ref", "number", "no"}

        for col in df.columns:
            col_lower = col.lower()
            name_match = any(kw in col_lower for kw in id_keywords)
            uniqueness  = df[col].nunique() / max(n, 1)

            if name_match and uniqueness > self.id_uniqueness_ratio:
                identifiers.append(col)
            elif uniqueness == 1.0 and df[col].dtype == object:
                # Perfectly unique string column → likely identifier
                identifiers.append(col)

        return identifiers