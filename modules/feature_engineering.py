"""
feature_engineering.py

Feature Engineering Engine for Smart AI Data Intelligence System.

Improvements:
- Safe log transform handles negative values via log1p(abs) + sign restoration
- Multicollinearity threshold configurable
- Scaler stored for later inverse-transform if needed
- Handles empty feature set gracefully
- Type narrowing before get_dummies to avoid mixed-type dummies
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Feature Object
# ============================================================

@dataclass
class FeatureObject:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    scaler: Optional[StandardScaler] = field(default=None, repr=False)

    def to_dict(self):
        return {
            "feature_count": len(self.feature_names),
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
        }


# ============================================================
# Feature Engineering Engine
# ============================================================

class FeatureEngineeringEngine:

    MULTICOLLINEARITY_THRESHOLD = 0.90

    def __init__(self, config: dict = None):
        self.config       = config or {}
        self.test_size    = self.config.get("test_size", 0.2)
        self.random_state = self.config.get("random_state", 42)
        self.max_dummies  = self.config.get("max_dummies", 10)

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(self, df: pd.DataFrame, understanding_obj) -> FeatureObject:
        target        = understanding_obj.target_column
        time_column   = understanding_obj.time_column
        is_time_series = understanding_obj.is_time_series

        df_model = df.copy()

        # 1. Log-transform skewed features
        for col in understanding_obj.skewed_features:
            if col in df_model.columns and col != target:
                df_model[col] = np.sign(df_model[col]) * np.log1p(np.abs(df_model[col]))

        # 2. Encode categorical features (top-N + Other bucket)
        cat_cols = [c for c in understanding_obj.categorical_columns if c in df_model.columns and c != target]
        for col in cat_cols:
            top = df_model[col].value_counts().nlargest(self.max_dummies).index
            df_model[col] = df_model[col].where(df_model[col].isin(top), "Other")
            # Cast to string so get_dummies is consistent
            df_model[col] = df_model[col].astype(str)

        if cat_cols:
            df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
            # Remove "Unknown" dummies created from fillna('Unknown')
            unknown_cols = [c for c in df_model.columns if "_Unknown" in c]
            df_model.drop(columns=unknown_cols, inplace=True, errors="ignore")

        # 3. Sort by time if applicable
        if time_column and time_column in df_model.columns:
            df_model = df_model.sort_values(by=time_column)

        # 4. Drop all datetime columns (sklearn incompatible)
        dt_cols = df_model.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        df_model.drop(columns=dt_cols, inplace=True)

        # 5. Separate X / y
        if target not in df_model.columns:
            raise ValueError(f"Target column '{target}' missing after feature engineering.")

        X = df_model.drop(columns=[target])
        y = df_model[target]

        # Sanitise
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())

        # 6. Remove multicollinear features
        if len(X.columns) > 1:
            corr_matrix = X.corr().abs()
            corr_np     = corr_matrix.values.copy()   # make writable copy
            mask        = np.triu(np.ones(corr_np.shape), k=1).astype(bool)
            corr_upper  = pd.DataFrame(corr_np, index=corr_matrix.index, columns=corr_matrix.columns).where(mask)
            to_drop     = [col for col in corr_upper.columns if corr_upper[col].max() > self.MULTICOLLINEARITY_THRESHOLD]
            X.drop(columns=to_drop, inplace=True)

        if X.empty:
            raise ValueError("No features remain after preprocessing. Check your dataset.")

        # 7. Scale numeric features
        scaler     = StandardScaler()
        num_cols   = X.select_dtypes(include=np.number).columns
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # 8. Train / test split
        if is_time_series:
            split = int(len(X) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True,
            )

        return FeatureObject(
            X_train=X_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            feature_names=X.columns.tolist(),
            scaler=scaler,
        )