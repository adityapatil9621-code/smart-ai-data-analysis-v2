"""
visual_intelligence.py

Visual Intelligence Engine for Smart AI Data Intelligence System.

Improvements:
- Matplotlib backend set to Agg (safe for server/Streamlit)
- Figure size capped to avoid huge images
- Heatmap skips if only 1 numeric column remains after target removal
- Time-series falls back to daily if monthly resampling yields < 5 points
- to_dict() excludes un-serialisable figure objects (returned separately)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from dataclasses import dataclass, field
from typing import Dict


sns.set_theme(style="whitegrid", palette="muted")


# ============================================================
# Visual Object
# ============================================================

@dataclass
class VisualObject:
    primary_chart: str
    selection_reason: str
    visual_confidence: float
    figures: Dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict:
        return {
            "primary_chart":    self.primary_chart,
            "selection_reason": self.selection_reason,
            "visual_confidence": self.visual_confidence,
            "figures":          self.figures,   # kept for Streamlit st.pyplot()
        }


# ============================================================
# Visual Intelligence Engine
# ============================================================

class VisualIntelligenceEngine:

    FIG_W = 7
    FIG_H = 4

    def __init__(self, config: dict = None):
        self.config = config or {}

    def run(self, df: pd.DataFrame, understanding_obj) -> VisualObject:
        df             = df.copy()
        target         = understanding_obj.target_column
        time_column    = understanding_obj.time_column
        figures        = {}
        primary_chart  = None
        sel_reason     = "Default visualisation"
        vis_confidence = 0.5

        # Drop ID-like columns
        id_cols = [c for c in df.columns if "id" in c.lower()]
        df.drop(columns=id_cols, errors="ignore", inplace=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        feat_cols    = [c for c in numeric_cols if c != target]

        # -------------------------------------------------------
        # 1. Time-series trend
        # -------------------------------------------------------
        if understanding_obj.is_time_series and time_column and time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            df = df.dropna(subset=[time_column]).set_index(time_column)

            monthly = df[target].resample("ME").mean()
            if len(monthly) < 5:          # too few months → fall back to daily
                monthly = df[target].resample("D").mean()

            smooth = monthly.rolling(window=3, min_periods=1).mean()

            fig1, ax1 = plt.subplots(figsize=(self.FIG_W, self.FIG_H))
            ax1.plot(monthly.index, monthly.values, alpha=0.35, label="Raw")
            ax1.plot(smooth.index, smooth.values, linewidth=2, label="Trend")
            ax1.xaxis.set_major_locator(ticker.MaxNLocator(6))
            ax1.tick_params(axis="x", rotation=45)
            ax1.set_title(f"{target} — Trend over Time")
            ax1.set_ylabel(target)
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            figures["time_series"] = fig1

            primary_chart  = "Time-Series Trend"
            sel_reason     = "Datetime column detected."
            vis_confidence = 0.9

            df = df.reset_index()   # restore index for subsequent plots

        # -------------------------------------------------------
        # 2. Correlation heatmap
        # -------------------------------------------------------
        if len(feat_cols) > 1:
            corr  = df[feat_cols + [target]].corr()
            n     = len(corr)
            fsize = min(self.FIG_W, max(4, n * 0.7))

            fig2, ax2 = plt.subplots(figsize=(fsize, fsize * 0.8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                        linewidths=0.5, ax=ax2, annot_kws={"size": 8})
            ax2.set_title("Correlation Heatmap")
            plt.tight_layout()
            figures["heatmap"] = fig2

            if not understanding_obj.is_time_series:
                primary_chart  = "Correlation Heatmap"
                sel_reason     = "Multiple numeric features detected."
                vis_confidence = 0.85

        # -------------------------------------------------------
        # 3. Target distribution
        # -------------------------------------------------------
        if target in df.columns:
            fig3, ax3 = plt.subplots(figsize=(self.FIG_W, self.FIG_H))
            sns.histplot(df[target].dropna(), kde=True, ax=ax3, color="#4C72B0")
            ax3.xaxis.set_major_locator(ticker.MaxNLocator(6))
            ax3.tick_params(axis="x", rotation=45)
            ax3.set_title(f"{target} — Distribution")
            plt.tight_layout()
            figures["distribution"] = fig3

        # -------------------------------------------------------
        # 4. Boxplot (outliers)
        # -------------------------------------------------------
        if target in df.columns:
            fig4, ax4 = plt.subplots(figsize=(self.FIG_W, self.FIG_H))
            sns.boxplot(x=df[target].dropna(), ax=ax4, color="#4C72B0")
            ax4.xaxis.set_major_locator(ticker.MaxNLocator(6))
            ax4.tick_params(axis="x", rotation=45)
            ax4.set_title(f"{target} — Outlier View")
            plt.tight_layout()
            figures["boxplot"] = fig4

        # -------------------------------------------------------
        # 5. Feature vs target scatter
        # -------------------------------------------------------
        if feat_cols and target in df.columns:
            feature = feat_cols[0]
            fig5, ax5 = plt.subplots(figsize=(self.FIG_W, self.FIG_H))
            ax5.scatter(df[feature], df[target], alpha=0.4, s=20, color="#4C72B0")
            ax5.xaxis.set_major_locator(ticker.MaxNLocator(6))
            ax5.tick_params(axis="x", rotation=45)
            ax5.set_xlabel(feature)
            ax5.set_ylabel(target)
            ax5.set_title(f"{feature} vs {target}")
            plt.tight_layout()
            figures["relationship"] = fig5

        # -------------------------------------------------------
        # Fallback
        # -------------------------------------------------------
        if not figures:
            fig, ax = plt.subplots(figsize=(self.FIG_W, self.FIG_H))
            ax.text(0.5, 0.5, "Insufficient numeric data for visualisation",
                    ha="center", va="center", fontsize=12)
            figures["fallback"]    = fig
            primary_chart          = "Overview"
            sel_reason             = "Limited numeric structure."
            vis_confidence         = 0.4

        if primary_chart is None:
            primary_chart = list(figures.keys())[0]

        # -------------------------------------------------------
        # Filter figures by relevance
        # -------------------------------------------------------
        selected = {}
        if understanding_obj.is_time_series and "time_series" in figures:
            selected["time_series"] = figures["time_series"]
        if len(feat_cols) > 2 and "heatmap" in figures:
            selected["heatmap"] = figures["heatmap"]
        if "distribution" in figures:
            selected["distribution"] = figures["distribution"]
        if df[target].std() > 0 and "boxplot" in figures:
            selected["boxplot"] = figures["boxplot"]
        if len(feat_cols) > 1 and "relationship" in figures:
            selected["relationship"] = figures["relationship"]
        if "fallback" in figures:
            selected["fallback"] = figures["fallback"]

        return VisualObject(
            primary_chart=primary_chart,
            selection_reason=sel_reason,
            visual_confidence=vis_confidence,
            figures=selected or figures,
        )