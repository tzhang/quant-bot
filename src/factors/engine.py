from typing import Dict, List

import pandas as pd

from .technical import TechnicalFactors
from .risk import RiskFactors


class FactorEngine:
    """Orchestrates factor calculations across technical and risk factors."""

    def __init__(self) -> None:
        self.tech = TechnicalFactors()
        self.risk = RiskFactors()

    def compute_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.tech.calculate_all_factors(df)

    def compute_risk(self, df: pd.DataFrame, benchmark_returns: pd.Series = None) -> pd.DataFrame:
        """Compute risk factors; optionally include Beta if benchmark provided."""
        risk_factors = self.risk.calculate_all_factors(df)
        if benchmark_returns is not None:
            beta = self.risk.calculate_beta(df, benchmark_returns, window=60)
            risk_factors["BETA60"] = beta
        return risk_factors

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.compute_technical(df)
        risk = self.compute_risk(df)
        return out.join(risk, how="left")

    def normalize_factors(self, factors: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """Normalize factors using z-score standardization."""
        if method == "zscore":
            return (factors - factors.mean()) / factors.std()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def winsorize_factors(self, factors: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
        """Apply winsorization to remove extreme values."""
        return factors.clip(
            lower=factors.quantile(lower, axis=0),
            upper=factors.quantile(upper, axis=0),
            axis=1,
        )

    def compute_factor_score(
        self,
        df: pd.DataFrame,
        weights: Dict[str, float] | None = None,
        normalize: bool = True,
        winsorize: bool = True,
        benchmark_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Compute all factors and synthesize into FACTOR_SCORE."""
        # Get all factors
        tech_factors = self.compute_technical(df)
        risk_factors = self.compute_risk(df, benchmark_returns)
        all_factors = tech_factors.join(risk_factors, how="left")

        # Define default weights if not provided
        if weights is None:
            weights = {
                "RSI14": 0.15,
                "MACD_12_26_9": 0.20,
                "VOL20_ANN": -0.10,  # Lower volatility is better
                "VAR95_ANN": -0.15,  # Lower VaR is better
                "RET_DAILY": 0.25,
                "SMA20": 0.10,
                "EMA20": 0.15,
            }
            # Add BETA if available
            if "BETA60" in all_factors.columns:
                weights["BETA60"] = 0.05

        # Select only factors present in weights
        factor_cols = [col for col in weights.keys() if col in all_factors.columns]
        selected_factors = all_factors[factor_cols].copy()

        # Apply preprocessing
        if winsorize and not selected_factors.empty:
            selected_factors = self.winsorize_factors(selected_factors)
        if normalize and not selected_factors.empty:
            selected_factors = self.normalize_factors(selected_factors)

        # Compute weighted score
        factor_score = pd.Series(0.0, index=selected_factors.index)
        for col in factor_cols:
            factor_score = factor_score.add(selected_factors[col].fillna(0) * weights[col], fill_value=0)

        # Return combined result
        result = all_factors.copy()
        result["FACTOR_SCORE"] = factor_score
        return result