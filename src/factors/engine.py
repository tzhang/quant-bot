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

    def compute_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.risk.calculate_all_factors(df)

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.compute_technical(df)
        risk = self.compute_risk(df)
        return out.join(risk, how="left")