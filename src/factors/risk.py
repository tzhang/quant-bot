from typing import Dict

import pandas as pd


class RiskFactors:
    """Compute basic risk-related factors from OHLCV."""

    def calculate_daily_returns(self, df: pd.DataFrame) -> pd.Series:
        return df["Close"].pct_change()

    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        returns = self.calculate_daily_returns(df)
        return returns.rolling(window).std() * (252 ** 0.5)

    def calculate_drawdown(self, df: pd.DataFrame) -> pd.Series:
        cum = (1 + df["Close"].pct_change().fillna(0)).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["RET_DAILY"] = self.calculate_daily_returns(df)
        out["VOL20_ANN"] = self.calculate_volatility(df, window=20)
        out["DRAWDOWN"] = self.calculate_drawdown(df)
        return out