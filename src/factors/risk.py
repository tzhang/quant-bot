from typing import Dict, Union

import pandas as pd


class RiskFactors:
    """Compute basic risk-related factors from OHLCV."""

    def calculate_daily_returns(self, df: pd.DataFrame) -> pd.Series:
        close = self._ensure_series(df["Close"]) if hasattr(self, "_ensure_series") else (
            df["Close"] if isinstance(df["Close"], pd.Series) else pd.Series(df["Close"]).astype(float)
        )
        return close.pct_change()

    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        returns = self.calculate_daily_returns(df)
        return returns.rolling(window).std() * (252 ** 0.5)

    def calculate_drawdown(self, df: pd.DataFrame) -> pd.Series:
        cum = (1 + df["Close"].pct_change().fillna(0)).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd

    def calculate_var(
        self,
        df: pd.DataFrame,
        window: int = 252,
        alpha: float = 0.05,
        annualize: bool = True,
    ) -> pd.Series:
        returns = self.calculate_daily_returns(df)
        var = -returns.rolling(window).quantile(alpha)
        if annualize:
            var = var * (252 ** 0.5)
        return var

    def _ensure_series(self, obj: Union[pd.Series, pd.DataFrame, list, tuple]) -> pd.Series:
        """Ensure input is a 1D numeric Series.

        - DataFrame with one column: use that column
        - DataFrame with multiple columns: use row-wise mean
        - Array-like: convert to Series
        """
        if isinstance(obj, pd.Series):
            s = obj
        elif isinstance(obj, pd.DataFrame):
            if obj.shape[1] == 1:
                s = obj.iloc[:, 0]
            else:
                s = obj.mean(axis=1)
        else:
            s = pd.Series(obj)
        return pd.to_numeric(s, errors="coerce")

    def calculate_beta(
        self,
        df: pd.DataFrame,
        benchmark_returns: Union[pd.Series, pd.DataFrame, list, tuple],
        window: int = 60,
    ) -> pd.Series:
        # Coerce benchmark returns to numeric Series and align to asset index
        br_raw = self._ensure_series(benchmark_returns)
        br = br_raw.reindex(df.index)
        ar = pd.to_numeric(self.calculate_daily_returns(df), errors="coerce")
        cov = ar.rolling(window).cov(br)
        var_b = br.rolling(window).var()
        beta = cov / (var_b + 1e-12)
        return beta

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["RET_DAILY"] = self.calculate_daily_returns(df)
        out["VOL20_ANN"] = self.calculate_volatility(df, window=20)
        out["DRAWDOWN"] = self.calculate_drawdown(df)
        out["VAR95_ANN"] = self.calculate_var(df, window=252, alpha=0.05, annualize=True)
        return out