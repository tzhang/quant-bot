from typing import Dict, List, Optional

import pandas as pd
import pandas_ta as ta


class TechnicalFactors:
    """Compute common technical indicators using pandas-ta."""

    @staticmethod
    def _as_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure input is a 1D Series; if DataFrame with one column, take that column."""
        if isinstance(obj, pd.Series):
            return obj
        if isinstance(obj, pd.DataFrame):
            if obj.shape[1] >= 1:
                return obj.iloc[:, 0]
        # Fallback: construct series if possible
        return pd.Series(obj)

    def calculate_sma(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        return ta.sma(df["Close"], length=window)

    def calculate_ema(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        return ta.ema(df["Close"], length=window)

    def calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        return ta.rsi(df["Close"], length=window)

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        macd = ta.macd(self._as_series(df["Close"]), fast=fast, slow=slow, signal=signal)
        if isinstance(macd, pd.DataFrame):
            return macd
        # Fallback: manual MACD using EMA if pandas-ta returns None
        close = self._as_series(df["Close"])  # ensure 1D
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = (ema_fast - ema_slow).astype(float)
        signal_line = macd_line.ewm(span=signal, adjust=False).mean().astype(float)
        hist = (macd_line - signal_line).astype(float)
        macd_line.name = f"MACD_{fast}_{slow}_{signal}"
        signal_line.name = f"MACDs_{fast}_{slow}_{signal}"
        hist.name = f"MACDh_{fast}_{slow}_{signal}"
        out = pd.concat([macd_line, signal_line, hist], axis=1)
        return out

    def calculate_bbands(self, df: pd.DataFrame, window: int = 20, std: float = 2.0) -> pd.DataFrame:
        bb = ta.bbands(self._as_series(df["Close"]), length=window, std=std)
        if isinstance(bb, pd.DataFrame):
            return bb
        # Fallback: manual Bollinger Bands
        close = self._as_series(df["Close"])  # ensure 1D
        ma = close.rolling(window).mean()
        s = close.rolling(window).std()
        upper = ma + std * s
        lower = ma - std * s
        lower.name = f"BBL_{window}_{std}"
        ma.name = f"BBM_{window}_{std}"
        upper.name = f"BBU_{window}_{std}"
        width = (upper - lower).rename(f"BBB_{window}_{std}")
        out = pd.concat([lower.astype(float), ma.astype(float), upper.astype(float), width.astype(float)], axis=1)
        return out

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # SMA/EMA always have robust fallback using pandas
        out["SMA20"] = self.calculate_sma(df, window=20)
        out["EMA20"] = self.calculate_ema(df, window=20)
        # RSI: skip if None
        rsi = self.calculate_rsi(df, window=14)
        if rsi is not None:
            out["RSI14"] = rsi
        # MACD/BBANDS: robust fallbacks already inside functions
        macd = self.calculate_macd(df)
        if isinstance(macd, pd.DataFrame) and not macd.empty:
            out = pd.concat([out, macd], axis=1)
        bb = self.calculate_bbands(df)
        if isinstance(bb, pd.DataFrame) and not bb.empty:
            out = pd.concat([out, bb], axis=1)
        return out