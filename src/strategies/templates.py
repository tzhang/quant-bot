import pandas as pd


class MeanReversionStrategy:
    """Mean reversion using z-score of price vs SMA."""

    def __init__(self, lookback: int = 20, entry_z: float = -1.0, exit_z: float = 0.0):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def signal(self, df: pd.DataFrame) -> pd.Series:
        sma = df["Close"].rolling(self.lookback).mean()
        std = df["Close"].rolling(self.lookback).std()
        z = (df["Close"] - sma) / (std + 1e-12)
        sig = pd.Series(0.0, index=df.index)
        sig = sig.where(z > self.entry_z, 1.0)
        sig = sig.where(z < self.exit_z, 0.0)
        return sig.clip(0.0, 1.0)


class MomentumStrategy:
    """Momentum strategy based on EMA crossover."""

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def signal(self, df: pd.DataFrame) -> pd.Series:
        ema_fast = df["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.slow, adjust=False).mean()
        sig = (ema_fast > ema_slow).astype(float)
        return sig.clip(0.0, 1.0)