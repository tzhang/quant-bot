import numpy as np
import pandas as pd

from src.factors import FactorEngine


def make_trend_ohlcv(n: int = 120, trend: str = "flat", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100.0
    noise = rng.normal(0, 0.2, size=n)

    if trend == "up":
        drift = np.linspace(0, 20, n) / n
    elif trend == "down":
        drift = -np.linspace(0, 20, n) / n
    else:
        drift = np.zeros(n)

    close = base * (1 + drift) + noise
    close = pd.Series(close)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = close + np.abs(rng.normal(0.1, 0.05, size=n))
    low = close - np.abs(rng.normal(0.1, 0.05, size=n))
    vol = rng.integers(10_000, 50_000, size=n)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })
    df.index = pd.date_range("2022-01-01", periods=n, freq="B")
    return df


def test_factor_score_ranking_snapshot_from_example_style():
    # 仿照 examples 中的选股逻辑：根据因子分数选择更优趋势
    fe = FactorEngine()

    up = fe.compute_factor_score(make_trend_ohlcv(trend="up", seed=1))
    flat = fe.compute_factor_score(make_trend_ohlcv(trend="flat", seed=2))
    down = fe.compute_factor_score(make_trend_ohlcv(trend="down", seed=3))

    # 仅比较最后一个交易日的因子分数，形成快照式断言
    scores = {
        "UP": float(up["FACTOR_SCORE"].iloc[-1]),
        "FLAT": float(flat["FACTOR_SCORE"].iloc[-1]),
        "DOWN": float(down["FACTOR_SCORE"].iloc[-1]),
    }

    # 快照：趋势应满足 UP > FLAT > DOWN
    assert scores["UP"] > scores["FLAT"] > scores["DOWN"]