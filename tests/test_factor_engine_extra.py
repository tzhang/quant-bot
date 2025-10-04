import numpy as np
import pandas as pd

from src.factors import FactorEngine


def make_dummy_ohlcv(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.0004, scale=0.01, size=n)
    prices = 100 * np.exp(np.cumsum(rets))
    close = pd.Series(prices)
    df = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * (1 + rng.normal(0.001, 0.002, size=n)),
            "Low": close * (1 - rng.normal(0.001, 0.002, size=n)),
            "Close": close,
            "Volume": rng.integers(50_000, 100_000, size=n),
        }
    )
    df.index = pd.date_range("2021-01-01", periods=n, freq="B")
    return df


def test_normalize_factors_invalid_method_raises():
    fe = FactorEngine()
    df = make_dummy_ohlcv()
    factors = fe.compute_all(df)
    try:
        fe.normalize_factors(factors[["SMA20", "EMA20"]], method="unsupported")
    except ValueError as e:
        assert "Unsupported normalization method" in str(e)
    else:
        assert False, "Expected ValueError for unsupported normalization method"


def test_compute_factor_score_beta_included_when_benchmark():
    df = make_dummy_ohlcv()
    bench = df["Close"].pct_change()
    fe = FactorEngine()
    res = fe.compute_factor_score(df, benchmark_returns=bench)
    assert "BETA60" in res.columns
    assert res["BETA60"].notna().sum() > 0
    assert "FACTOR_SCORE" in res.columns


def test_compute_factor_score_no_beta_without_benchmark():
    df = make_dummy_ohlcv()
    fe = FactorEngine()
    res = fe.compute_factor_score(df, benchmark_returns=None)
    assert "BETA60" not in res.columns
    assert "FACTOR_SCORE" in res.columns