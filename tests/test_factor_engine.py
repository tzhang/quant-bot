import numpy as np
import pandas as pd

from src.factors import FactorEngine


def make_dummy_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Generate returns with small drift
    rets = rng.normal(loc=0.0005, scale=0.01, size=n)
    prices = 100 * np.exp(np.cumsum(rets))
    close = pd.Series(prices)
    df = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * (1 + rng.normal(0.001, 0.002, size=n)),
            "Low": close * (1 - rng.normal(0.001, 0.002, size=n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, size=n),
        }
    )
    df.index = pd.date_range("2020-01-01", periods=n, freq="B")
    return df


def test_factor_synthesis_outputs_score_and_columns():
    df = make_dummy_ohlcv()
    fe = FactorEngine()
    res = fe.compute_factor_score(df)
    # Must contain FACTOR_SCORE and several base factor columns
    assert "FACTOR_SCORE" in res.columns
    assert res["FACTOR_SCORE"].notna().sum() > 0
    for col in ("RSI14", "VOL20_ANN", "VAR95_ANN"):
        assert col in res.columns


def test_factor_synthesis_toggle_affects_score_values():
    df = make_dummy_ohlcv()
    fe = FactorEngine()
    res_norm = fe.compute_factor_score(df, normalize=True, winsorize=True)
    res_raw = fe.compute_factor_score(df, normalize=False, winsorize=False)
    # Scores should differ when preprocessing toggles change
    s1 = res_norm["FACTOR_SCORE"].fillna(0)
    s2 = res_raw["FACTOR_SCORE"].fillna(0)
    assert not np.allclose(s1.values, s2.values)


def test_compute_risk_includes_beta_when_benchmark_provided():
    df = make_dummy_ohlcv()
    # Use the same asset returns as a proxy benchmark for test simplicity
    benchmark_returns = df["Close"].pct_change()
    fe = FactorEngine()
    risk = fe.compute_risk(df, benchmark_returns=benchmark_returns)
    assert "BETA60" in risk.columns
    # After sufficient window, beta values should be finite for some rows
    assert risk["BETA60"].notna().sum() > 0