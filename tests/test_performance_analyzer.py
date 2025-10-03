import numpy as np
import pandas as pd

from src.performance.analyzer import PerformanceAnalyzer


def make_returns_series(n: int = 252, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Daily returns with small drift
    ret = rng.normal(loc=0.0004, scale=0.01, size=n)
    return pd.Series(ret)


def test_metrics_accepts_series_dataframe_array_and_returns_floats():
    pa = PerformanceAnalyzer()

    # Series input
    ret_s = make_returns_series()
    m_s = pa.metrics(ret_s)

    # DataFrame single column input
    ret_df1 = pd.DataFrame({"r": ret_s})
    m_df1 = pa.metrics(ret_df1)

    # DataFrame multi-column input (mean across columns)
    ret_df2 = pd.DataFrame({"a": ret_s, "b": ret_s.shift(1).fillna(0)})
    m_df2 = pa.metrics(ret_df2)

    # Array-like input
    ret_arr = ret_s.values
    m_arr = pa.metrics(ret_arr)

    for m in (m_s, m_df1, m_df2, m_arr):
        # Expected keys
        for key in ("cum_return", "ann_return", "ann_vol", "sharpe", "max_drawdown"):
            assert key in m
            assert isinstance(m[key], float)
            # Values should be finite numbers
            assert np.isfinite(m[key])


def test_metrics_sharpe_zero_when_volatility_zero():
    pa = PerformanceAnalyzer()
    # Zero returns -> volatility zero
    ret_zero = pd.Series([0.0] * 100)
    m = pa.metrics(ret_zero)
    assert m["ann_vol"] == 0.0
    assert m["sharpe"] == 0.0