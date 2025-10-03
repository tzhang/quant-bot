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


def test_metrics_extended_indicators_present_and_valid():
    pa = PerformanceAnalyzer()
    ret = make_returns_series(n=252)
    m = pa.metrics(ret)
    for key in ("sortino", "calmar", "hit_rate"):
        assert key in m
        assert isinstance(m[key], float)
        assert np.isfinite(m[key])


def test_plot_functions_return_figures():
    pa = PerformanceAnalyzer()
    # Generate synthetic returns and price series
    ret = make_returns_series(n=120, seed=42)
    # Price from cumulative returns
    price = (1 + ret).cumprod() * 100
    df = pd.DataFrame({"Close": price})
    # Simple signal based on normalized returns
    sig = ((ret - ret.rolling(20).min()) / (ret.rolling(20).max() - ret.rolling(20).min() + 1e-12)).fillna(0).clip(0, 1)
    # Benchmark returns: slightly different series
    bench = ret.shift(1).fillna(0) * 0.8

    fig_eq = pa.plot_equity(ret)
    assert fig_eq is not None

    fig_dd = pa.plot_drawdown(ret)
    assert fig_dd is not None

    fig_vs = pa.plot_equity_vs_benchmark(ret, benchmark_returns=bench)
    assert fig_vs is not None

    fig_sig = pa.plot_signal_price(df, sig)
    assert fig_sig is not None

    # Factor score and rolling beta plots
    score = ((ret - ret.mean()) / (ret.std() + 1e-12)).fillna(0)
    fig_score = pa.plot_factor_score(score, window=20)
    assert fig_score is not None

    beta_series = pd.Series(0.8, index=df.index)
    fig_beta = pa.plot_rolling_beta(beta_series)
    assert fig_beta is not None