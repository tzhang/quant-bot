import numpy as np
import pandas as pd

from src.factors.risk import RiskFactors


def make_price_df(n: int = 250, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 模拟带轻微漂移的收益率序列
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100.0 * (1 + pd.Series(returns)).cumprod()
    return pd.DataFrame({"Close": close.astype(float)})


def test_daily_returns_matches_pct_change() -> None:
    risk = RiskFactors()
    df = make_price_df()
    ret = risk.calculate_daily_returns(df)
    expected = df["Close"].pct_change()
    assert ret.equals(expected)


def test_volatility_matches_rolling_std_window20() -> None:
    risk = RiskFactors()
    df = make_price_df()
    vol = risk.calculate_volatility(df, window=20)
    returns = df["Close"].pct_change()
    expected = returns.rolling(20).std() * (252 ** 0.5)
    # 对齐NaN并比较数值（前19个为NaN）
    vol_f = vol.fillna(0).to_numpy()
    exp_f = expected.fillna(0).to_numpy()
    assert np.allclose(vol_f, exp_f, rtol=1e-6, atol=1e-8)


def test_drawdown_is_non_positive_and_bounds() -> None:
    risk = RiskFactors()
    df = make_price_df()
    dd = risk.calculate_drawdown(df)
    # 回撤应不为正，最大为0，最小大于等于-1
    assert dd.index.equals(df.index)
    assert np.isfinite(dd).all()
    assert dd.max() <= 1e-12  # 允许极小数值误差
    assert dd.min() >= -1.0 - 1e-12


def test_calculate_all_factors_contains_expected_columns() -> None:
    risk = RiskFactors()
    df = make_price_df()
    out = risk.calculate_all_factors(df)
    expected_cols = {"RET_DAILY", "VOL20_ANN", "DRAWDOWN", "VAR95_ANN"}
    assert expected_cols.issubset(set(out.columns))
    # 行数与输入一致，且类型正确
    assert out.shape[0] == df.shape[0]
    assert out["RET_DAILY"].dtype.kind in ("f", "d")
    assert out["VOL20_ANN"].dtype.kind in ("f", "d")
    assert out["DRAWDOWN"].dtype.kind in ("f", "d")


def test_var95_matches_rolling_quantile_ann_scaling():
    risk = RiskFactors()
    df = make_price_df(n=400)
    returns = df["Close"].pct_change()
    expected = -returns.rolling(252).quantile(0.05) * (252 ** 0.5)
    var = risk.calculate_var(df, window=252, alpha=0.05, annualize=True)
    var_f = var.fillna(0).to_numpy()
    exp_f = expected.fillna(0).to_numpy()
    assert np.allclose(var_f, exp_f, rtol=1e-6, atol=1e-8)


def test_beta_rolling_cov_over_var_benchmark():
    risk = RiskFactors()
    # 生成资产与基准的价格序列
    df = make_price_df(n=300, seed=123)
    # 基准收益率为资产收益率的线性组合 + 噪声，以构造正协方差
    rng = np.random.default_rng(456)
    asset_ret = df["Close"].pct_change().fillna(0)
    bench_ret = 0.8 * asset_ret + rng.normal(0, 0.005, size=len(asset_ret))
    beta = risk.calculate_beta(df, benchmark_returns=bench_ret, window=60)
    # Beta在足够窗口后应为有限数值，且大体接近0.8（受噪声影响允许误差较大）
    finite_mask = beta.notna()
    assert finite_mask.sum() > 0
    mean_beta = beta[finite_mask].mean()
    assert np.isfinite(mean_beta)
    assert 0.4 <= mean_beta <= 1.2