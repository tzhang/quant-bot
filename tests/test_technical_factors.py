import numpy as np
import pandas as pd

from src.factors.technical import TechnicalFactors


def make_price_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Simulate a price series with slight drift
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100.0 * (1 + pd.Series(returns)).cumprod()
    volume = pd.Series(rng.integers(1_000, 10_000, size=n), dtype=float)
    df = pd.DataFrame({"Close": close.astype(float), "Volume": volume.astype(float)})
    return df


def test_macd_and_bbands_outputs_have_expected_columns():
    tf = TechnicalFactors()
    df = make_price_df()

    macd = tf.calculate_macd(df)
    bb = tf.calculate_bbands(df)

    # MACD should be a DataFrame with 3 columns
    assert isinstance(macd, pd.DataFrame)
    assert macd.shape[1] == 3
    # Columns should match pandas-ta naming or fallback naming
    macd_expected_cols = {"MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"}
    assert macd_expected_cols.issubset(set(macd.columns))

    # BBANDS may have 4 (fallback) or 5 (pandas_ta) columns
    assert isinstance(bb, pd.DataFrame)
    assert bb.shape[1] in (4, 5)
    cols = set(bb.columns)
    # Accept names with potential duplicated std suffix (e.g., _2.0_2.0)
    assert any(c.startswith("BBL_20") for c in cols)
    assert any(c.startswith("BBM_20") for c in cols)
    assert any(c.startswith("BBU_20") for c in cols)
    assert any(c.startswith("BBB_20") for c in cols)


def test_calculate_all_factors_contains_technical_columns():
    tf = TechnicalFactors()
    df = make_price_df()
    out = tf.calculate_all_factors(df)

    # Basic technical factors
    assert "SMA20" in out.columns
    assert "EMA20" in out.columns
    # RSI may be present if library returns it
    # MACD/BBANDS columns from concat
    expected_prefixes = [
        "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
        "BBU_20", "BBM_20", "BBL_20", "BBB_20"
    ]
    for pref in expected_prefixes:
        assert any(c.startswith(pref) for c in out.columns)