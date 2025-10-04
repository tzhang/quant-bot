import numpy as np
import pandas as pd

from src.factors.technical import TechnicalFactors


def test_as_series_from_dataframe_returns_first_column():
    tf = TechnicalFactors()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s = tf._as_series(df)
    assert isinstance(s, pd.Series)
    assert s.equals(df.iloc[:, 0])


def test_as_series_from_ndarray_constructs_series_length_match():
    tf = TechnicalFactors()
    arr = np.array([0.1, 0.2, 0.3, 0.4])
    s = tf._as_series(arr)
    assert isinstance(s, pd.Series)
    assert len(s) == len(arr)


def test_as_series_handles_series_passthrough():
    tf = TechnicalFactors()
    s0 = pd.Series([10, 20, 30])
    s = tf._as_series(s0)
    assert s.equals(s0)