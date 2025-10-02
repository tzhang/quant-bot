import datetime as dt

import pandas as pd

from src.data import DataManager
from src.factors import FactorEngine
from src.strategies import MomentumStrategy
from src.backtest import BacktestEngine
from src.performance import PerformanceAnalyzer


def main() -> None:
    dm = DataManager(use_cache=True)
    start = dt.date.today().replace(year=dt.date.today().year - 1)
    end = dt.date.today()
    data = dm.get_stock_data("SPY", start, end)
    df = data["SPY"]
    fe = FactorEngine()
    factors = fe.compute_all(df)
    strat = MomentumStrategy(fast=12, slow=26)
    signal = strat.signal(df)
    bt = BacktestEngine(trading_cost_bps=10)
    res = bt.run(df, signal)
    perf = PerformanceAnalyzer()
    metrics = perf.metrics(res["returns"])
    print("Metrics:", metrics)
    perf.plot_equity(res["returns"], save_path="examples/mvp_equity.png")
    perf.plot_drawdown(res["returns"], save_path="examples/mvp_drawdown.png")


if __name__ == "__main__":
    main()