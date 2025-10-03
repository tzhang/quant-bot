import datetime as dt

from src.data import DataManager
from src.factors import FactorEngine
from src.backtest import BacktestEngine
from src.performance import PerformanceAnalyzer


def main() -> None:
    dm = DataManager(use_cache=True)
    start = dt.date.today().replace(year=dt.date.today().year - 1)
    end = dt.date.today()
    data = dm.get_stock_data("SPY", start, end)
    df = data["SPY"]
    # 获取基准（标普500）以用于Beta计算
    benchmark = dm.get_stock_data("^GSPC", start, end)["^GSPC"]
    benchmark_returns = benchmark["Close"].pct_change().fillna(0.0)

    # 计算所有因子并合成FACTOR_SCORE（包含可选的Beta）
    fe = FactorEngine()
    factors = fe.compute_factor_score(df, benchmark_returns=benchmark_returns)

    # 使用FACTOR_SCORE生成连续仓位信号：滚动60日Min-Max归一化至[0,1]
    score = factors["FACTOR_SCORE"].fillna(0.0)
    roll_min = score.rolling(60).min()
    roll_max = score.rolling(60).max()
    signal = ((score - roll_min) / (roll_max - roll_min + 1e-12)).clip(0.0, 1.0).fillna(0.0)

    bt = BacktestEngine(trading_cost_bps=10)
    res = bt.run(df, signal)
    perf = PerformanceAnalyzer()
    metrics = perf.metrics(res["returns"])
    print("Metrics:", metrics)
    perf.plot_equity(res["returns"], save_path="examples/mvp_equity.png")
    perf.plot_drawdown(res["returns"], save_path="examples/mvp_drawdown.png")
    # 新增图表：权益对比基准、价格与信号、因子得分与归一、滚动Beta
    perf.plot_equity_vs_benchmark(
        res["returns"], benchmark_returns=benchmark_returns, save_path="examples/mvp_equity_vs_benchmark.png"
    )
    perf.plot_signal_price(df, signal, save_path="examples/mvp_price_signal.png")
    perf.plot_factor_score(score, window=60, save_path="examples/mvp_factor_score.png")
    if "BETA60" in factors.columns:
        perf.plot_rolling_beta(factors["BETA60"], save_path="examples/mvp_beta.png")


if __name__ == "__main__":
    main()