import datetime as dt
import sys
import os
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_adapter import create_data_adapter
from src.factors import FactorEngine
from src.backtest import BacktestEngine
from src.performance import PerformanceAnalyzer


def main() -> None:
    print("🚀 启动量化交易系统MVP演示...")
    
    # 创建数据适配器，启用多个数据源作为回退
    print("📊 初始化数据适配器...")
    adapter = create_data_adapter(
        enable_alpaca=True,
        fallback_to_yfinance=True,  # 添加yfinance作为回退
        enable_openbb=False,
        enable_ib=False,
        prefer_qlib=False
    )
    
    # 设置时间范围（使用历史数据避免API限速）
    end = dt.date(2024, 10, 1)  # 使用历史日期
    start = end.replace(year=end.year - 1)
    
    # 使用Alpaca获取SPY数据
    print("📈 获取SPY股票数据...")
    data = adapter.get_stock_data("SPY", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    
    # 检查数据是否获取成功
    if data is None or data.empty:
        print("❌ 无法获取SPY数据，请检查数据源配置或网络连接")
        print("💡 提示：Alpaca免费账户可能无法获取最新数据，请考虑使用其他数据源")
        return
    
    df = data  # 直接使用返回的DataFrame
    
    # 获取基准（标准普尔500）以用于Beta计算
    print("📊 获取基准数据...")
    benchmark = adapter.get_stock_data("SPY", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))  # 使用SPY作为基准
    
    # 检查基准数据是否获取成功
    if benchmark is None or benchmark.empty:
        print("❌ 无法获取基准数据，使用模拟基准数据")
        # 创建模拟基准收益率
        benchmark_returns = pd.Series(0.0, index=df.index)
    else:
        # 检查列名并使用正确的列名
        if "Close" in benchmark.columns:
            benchmark_returns = benchmark["Close"].pct_change().fillna(0.0)
        elif "close" in benchmark.columns:
            benchmark_returns = benchmark["close"].pct_change().fillna(0.0)
        else:
            print(f"⚠️ 基准数据列名不匹配，可用列: {list(benchmark.columns)}")
            benchmark_returns = pd.Series(0.0, index=df.index)

    # 计算所有因子并合成FACTOR_SCORE（包含可选的Beta）
    print("🔧 计算技术因子...")
    fe = FactorEngine()
    factors = fe.compute_factor_score(df, benchmark_returns=benchmark_returns)
    print(f"✅ 因子计算完成，包含列: {list(factors.columns)}")

    # 使用FACTOR_SCORE生成连续仓位信号：滚动60日Min-Max归一化至[0,1]
    print("📈 生成交易信号...")
    score = factors["FACTOR_SCORE"].fillna(0.0)
    roll_min = score.rolling(60).min()
    roll_max = score.rolling(60).max()
    signal = ((score - roll_min) / (roll_max - roll_min + 1e-12)).clip(0.0, 1.0).fillna(0.0)
    print(f"✅ 信号生成完成，平均信号强度: {signal.mean():.3f}")

    print("🔄 运行回测...")
    bt = BacktestEngine(commission=0.001)  # 使用正确的参数名
    res = bt.run(df, signal)
    
    print("📊 分析性能指标...")
    perf = PerformanceAnalyzer()
    metrics = perf.metrics(res["returns"])
    
    print("\n" + "="*50)
    print("📈 量化交易MVP演示结果")
    print("="*50)
    print(f"总收益率: {metrics.get('total_return', 0):.2%}")
    print(f"年化收益率: {metrics.get('annual_return', 0):.2%}")
    print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
    print(f"胜率: {metrics.get('win_rate', 0):.2%}")
    print("="*50)
    
    print("📊 生成性能图表...")
    try:
        perf.plot_equity(res["returns"], save_path="examples/mvp_equity.png")
        print("✅ 权益曲线图已保存: examples/mvp_equity.png")
    except Exception as e:
        print(f"⚠️  权益曲线图生成失败: {e}")
    
    try:
        perf.plot_drawdown(res["returns"], save_path="examples/mvp_drawdown.png")
        print("✅ 回撤图已保存: examples/mvp_drawdown.png")
    except Exception as e:
        print(f"⚠️  回撤图生成失败: {e}")
    
    # 新增图表：权益对比基准、价格与信号、因子得分与归一、滚动Beta
    try:
        perf.plot_equity_vs_benchmark(
            res["returns"], benchmark_returns=benchmark_returns, save_path="examples/mvp_equity_vs_benchmark.png"
        )
        print("✅ 基准对比图已保存: examples/mvp_equity_vs_benchmark.png")
    except Exception as e:
        print(f"⚠️  基准对比图生成失败: {e}")
    
    try:
        perf.plot_signal_price(df, signal, save_path="examples/mvp_price_signal.png")
        print("✅ 价格信号图已保存: examples/mvp_price_signal.png")
    except Exception as e:
        print(f"⚠️  价格信号图生成失败: {e}")
    
    try:
        perf.plot_factor_score(score, window=60, save_path="examples/mvp_factor_score.png")
        print("✅ 因子得分图已保存: examples/mvp_factor_score.png")
    except Exception as e:
        print(f"⚠️  因子得分图生成失败: {e}")
    
    if "BETA60" in factors.columns:
        try:
            perf.plot_rolling_beta(factors["BETA60"], save_path="examples/mvp_beta.png")
            print("✅ 滚动Beta图已保存: examples/mvp_beta.png")
        except Exception as e:
            print(f"⚠️  滚动Beta图生成失败: {e}")
    
    print("\n🎉 MVP演示完成！请查看examples/目录下的图表文件。")


if __name__ == "__main__":
    main()