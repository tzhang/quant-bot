# 使用指南 (Usage Guide) - v1.6.0 数据源修复版

本指南将帮助您快速上手量化交易系统的各项功能，包括最新的数据源修复和MVP演示系统。

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/quant-system.git
cd quant-system

# 创建虚拟环境 (强制要求 Python 3.12)
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 1.1 MVP演示系统测试 (v1.6.0 新增)

```bash
# 运行MVP演示系统 - 端到端量化交易演示
python mvp_demo.py

# 生成的专业图表文件:
# - mvp_net_value_curve.png (净值曲线图)
# - mvp_drawdown_analysis.png (回撤分析图)
# - mvp_returns_distribution.png (收益分布图)
# - mvp_rolling_metrics.png (滚动指标图)
# - mvp_risk_return_scatter.png (风险收益散点图)
# - mvp_monthly_returns_heatmap.png (月度收益热力图)
```

### 1.2 性能优化系统测试 (v3.0.0)

```bash
# 运行集成优化系统测试
python examples/final_integration_test.py

# 运行性能基准测试
python examples/test_optimized_parallel_performance.py

# 运行大规模性能测试
python examples/test_large_scale_performance.py
```

### 2. 基本配置

创建 `.env` 文件：
```env
# API密钥
ALPHA_VANTAGE_API_KEY=your_key_here
QUANDL_API_KEY=your_key_here
OPENBB_API_KEY=your_key_here  # v1.5.0 新增
ALPACA_API_KEY=your_key_here  # v1.6.0 新增
ALPACA_SECRET_KEY=your_secret_here  # v1.6.0 新增

# 数据库配置（可选）
DATABASE_URL=postgresql://user:password@localhost:5432/quant_db
REDIS_URL=redis://localhost:6379/0

# 缓存配置
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# 数据源配置 (v1.6.0 更新 - 智能回退机制)
DEFAULT_DATA_SOURCE=auto  # auto, ib, qlib, openbb
DATA_SOURCE_PRIORITY=ib,qlib,openbb  # v1.6.0 新增智能回退顺序
ENABLE_DATA_FALLBACK=true  # v1.6.0 新增 - 启用数据源回退
COLUMN_COMPATIBILITY_MODE=true  # v1.6.0 新增 - 列名兼容性处理
API_RATE_LIMIT_ENABLED=true  # v1.6.0 新增 - API限流保护
```

### 3. 第一个示例

```python
import src as quant

# 创建数据适配器 (v1.6.0 更新 - 智能数据源回退)
from src.data.data_adapter import DataAdapter
data_adapter = DataAdapter()

# 获取股票数据 - 自动选择最佳数据源，支持智能回退
data = data_adapter.get_data(['AAPL', 'GOOGL'], period='1y')
print(f"获取到 {len(data)} 条数据")
print(f"使用的数据源: {data_adapter.get_last_used_source()}")

# 查看系统版本
print(f"系统版本: {quant.get_version()}")
```

### 3.1 MVP演示系统使用 (v1.6.0 新增)

```python
from mvp_demo import MVPDemo

# 创建MVP演示实例
mvp = MVPDemo()

# 运行完整的量化交易演示
results = mvp.run_full_demo()

# 查看演示结果
print(f"策略总收益: {results['total_return']:.2%}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")

# 生成的图表文件将保存在当前目录
print("生成的专业图表:")
for chart in results['generated_charts']:
    print(f"- {chart}")
```

### 3.1 性能优化系统使用 (v3.0.0 最新)

```python
from optimization.cache_system import SmartCacheSystem
from optimization.memory_pool import MemoryPoolManager
from optimization.performance_profiler import PerformanceProfiler
from optimization.adaptive_executor import AdaptiveExecutor

# 初始化性能优化组件
cache_system = SmartCacheSystem()
memory_pool = MemoryPoolManager()
profiler = PerformanceProfiler()
executor = AdaptiveExecutor()

# 使用智能缓存系统
cache_system.set('stock_data_AAPL', data)
cached_data = cache_system.get('stock_data_AAPL')
print(f"缓存命中率: {cache_system.get_hit_rate():.2%}")

# 使用内存池管理器
with memory_pool.get_buffer(1024*1024) as buffer:  # 1MB缓冲区
    # 在这里进行数据处理
    processed_data = process_large_dataset(data, buffer)
print(f"内存使用优化: {memory_pool.get_memory_savings():.1f}MB")

# 使用性能分析器
with profiler.profile('data_processing'):
    result = complex_calculation(data)
print(f"处理时间: {profiler.get_last_duration():.3f}秒")

# 使用自适应执行器
optimized_result = executor.execute_adaptive(
    func=calculate_factors,
    data=data,
    auto_optimize=True
)
print(f"性能提升: {executor.get_performance_gain():.1f}x")
```

## 📊 数据管理 (v1.6.0 数据源修复版)

### 智能数据源回退系统 (v1.6.0 新增)

**重大更新**: 系统现在支持 IB TWS API → Qlib → OpenBB 智能回退机制，确保数据获取的稳定性

```python
from src.data.data_adapter import DataAdapter

# 创建数据适配器 (v1.6.0 - 支持智能回退)
adapter = DataAdapter()

# 自动选择最佳数据源，支持智能回退
data = adapter.get_data('AAPL', start='2023-01-01', end='2024-01-01')
print(f"使用的数据源: {adapter.get_last_used_source()}")
print(f"回退次数: {adapter.get_fallback_count()}")

# 检查数据可用性和回退状态
availability = adapter.check_data_availability(['AAPL', 'GOOGL', 'MSFT'])
print("数据源可用性:", availability)

# 强制使用特定数据源
data_alpaca = adapter.get_data('AAPL', source='alpaca')  # v1.6.0 新增
data_ib = adapter.get_data('AAPL', source='ib')
data_qlib = adapter.get_data('AAPL', source='qlib')
data_openbb = adapter.get_data('AAPL', source='openbb')
```

### 数据源修复功能 (v1.6.0 核心特性)

```python
# 列名兼容性处理
data = adapter.get_data('AAPL', normalize_columns=True)  # 自动标准化列名
print("标准化后的列名:", data.columns.tolist())

# API限流处理
adapter.set_rate_limit(requests_per_minute=60)  # 设置API限流
data = adapter.get_data_with_retry('AAPL', max_retries=3)  # 支持重试

# 数据质量验证
quality_report = adapter.validate_data_quality(data)
print(f"数据完整性: {quality_report['completeness']:.2%}")
print(f"数据一致性: {quality_report['consistency']:.2%}")
```

### 数据源性能对比 (v1.6.0 更新)

| 数据源 | 获取速度 | 加速比 | 适用场景 | 稳定性 | v1.6.0改进 |
|--------|----------|--------|----------|--------|------------|
| Alpaca API | 0.45秒 | 5.2x | 实时交易数据 | 99.5% | 新增主数据源 |
| yfinance | 2.34秒 | 1.0x | 通用股票数据 | 95.8% | 智能回退 |
| Qlib 本地数据 | 0.03秒 | 78.0x | 本地量化研究 | 99.9% | 列名兼容 |
| OpenBB 平台 | 0.89秒 | 2.6x | 专业金融分析 | 97.2% | 错误处理 |
| 智能缓存 | 0.08秒 | 29.3x | 重复查询 | 100% | 缓存优化 |

### 批量获取多股票数据 (v1.6.0 增强)

```python
# 批量获取多只股票数据 - 支持智能回退
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
data_dict = adapter.get_multiple_data(
    symbols=symbols,
    start='2023-01-01',
    end='2024-01-01',
    enable_fallback=True  # v1.6.0 新增
)

# 查看每个股票使用的数据源
for symbol, data in data_dict.items():
    source_info = adapter.get_source_info(symbol)
    print(f"{symbol}: {len(data)} 条数据, 数据源: {source_info['source']}")
    if source_info['fallback_used']:
        print(f"  - 使用了回退机制: {source_info['fallback_chain']}")
```

for symbol, data in data_dict.items():
    print(f"{symbol}: {len(data)} 条数据")
```

### 传统数据获取方法 (兼容性保持)

**重要提示**: 请使用新的 `DataAdapter` 类进行数据获取，但旧的 `DataManager` 仍然可用

```python
from src.data.data_manager import DataManager

# 创建数据管理器 (旧版本兼容)
data_manager = DataManager()

# 获取单只股票数据
data = data_manager.get_data(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    data_type='ohlcv'
)

# 获取多只股票数据
symbols = ['AAPL', 'GOOGL', 'MSFT']
for symbol in symbols:
    data = data_manager.get_data(
        symbol=symbol,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    print(f"{symbol} 数据形状: {data.shape}")
```

### 使用缓存数据避免API限制

```python
import pandas as pd
from pathlib import Path

def load_cached_data(symbol='AAPL'):
    """加载缓存的股票数据"""
    cache_dir = Path('data_cache')
    cache_files = list(cache_dir.glob(f'ohlcv_{symbol}_*.csv'))
    
    if cache_files:
        cache_file = cache_files[0]
        # 读取CSV文件，跳过前两行
        df = pd.read_csv(cache_file, skiprows=2)
        df = df.iloc[:, 1:]  # 去掉第一列
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        return df
    return None

# 使用缓存数据
cached_data = load_cached_data('AAPL')
if cached_data is not None:
    print(f"成功加载缓存数据: {cached_data.shape}")
```

### 数据获取最佳实践

```python
import time

def safe_get_data(data_manager, symbol, start_date, end_date, max_retries=3):
    """安全的数据获取函数，包含重试机制"""
    for attempt in range(max_retries):
        try:
            data = data_manager.get_data(symbol, start_date, end_date)
            return data
        except Exception as e:
            print(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"❌ {symbol} 数据获取最终失败")
                return None

# 批量获取数据时添加延迟
symbols = ['AAPL', 'GOOGL', 'MSFT']
for symbol in symbols:
    data = safe_get_data(data_manager, symbol, '2023-01-01', '2024-01-01')
    if data is not None:
        print(f"✅ {symbol} 数据获取成功")
    time.sleep(1)  # 避免API限制
```

## 🧮 因子计算

### 正确的因子计算流程

```python
from src.factors.engine import FactorEngine
from src.data.data_adapter import DataAdapter  # v1.5.0 更新

# 1. 获取数据 (使用新的数据适配器)
data_adapter = DataAdapter()
data = data_adapter.get_data('AAPL', start='2023-01-01', end='2024-01-01')

# 2. 创建因子引擎
factor_engine = FactorEngine()

# 3. 计算技术因子
tech_factors = factor_engine.compute_technical(data)
print(f"计算了 {len(tech_factors.columns)} 个技术因子")

# 4. 计算风险因子
risk_factors = factor_engine.compute_risk(data)
print(f"计算了 {len(risk_factors.columns)} 个风险因子")

# 5. 计算所有因子
all_factors = factor_engine.compute_all(data)
print(f"总共计算了 {len(all_factors.columns)} 个因子")
```

### 因子标准化和处理

```python
# 因子标准化
normalized_factors = factor_engine.normalize_factors(all_factors)

# 因子去极值
winsorized_factors = factor_engine.winsorize_factors(all_factors)

# 计算因子得分
factor_scores = factor_engine.compute_factor_score(all_factors)
print(f"因子得分: {factor_scores}")
```

## 📈 策略开发与回测 (v1.4.0 新增)

### 策略框架

```python
from src.strategies.base_strategy import BaseStrategy
from src.strategies.multi_factor_strategy import MultiFactorStrategy

# 创建多因子策略
strategy = MultiFactorStrategy(
    factors=['momentum', 'value', 'quality'],
    weights=[0.4, 0.3, 0.3],
    rebalance_freq='monthly'
)

# 自定义策略
class MyCustomStrategy(BaseStrategy):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data):
        # 实现信号生成逻辑
        signals = {}
        # ... 策略逻辑
        return signals
    
    def calculate_positions(self, signals, current_positions):
        # 实现仓位计算逻辑
        new_positions = {}
        # ... 仓位逻辑
        return new_positions
```

### 回测引擎

```python
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_analyzer import PerformanceAnalyzer

# 创建回测引擎
backtest_engine = BacktestEngine(
    initial_capital=1000000,
    commission_rate=0.001,
    slippage_rate=0.0005,
    benchmark='SPY'
)

# 运行回测
results = backtest_engine.run_backtest(
    strategy=strategy,
    start_date='2020-01-01',
    end_date='2023-12-31',
    universe=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
)

# 性能分析
analyzer = PerformanceAnalyzer()
performance_metrics = analyzer.analyze(results)

print(f"总收益率: {performance_metrics['total_return']:.2%}")
print(f"年化收益率: {performance_metrics['annual_return']:.2%}")
print(f"夏普比率: {performance_metrics['sharpe_ratio']:.2f}")
print(f"最大回撤: {performance_metrics['max_drawdown']:.2%}")
print(f"信息比率: {performance_metrics['information_ratio']:.2f}")
```

### 策略性能可视化

```python
from src.visualization.strategy_visualizer import StrategyVisualizer

# 创建可视化器
visualizer = StrategyVisualizer()

# 生成策略报告
visualizer.generate_strategy_report(
    results=results,
    save_path='reports/strategy_performance.html'
)

# 绘制净值曲线
visualizer.plot_equity_curve(results)

# 绘制回撤分析
visualizer.plot_drawdown_analysis(results)

# 绘制月度收益热力图
visualizer.plot_monthly_returns_heatmap(results)
```

## 📊 投资组合优化

### 基本优化

```python
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.constraints import OptimizationConstraint

# 创建优化器
optimizer = PortfolioOptimizer()

# 设置约束条件
constraints = [
    OptimizationConstraint('weight_sum', target=1.0),  # 权重和为1
    OptimizationConstraint('long_only', bounds=(0, 1)),  # 只做多
    OptimizationConstraint('max_weight', bounds=(0, 0.1))  # 单只股票最大权重10%
]

# 执行优化
result = optimizer.optimize(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    constraints=constraints,
    objective='max_sharpe'  # 'max_return', 'min_risk', 'max_sharpe'
)

print(f"最优权重: {result.weights}")
print(f"预期收益: {result.expected_return:.4f}")
print(f"预期风险: {result.expected_risk:.4f}")
```

### 高级优化策略

```python
# Black-Litterman优化
from src.portfolio.black_litterman import BlackLittermanOptimizer

bl_optimizer = BlackLittermanOptimizer()
bl_result = bl_optimizer.optimize(
    market_caps=market_caps,
    returns=historical_returns,
    views=investor_views,  # 投资者观点
    view_confidences=confidences
)

# 风险平价优化
from src.portfolio.risk_parity import RiskParityOptimizer

rp_optimizer = RiskParityOptimizer()
rp_result = rp_optimizer.optimize(covariance_matrix=cov_matrix)
```

## 🔄 传统回测方法 (兼容性保持)

### 基本回测

```python
# 创建回测引擎 (旧版本兼容)
backtest_engine = quant.create_backtest_engine(
    initial_capital=100000,
    commission=0.001,  # 手续费率
    slippage=0.001     # 滑点
)

# 定义简单策略
from src.strategies import SimpleMovingAverageStrategy

strategy = SimpleMovingAverageStrategy(
    short_window=20,
    long_window=50
)

# 运行回测
results = backtest_engine.run_backtest(
    strategy=strategy,
    data=data,
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# 查看结果
print(f"总收益率: {results.total_return:.2%}")
print(f"年化收益率: {results.annual_return:.2%}")
print(f"夏普比率: {results.sharpe_ratio:.2f}")
print(f"最大回撤: {results.max_drawdown:.2%}")
```

### 增强回测

```python
from src.backtesting import EnhancedBacktestEngine

# 创建增强回测引擎
enhanced_engine = EnhancedBacktestEngine(
    initial_capital=100000,
    commission_rate=0.001,
    slippage_rate=0.001,
    risk_free_rate=0.02
)

# 添加风险管理
enhanced_engine.add_risk_manager(
    max_position_size=0.1,  # 单只股票最大仓位
    max_drawdown=0.2,       # 最大回撤限制
    stop_loss=0.05          # 止损比例
)

# 运行增强回测
enhanced_results = enhanced_engine.run_backtest(
    strategy=strategy,
    data=data,
    benchmark_data=benchmark_data
)
```

## ⚠️ 风险管理

### VaR计算

```python
from src.risk.risk_manager import RiskManager

# 创建风险管理器
risk_manager = RiskManager()

# 计算投资组合VaR
var_result = risk_manager.calculate_portfolio_var(
    weights=portfolio_weights,
    returns=return_data,
    confidence_level=0.95,
    method='historical'  # 'historical', 'parametric', 'monte_carlo'
)

print(f"95% VaR: {var_result['var']:.4f}")
print(f"预期损失: {var_result['expected_shortfall']:.4f}")
```

### 压力测试

```python
# 定义压力测试情景
from src.risk.stress_test import StressTestScenario

scenarios = [
    StressTestScenario(
        name="市场崩盘",
        market_shock=-0.3,
        volatility_shock=2.0
    ),
    StressTestScenario(
        name="利率上升",
        interest_rate_shock=0.02,
        market_shock=-0.1
    )
]

# 运行压力测试
stress_results = risk_manager.run_stress_test(
    portfolio_weights=weights,
    scenarios=scenarios,
    historical_data=return_data
)

for result in stress_results:
    print(f"{result.scenario_name}: {result.portfolio_loss:.2%}")
```

### 风险归因

```python
# 风险归因分析
attribution_result = risk_manager.get_risk_attribution(
    portfolio_weights=weights,
    factor_exposures=factor_exposures,
    factor_covariance=factor_cov
)

print("风险贡献:")
for factor, contribution in attribution_result.items():
    print(f"  {factor}: {contribution:.2%}")
```

## 🗄️ 数据库优化 (v1.5.0 增强)

### 缓存管理

```python
from src.data.data_adapter import DataAdapter

# 创建数据适配器
adapter = DataAdapter()

# 获取缓存数据
cached_data = adapter.get_cached_data(
    symbol='AAPL',
    start='2023-01-01',
    end='2024-01-01'
)

# 清理过期缓存
adapter.cleanup_cache(max_age_days=30)

# 获取缓存统计
cache_stats = adapter.get_cache_stats()
print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")
print(f"缓存大小: {cache_stats['size_mb']:.1f} MB")
```

## 📊 可视化分析 (v1.4.0 增强)

### 基本图表

```python
from src.visualization.visualizer import Visualizer

viz = Visualizer()

# 绘制净值曲线
viz.plot_equity_curve(
    equity_data=backtest_results.equity_curve,
    benchmark_data=benchmark_equity
)

# 绘制回撤图
viz.plot_drawdown(backtest_results.drawdown)

# 绘制收益分布
viz.plot_returns_distribution(backtest_results.returns)
```

### 因子分析图表

```python
# IC分析图
viz.plot_ic_analysis(
    ic_data=factor_ic_results,
    factor_name="momentum_20"
)

# 分层收益图
viz.plot_quantile_returns(
    quantile_returns=layered_returns,
    factor_name="value_factor"
)

# 因子暴露图
viz.plot_factor_exposure(
    exposures=factor_exposures,
    portfolio_weights=weights
)
```

### 策略性能仪表板 (v1.4.0 新增)

```python
from src.visualization.dashboard import StrategyDashboard

# 创建策略仪表板
dashboard = StrategyDashboard()

# 生成完整报告
dashboard.generate_full_report(
    strategy_results=results,
    output_path='reports/strategy_dashboard.html'
)

# 实时监控
dashboard.start_live_monitoring(
    strategy=strategy,
    update_interval=60  # 秒
)
```

## 🔧 高级功能

### 性能优化系统高级用法 (v3.0.0 新增)

```python
from optimization.integrated_optimizer import IntegratedOptimizer
from optimization.performance_profiler import PerformanceProfiler
from optimization.adaptive_executor import AdaptiveExecutor

# 创建集成优化器
optimizer = IntegratedOptimizer()

# 配置优化参数
optimizer.configure({
    'cache_size': 1000,
    'memory_pool_size': 512,  # MB
    'profiling_enabled': True,
    'adaptive_execution': True,
    'parallel_workers': 4
})

# 优化因子计算
@optimizer.optimize
def calculate_complex_factors(data, factor_list):
    """使用优化器装饰器自动优化函数"""
    results = {}
    for factor_name in factor_list:
        # 复杂的因子计算逻辑
        results[factor_name] = compute_factor(data, factor_name)
    return results

# 批量优化处理
optimized_results = optimizer.batch_process(
    func=calculate_complex_factors,
    data_batches=[data1, data2, data3],
    factor_list=['momentum', 'value', 'quality']
)

# 获取优化报告
optimization_report = optimizer.get_performance_report()
print(f"总体性能提升: {optimization_report['overall_speedup']:.1f}x")
print(f"内存使用减少: {optimization_report['memory_savings']:.1f}%")
print(f"缓存命中率: {optimization_report['cache_hit_rate']:.1%}")
```

### 大规模数据处理优化

```python
from optimization.large_scale_processor import LargeScaleProcessor

# 创建大规模处理器
processor = LargeScaleProcessor(
    chunk_size=10000,
    parallel_workers=8,
    memory_limit='2GB'
)

# 处理大规模股票数据
large_dataset = load_large_stock_data()  # 假设有100万条数据
processed_results = processor.process_in_chunks(
    data=large_dataset,
    processing_func=calculate_all_factors,
    progress_callback=lambda p: print(f"处理进度: {p:.1%}")
)

print(f"处理完成，结果包含 {len(processed_results)} 条记录")
```

### 自定义策略 (v1.4.0 框架)

```python
from src.strategies.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    """自定义策略示例"""
    
    def __init__(self, param1=10, param2=0.05):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data):
        """生成交易信号"""
        signals = pd.DataFrame(index=data.index, columns=data.columns)
        
        # 实现您的策略逻辑
        # ...
        
        return signals
    
    def calculate_positions(self, signals, current_positions):
        """计算目标仓位"""
        # 实现仓位计算逻辑
        # ...
        
        return target_positions

# 使用自定义策略
custom_strategy = MyCustomStrategy(param1=15, param2=0.03)
results = backtest_engine.run_backtest(custom_strategy, data)
```

### 自定义因子

```python
from src.factors.base_factor import BaseFactor

class MyCustomFactor(BaseFactor):
    """自定义因子示例"""
    
    def __init__(self, window=20):
        super().__init__()
        self.window = window
    
    def calculate(self, data):
        """计算因子值"""
        # 实现您的因子计算逻辑
        factor_values = data['Close'].rolling(self.window).apply(
            lambda x: your_custom_calculation(x)
        )
        
        return factor_values

# 注册自定义因子
factor_engine.register_factor('my_custom_factor', MyCustomFactor)
```

## 🚀 券商集成 (v1.2.0 新增)

### 支持的券商

```python
from src.brokers import BrokerFactory

# 创建券商连接
broker = BrokerFactory.create_broker(
    broker_type='td_ameritrade',  # 'charles_schwab', 'etrade', 'robinhood'
    api_key='your_api_key',
    secret_key='your_secret_key'
)

# 获取账户信息
account_info = broker.get_account_info()
print(f"账户余额: ${account_info['balance']:,.2f}")

# 获取持仓
positions = broker.get_positions()
for position in positions:
    print(f"{position['symbol']}: {position['quantity']} 股")

# 下单
order_result = broker.place_order(
    symbol='AAPL',
    quantity=100,
    order_type='market',
    side='buy'
)
```

### 实盘交易监控

```python
from src.brokers.monitor import TradingMonitor

# 创建交易监控
monitor = TradingMonitor(broker=broker)

# 启动实时监控
monitor.start_monitoring(
    strategies=[strategy1, strategy2],
    risk_limits={
        'max_daily_loss': 0.02,
        'max_position_size': 0.1
    }
)
```

## 📱 移动端支持 (v1.4.0 新增)

### 移动端API

```python
from src.mobile.api import MobileAPI

# 创建移动端API
mobile_api = MobileAPI()

# 获取简化的投资组合信息
portfolio_summary = mobile_api.get_portfolio_summary()

# 获取关键指标
key_metrics = mobile_api.get_key_metrics()

# 发送推送通知
mobile_api.send_notification(
    title="策略提醒",
    message="AAPL 触发买入信号",
    priority="high"
)
```

## 🔍 故障排除

### 常见问题

1. **数据获取失败**
   ```python
   # 检查数据源状态
   adapter = DataAdapter()
   status = adapter.check_data_sources_status()
   print("数据源状态:", status)
   
   # 使用备用数据源
   data = adapter.get_data('AAPL', source='yfinance', fallback=True)
   ```

2. **API限制问题**
   ```python
   # 使用缓存数据
   cached_data = adapter.get_cached_data('AAPL')
   if cached_data is not None:
       print("使用缓存数据")
   ```

3. **性能优化**
   ```python
   # 批量获取数据
   symbols = ['AAPL', 'GOOGL', 'MSFT']
   data_dict = adapter.get_multiple_data(symbols, batch_size=10)
   ```

### 日志和调试

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 查看系统状态
from src.utils.system_info import SystemInfo
system_info = SystemInfo()
print(system_info.get_system_status())
```

## 📚 更多资源

- 📖 [完整API文档](docs/api_reference.md)
- 🎯 [策略开发指南](docs/strategy_development.md)
- 📊 [因子研究手册](docs/factor_research.md)
- 🔧 [系统配置指南](docs/configuration.md)
- 🚀 [部署指南](docs/deployment.md)
- ⚡ [性能优化指南](docs/OPTIMIZATION_GUIDE.md) (v3.0.0 新增)
- 📈 [性能优化报告](docs/PERFORMANCE_REPORT.md) (v3.0.0 新增)
- 🔗 [集成优化指南](docs/INTEGRATION_GUIDE.md) (v3.0.0 新增)

---

**版本信息**: 本指南适用于 v3.0.0 性能优化版

**更新日志**:
- v3.0.0: 新增性能优化系统，包含智能缓存、内存池管理、性能分析器等
- v1.5.0: 新增三数据源集成系统
- v1.4.0: 新增策略开发与回测框架
- v1.3.0: 新增高级数据抓取优化
- v1.2.0: 新增券商集成支持

## 📋 示例脚本

系统提供了多个示例脚本帮助您快速上手：

### 1. 性能优化系统测试 (v3.0.0 新增)
```bash
# 集成优化系统测试
python examples/final_integration_test.py

# 性能基准测试
python examples/test_optimized_parallel_performance.py

# 大规模性能测试
python examples/test_large_scale_performance.py
```

### 2. 数据获取教程
```bash
# 基础数据获取演示
python examples/data_tutorial.py

# 数据获取演示（避免API限制）
python examples/data_fetch_demo.py

# 缓存数据使用演示
python examples/cached_data_demo.py
```

### 3. 因子分析教程
```bash
# 因子计算教程
python examples/factor_tutorial.py

# 因子评估演示
python examples/factor_evaluation.py
```

### 4. 策略测试
```bash
# 策略测试演示
python examples/strategy_testing_demo.py

# MVP演示
python examples/mvp_demo.py
```

## 🚨 常见问题

### Q: 数据获取失败怎么办？
**A**: 
1. 检查网络连接
2. 使用缓存数据：`python examples/cached_data_demo.py`
3. 添加请求延迟避免API限制
4. 检查 yfinance 库版本

### Q: FactorEngine 没有 get_data 方法？
**A**: 
请使用 `DataManager` 类获取数据：
```python
from src.data.data_manager import DataManager
data_manager = DataManager()
data = data_manager.get_data('AAPL', '2023-01-01', '2024-01-01')
```

### Q: yfinance 出现 YFRateLimitError？
**A**: 
1. 使用缓存数据避免频繁请求
2. 添加请求间隔：`time.sleep(1)`
3. 使用重试机制
4. 考虑使用代理服务器

### Q: 数据库连接失败？
**A**: 
系统可以在没有数据库的情况下正常运行，使用文件缓存作为替代。

### Q: 缓存数据格式问题？
**A**: 
使用提供的 `load_cached_data()` 函数正确加载缓存数据。

## 📚 更多资源

- [详细数据使用指南](docs/DATA_USAGE_GUIDE.md)
- [API文档](docs/api.md)
- [策略开发指南](docs/strategy_development.md)
- [因子开发指南](docs/factor_development.md)
- [最佳实践](docs/best_practices.md)
- [常见问题解答](docs/FAQ_TROUBLESHOOTING.md)

## 💡 提示和技巧 (v3.0.0 优化版)

1. **优先使用性能优化**: 充分利用智能缓存系统和内存池管理器
2. **监控性能指标**: 使用性能分析器识别瓶颈并优化
3. **自适应执行**: 启用自适应执行器自动优化计算密集型任务
4. **批量处理优化**: 使用大规模处理器处理大量数据
5. **内存管理**: 利用内存池减少内存分配开销
6. **缓存策略**: 合理配置缓存大小和过期时间
7. **并行计算**: 充分利用多核CPU进行并行处理
8. **避免API限制**: 添加适当的请求延迟，使用重试机制
9. **正确的数据获取**: 使用 `DataManager` 而不是 `FactorEngine.get_data()`
10. **参数调优**: 使用网格搜索或贝叶斯优化进行参数调优
11. **风险控制**: 始终设置合理的风险限制和止损机制

### 性能优化最佳实践

```python
# 1. 使用集成优化器
from optimization.integrated_optimizer import IntegratedOptimizer
optimizer = IntegratedOptimizer()

# 2. 配置合适的参数
optimizer.configure({
    'cache_size': 2000,      # 根据内存大小调整
    'memory_pool_size': 1024, # MB
    'parallel_workers': 8     # 根据CPU核心数调整
})

# 3. 使用装饰器自动优化
@optimizer.optimize
def your_compute_intensive_function(data):
    # 您的计算逻辑
    return results

# 4. 监控性能
performance_report = optimizer.get_performance_report()
print(f"性能提升: {performance_report['overall_speedup']:.1f}x")
```

祝您使用愉快！如有问题，请参考 [FAQ文档](docs/FAQ_TROUBLESHOOTING.md) 或查看示例脚本。