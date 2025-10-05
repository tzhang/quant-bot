# 使用指南 (Usage Guide)

本指南将帮助您快速上手量化交易系统的各项功能。

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/quant-system.git
cd quant-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 基本配置

创建 `.env` 文件：
```env
# API密钥
ALPHA_VANTAGE_API_KEY=your_key_here
QUANDL_API_KEY=your_key_here

# 数据库配置（可选）
DATABASE_URL=postgresql://user:password@localhost:5432/quant_db
REDIS_URL=redis://localhost:6379/0

# 缓存配置
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
```

### 3. 第一个示例

```python
import src as quant

# 创建数据管理器
data_manager = quant.create_data_manager()

# 获取股票数据
data = data_manager.get_data(['AAPL', 'GOOGL'], period='1y')
print(f"获取到 {len(data)} 条数据")

# 查看系统版本
print(f"系统版本: {quant.get_version()}")
```

## 📊 数据管理

### 正确的数据获取方法

**重要提示**: 请使用 `DataManager` 类进行数据获取，而不是 `FactorEngine.get_data()`

```python
from src.data.data_manager import DataManager

# 创建数据管理器
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
from src.data.data_manager import DataManager

# 1. 获取数据
data_manager = DataManager()
data = data_manager.get_data('AAPL', '2023-01-01', '2024-01-01')

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

## 📈 投资组合优化

### 基本优化

```python
from src.factors import PortfolioOptimizer, OptimizationConstraint

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
from src.factors import BlackLittermanOptimizer

bl_optimizer = BlackLittermanOptimizer()
bl_result = bl_optimizer.optimize(
    market_caps=market_caps,
    returns=historical_returns,
    views=investor_views,  # 投资者观点
    view_confidences=confidences
)

# 风险平价优化
from src.factors import RiskParityOptimizer

rp_optimizer = RiskParityOptimizer()
rp_result = rp_optimizer.optimize(covariance_matrix=cov_matrix)
```

## 🔄 回测引擎

### 基本回测

```python
# 创建回测引擎
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
# 创建风险管理器
risk_manager = quant.create_risk_manager()

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
from src.risk import StressTestScenario

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

## 🗄️ 数据库优化

### 缓存管理

```python
# 获取缓存数据
cached_data = data_manager.get_cached_data(
    key="AAPL_1y_daily",
    fallback_func=lambda: data_manager.get_data(['AAPL'], period='1y')
)

# 优化查询
optimized_query = data_manager.optimize_query(
    "SELECT * FROM stock_prices WHERE symbol = 'AAPL'"
)

# 获取性能统计
perf_stats = data_manager.get_performance_stats()
print(f"平均查询时间: {perf_stats['avg_query_time']:.2f}ms")
```

## 📊 可视化分析

### 基本图表

```python
from src.utils import Visualizer

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

## 🔧 高级功能

### 自定义策略

```python
from src.strategies import BaseStrategy

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
from src.factors import BaseFactor

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

## 📋 示例脚本

系统提供了多个示例脚本帮助您快速上手：

### 1. 数据获取教程
```bash
# 基础数据获取演示
python examples/data_tutorial.py

# 数据获取演示（避免API限制）
python examples/data_fetch_demo.py

# 缓存数据使用演示
python examples/cached_data_demo.py
```

### 2. 因子分析教程
```bash
# 因子计算教程
python examples/factor_tutorial.py

# 因子评估演示
python examples/factor_evaluation.py
```

### 3. 策略测试
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

## 💡 提示和技巧

1. **优先使用缓存**: 充分利用 `data_cache` 目录中的缓存数据
2. **避免API限制**: 添加适当的请求延迟，使用重试机制
3. **正确的数据获取**: 使用 `DataManager` 而不是 `FactorEngine.get_data()`
4. **批量处理**: 尽量批量处理多只股票，提高效率
5. **内存管理**: 及时释放不需要的大型数据对象
6. **参数调优**: 使用网格搜索或贝叶斯优化进行参数调优
7. **风险控制**: 始终设置合理的风险限制和止损机制

祝您使用愉快！如有问题，请参考 [FAQ文档](docs/FAQ_TROUBLESHOOTING.md) 或查看示例脚本。