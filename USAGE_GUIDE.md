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

### 获取股票数据

```python
# 基本用法
data = data_manager.get_data(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    period='2y',  # 时间周期
    interval='1d'  # 数据频率
)

# 指定日期范围
data = data_manager.get_data(
    symbols=['AAPL'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# 获取基本面数据
fundamental_data = data_manager.get_fundamental_data(['AAPL'])
print(fundamental_data['AAPL']['market_cap'])
```

### 数据缓存管理

```python
# 清理缓存
data_manager.clear_cache()

# 获取缓存统计
cache_stats = data_manager.get_cache_stats()
print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")

# 预热缓存
data_manager.warm_cache(['AAPL', 'MSFT'], period='1y')
```

## 🧮 因子计算

### 创建因子引擎

```python
# 创建因子引擎
factor_engine = quant.create_factor_engine()

# 计算技术因子
tech_factors = factor_engine.compute_technical(data)
print(f"计算了 {len(tech_factors.columns)} 个技术因子")

# 计算基本面因子
fundamental_factors = factor_engine.compute_fundamental(
    price_data=data,
    fundamental_data=fundamental_data
)
```

### 多因子模型

```python
from src.factors import MultiFactorModel, FactorConfig, ModelConfig

# 配置因子
factor_config = FactorConfig(
    technical_factors=['momentum_20', 'rsi_14', 'volatility_20'],
    fundamental_factors=['pe_ratio', 'pb_ratio', 'roe'],
    risk_factors=['market_beta', 'size_factor']
)

# 配置模型
model_config = ModelConfig(
    model_type='ridge',  # 'linear', 'ridge', 'lasso', 'random_forest'
    alpha=0.1,
    lookback_window=252,
    rebalance_frequency='monthly'
)

# 创建和训练模型
model = MultiFactorModel(factor_config, model_config)
model.fit(factor_data, return_data)

# 预测收益
predictions = model.predict(new_factor_data)
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

## 🚨 常见问题

### Q: 数据获取失败怎么办？
A: 检查网络连接和API密钥配置，可以尝试清理缓存后重新获取。

### Q: 回测结果不准确？
A: 确保数据质量，检查交易成本设置，验证策略逻辑的正确性。

### Q: 内存使用过高？
A: 使用数据分批处理，及时清理不需要的变量，调整缓存大小。

### Q: 计算速度慢？
A: 启用并行计算，使用缓存机制，优化数据结构。

## 📚 更多资源

- [API文档](docs/api.md)
- [策略开发指南](docs/strategy_development.md)
- [因子开发指南](docs/factor_development.md)
- [最佳实践](docs/best_practices.md)
- [常见问题解答](docs/faq.md)

## 💡 提示和技巧

1. **数据缓存**: 充分利用缓存机制，避免重复获取相同数据
2. **批量处理**: 尽量批量处理多只股票，提高效率
3. **内存管理**: 及时释放不需要的大型数据对象
4. **参数调优**: 使用网格搜索或贝叶斯优化进行参数调优
5. **风险控制**: 始终设置合理的风险限制和止损机制

祝您使用愉快！如有问题，请随时联系我们。