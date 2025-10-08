# 系统架构文档

## 概述

本量化交易系统采用模块化设计，通过清晰的分层架构实现了数据管理、因子计算、策略执行、回测分析和风险管理的完整流程。系统支持多种数据源、多种策略和灵活的扩展机制。

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    主入口程序 (main.py)                      │
├─────────────────────────────────────────────────────────────┤
│                    配置管理层                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   ConfigManager │  │   LoggerManager │  │   CacheManager  ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    数据管理层                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   DataManager   │  │   DataFetcher   │  │   DataValidator ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    因子计算层                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │  FactorEngine   │  │ TechnicalFactors│  │  FactorValidator││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    策略执行层                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │ MomentumStrategy│  │  BaseStrategy   │  │ StrategyManager ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    回测分析层                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │ BacktestEngine  │  │PerformanceAnalyzer│ │  RiskManager   ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    工具支持层                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   Visualizer    │  │   Optimizer     │  │   Monitor       ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 核心模块详解

### 1. 主入口程序 (main.py)

**职责**: 系统统一入口，协调各模块工作

**核心类**: `QuantTradingSystem`

**主要功能**:
- 系统初始化和配置管理
- 模块间协调和数据流控制
- 命令行界面提供
- 快速演示和系统信息展示

**关键方法**:
```python
class QuantTradingSystem:
    def initialize(self)           # 初始化所有模块
    def get_system_info(self)      # 获取系统信息
    def quick_start_demo(self)     # 快速演示
    def get_data(self, symbols)    # 数据获取接口
    def calculate_factors(self)    # 因子计算接口
    def run_backtest(self)         # 回测执行接口
```

### 2. 配置管理层

#### ConfigManager
- **位置**: `src/core/utils.py`
- **职责**: 统一配置管理
- **功能**: 配置文件加载、环境变量管理、参数验证

#### LoggerManager
- **位置**: `src/utils/logger.py`
- **职责**: 日志系统管理
- **功能**: 多级别日志、日志轮转、性能监控

#### CacheManager
- **职责**: 缓存系统管理
- **功能**: 数据缓存、缓存清理、缓存策略

### 3. 数据管理层

#### DataManager
- **位置**: `src/data/manager.py`
- **职责**: 数据管理总控制器
- **功能**: 
  - 多数据源统一接口
  - 数据缓存管理
  - 数据质量控制

**核心方法**:
```python
class DataManager:
    def get_data(self, symbols, start_date, end_date)  # 获取数据
    def cache_data(self, data, key)                    # 缓存数据
    def validate_data(self, data)                      # 数据验证
```

#### DataFetcher
- **位置**: `src/data/fetcher.py`
- **职责**: 数据获取实现
- **支持的数据源**:
  - Qlib (量化数据平台)
  - OpenBB (开源金融数据)
  - yfinance (Yahoo Finance)
  - 自定义数据源

### 4. 因子计算层

#### FactorEngine
- **位置**: `src/factors/engine.py`
- **职责**: 因子计算引擎
- **功能**:
  - 批量因子计算
  - 因子有效性验证
  - 因子缓存管理

**核心方法**:
```python
class FactorEngine:
    def compute_technical(self, data)    # 计算技术因子
    def compute_fundamental(self, data)  # 计算基本面因子
    def validate_factors(self, factors)  # 因子验证
```

#### TechnicalIndicators
- **位置**: `src/factors/technical.py`
- **职责**: 技术指标计算
- **支持的指标**:
  - RSI (相对强弱指数)
  - MACD (移动平均收敛散度)
  - 布林带 (Bollinger Bands)
  - 移动平均线 (MA)
  - 随机指标 (Stochastic)

### 5. 策略执行层

#### BaseStrategy
- **位置**: `src/strategies/templates.py`
- **职责**: 策略基类
- **功能**: 策略接口定义、通用方法实现

#### MomentumStrategy
- **位置**: `src/strategies/templates.py`
- **职责**: 动量策略实现
- **参数**:
  - `fast`: 快速EMA周期
  - `slow`: 慢速EMA周期

**策略逻辑**:
```python
def signal(self, data):
    # 计算快慢EMA
    fast_ema = data['close'].ewm(span=self.fast).mean()
    slow_ema = data['close'].ewm(span=self.slow).mean()
    
    # 生成信号
    signals = np.where(fast_ema > slow_ema, 1, -1)
    return signals
```

### 6. 回测分析层

#### BacktestEngine
- **位置**: `src/backtest/engine.py`
- **职责**: 回测引擎
- **功能**:
  - 历史数据回测
  - 交易成本计算
  - 仓位管理
  - 风险控制

**核心方法**:
```python
class BacktestEngine:
    def run(self, strategy, data, initial_capital)  # 执行回测
    def calculate_returns(self, signals, prices)    # 计算收益
    def apply_transaction_costs(self, returns)      # 应用交易成本
```

#### PerformanceAnalyzer
- **位置**: `src/performance/analyzer.py`
- **职责**: 性能分析
- **计算指标**:
  - 累计收益率 (Cumulative Return)
  - 年化收益率 (Annualized Return)
  - 年化波动率 (Annualized Volatility)
  - 夏普比率 (Sharpe Ratio)
  - 最大回撤 (Maximum Drawdown)
  - 索提诺比率 (Sortino Ratio)
  - 卡尔玛比率 (Calmar Ratio)
  - 胜率 (Hit Rate)

### 7. 风险管理层

#### RiskManager
- **位置**: `src/risk/manager.py`
- **职责**: 风险管理
- **功能**:
  - 风险评估
  - 仓位限制
  - 止损止盈
  - 风险预警

## 数据流架构

### 1. 数据获取流程

```
用户请求 → DataManager → DataFetcher → 数据源API
                ↓
            缓存检查 ← CacheManager
                ↓
            数据验证 ← DataValidator
                ↓
            返回数据
```

### 2. 因子计算流程

```
原始数据 → FactorEngine → TechnicalIndicators
              ↓
          因子验证 ← FactorValidator
              ↓
          因子缓存 ← CacheManager
              ↓
          返回因子
```

### 3. 策略回测流程

```
历史数据 + 策略参数 → Strategy.signal() → 交易信号
                                        ↓
                    BacktestEngine ← 交易信号
                         ↓
                    PerformanceAnalyzer ← 回测结果
                         ↓
                    RiskManager ← 性能指标
                         ↓
                    最终报告
```

## 扩展机制

### 1. 添加新数据源

```python
# 在 src/data/fetcher.py 中添加
class NewDataSource:
    def fetch_data(self, symbols, start_date, end_date):
        # 实现数据获取逻辑
        return data

# 在 DataFetcher 中注册
self.sources['new_source'] = NewDataSource()
```

### 2. 添加新策略

```python
# 继承 BaseStrategy
class NewStrategy(BaseStrategy):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def signal(self, data):
        # 实现策略逻辑
        return signals
```

### 3. 添加新因子

```python
# 在 TechnicalIndicators 中添加
def new_factor(self, data, period=20):
    # 实现因子计算
    return factor_values
```

## 性能优化

### 1. 缓存策略

- **数据缓存**: 原始数据本地缓存，避免重复下载
- **因子缓存**: 计算结果缓存，提升重复计算效率
- **策略缓存**: 策略结果缓存，支持增量更新

### 2. 并行计算

- **多进程**: 多股票并行处理
- **向量化**: NumPy/Pandas向量化计算
- **批处理**: 批量数据处理

### 3. 内存管理

- **数据分片**: 大数据集分片处理
- **垃圾回收**: 及时释放不用的对象
- **内存监控**: 实时内存使用监控

## 配置管理

### 1. 配置文件结构

```yaml
# config/config.yaml
data:
  sources: ['qlib', 'openbb', 'yfinance']
  cache_dir: './data_cache'
  
strategies:
  momentum:
    fast: 12
    slow: 26
    
backtest:
  initial_capital: 100000
  transaction_cost: 0.001
  
logging:
  level: INFO
  file: './logs/system.log'
```

### 2. 环境变量

```bash
# 数据源API密钥
export OPENBB_API_KEY="your_api_key"
export ALPHA_VANTAGE_API_KEY="your_api_key"

# 系统配置
export QUANT_CACHE_DIR="./data_cache"
export QUANT_LOG_LEVEL="INFO"
```

## 监控和调试

### 1. 日志系统

- **分级日志**: DEBUG, INFO, WARNING, ERROR
- **模块日志**: 每个模块独立日志
- **性能日志**: 执行时间和资源使用

### 2. 性能监控

- **执行时间**: 各模块执行时间统计
- **内存使用**: 内存使用情况监控
- **缓存命中率**: 缓存效率统计

### 3. 错误处理

- **异常捕获**: 全局异常处理机制
- **错误恢复**: 自动错误恢复策略
- **错误报告**: 详细错误信息记录

## 部署架构

### 1. 开发环境

```
本地开发 → Git版本控制 → 单元测试 → 集成测试
```

### 2. 生产环境

```
代码部署 → 环境配置 → 数据初始化 → 服务启动 → 监控告警
```

### 3. 容器化部署

```dockerfile
# Dockerfile
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "main.py", "--demo"]
```

---

*本文档描述了系统的整体架构设计，为开发者提供了全面的技术参考。*