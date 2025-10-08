# 主入口程序使用指南 (main.py)

## 概述

`main.py` 是量化交易系统的统一入口程序，整合了数据管理、因子计算、策略回测、性能分析和风险评估等所有核心功能。通过简单的命令行参数，您可以快速体验完整的量化交易流程。

## 快速开始

### 基本命令

```bash
# 查看系统信息
python main.py --info

# 运行快速演示
python main.py --demo

# 自定义参数运行
python main.py --demo --cache-dir ./my_cache --initial-capital 50000
```

### 命令行参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--info` | 显示系统信息 | - | `python main.py --info` |
| `--demo` | 运行快速演示 | - | `python main.py --demo` |
| `--cache-dir` | 指定缓存目录 | `./data_cache` | `--cache-dir ./my_cache` |
| `--initial-capital` | 设置初始资金 | `100000` | `--initial-capital 50000` |

## 系统架构

### 核心模块

1. **数据管理层 (DataManager)**
   - 多数据源支持 (Qlib, OpenBB, yfinance)
   - 智能缓存机制
   - 自动数据清理

2. **因子计算层 (FactorEngine)**
   - 技术因子计算 (RSI, MACD, 布林带等)
   - 批量因子处理
   - 因子有效性验证

3. **策略层 (Strategy)**
   - 动量策略
   - 均值回归策略
   - 自定义策略支持

4. **回测层 (BacktestEngine)**
   - 完整回测框架
   - 风险管理集成
   - 仓位管理

5. **分析层 (PerformanceAnalyzer)**
   - 性能指标计算
   - 风险调整收益
   - 可视化分析

6. **风险管理层 (RiskManager)**
   - 风险评估
   - 风险限制
   - 动态风险调整

## 功能详解

### 1. 系统信息查看

```bash
python main.py --info
```

输出内容包括：
- 系统名称和版本
- 作者信息
- 系统描述
- 包含的核心模块

### 2. 快速演示流程

```bash
python main.py --demo
```

演示流程包括：

#### 步骤1: 数据获取
- 获取AAPL、GOOGL、MSFT三只股票的2023年数据
- 自动缓存数据，提升后续访问速度
- 数据质量检查和验证

#### 步骤2: 技术因子计算
- 计算RSI、MACD、布林带等技术指标
- 批量处理多只股票
- 因子有效性验证

#### 步骤3: 策略回测
- 运行动量策略回测
- 生成交易信号
- 计算策略收益

#### 步骤4: 性能分析
- 计算关键性能指标：
  - 累计收益率 (cum_return)
  - 年化收益率 (ann_return)
  - 年化波动率 (ann_vol)
  - 夏普比率 (sharpe)
  - 最大回撤 (max_drawdown)
  - 索提诺比率 (sortino)
  - 卡尔玛比率 (calmar)
  - 胜率 (hit_rate)

#### 步骤5: 风险评估
- 评估投资风险
- 风险调整收益分析
- 风险预警

## 编程接口使用

### 基本用法

```python
from main import QuantTradingSystem

# 初始化系统
system = QuantTradingSystem(
    cache_dir="./data_cache",
    initial_capital=100000
)

# 初始化所有模块
system.initialize()

# 运行快速演示
system.quick_start_demo()

# 获取系统信息
info = system.get_system_info()
print(info)
```

### 高级用法

```python
# 自定义数据获取
data = system.get_data(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# 计算技术因子
factors = system.calculate_factors(data)

# 运行回测
results = system.run_backtest(
    strategy_name='momentum',
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# 分析性能
performance = system.analyze_performance(results)

# 评估风险
risk_metrics = system.assess_risk(results)
```

## 配置选项

### 缓存配置

```python
# 自定义缓存目录
system = QuantTradingSystem(cache_dir="./custom_cache")

# 禁用缓存
system = QuantTradingSystem(use_cache=False)
```

### 资金配置

```python
# 设置初始资金
system = QuantTradingSystem(initial_capital=50000)
```

### 日志配置

系统使用统一的日志管理，支持：
- 多级别日志 (DEBUG, INFO, WARNING, ERROR)
- 自动日志轮转
- 性能监控日志

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ImportError: cannot import name 'Logger' from 'src.utils'
   ```
   解决方案：确保使用正确的导入路径，系统已自动修复此问题。

2. **数据获取失败**
   - 检查网络连接
   - 确认股票代码正确
   - 查看缓存目录权限

3. **内存不足**
   - 减少处理的股票数量
   - 缩短时间范围
   - 清理缓存数据

### 性能优化建议

1. **使用缓存**: 启用数据缓存可显著提升性能
2. **批量处理**: 一次处理多只股票比单独处理更高效
3. **合理的时间范围**: 避免处理过长的时间序列
4. **定期清理**: 定期清理过期的缓存数据

## 扩展开发

### 添加新策略

```python
from src.strategies.templates import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def signal(self, data):
        # 实现您的策略逻辑
        return signals
```

### 添加新因子

```python
from src.factors.technical import TechnicalIndicators

class MyFactors(TechnicalIndicators):
    def my_custom_factor(self, data):
        # 实现您的因子计算
        return factor_values
```

## 更新日志

### v1.0.0 (当前版本)
- 统一主入口程序
- 完整的量化交易流程
- 智能缓存系统
- 多模块集成
- 命令行界面

## 支持与反馈

如果您在使用过程中遇到问题或有改进建议，请：

1. 查看本文档的故障排除部分
2. 检查系统日志文件
3. 提交Issue或Pull Request
4. 联系开发团队

---

*本文档持续更新中，最后更新时间：2025-10-08*