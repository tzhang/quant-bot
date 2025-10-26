# 自动化交易系统使用指南

## 系统概述

这是一个完整的自动化交易系统，支持多策略信号聚合、智能风险管理、实时监控和灵活配置。系统支持模拟交易、实盘交易和回测模式。

## 核心功能

### 🎯 多策略支持
- **动量策略 (Momentum)**: 基于价格趋势的交易策略
- **均值回归策略 (Mean Reversion)**: 基于价格回归的交易策略  
- **突破策略 (Breakout)**: 基于价格突破的交易策略
- **机器学习策略 (ML Prediction)**: 基于机器学习预测的交易策略

### 🛡️ 风险管理
- 单个持仓最大价值限制
- 每日最大亏损限制
- 单个标的最大暴露限制
- 每日最大交易次数限制
- 动态止损止盈
- 投资组合风险控制

### 📊 实时监控
- 实时性能报告
- 持仓监控
- 风险警报
- 交易历史记录
- 系统健康检查

### ⚙️ 灵活配置
- YAML配置文件
- 命令行参数覆盖
- 交互式配置
- 策略参数调整

## 快速开始

### 1. 安装依赖

```bash
pip install schedule pyyaml psutil yfinance pandas numpy
```

### 2. 快速启动（模拟交易）

```bash
python run_trading.py --quick
```

### 3. 交互式配置启动

```bash
python run_trading.py --interactive
```

### 4. 命令行参数启动

```bash
# 模拟交易模式
python run_trading.py --mode paper --capital 50000 --symbols AAPL GOOGL MSFT

# 实盘交易模式（谨慎使用）
python run_trading.py --mode live --capital 10000 --symbols AAPL

# 回测模式
python run_trading.py --mode backtest --symbols AAPL GOOGL MSFT TSLA AMZN
```

## 配置文件说明

系统使用 `trading_config.yaml` 配置文件，主要配置项包括：

### 交易设置
```yaml
trading:
  mode: paper  # paper, live, backtest
  initial_capital: 100000.0
  symbols: [AAPL, GOOGL, MSFT, TSLA, AMZN]
  trading_hours:
    start: "09:30"
    end: "16:00"
  data_update_interval: 60  # 秒
  signal_generation_interval: 300  # 秒
```

### 风险管理
```yaml
risk_management:
  max_position_value: 50000.0
  max_daily_loss: 5000.0
  max_symbol_exposure: 20000.0
  max_daily_trades: 100
  stop_loss_pct: 0.05
  take_profit_pct: 0.10
```

### 策略配置
```yaml
strategies:
  momentum:
    enabled: true
    weight: 1.0
    parameters:
      base_position_size: 100
      max_position_size: 500
```

## 系统架构

### 核心组件

1. **AutomatedTradingSystem**: 主系统类，协调所有组件
2. **EnhancedTradingEngine**: 交易执行引擎，处理订单和风险管理
3. **StrategyManager**: 策略管理器，聚合多个策略信号
4. **DataProvider**: 数据提供者，获取市场数据
5. **TradingConfig**: 配置管理器，处理系统配置

### 数据流

```
市场数据 → 策略分析 → 信号生成 → 风险检查 → 订单执行 → 持仓更新
```

## 运行时命令

系统启动后，可以使用以下命令：

- `s` 或 `status`: 显示系统状态
- `r` 或 `report`: 生成性能报告
- `q` 或 `quit`: 退出系统
- `h` 或 `help`: 显示帮助

## 测试结果

✅ **系统启动测试**: 成功
- 配置文件自动生成
- 多线程启动正常
- 策略管理器初始化成功
- 交易引擎启动正常

✅ **数据获取测试**: 部分成功
- Yahoo Finance API有频率限制
- 模拟数据生成正常
- 数据缓存机制工作正常

✅ **风险管理测试**: 成功
- 风险限制配置正确
- 止损止盈机制正常

✅ **性能监控测试**: 成功
- 实时性能报告生成
- 系统状态监控正常
- 日志记录完整

## 注意事项

### ⚠️ 重要警告

1. **实盘交易风险**: 实盘交易模式会使用真实资金，请确保充分测试策略
2. **数据源限制**: Yahoo Finance有API频率限制，建议配置备用数据源
3. **策略风险**: 所有策略都有风险，过往表现不代表未来收益

### 🔧 系统要求

- Python 3.8+
- 足够的内存运行多线程
- 稳定的网络连接
- 有效的数据源API密钥（可选）

### 📈 性能优化建议

1. **数据更新频率**: 根据策略需求调整数据更新间隔
2. **信号生成频率**: 平衡信号质量和计算资源
3. **风险参数**: 根据风险承受能力调整风险限制
4. **策略权重**: 根据历史表现调整策略权重

## 扩展功能

### 添加新策略

1. 继承 `BaseStrategy` 类
2. 实现 `generate_signal` 方法
3. 在 `StrategyManager` 中注册策略

### 添加新数据源

1. 在 `DataProvider` 中添加新的数据获取方法
2. 更新配置文件中的数据源选项
3. 实现数据格式转换

### 添加通知功能

1. 实现邮件/Slack通知方法
2. 在配置文件中添加通知设置
3. 在风险警报回调中调用通知方法

## 故障排除

### 常见问题

1. **模块导入错误**: 检查依赖包是否安装完整
2. **数据获取失败**: 检查网络连接和API限制
3. **配置文件错误**: 检查YAML格式是否正确
4. **权限错误**: 检查文件读写权限

### 日志分析

系统会生成详细的日志文件 `automated_trading.log`，包含：
- 系统启动和停止信息
- 交易信号生成记录
- 订单执行详情
- 风险警报信息
- 错误和异常信息

## 联系支持

如有问题或建议，请查看日志文件或联系技术支持。

---

**免责声明**: 本系统仅供学习和研究使用，不构成投资建议。使用者需自行承担投资风险。