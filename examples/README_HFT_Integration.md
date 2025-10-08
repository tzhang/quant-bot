# Citadel高频交易系统 - Interactive Brokers API集成

## 概述

本项目实现了一个完整的高频交易系统，结合Citadel交易策略与Interactive Brokers API，提供实时市场数据获取、策略执行、风险管理、监控日志等全套功能。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    完整HFT交易系统                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ GUI监控系统  │  │ 日志监控系统 │  │ 风险管理器   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Citadel-IB集成引擎                        │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 市场数据流   │  │ Citadel策略  │  │ IB适配器     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Interactive Brokers API                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. IB适配器 (`ib_adapter.py`)
- **功能**: Interactive Brokers API的Python封装
- **特性**: 
  - 连接管理和自动重连
  - 市场数据订阅
  - 订单管理和执行
  - 持仓和账户信息获取
  - 异步事件处理

### 2. 市场数据流 (`ib_market_data_stream.py`)
- **功能**: 实时市场数据获取和处理
- **特性**:
  - Tick数据订阅
  - 实时K线数据
  - 数据回调机制
  - 多品种同时订阅
  - 数据质量监控

### 3. Citadel策略集成 (`citadel_ib_integration.py`)
- **功能**: 策略信号生成与交易执行的桥梁
- **特性**:
  - 策略信号实时生成
  - 智能订单路由
  - 仓位管理
  - 性能统计
  - 风险控制集成

### 4. 高级风险管理器 (`advanced_risk_manager.py`)
- **功能**: 全面的风险控制和合规监控
- **特性**:
  - 实时风险监控
  - 多层级风险限制
  - VaR计算
  - 流动性评估
  - 自动风险处置

### 5. 监控日志系统 (`hft_monitoring_logger.py`)
- **功能**: 全方位的交易活动监控和日志记录
- **特性**:
  - 多类型事件记录
  - 实时性能分析
  - 数据库存储
  - 邮件警报
  - 日报生成

### 6. GUI监控系统 (`hft_monitor_system.py`)
- **功能**: 可视化实时监控界面
- **特性**:
  - 实时数据展示
  - 交易信号可视化
  - 风险状态监控
  - 性能图表
  - 操作控制面板

### 7. 完整系统示例 (`complete_hft_example.py`)
- **功能**: 整合所有模块的完整运行示例
- **特性**:
  - 一键启动
  - 配置管理
  - 系统健康监控
  - 优雅关闭
  - 演示模式

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保IB TWS或Gateway正在运行
# TWS Demo端口: 7497
# TWS Live端口: 7496
# Gateway Demo端口: 4002
# Gateway Live端口: 4001
```

### 2. 配置设置

编辑配置文件或在代码中修改配置:

```python
config = {
    'ib_config': {
        'host': '127.0.0.1',
        'port': 7497,  # 根据你的IB设置调整
        'client_id': 1
    },
    'strategy_config': {
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'parameters': {
            'max_position_size': 1000,
            'risk_per_trade': 0.02
        }
    },
    'paper_trading': True  # 模拟交易
}
```

### 3. 运行系统

```bash
# 基本运行
python examples/complete_hft_example.py

# 演示模式（包含GUI）
python examples/complete_hft_example.py --demo
```

## 详细使用指南

### 市场数据获取

```python
from examples.ib_market_data_stream import IBMarketDataStream
from examples.ib_adapter import IBAdapter

# 创建IB连接
ib_adapter = IBAdapter(host='127.0.0.1', port=7497, client_id=1)
ib_adapter.connect()

# 创建数据流
data_stream = IBMarketDataStream(ib_adapter)

# 设置回调函数
def on_tick(symbol, data):
    print(f"{symbol}: Bid={data['bid']}, Ask={data['ask']}")

data_stream.set_tick_callback(on_tick)

# 订阅数据
data_stream.subscribe_market_data('AAPL')
```

### 策略集成

```python
from examples.citadel_ib_integration import CitadelIBIntegration
from strategies.citadel_hft_strategy import CitadelHFTStrategy

# 创建策略
strategy = CitadelHFTStrategy(symbols=['AAPL', 'MSFT'])

# 创建集成引擎
integration = CitadelIBIntegration(
    ib_adapter=ib_adapter,
    strategy=strategy,
    risk_manager=risk_manager,
    monitoring_logger=logger
)

# 启动系统
integration.start()
```

### 风险管理

```python
from examples.advanced_risk_manager import AdvancedRiskManager

# 风险配置
risk_config = {
    'max_portfolio_risk': 0.05,
    'max_position_size': 10000,
    'max_daily_loss': 5000,
    'position_limits': {
        'AAPL': 2000,
        'MSFT': 2000
    }
}

# 创建风险管理器
risk_manager = AdvancedRiskManager(risk_config)

# 检查交易风险
can_trade = risk_manager.check_trade_risk('AAPL', 'BUY', 100, 150.0)
```

### 监控日志

```python
from examples.hft_monitoring_logger import HFTMonitoringLogger

# 监控配置
monitoring_config = {
    'log_dir': './logs',
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.gmail.com',
        'to_emails': ['alert@company.com']
    }
}

# 创建监控系统
monitor = HFTMonitoringLogger(monitoring_config)
monitor.start()

# 记录交易事件
monitor.log_trade_event(
    event_type='FILL',
    symbol='AAPL',
    action='BUY',
    quantity=100,
    price=150.0,
    order_id='ORDER_001',
    strategy='CitadelHFT',
    pnl=50.0
)
```

## 配置参数详解

### IB连接配置
- `host`: IB TWS/Gateway主机地址
- `port`: 连接端口
- `client_id`: 客户端ID（避免冲突）

### 策略配置
- `symbols`: 交易品种列表
- `max_position_size`: 最大持仓数量
- `risk_per_trade`: 每笔交易风险比例
- `stop_loss_pct`: 止损百分比
- `take_profit_pct`: 止盈百分比

### 风险管理配置
- `max_portfolio_risk`: 最大组合风险
- `max_daily_loss`: 最大日损失
- `position_limits`: 品种持仓限制
- `sector_limits`: 行业配置限制

### 监控配置
- `log_dir`: 日志目录
- `email`: 邮件警报配置
- `alert_thresholds`: 警报阈值设置

## 性能优化

### 1. 延迟优化
- 使用异步处理
- 优化数据结构
- 减少不必要的计算
- 网络连接优化

### 2. 内存管理
- 限制历史数据缓存
- 定期清理过期数据
- 使用高效的数据结构

### 3. 并发处理
- 多线程数据处理
- 异步订单执行
- 并行风险检查

## 风险提示

### 1. 交易风险
- 高频交易具有高风险
- 请在充分测试后使用
- 建议先使用模拟账户

### 2. 技术风险
- 网络连接稳定性
- 系统故障处理
- 数据质量监控

### 3. 合规风险
- 遵守相关法规
- 风险控制措施
- 交易记录保存

## 故障排除

### 常见问题

1. **IB连接失败**
   - 检查TWS/Gateway是否运行
   - 确认端口配置正确
   - 检查API权限设置

2. **数据订阅失败**
   - 确认品种代码正确
   - 检查市场数据权限
   - 验证交易时间

3. **订单执行失败**
   - 检查账户资金
   - 确认交易权限
   - 验证订单参数

4. **性能问题**
   - 监控系统资源使用
   - 优化数据处理逻辑
   - 调整并发参数

### 日志分析

系统会生成多种日志文件：
- `hft_main_YYYYMMDD.log`: 主系统日志
- `hft_trades_YYYYMMDD.log`: 交易日志
- `hft_risk_YYYYMMDD.log`: 风险日志
- `hft_data.db`: SQLite数据库

## 扩展开发

### 添加新策略
1. 继承`CitadelHFTStrategy`基类
2. 实现`generate_signals`方法
3. 在集成引擎中注册

### 自定义风险规则
1. 扩展`AdvancedRiskManager`类
2. 添加新的风险检查方法
3. 更新风险配置

### 新增监控指标
1. 扩展`HFTMonitoringLogger`类
2. 添加新的事件类型
3. 更新数据库结构

## 技术支持

如有问题或建议，请：
1. 查看日志文件
2. 检查配置设置
3. 参考API文档
4. 联系技术支持

## 版本历史

- v1.0.0: 初始版本，基础功能实现
- v1.1.0: 添加GUI监控系统
- v1.2.0: 增强风险管理功能
- v1.3.0: 优化性能和稳定性

## 许可证

本项目仅供学习和研究使用，请勿用于实际交易，风险自负。