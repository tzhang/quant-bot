# Interactive Brokers (IB) 交易系统用户手册

## 目录

1. [系统介绍](#系统介绍)
2. [安装和配置](#安装和配置)
3. [快速入门](#快速入门)
4. [功能详解](#功能详解)
5. [高级功能](#高级功能)
6. [故障排除](#故障排除)
7. [常见问题](#常见问题)

## 系统介绍

Interactive Brokers (IB) 交易系统是一个专为量化交易设计的Python平台，提供以下核心功能：

### 主要特性

- **自动化交易**: 支持市价单、限价单、止损单等多种订单类型
- **风险管理**: 内置风险控制机制，包括仓位管理、止损设置等
- **技术分析**: 提供50+种技术指标和交易策略
- **实时监控**: Web界面实时监控交易状态和系统性能
- **回测功能**: 历史数据回测验证交易策略
- **模拟交易**: 支持模拟模式，无风险测试策略

### 系统要求

- **Python 3.12** (强制要求)
- macOS 10.15+ / Windows 10+ / Linux
- 内存: 最少4GB，推荐8GB+
- 网络: 稳定的互联网连接
- Interactive Brokers账户（用于实盘交易）
- IB TWS (Trader Workstation) 或 IB Gateway

## 安装和配置

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/ib-trading-system.git
cd ib-trading-system

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置文件设置

创建配置文件 `config/config.yaml`:

```yaml
# IB连接配置
ib_connection:
  host: "127.0.0.1"
  port: 7497  # TWS: 7497, Gateway: 4001
  client_id: 1

# 交易配置
trading:
  dry_run: true  # 模拟模式，设为false启用实盘交易
  max_order_size: 10000  # 最大订单金额
  default_quantity: 100  # 默认交易数量

# 风险管理配置
risk_management:
  max_daily_loss: 5000  # 最大日损失
  max_position_size: 0.1  # 最大仓位比例
  stop_loss_pct: 0.02  # 止损百分比
  allowed_symbols:  # 允许交易的股票
    - "AAPL"
    - "GOOGL"
    - "MSFT"
    - "TSLA"

# 系统配置
system:
  log_level: "INFO"
  cache_enabled: true
  max_retries: 3
  timeout: 30

# 监控配置
monitoring:
  enabled: true
  port: 8080
  update_interval: 5  # 秒
```

### 3. 环境变量设置

为了安全起见，建议使用环境变量存储敏感信息。创建 `.env` 文件：

```bash
# IB连接参数
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

# API密钥（如果使用第三方数据源）
ALPHA_VANTAGE_API_KEY=your_api_key
POLYGON_API_KEY=your_api_key
```

## 快速入门

### 第一次运行

1. **启动IB TWS或Gateway**

首先确保IB TWS (Trader Workstation) 或 IB Gateway已启动并配置好API连接。

2. **启动系统**

```bash
cd examples
python ib_automated_trading_system.py
```

3. **基本交易操作**

```python
from ib_automated_trading_system import IBAutomatedTradingSystem

# 创建交易系统实例
trading_system = IBAutomatedTradingSystem(
    host="127.0.0.1",
    port=7497,
    client_id=1,
    use_ib=True  # 使用IB实盘交易，设为False使用模拟模式
)

# 连接
if trading_system.connect():
    print("✅ 连接成功")
    
    # 查看账户信息
    account = trading_system.get_account_info()
    print(f"💰 账户余额: ${account['cash']:,.2f}")
    
    # 获取股票报价
    quote = trading_system.get_quote("AAPL")
    print(f"📈 AAPL价格: ${quote['price']:.2f}")
    
    # 下单买入
    result = trading_system.place_order(
        symbol="AAPL",
        quantity=10,
        order_type="MKT",
        action="BUY"
    )
    
    if result['status'] == 'success':
        print(f"✅ 订单成功: {result['order_id']}")
    else:
        print(f"❌ 订单失败: {result['message']}")
        
    # 断开连接
    trading_system.disconnect()
```

### 启动监控面板

```bash
cd examples
python monitoring_dashboard.py
```

然后在浏览器中访问 `http://localhost:8080` 查看实时监控界面。

## 功能详解

### 1. 账户管理

#### 查看账户信息

```python
# 获取完整账户信息
account_info = trading_system.get_account_info()
print(f"现金余额: ${account_info['cash']:,.2f}")
print(f"购买力: ${account_info['buying_power']:,.2f}")
print(f"总资产: ${account_info['total_value']:,.2f}")

# 获取持仓信息
positions = trading_system.get_positions()
for position in positions:
    print(f"{position['symbol']}: {position['quantity']}股, "
          f"成本${position['avg_price']:.2f}, "
          f"市值${position['market_value']:,.2f}")
```

#### 计算投资组合表现

```python
# 获取投资组合表现
performance = trading_system.calculate_portfolio_performance()
print(f"总收益率: {performance['total_return']:.2%}")
print(f"年化收益率: {performance['annualized_return']:.2%}")
print(f"夏普比率: {performance['sharpe_ratio']:.2f}")
print(f"最大回撤: {performance['max_drawdown']:.2%}")
```

### 2. 市场数据

#### 获取实时报价

```python
# 单个股票报价
quote = trading_system.get_quote("AAPL")
print(f"价格: ${quote['price']:.2f}")
print(f"买价: ${quote['bid']:.2f}")
print(f"卖价: ${quote['ask']:.2f}")
print(f"成交量: {quote['volume']:,}")

# 批量获取报价
symbols = ["AAPL", "GOOGL", "MSFT"]
quotes = trading_system.get_quotes_batch(symbols)
for symbol, quote in quotes.items():
    print(f"{symbol}: ${quote['price']:.2f}")
```

#### 获取历史数据

```python
# 获取历史K线数据
historical_data = trading_system.get_historical_data("AAPL", "1 Y")
print(f"获取到 {len(historical_data)} 条历史数据")

# 数据格式
for data_point in historical_data[-5:]:  # 显示最近5天
    print(f"{data_point['date']}: "
          f"开盘${data_point['open']:.2f}, "
          f"收盘${data_point['close']:.2f}, "
          f"成交量{data_point['volume']:,}")
```

### 3. 订单管理

#### 下单操作

```python
# 市价买单
buy_order = trading_system.place_order(
    symbol="AAPL",
    quantity=100,
    order_type="MKT",
    action="BUY"
)

# 限价卖单
sell_order = trading_system.place_order(
    symbol="AAPL",
    quantity=50,
    order_type="LMT",
    action="SELL",
    price=155.00
)

# 止损单
stop_order = trading_system.place_order(
    symbol="AAPL",
    quantity=100,
    order_type="STP",
    action="SELL",
    price=145.00  # 止损价格
)
```

#### 订单管理

```python
# 查看所有订单
all_orders = trading_system.get_orders("all")
print(f"总订单数: {len(all_orders)}")

# 查看未成交订单
open_orders = trading_system.get_orders("open")
for order in open_orders:
    print(f"订单ID: {order['order_id']}")
    print(f"股票: {order['symbol']}")
    print(f"数量: {order['quantity']}")
    print(f"状态: {order['status']}")

# 取消订单
if open_orders:
    cancel_result = trading_system.cancel_order(open_orders[0]['order_id'])
    if cancel_result['status'] == 'success':
        print("✅ 订单取消成功")
```

### 4. 风险管理

#### 设置风险参数

```python
# 配置风险管理
risk_config = {
    'max_order_size': 5000,      # 单笔最大订单金额
    'max_daily_loss': 1000,      # 日最大损失
    'max_position_size': 0.05,   # 单只股票最大仓位
    'stop_loss_pct': 0.02,       # 止损百分比
    'allowed_symbols': ['AAPL', 'GOOGL', 'MSFT']
}

trading_system.update_risk_config(risk_config)
```

#### 风险检查

```python
# 检查订单风险
order_data = {
    'symbol': 'AAPL',
    'quantity': 100,
    'price': 150.00,
    'action': 'BUY'
}

is_valid, error_msg = trading_system.validate_order_risk(order_data)
if not is_valid:
    print(f"❌ 风险检查失败: {error_msg}")
else:
    print("✅ 风险检查通过")
```

## 高级功能

### 1. 技术分析和交易策略

#### 使用技术指标

```python
from technical_indicators import TechnicalIndicators

# 创建技术指标分析器
indicators = TechnicalIndicators()

# 获取历史数据
data = trading_system.get_historical_data("AAPL", "6 M")

# 计算技术指标
sma_20 = indicators.sma(data, 20)  # 20日简单移动平均
ema_12 = indicators.ema(data, 12)  # 12日指数移动平均
rsi = indicators.rsi(data, 14)     # 14日RSI
macd = indicators.macd(data)       # MACD指标
bb = indicators.bollinger_bands(data, 20, 2)  # 布林带

# 打印最新指标值
print(f"SMA(20): ${sma_20[-1]:.2f}")
print(f"EMA(12): ${ema_12[-1]:.2f}")
print(f"RSI(14): {rsi[-1]:.2f}")
print(f"MACD: {macd['macd'][-1]:.4f}")
print(f"布林带上轨: ${bb['upper'][-1]:.2f}")
print(f"布林带下轨: ${bb['lower'][-1]:.2f}")
```

#### 实施交易策略

```python
# RSI策略示例
def rsi_strategy(symbol, data):
    """RSI超买超卖策略"""
    indicators = TechnicalIndicators()
    rsi = indicators.rsi(data, 14)
    current_rsi = rsi[-1]
    
    if current_rsi < 30:  # 超卖
        return "BUY", f"RSI超卖 ({current_rsi:.2f})"
    elif current_rsi > 70:  # 超买
        return "SELL", f"RSI超买 ({current_rsi:.2f})"
    else:
        return "HOLD", f"RSI中性 ({current_rsi:.2f})"

# 应用策略
symbol = "AAPL"
data = trading_system.get_historical_data(symbol, "3 M")
signal, reason = rsi_strategy(symbol, data)

print(f"交易信号: {signal}")
print(f"理由: {reason}")

# 根据信号执行交易
if signal == "BUY":
    result = trading_system.place_order(
        symbol=symbol,
        quantity=100,
        order_type="MKT",
        action="BUY"
    )
elif signal == "SELL":
    # 检查是否有持仓
    positions = trading_system.get_positions()
    position = next((p for p in positions if p['symbol'] == symbol), None)
    if position and position['quantity'] > 0:
        result = trading_system.place_order(
            symbol=symbol,
            quantity=position['quantity'],
            order_type="MKT",
            action="SELL"
        )
```

### 2. 策略回测

```python
from technical_indicators import StrategyBacktester

# 创建回测器
backtester = StrategyBacktester(initial_capital=100000)

# 定义策略
def moving_average_strategy(data, short_window=5, long_window=20):
    """移动平均线交叉策略"""
    indicators = TechnicalIndicators()
    short_ma = indicators.sma(data, short_window)
    long_ma = indicators.sma(data, long_window)
    
    signals = []
    for i in range(len(data)):
        if i < long_window:
            signals.append("HOLD")
        elif short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
            signals.append("BUY")
        elif short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    
    return signals

# 获取历史数据进行回测
historical_data = trading_system.get_historical_data("AAPL", "2 Y")

# 执行回测
results = backtester.backtest(
    data=historical_data,
    strategy_func=moving_average_strategy,
    strategy_params={'short_window': 5, 'long_window': 20}
)

# 显示回测结果
print("=== 回测结果 ===")
print(f"总收益率: {results['total_return']:.2%}")
print(f"年化收益率: {results['annualized_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
print(f"胜率: {results['win_rate']:.2%}")
print(f"总交易次数: {results['total_trades']}")
```

### 3. 自动化交易

#### 创建交易机器人

```python
import time
import schedule
from datetime import datetime

class TradingBot:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.strategies = {}
        self.running = False
    
    def add_strategy(self, name, strategy_func, symbols, params=None):
        """添加交易策略"""
        self.strategies[name] = {
            'func': strategy_func,
            'symbols': symbols,
            'params': params or {}
        }
    
    def run_strategies(self):
        """执行所有策略"""
        if not self.trading_system.is_market_open():
            print("市场未开放，跳过策略执行")
            return
        
        for name, strategy in self.strategies.items():
            print(f"执行策略: {name}")
            
            for symbol in strategy['symbols']:
                try:
                    # 获取数据
                    data = self.trading_system.get_historical_data(symbol, "1 min")
                    
                    # 执行策略
                    signal, reason = strategy['func'](
                        symbol, data, **strategy['params']
                    )
                    
                    print(f"{symbol}: {signal} - {reason}")
                    
                    # 执行交易
                    if signal == "BUY":
                        self.execute_buy(symbol)
                    elif signal == "SELL":
                        self.execute_sell(symbol)
                        
                except Exception as e:
                    print(f"策略执行错误 {name}-{symbol}: {e}")
    
    def execute_buy(self, symbol, quantity=100):
        """执行买入"""
        result = self.trading_system.place_order(
            symbol=symbol,
            quantity=quantity,
            order_type="MKT",
            action="BUY"
        )
        
        if result['status'] == 'success':
            print(f"✅ 买入 {symbol} {quantity}股")
        else:
            print(f"❌ 买入失败: {result['message']}")
    
    def execute_sell(self, symbol):
        """执行卖出"""
        positions = self.trading_system.get_positions()
        position = next((p for p in positions if p['symbol'] == symbol), None)
        
        if position and position['quantity'] > 0:
            result = self.trading_system.place_order(
                symbol=symbol,
                quantity=position['quantity'],
                order_type="MKT",
                action="SELL"
            )
            
            if result['status'] == 'success':
                print(f"✅ 卖出 {symbol} {position['quantity']}股")
            else:
                print(f"❌ 卖出失败: {result['message']}")
    
    def start(self):
        """启动机器人"""
        self.running = True
        
        # 设置定时任务
        schedule.every(5).minutes.do(self.run_strategies)
        
        print("🤖 交易机器人启动")
        
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def stop(self):
        """停止机器人"""
        self.running = False
        print("🛑 交易机器人停止")

# 使用示例
bot = TradingBot(trading_system)

# 添加RSI策略
bot.add_strategy(
    name="RSI策略",
    strategy_func=rsi_strategy,
    symbols=["AAPL", "GOOGL", "MSFT"],
    params={'rsi_period': 14}
)

# 启动机器人
try:
    bot.start()
except KeyboardInterrupt:
    bot.stop()
```

### 4. 性能优化

#### 启用缓存和批量处理

```python
from performance_optimizer import PerformanceOptimizer

# 创建性能优化器
optimizer = PerformanceOptimizer()

# 启用缓存
optimizer.enable_cache(ttl=60)  # 缓存60秒

# 批量获取报价
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
quotes = optimizer.get_quotes_batch(symbols)

# 批量下单
orders = [
    {'symbol': 'AAPL', 'quantity': 10, 'action': 'BUY'},
    {'symbol': 'GOOGL', 'quantity': 5, 'action': 'BUY'},
    {'symbol': 'MSFT', 'quantity': 15, 'action': 'BUY'}
]

results = optimizer.place_orders_batch(orders)
for result in results:
    print(f"{result['symbol']}: {result['status']}")
```

## 故障排除

### 常见错误及解决方案

#### 1. 连接失败

**错误信息**: `ConnectionException: IB连接失败`

**可能原因**:
- IB TWS/Gateway未启动
- 端口配置错误
- API连接未启用
- 客户端ID冲突

**解决方案**:
```python
# 检查连接
print("检查IB连接...")
try:
    trading_system.connect()
except ConnectionException as e:
    print(f"连接失败: {e}")
    # 1. 确认TWS/Gateway已启动
    # 2. 检查端口配置 (TWS: 7497, Gateway: 4001)
    # 3. 启用API连接设置
    # 4. 更换客户端ID
```

#### 2. 网络连接错误

**错误信息**: `NetworkException: 网络连接超时`

**解决方案**:
```python
import time

def retry_with_backoff(func, max_retries=3, backoff_factor=2):
    """带退避的重试机制"""
    for attempt in range(max_retries):
        try:
            return func()
        except NetworkException as e:
            if attempt == max_retries - 1:
                raise e
            
            wait_time = backoff_factor ** attempt
            print(f"网络错误，{wait_time}秒后重试...")
            time.sleep(wait_time)

# 使用重试机制
quote = retry_with_backoff(
    lambda: trading_system.get_quote("AAPL")
)
```

#### 3. 订单被拒绝

**错误信息**: `OrderException: 订单被拒绝`

**可能原因**:
- 余额不足
- 股票代码无效
- 市场已关闭
- 违反风险规则

**解决方案**:
```python
# 检查订单前置条件
def validate_order_preconditions(symbol, quantity, action):
    # 1. 检查市场状态
    if not trading_system.is_market_open():
        return False, "市场未开放"
    
    # 2. 检查账户余额
    account = trading_system.get_account_info()
    if action == "BUY":
        quote = trading_system.get_quote(symbol)
        required_cash = quantity * quote['price']
        if account['cash'] < required_cash:
            return False, f"余额不足，需要${required_cash:.2f}"
    
    # 3. 检查持仓
    if action == "SELL":
        positions = trading_system.get_positions()
        position = next((p for p in positions if p['symbol'] == symbol), None)
        if not position or position['quantity'] < quantity:
            return False, "持仓不足"
    
    return True, "检查通过"

# 使用前置检查
is_valid, message = validate_order_preconditions("AAPL", 100, "BUY")
if is_valid:
    result = trading_system.place_order(
        symbol="AAPL",
        quantity=100,
        order_type="MKT",
        action="BUY"
    )
else:
    print(f"订单前置检查失败: {message}")
```

### 日志分析

#### 启用详细日志

```python
import logging

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_debug.log'),
        logging.StreamHandler()
    ]
)

# 查看日志
tail -f trading_debug.log
```

#### 性能监控

```python
from monitoring_dashboard import MonitoringDashboard

# 启动监控面板
monitor = MonitoringDashboard()
monitor.start()

# 访问 http://localhost:8080 查看监控数据
```

## 常见问题

### Q1: 如何在实盘和模拟模式之间切换？

**A**: 在创建交易系统时设置相应的连接参数：

```python
# 模拟模式 (Paper Trading)
trading_system = IBAutomatedTradingSystem(
    host="127.0.0.1",
    port=7497,  # TWS Paper Trading端口
    client_id=1
)

# 实盘模式
trading_system = IBAutomatedTradingSystem(
    host="127.0.0.1", 
    port=7496,  # TWS Live Trading端口
    client_id=1
)
```

### Q2: 如何设置止损？

**A**: 可以通过多种方式设置止损：

```python
# 方法1: 下止损单
stop_order = trading_system.place_order(
    symbol="AAPL",
    quantity=100,
    order_type="STP",
    action="SELL",
    aux_price=145.00  # 止损价格
)

# 方法2: 在风险管理中设置全局止损
risk_config = {
    'stop_loss_pct': 0.05  # 5%止损
}
trading_system.update_risk_config(risk_config)

# 方法3: 程序化止损
def check_stop_loss():
    positions = trading_system.get_positions()
    for position in positions:
        current_price = trading_system.get_quote(position['symbol'])['price']
        loss_pct = (position['avg_price'] - current_price) / position['avg_price']
        
        if loss_pct > 0.05:  # 损失超过5%
            trading_system.place_order(
                symbol=position['symbol'],
                quantity=position['quantity'],
                order_type="MKT",
                action="SELL"
            )
            print(f"触发止损: {position['symbol']}")
```

### Q3: 如何获取更多的历史数据？

**A**: 系统支持多种数据源：

```python
# 使用不同的时间周期
data_1d = trading_system.get_historical_data("AAPL", "1 D")    # 1天
data_1w = trading_system.get_historical_data("AAPL", "1 W")    # 1周  
data_1m = trading_system.get_historical_data("AAPL", "1 M")    # 1个月
data_1y = trading_system.get_historical_data("AAPL", "1 Y")    # 1年
data_5y = trading_system.get_historical_data("AAPL", "5 Y")    # 5年

# 配置外部数据源
trading_system.set_data_source("alpha_vantage", api_key="your_key")
```

### Q4: 如何处理盘后交易？

**A**: 系统支持盘后交易检测：

```python
# 检查市场状态
market_status = trading_system.get_market_status()
print(f"市场状态: {market_status['status']}")
print(f"下次开盘: {market_status['next_open']}")

# 盘后交易设置
if market_status['status'] == 'after_hours':
    # 盘后交易通常使用限价单
    result = trading_system.place_order(
        symbol="AAPL",
        quantity=100,
        order_type="LMT",
        action="BUY",
        lmt_price=150.00,
        outside_rth=True  # 启用盘后交易
    )
```

### Q5: 如何备份和恢复交易数据？

**A**: 系统提供数据备份功能：

```python
# 备份交易数据
backup_data = trading_system.export_data()
with open('trading_backup.json', 'w') as f:
    json.dump(backup_data, f, indent=2)

# 恢复交易数据
with open('trading_backup.json', 'r') as f:
    backup_data = json.load(f)
trading_system.import_data(backup_data)
```

### Q6: 如何优化策略性能？

**A**: 几个优化建议：

```python
# 1. 使用向量化计算
import numpy as np
import pandas as pd

def optimized_sma(prices, window):
    """优化的移动平均计算"""
    return pd.Series(prices).rolling(window=window).mean().values

# 2. 缓存计算结果
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_indicator(symbol, period, indicator_type):
    """缓存技术指标计算"""
    data = trading_system.get_historical_data(symbol, period)
    # 计算指标...
    return result

# 3. 批量处理
symbols = ["AAPL", "GOOGL", "MSFT"]
quotes = trading_system.get_quotes_batch(symbols)  # 批量获取报价
```

---

如需更多帮助，请参考：
- [API文档](API_Documentation.md)
- [示例代码](../examples/)
- [GitHub Issues](https://github.com/your-repo/ib-trading-system/issues)

*最后更新: 2024年1月*