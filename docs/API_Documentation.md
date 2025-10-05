# Firstrade 交易系统 API 文档

## 概述

Firstrade 交易系统是一个功能完整的量化交易平台，提供股票交易、风险管理、投资组合分析等功能。本文档详细介绍了系统的API接口和使用方法。

## 目录

1. [系统架构](#系统架构)
2. [快速开始](#快速开始)
3. [核心模块](#核心模块)
4. [API参考](#api参考)
5. [示例代码](#示例代码)
6. [错误处理](#错误处理)
7. [最佳实践](#最佳实践)

## 系统架构

```
Firstrade交易系统
├── FirstradeTradingSystem (主系统)
├── FirstradeConnector (连接器)
├── RiskManager (风险管理)
├── OrderManager (订单管理)
├── PerformanceOptimizer (性能优化)
├── TechnicalIndicators (技术指标)
└── MonitoringDashboard (监控面板)
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from firstrade_trading_system import FirstradeTradingSystem

# 创建交易系统实例
trading_system = FirstradeTradingSystem(
    username="your_username",
    password="your_password", 
    pin="your_pin",
    dry_run=True  # 模拟模式
)

# 登录
if trading_system.login():
    print("登录成功")
    
    # 获取账户信息
    account_info = trading_system.get_account_info()
    print(f"账户余额: {account_info.get('cash', 0)}")
    
    # 获取股票报价
    quote = trading_system.get_quote("AAPL")
    print(f"AAPL 当前价格: {quote.get('price', 0)}")
    
    # 下单
    order_result = trading_system.place_order(
        symbol="AAPL",
        quantity=10,
        order_type="market",
        side="buy"
    )
    print(f"订单结果: {order_result}")
```

## 核心模块

### FirstradeTradingSystem

主交易系统类，提供完整的交易功能。

#### 构造函数

```python
def __init__(self, username: str, password: str, pin: str, dry_run: bool = False):
    """
    初始化交易系统
    
    参数:
        username (str): Firstrade用户名
        password (str): 密码
        pin (str): PIN码
        dry_run (bool): 是否为模拟模式，默认False
    """
```

#### 主要方法

##### 登录/登出

```python
def login(self) -> bool:
    """
    登录到Firstrade系统
    
    返回:
        bool: 登录是否成功
    """

def logout(self) -> bool:
    """
    登出系统
    
    返回:
        bool: 登出是否成功
    """
```

##### 账户信息

```python
def get_account_info(self) -> dict:
    """
    获取账户信息
    
    返回:
        dict: 包含账户余额、持仓等信息
        {
            'cash': float,           # 现金余额
            'buying_power': float,   # 购买力
            'total_value': float,    # 总资产
            'positions': list        # 持仓列表
        }
    """

def get_positions(self) -> list:
    """
    获取当前持仓
    
    返回:
        list: 持仓列表
        [
            {
                'symbol': str,      # 股票代码
                'quantity': int,    # 持仓数量
                'avg_price': float, # 平均成本
                'market_value': float # 市值
            }
        ]
    """
```

##### 市场数据

```python
def get_quote(self, symbol: str) -> dict:
    """
    获取股票实时报价
    
    参数:
        symbol (str): 股票代码
        
    返回:
        dict: 报价信息
        {
            'symbol': str,
            'price': float,
            'bid': float,
            'ask': float,
            'volume': int,
            'timestamp': datetime
        }
    """

def get_historical_data(self, symbol: str, period: str = "1y") -> list:
    """
    获取历史数据
    
    参数:
        symbol (str): 股票代码
        period (str): 时间周期 ('1d', '1w', '1m', '1y')
        
    返回:
        list: 历史数据列表
    """
```

##### 订单管理

```python
def place_order(self, symbol: str, quantity: int, order_type: str, 
                side: str, price: float = None) -> dict:
    """
    下单
    
    参数:
        symbol (str): 股票代码
        quantity (int): 数量
        order_type (str): 订单类型 ('market', 'limit', 'stop')
        side (str): 买卖方向 ('buy', 'sell')
        price (float): 价格（限价单必需）
        
    返回:
        dict: 订单结果
        {
            'order_id': str,
            'status': str,
            'message': str
        }
    """

def cancel_order(self, order_id: str) -> dict:
    """
    取消订单
    
    参数:
        order_id (str): 订单ID
        
    返回:
        dict: 取消结果
    """

def get_orders(self, status: str = "all") -> list:
    """
    获取订单列表
    
    参数:
        status (str): 订单状态过滤 ('all', 'open', 'filled', 'cancelled')
        
    返回:
        list: 订单列表
    """
```

### FirstradeConnector

底层连接器，处理与Firstrade API的通信。

```python
class FirstradeConnector:
    def __init__(self, username: str, password: str, pin: str):
        """初始化连接器"""
        
    def login(self) -> bool:
        """登录"""
        
    def logout(self) -> bool:
        """登出"""
        
    def get_account_info(self) -> dict:
        """获取账户信息"""
        
    def get_quote(self, symbol: str) -> dict:
        """获取报价"""
        
    def place_order(self, order_data: dict) -> dict:
        """下单"""
```

### RiskManager

风险管理模块，提供订单风险检查功能。

```python
class RiskManager:
    def __init__(self, config: dict):
        """
        初始化风险管理器
        
        参数:
            config (dict): 风险配置
            {
                'max_order_size': float,      # 最大订单金额
                'max_daily_loss': float,      # 最大日损失
                'allowed_symbols': list,      # 允许交易的股票
                'position_limit': float       # 单只股票最大仓位
            }
        """
        
    def validate_order(self, order: dict) -> tuple:
        """
        验证订单
        
        参数:
            order (dict): 订单信息
            
        返回:
            tuple: (是否通过, 错误信息)
        """
        
    def check_daily_limits(self) -> bool:
        """检查日限制"""
        
    def calculate_position_size(self, symbol: str, target_weight: float) -> int:
        """计算仓位大小"""
```

### OrderManager

订单管理模块，处理订单的生命周期。

```python
class OrderManager:
    def __init__(self, connector: FirstradeConnector):
        """初始化订单管理器"""
        
    def create_order(self, symbol: str, quantity: int, order_type: str, 
                    side: str, price: float = None) -> dict:
        """创建订单"""
        
    def execute_order(self, order: dict) -> dict:
        """执行订单"""
        
    def cancel_order(self, order_id: str) -> dict:
        """取消订单"""
        
    def get_order_status(self, order_id: str) -> dict:
        """获取订单状态"""
```

## API参考

### 错误代码

| 错误代码 | 描述 | 解决方案 |
|---------|------|----------|
| 1001 | 登录失败 | 检查用户名、密码和PIN |
| 1002 | 网络连接错误 | 检查网络连接 |
| 2001 | 订单金额超限 | 减少订单金额 |
| 2002 | 股票代码无效 | 检查股票代码 |
| 2003 | 余额不足 | 检查账户余额 |
| 3001 | 市场已关闭 | 等待市场开放 |

### 配置参数

```python
# 风险管理配置
RISK_CONFIG = {
    'max_order_size': 10000,        # 最大订单金额
    'max_daily_loss': 5000,         # 最大日损失
    'max_position_size': 0.1,       # 最大仓位比例
    'allowed_symbols': ['AAPL', 'GOOGL', 'MSFT'],  # 允许交易的股票
    'trading_hours': {
        'start': '09:30',
        'end': '16:00'
    }
}

# 系统配置
SYSTEM_CONFIG = {
    'dry_run': True,                # 模拟模式
    'log_level': 'INFO',           # 日志级别
    'cache_enabled': True,         # 启用缓存
    'max_retries': 3,              # 最大重试次数
    'timeout': 30                  # 超时时间（秒）
}
```

## 示例代码

### 基本交易流程

```python
import logging
from firstrade_trading_system import FirstradeTradingSystem

# 配置日志
logging.basicConfig(level=logging.INFO)

# 创建交易系统
trading_system = FirstradeTradingSystem(
    username="your_username",
    password="your_password",
    pin="your_pin",
    dry_run=True
)

try:
    # 登录
    if not trading_system.login():
        raise Exception("登录失败")
    
    # 获取账户信息
    account = trading_system.get_account_info()
    print(f"可用资金: ${account['cash']:.2f}")
    
    # 获取股票报价
    symbol = "AAPL"
    quote = trading_system.get_quote(symbol)
    current_price = quote['price']
    print(f"{symbol} 当前价格: ${current_price:.2f}")
    
    # 计算购买数量（使用10%的资金）
    investment_amount = account['cash'] * 0.1
    quantity = int(investment_amount / current_price)
    
    if quantity > 0:
        # 下市价买单
        order_result = trading_system.place_order(
            symbol=symbol,
            quantity=quantity,
            order_type="market",
            side="buy"
        )
        
        if order_result['status'] == 'success':
            print(f"订单成功: {order_result['order_id']}")
        else:
            print(f"订单失败: {order_result['message']}")
    
except Exception as e:
    print(f"交易过程中发生错误: {e}")
    
finally:
    # 登出
    trading_system.logout()
```

### 技术指标分析

```python
from technical_indicators import TechnicalIndicators

# 创建技术指标分析器
indicators = TechnicalIndicators()

# 获取历史数据
historical_data = trading_system.get_historical_data("AAPL", "3m")

# 计算技术指标
sma_20 = indicators.sma(historical_data, 20)
rsi = indicators.rsi(historical_data, 14)
macd = indicators.macd(historical_data)

# 生成交易信号
if rsi[-1] < 30 and historical_data[-1]['close'] > sma_20[-1]:
    print("买入信号：RSI超卖且价格在20日均线上方")
elif rsi[-1] > 70:
    print("卖出信号：RSI超买")
```

### 投资组合管理

```python
# 获取当前持仓
positions = trading_system.get_positions()

# 计算投资组合指标
total_value = sum(pos['market_value'] for pos in positions)
portfolio_weights = {pos['symbol']: pos['market_value']/total_value 
                    for pos in positions}

print("投资组合权重:")
for symbol, weight in portfolio_weights.items():
    print(f"{symbol}: {weight:.2%}")

# 重新平衡投资组合
target_weights = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4}

for symbol, target_weight in target_weights.items():
    current_weight = portfolio_weights.get(symbol, 0)
    weight_diff = target_weight - current_weight
    
    if abs(weight_diff) > 0.05:  # 权重差异超过5%时调整
        target_value = total_value * target_weight
        current_pos = next((p for p in positions if p['symbol'] == symbol), None)
        current_value = current_pos['market_value'] if current_pos else 0
        
        adjustment_value = target_value - current_value
        quote = trading_system.get_quote(symbol)
        adjustment_quantity = int(adjustment_value / quote['price'])
        
        if adjustment_quantity != 0:
            side = "buy" if adjustment_quantity > 0 else "sell"
            quantity = abs(adjustment_quantity)
            
            order_result = trading_system.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type="market",
                side=side
            )
            print(f"{symbol} 调整订单: {side} {quantity}股")
```

## 错误处理

### 异常类型

```python
class FirstradeException(Exception):
    """Firstrade系统基础异常"""
    pass

class LoginException(FirstradeException):
    """登录异常"""
    pass

class OrderException(FirstradeException):
    """订单异常"""
    pass

class NetworkException(FirstradeException):
    """网络异常"""
    pass

class RiskException(FirstradeException):
    """风险管理异常"""
    pass
```

### 错误处理示例

```python
from firstrade_trading_system import (
    FirstradeTradingSystem, 
    LoginException, 
    OrderException,
    NetworkException
)

try:
    trading_system = FirstradeTradingSystem(
        username="user", 
        password="pass", 
        pin="1234"
    )
    
    trading_system.login()
    
    # 执行交易操作
    result = trading_system.place_order(
        symbol="AAPL",
        quantity=10,
        order_type="market",
        side="buy"
    )
    
except LoginException as e:
    print(f"登录失败: {e}")
    # 重新尝试登录或检查凭据
    
except OrderException as e:
    print(f"订单错误: {e}")
    # 检查订单参数或账户状态
    
except NetworkException as e:
    print(f"网络错误: {e}")
    # 重试或检查网络连接
    
except Exception as e:
    print(f"未知错误: {e}")
    # 记录错误并进行适当处理
```

## 最佳实践

### 1. 安全性

- 不要在代码中硬编码凭据
- 使用环境变量或配置文件存储敏感信息
- 定期更换密码和PIN

```python
import os
from dotenv import load_dotenv

load_dotenv()

trading_system = FirstradeTradingSystem(
    username=os.getenv('FIRSTRADE_USERNAME'),
    password=os.getenv('FIRSTRADE_PASSWORD'),
    pin=os.getenv('FIRSTRADE_PIN')
)
```

### 2. 风险管理

- 始终设置止损
- 分散投资，不要把所有资金投入单一股票
- 定期检查和调整风险参数

```python
# 设置风险参数
risk_config = {
    'max_order_size': 5000,      # 单笔订单最大金额
    'max_daily_loss': 1000,      # 日最大损失
    'max_position_size': 0.05,   # 单只股票最大仓位5%
    'stop_loss_pct': 0.02        # 2%止损
}
```

### 3. 性能优化

- 使用缓存减少API调用
- 批量处理订单
- 合理设置重试机制

```python
from performance_optimizer import PerformanceOptimizer

# 启用性能优化
optimizer = PerformanceOptimizer()
trading_system.set_optimizer(optimizer)

# 批量获取报价
symbols = ['AAPL', 'GOOGL', 'MSFT']
quotes = trading_system.get_quotes_batch(symbols)
```

### 4. 监控和日志

- 启用详细日志记录
- 监控系统性能和错误
- 设置告警机制

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

# 启用监控
from monitoring_dashboard import MonitoringDashboard
monitor = MonitoringDashboard()
monitor.start()
```

### 5. 测试

- 始终在模拟模式下测试新策略
- 编写单元测试和集成测试
- 进行回测验证策略有效性

```python
# 模拟模式测试
trading_system = FirstradeTradingSystem(
    username="test_user",
    password="test_pass", 
    pin="0000",
    dry_run=True  # 启用模拟模式
)

# 运行测试
python -m pytest tests/
```

## 支持和联系

如有问题或建议，请通过以下方式联系：

- 邮箱: support@example.com
- 文档: https://docs.example.com
- GitHub: https://github.com/example/firstrade-trading-system

---

*本文档最后更新时间: 2024年1月*