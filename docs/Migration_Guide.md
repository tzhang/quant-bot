# 从 Firstrade 到 Interactive Brokers 迁移指南

## 概述

本指南将帮助您将现有的 Firstrade 交易系统迁移到 Interactive Brokers (IB) 平台。IB 提供更强大的 API 功能、更低的交易成本和更广泛的市场覆盖。

## 迁移前准备

### 1. 开设 IB 账户

1. 访问 [Interactive Brokers 官网](https://www.interactivebrokers.com)
2. 开设个人或机构账户
3. 完成身份验证和资金转入
4. 申请 API 访问权限

### 2. 安装必要软件

```bash
# 下载并安装 TWS (Trader Workstation) 或 IB Gateway
# 从 IB 官网下载对应版本

# 安装 Python IB API
pip install ibapi
```

### 3. 配置 TWS/Gateway

1. 启动 TWS 或 IB Gateway
2. 进入 **配置** → **API** → **设置**
3. 启用 **Enable ActiveX and Socket Clients**
4. 设置端口号：
   - TWS 实盘：7496
   - TWS 模拟：7497
   - Gateway 实盘：4001
   - Gateway 模拟：4002
5. 添加可信 IP 地址（通常为 127.0.0.1）

## 核心差异对比

### 连接方式

| 功能 | Firstrade | Interactive Brokers |
|------|-----------|-------------------|
| 认证方式 | 用户名/密码/PIN | Host/Port/Client ID |
| 连接协议 | HTTP/Web Scraping | Socket/API |
| 会话管理 | Cookie/Session | 持久连接 |

**Firstrade 连接**:
```python
trading_system = FirstradeTradingSystem(
    username="your_username",
    password="your_password", 
    pin="your_pin"
)
trading_system.login()
```

**IB 连接**:
```python
trading_system = IBAutomatedTradingSystem(
    host="127.0.0.1",
    port=7497,  # 模拟账户
    client_id=1
)
trading_system.connect()
```

### 订单参数

| 参数 | Firstrade | Interactive Brokers |
|------|-----------|-------------------|
| 买卖方向 | `side`: "buy"/"sell" | `action`: "BUY"/"SELL" |
| 订单类型 | `order_type`: "market"/"limit"/"stop" | `order_type`: "MKT"/"LMT"/"STP" |
| 限价 | `price` | `lmt_price` |
| 止损价 | `price` | `aux_price` |

**Firstrade 下单**:
```python
order = trading_system.place_order(
    symbol="AAPL",
    quantity=100,
    side="buy",
    order_type="limit",
    price=150.00
)
```

**IB 下单**:
```python
order = trading_system.place_order(
    symbol="AAPL",
    quantity=100,
    action="BUY",
    order_type="LMT",
    lmt_price=150.00
)
```

### 历史数据格式

| 参数 | Firstrade | Interactive Brokers |
|------|-----------|-------------------|
| 时间周期 | "1d", "1w", "1m", "1y" | "1 D", "1 W", "1 M", "1 Y" |
| 分钟数据 | "1m", "5m", "15m" | "1 min", "5 mins", "15 mins" |

## 逐步迁移流程

### 第一步：更新依赖

```bash
# 卸载 Firstrade 相关包
pip uninstall firstrade-api

# 安装 IB API
pip install ibapi
pip install ib_insync  # 可选：更高级的封装
```

### 第二步：更新配置文件

**旧配置 (config.yaml)**:
```yaml
firstrade:
  username: "your_username"
  password: "your_password"
  pin: "your_pin"
  
trading:
  dry_run: true
```

**新配置 (config.yaml)**:
```yaml
ib:
  host: "127.0.0.1"
  port: 7497  # 模拟账户
  client_id: 1
  
trading:
  paper_trading: true
```

### 第三步：更新环境变量

**旧环境变量**:
```bash
export FIRSTRADE_USERNAME="your_username"
export FIRSTRADE_PASSWORD="your_password"
export FIRSTRADE_PIN="your_pin"
```

**新环境变量**:
```bash
export IB_HOST="127.0.0.1"
export IB_PORT="7497"
export IB_CLIENT_ID="1"
```

### 第四步：更新代码

#### 4.1 导入模块

**旧代码**:
```python
from firstrade_trading_system import FirstradeTradingSystem
from firstrade_connector import FirstradeConnector
```

**新代码**:
```python
from ib_automated_trading_system import IBAutomatedTradingSystem
from ib_trading_manager import IBTradingManager
```

#### 4.2 初始化系统

**旧代码**:
```python
trading_system = FirstradeTradingSystem(
    username=os.getenv("FIRSTRADE_USERNAME"),
    password=os.getenv("FIRSTRADE_PASSWORD"),
    pin=os.getenv("FIRSTRADE_PIN"),
    dry_run=True
)
```

**新代码**:
```python
trading_system = IBAutomatedTradingSystem(
    host=os.getenv("IB_HOST", "127.0.0.1"),
    port=int(os.getenv("IB_PORT", "7497")),
    client_id=int(os.getenv("IB_CLIENT_ID", "1"))
)
```

#### 4.3 连接和断开

**旧代码**:
```python
# 连接
trading_system.login()

# 断开
trading_system.logout()
```

**新代码**:
```python
# 连接
trading_system.connect()

# 断开
trading_system.disconnect()
```

#### 4.4 下单操作

**旧代码**:
```python
# 市价买单
order = trading_system.place_order(
    symbol="AAPL",
    quantity=100,
    side="buy",
    order_type="market"
)

# 限价卖单
order = trading_system.place_order(
    symbol="AAPL", 
    quantity=50,
    side="sell",
    order_type="limit",
    price=155.00
)

# 止损单
order = trading_system.place_order(
    symbol="AAPL",
    quantity=100,
    side="sell", 
    order_type="stop",
    price=145.00
)
```

**新代码**:
```python
# 市价买单
order = trading_system.place_order(
    symbol="AAPL",
    quantity=100,
    action="BUY",
    order_type="MKT"
)

# 限价卖单
order = trading_system.place_order(
    symbol="AAPL",
    quantity=50,
    action="SELL", 
    order_type="LMT",
    lmt_price=155.00
)

# 止损单
order = trading_system.place_order(
    symbol="AAPL",
    quantity=100,
    action="SELL",
    order_type="STP",
    aux_price=145.00
)
```

#### 4.5 历史数据获取

**旧代码**:
```python
# 获取日线数据
data = trading_system.get_historical_data("AAPL", "1y")

# 获取分钟数据
data = trading_system.get_historical_data("AAPL", "5m")
```

**新代码**:
```python
# 获取日线数据
data = trading_system.get_historical_data("AAPL", "1 Y")

# 获取分钟数据
data = trading_system.get_historical_data("AAPL", "5 mins")
```

### 第五步：更新异常处理

**旧代码**:
```python
try:
    trading_system.login()
except FirstradeException as e:
    print(f"Firstrade错误: {e}")
except LoginException as e:
    print(f"登录失败: {e}")
```

**新代码**:
```python
try:
    trading_system.connect()
except IBException as e:
    print(f"IB错误: {e}")
except ConnectionException as e:
    print(f"连接失败: {e}")
```

## 测试迁移

### 1. 模拟环境测试

```python
# 使用模拟账户测试
trading_system = IBAutomatedTradingSystem(
    host="127.0.0.1",
    port=7497,  # 模拟端口
    client_id=1
)

# 连接测试
try:
    trading_system.connect()
    print("✅ 连接成功")
    
    # 获取账户信息
    account = trading_system.get_account_info()
    print(f"账户余额: ${account['cash']:.2f}")
    
    # 获取报价
    quote = trading_system.get_quote("AAPL")
    print(f"AAPL价格: ${quote['price']:.2f}")
    
    # 模拟下单
    order = trading_system.place_order(
        symbol="AAPL",
        quantity=1,
        action="BUY",
        order_type="MKT"
    )
    print(f"订单ID: {order['order_id']}")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
finally:
    trading_system.disconnect()
```

### 2. 功能对比测试

创建测试脚本验证关键功能：

```python
def test_migration():
    """迁移功能测试"""
    
    # 测试项目
    tests = [
        ("连接测试", test_connection),
        ("账户信息", test_account_info),
        ("报价获取", test_quotes),
        ("历史数据", test_historical_data),
        ("下单功能", test_place_order),
        ("订单查询", test_order_status),
        ("持仓查询", test_positions)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            test_func()
            results[test_name] = "✅ 通过"
        except Exception as e:
            results[test_name] = f"❌ 失败: {e}"
    
    # 输出测试结果
    print("\n=== 迁移测试结果 ===")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
```

## 性能优化建议

### 1. 连接管理

```python
# 使用连接池
class IBConnectionPool:
    def __init__(self, max_connections=5):
        self.max_connections = max_connections
        self.connections = []
        self.available = []
    
    def get_connection(self):
        if self.available:
            return self.available.pop()
        
        if len(self.connections) < self.max_connections:
            conn = IBAutomatedTradingSystem(
                host="127.0.0.1",
                port=7497,
                client_id=len(self.connections) + 1
            )
            conn.connect()
            self.connections.append(conn)
            return conn
        
        raise Exception("连接池已满")
    
    def return_connection(self, conn):
        self.available.append(conn)
```

### 2. 数据缓存

```python
from functools import lru_cache
import time

class CachedTradingSystem:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self._quote_cache = {}
        self._cache_ttl = 5  # 5秒缓存
    
    def get_quote(self, symbol):
        now = time.time()
        if symbol in self._quote_cache:
            quote, timestamp = self._quote_cache[symbol]
            if now - timestamp < self._cache_ttl:
                return quote
        
        quote = self.trading_system.get_quote(symbol)
        self._quote_cache[symbol] = (quote, now)
        return quote
```

## 常见问题解决

### Q1: TWS 连接被拒绝

**问题**: `ConnectionException: Connection refused`

**解决方案**:
1. 确认 TWS/Gateway 已启动
2. 检查端口配置是否正确
3. 验证 API 设置已启用
4. 检查防火墙设置

### Q2: 客户端ID冲突

**问题**: `IBException: Duplicate client id`

**解决方案**:
```python
# 使用不同的客户端ID
trading_system = IBAutomatedTradingSystem(
    host="127.0.0.1",
    port=7497,
    client_id=2  # 更改为未使用的ID
)
```

### Q3: 订单被拒绝

**问题**: 订单状态显示 "Rejected"

**解决方案**:
1. 检查账户余额是否充足
2. 验证股票代码是否正确
3. 确认市场是否开放
4. 检查订单参数是否有效

### Q4: 历史数据获取失败

**问题**: `DataException: No historical data`

**解决方案**:
```python
# 添加重试机制
def get_historical_data_with_retry(symbol, duration, retries=3):
    for i in range(retries):
        try:
            return trading_system.get_historical_data(symbol, duration)
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(1)
```

## 迁移检查清单

- [ ] IB 账户已开设并激活
- [ ] TWS/Gateway 已安装并配置
- [ ] API 权限已启用
- [ ] Python 依赖已更新
- [ ] 配置文件已修改
- [ ] 环境变量已更新
- [ ] 代码已更新（导入、初始化、API调用）
- [ ] 异常处理已更新
- [ ] 模拟环境测试通过
- [ ] 关键功能验证完成
- [ ] 性能测试满足要求
- [ ] 错误处理机制完善
- [ ] 监控和日志配置完成

## 回滚计划

如果迁移过程中遇到问题，可以按以下步骤回滚：

1. **保留原始代码备份**
2. **恢复原始配置文件**
3. **重新安装 Firstrade 依赖**
4. **恢复环境变量设置**
5. **验证原系统功能正常**

```bash
# 回滚脚本示例
#!/bin/bash
echo "开始回滚到 Firstrade..."

# 恢复依赖
pip uninstall ibapi
pip install firstrade-api

# 恢复配置
cp config.yaml.backup config.yaml

# 恢复环境变量
source .env.firstrade

echo "回滚完成"
```

## 技术支持

如果在迁移过程中遇到问题，可以通过以下渠道获取帮助：

- **IB API 文档**: https://interactivebrokers.github.io/tws-api/
- **IB 客户支持**: https://www.interactivebrokers.com/en/support/
- **项目 GitHub**: https://github.com/your-repo/ib-trading-system
- **社区论坛**: https://www.interactivebrokers.com/en/community/

---

*最后更新: 2024年1月*