# 新券商支持功能

本项目已扩展支持以下新券商API：

## 支持的券商

### 现有券商
- **Firstrade** - 第一证券
- **Alpaca** - Alpaca Markets
- **Interactive Brokers** - 盈透证券

### 新增券商
- **TD Ameritrade** - 德美利证券
- **Charles Schwab** - 嘉信理财
- **E*TRADE** - 亿创理财
- **Robinhood** - 罗宾汉

## 文件结构

```
examples/
├── config.py                      # 配置文件（已更新）
├── broker_factory.py             # 券商工厂（已更新）
├── td_ameritrade_adapter.py      # TD Ameritrade 适配器
├── charles_schwab_adapter.py     # Charles Schwab 适配器
├── etrade_adapter.py             # E*TRADE 适配器
├── robinhood_adapter.py          # Robinhood 适配器
├── test_new_brokers.py           # 新券商功能测试
├── test_config_integration.py    # 配置集成测试
├── run_all_tests.py              # 运行所有测试
└── README_new_brokers.md         # 本说明文档
```

## 配置说明

### 环境变量配置

#### TD Ameritrade
```bash
export TD_AMERITRADE_CONSUMER_KEY="your_consumer_key"
export TD_AMERITRADE_CONSUMER_SECRET="your_consumer_secret"
export TD_AMERITRADE_ACCESS_TOKEN="your_access_token"
export TD_AMERITRADE_ACCESS_SECRET="your_access_secret"
export TD_AMERITRADE_SANDBOX="true"
export TD_AMERITRADE_DRY_RUN="true"
export TD_AMERITRADE_ENABLED="true"
```

#### Charles Schwab
```bash
export CHARLES_SCHWAB_APP_KEY="your_app_key"
export CHARLES_SCHWAB_APP_SECRET="your_app_secret"
export CHARLES_SCHWAB_ACCESS_TOKEN="your_access_token"
export CHARLES_SCHWAB_REFRESH_TOKEN="your_refresh_token"
export CHARLES_SCHWAB_SANDBOX="true"
export CHARLES_SCHWAB_DRY_RUN="true"
export CHARLES_SCHWAB_ENABLED="true"
```

#### E*TRADE
```bash
export ETRADE_CONSUMER_KEY="your_consumer_key"
export ETRADE_CONSUMER_SECRET="your_consumer_secret"
export ETRADE_ACCESS_TOKEN="your_access_token"
export ETRADE_ACCESS_SECRET="your_access_secret"
export ETRADE_SANDBOX="true"
export ETRADE_DRY_RUN="true"
export ETRADE_ENABLED="true"
```

#### Robinhood
```bash
export ROBINHOOD_USERNAME="your_username"
export ROBINHOOD_PASSWORD="your_password"
export ROBINHOOD_DEVICE_TOKEN="your_device_token"
export ROBINHOOD_CHALLENGE_TYPE="sms"
export ROBINHOOD_SANDBOX="true"
export ROBINHOOD_DRY_RUN="true"
export ROBINHOOD_ENABLED="true"
```

## 使用示例

### 创建券商实例

```python
from broker_factory import BrokerFactory

# 创建 TD Ameritrade 券商
td_broker = BrokerFactory.create_broker(
    'td_ameritrade',
    consumer_key='your_key',
    consumer_secret='your_secret',
    access_token='your_token',
    access_secret='your_secret',
    sandbox=True,
    dry_run=True
)

# 创建 Charles Schwab 券商
schwab_broker = BrokerFactory.create_broker(
    'charles_schwab',
    app_key='your_key',
    app_secret='your_secret',
    access_token='your_token',
    refresh_token='your_refresh',
    sandbox=True,
    dry_run=True
)

# 创建 E*TRADE 券商
etrade_broker = BrokerFactory.create_broker(
    'etrade',
    consumer_key='your_key',
    consumer_secret='your_secret',
    access_token='your_token',
    access_secret='your_secret',
    sandbox=True,
    dry_run=True
)

# 创建 Robinhood 券商
robinhood_broker = BrokerFactory.create_broker(
    'robinhood',
    username='your_username',
    password='your_password',
    device_token='your_device',
    challenge_type='sms',
    sandbox=True,
    dry_run=True
)
```

### 使用配置文件

```python
from config import Config
from broker_factory import BrokerFactory

# 加载配置
config = Config()

# 使用配置创建券商
if config.td_ameritrade.enabled:
    td_broker = BrokerFactory.create_broker(
        'td_ameritrade',
        consumer_key=config.td_ameritrade.consumer_key,
        consumer_secret=config.td_ameritrade.consumer_secret,
        access_token=config.td_ameritrade.access_token,
        access_secret=config.td_ameritrade.access_secret,
        sandbox=config.td_ameritrade.sandbox,
        dry_run=config.td_ameritrade.dry_run
    )
```

## 统一接口

所有券商适配器都实现了 `TradingSystemInterface` 接口，提供以下方法：

- `connect()` - 连接到券商
- `disconnect()` - 断开连接
- `is_connected()` - 检查连接状态
- `get_portfolio_status()` - 获取投资组合状态
- `get_positions()` - 获取持仓信息
- `get_performance()` - 获取交易表现
- `place_order()` - 下单
- `get_detailed_positions()` - 获取详细持仓
- `calculate_portfolio_performance()` - 计算投资组合表现

## 测试

### 运行单个测试
```bash
# 测试新券商功能
python test_new_brokers.py

# 测试配置集成
python test_config_integration.py
```

### 运行所有测试
```bash
python run_all_tests.py
```

## 特性

### 安全特性
- 支持沙盒模式测试
- 支持干运行模式（不实际执行交易）
- 敏感信息通过环境变量配置

### 扩展性
- 统一的适配器模式
- 工厂模式创建券商实例
- 易于添加新的券商支持

### 可靠性
- 完整的错误处理
- 连接状态管理
- 详细的日志记录

## 注意事项

1. **API凭据**: 请确保使用正确的API凭据，并妥善保管
2. **沙盒模式**: 建议在生产环境前先在沙盒模式下测试
3. **干运行模式**: 可以启用干运行模式来测试逻辑而不实际执行交易
4. **API限制**: 注意各券商的API调用频率限制
5. **合规要求**: 确保符合各券商的使用条款和监管要求

## 故障排除

### 常见问题

1. **连接失败**
   - 检查API凭据是否正确
   - 确认网络连接正常
   - 验证券商API服务状态

2. **认证错误**
   - 检查access token是否过期
   - 确认API密钥权限设置
   - 验证沙盒/生产环境配置

3. **配置加载失败**
   - 检查环境变量是否正确设置
   - 确认配置文件路径
   - 验证配置文件格式

### 调试建议

1. 启用详细日志记录
2. 使用沙盒模式进行测试
3. 检查API文档了解最新变更
4. 运行测试脚本验证功能

## 更新日志

### v1.0.0 (当前版本)
- 添加 TD Ameritrade 支持
- 添加 Charles Schwab 支持
- 添加 E*TRADE 支持
- 添加 Robinhood 支持
- 更新配置系统
- 更新券商工厂
- 添加完整测试套件