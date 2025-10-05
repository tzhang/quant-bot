# Firstrade自动化交易系统

一个功能完整的Firstrade自动化交易系统，集成了市场数据获取、投资策略分析、风险管理、订单执行等功能。

## 🚀 功能特性

### 核心功能
- **账户管理**: 安全的账户连接和认证
- **实时数据**: 股票报价、市场数据、投资组合信息
- **智能交易**: 市价单、限价单、止损单、批量订单
- **投资策略**: 技术分析、情感分析、投资组合优化
- **风险控制**: 多层风险管理机制
- **错误处理**: 完善的异常恢复和重试机制

### 高级特性
- **模拟交易**: 安全的模拟交易环境
- **自动化执行**: 基于策略的自动交易
- **性能监控**: 投资组合表现分析
- **安全机制**: 数据加密和会话管理
- **可配置性**: 灵活的配置文件系统

## 📋 系统要求

### Python版本
- Python 3.8 或更高版本

### 依赖包
```bash
pip install firstrade
pip install pandas numpy
pip install yfinance alpha_vantage
pip install scikit-learn
pip install requests beautifulsoup4
pip install pyyaml
pip install cryptography
```

### 可选依赖
```bash
# 用于高级技术分析
pip install ta-lib

# 用于机器学习策略
pip install tensorflow torch

# 用于数据可视化
pip install matplotlib seaborn plotly
```

## 🛠️ 安装和配置

### 1. 克隆项目
```bash
git clone <repository_url>
cd my-quant/examples
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置系统

#### 方法一：使用配置文件
1. 复制配置模板：
```bash
cp config.yaml.template config.yaml
```

2. 编辑配置文件：
```yaml
account:
  username: "your_firstrade_username"
  password: "your_firstrade_password"
  pin: "your_pin_if_required"

trading:
  dry_run: true  # 开始时建议使用模拟模式
```

#### 方法二：使用环境变量
```bash
export FIRSTRADE_USERNAME="your_username"
export FIRSTRADE_PASSWORD="your_password"
export FIRSTRADE_PIN="your_pin"
```

### 4. 创建必要目录
```bash
mkdir -p logs security data
```

## 🚦 快速开始

### 基础使用示例

```python
from firstrade_trading_system import FirstradeTradingSystem

# 创建交易系统实例（模拟模式）
system = FirstradeTradingSystem(
    username="your_username",
    password="your_password",
    pin="your_pin",
    dry_run=True  # 模拟模式
)

# 连接到Firstrade
if system.connect():
    print("成功连接到Firstrade")
    
    # 获取账户信息
    account_info = system.get_account_balance()
    print(f"账户余额: ${account_info.get('buying_power', 0):,.2f}")
    
    # 获取投资组合
    positions = system.get_detailed_positions()
    print(f"持仓数量: {len(positions)}")
    
    # 执行模拟交易
    order_result = system.executor.place_market_order(
        symbol="AAPL",
        quantity=10,
        side="buy",
        dry_run=True
    )
    print(f"订单结果: {order_result}")
```

### 自动化交易示例

```python
# 运行自动化交易策略
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

# 执行投资策略分析
analysis_result = system.run_investment_analysis(symbols)
print("策略分析完成")

# 执行自动化交易（模拟模式）
trading_result = system.run_automated_trading(
    symbols=symbols,
    dry_run=True
)
print(f"交易执行结果: {trading_result}")
```

## 📊 主要模块说明

### 1. FirstradeConnector
负责与Firstrade API的连接和数据获取。

```python
connector = FirstradeConnector(username, password, pin)
connector.login()

# 获取股票报价
quote = connector.get_quote("AAPL")

# 获取账户信息
account = connector.get_account_info()
```

### 2. FirstradeOrderExecutor
处理各种类型的订单执行。

```python
executor = FirstradeOrderExecutor(connector)

# 市价单
market_order = executor.place_market_order("AAPL", 100, "buy")

# 限价单
limit_order = executor.place_limit_order("AAPL", 100, "buy", 150.0)

# 止损单
stop_order = executor.place_stop_loss_order("AAPL", 100, 140.0)
```

### 3. FirstradeTradingSystem
主要的交易系统类，整合所有功能。

```python
system = FirstradeTradingSystem(username, password, pin, dry_run=True)

# 投资组合分析
portfolio = system.get_portfolio_status()

# 交易历史
history = system.get_trading_history(days=30)

# 性能分析
performance = system.calculate_portfolio_performance()
```

### 4. RiskManager
风险管理和控制。

```python
from firstrade_trading_system import RiskManager, SecurityConfig

config = SecurityConfig()
risk_manager = RiskManager(config)

# 验证交易
is_valid, message = risk_manager.get_risk_assessment(
    symbol="AAPL",
    quantity=100,
    price=150.0,
    current_positions=[],
    portfolio_value=100000
)
```

## ⚙️ 配置选项

### 风险管理配置
```yaml
risk_management:
  max_daily_trades: 10          # 每日最大交易次数
  max_position_size: 0.15       # 单股最大仓位比例
  max_daily_loss: 0.05          # 每日最大亏损比例
  stop_loss_threshold: 0.08     # 止损阈值
  take_profit_threshold: 0.20   # 止盈阈值
```

### 交易配置
```yaml
trading:
  dry_run: true                 # 模拟交易模式
  require_confirmation: true    # 需要用户确认
  default_symbols: ["AAPL", "GOOGL", "MSFT"]
```

### 数据源配置
```yaml
data_fetcher:
  sources:
    alpha_vantage_key: "your_key"
    use_yahoo_finance: true
  update_frequency: 300         # 数据更新频率（秒）
```

## 🔒 安全注意事项

### 1. 凭据安全
- **永远不要**在代码中硬编码用户名和密码
- 使用环境变量或加密的配置文件
- 定期更换密码和API密钥

### 2. 模拟交易
- 在真实交易前，务必在模拟模式下充分测试
- 验证所有策略和风险控制机制

### 3. 风险控制
- 设置合理的止损和止盈阈值
- 限制单笔交易和每日交易金额
- 保持足够的现金储备

### 4. 监控和日志
- 启用详细的交易日志
- 定期检查交易记录和性能
- 设置异常情况的通知机制

## 🧪 测试

### 运行测试套件
```bash
python test_firstrade_system.py
```

### 测试覆盖范围
- 单元测试：各个模块的功能测试
- 集成测试：模块间的协作测试
- 性能测试：系统性能基准测试
- 压力测试：高负载情况下的稳定性测试

### 自定义测试
```python
# 测试特定功能
import unittest
from test_firstrade_system import TestFirstradeConnector

# 运行特定测试类
suite = unittest.TestLoader().loadTestsFromTestCase(TestFirstradeConnector)
unittest.TextTestRunner(verbosity=2).run(suite)
```

## 📈 策略开发

### 技术指标策略
```python
def custom_strategy(data):
    """自定义技术指标策略"""
    # 计算移动平均线
    data['MA_20'] = data['close'].rolling(20).mean()
    data['MA_50'] = data['close'].rolling(50).mean()
    
    # 生成交易信号
    signals = []
    if data['MA_20'].iloc[-1] > data['MA_50'].iloc[-1]:
        signals.append({
            'symbol': data['symbol'],
            'action': 'buy',
            'confidence': 0.7
        })
    
    return signals
```

### 情感分析策略
```python
def sentiment_strategy(sentiment_data):
    """基于情感分析的策略"""
    signals = []
    
    for symbol, sentiment in sentiment_data.items():
        if sentiment['score'] > 0.6:  # 积极情感
            signals.append({
                'symbol': symbol,
                'action': 'buy',
                'confidence': sentiment['score']
            })
        elif sentiment['score'] < -0.6:  # 消极情感
            signals.append({
                'symbol': symbol,
                'action': 'sell',
                'confidence': abs(sentiment['score'])
            })
    
    return signals
```

## 🔧 故障排除

### 常见问题

#### 1. 登录失败
```
错误: 登录Firstrade时发生错误
解决: 检查用户名、密码和PIN码是否正确
```

#### 2. API限制
```
错误: API请求频率过高
解决: 增加请求间隔，启用重试机制
```

#### 3. 数据获取失败
```
错误: 无法获取市场数据
解决: 检查网络连接，验证API密钥
```

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用调试模式
system = FirstradeTradingSystem(
    username, password, pin,
    dry_run=True
)
system.debug = True
```

### 日志分析
```bash
# 查看最新日志
tail -f logs/firstrade_system.log

# 搜索错误
grep "ERROR" logs/firstrade_system.log

# 分析交易记录
grep "订单执行" logs/firstrade_system.log
```

## 📚 API参考

### FirstradeTradingSystem主要方法

| 方法 | 描述 | 参数 | 返回值 |
|------|------|------|--------|
| `connect()` | 连接到Firstrade | 无 | bool |
| `get_account_balance()` | 获取账户余额 | 无 | Dict |
| `get_detailed_positions()` | 获取详细持仓 | 无 | List[Dict] |
| `run_investment_analysis()` | 运行投资策略分析 | symbols: List[str] | Dict |
| `run_automated_trading()` | 执行自动化交易 | symbols, dry_run | Dict |

### 配置参数参考

详细的配置参数说明请参考 `config.yaml` 文件中的注释。

## 🤝 贡献指南

### 开发环境设置
```bash
# 克隆开发分支
git clone -b develop <repository_url>

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/
```

### 代码规范
- 遵循PEP 8代码风格
- 添加详细的函数和类注释
- 编写相应的单元测试
- 更新文档和示例

### 提交流程
1. Fork项目
2. 创建功能分支
3. 提交代码更改
4. 运行测试套件
5. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。

## ⚠️ 免责声明

本软件仅供教育和研究目的使用。使用本软件进行实际交易存在财务风险，可能导致资金损失。用户应当：

1. 充分理解金融市场的风险
2. 在模拟环境中充分测试策略
3. 根据自身风险承受能力进行投资
4. 遵守相关法律法规

开发者不对使用本软件造成的任何损失承担责任。

## 📞 支持和联系

- 问题报告：[GitHub Issues](https://github.com/your-repo/issues)
- 功能请求：[GitHub Discussions](https://github.com/your-repo/discussions)
- 邮件支持：support@example.com

---

**祝您交易愉快！** 🎯📈