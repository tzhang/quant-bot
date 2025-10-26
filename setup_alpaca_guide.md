# Alpaca 券商设置指南

## 1. 注册Alpaca账户

1. 访问 [Alpaca官网](https://alpaca.markets/)
2. 点击"Get Started"注册账户
3. 完成身份验证和资金要求

## 2. 获取API密钥

1. 登录Alpaca账户
2. 进入"API Keys"页面
3. 创建新的API密钥对：
   - API Key ID
   - Secret Key
4. 选择环境：
   - Paper Trading (模拟交易) - 推荐先使用
   - Live Trading (实盘交易)

## 3. 配置系统

### 方法1：环境变量配置（推荐）
```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # 模拟交易
# export ALPACA_BASE_URL="https://api.alpaca.markets"      # 实盘交易
```

### 方法2：直接在代码中配置
```python
from examples.alpaca_adapter import AlpacaAdapter

# 创建Alpaca适配器
alpaca = AlpacaAdapter(
    api_key="your_api_key_here",
    secret_key="your_secret_key_here",
    base_url="https://paper-api.alpaca.markets"  # 模拟交易
)

# 连接并测试
if alpaca.connect():
    print("✅ Alpaca连接成功")
    
    # 获取账户信息
    account = alpaca.get_account_info()
    print(f"账户余额: ${account.get('cash', 0)}")
    
    # 获取持仓
    positions = alpaca.get_positions()
    print(f"当前持仓: {len(positions)}个")
else:
    print("❌ Alpaca连接失败")
```

## 4. 修改交易配置

更新 `trading_config.yaml` 文件：

```yaml
data_sources:
  primary: alpaca  # 改为alpaca
  backup: alpha_vantage
  api_keys:
    alpaca:
      api_key: "your_api_key_here"
      secret_key: "your_secret_key_here"
      base_url: "https://paper-api.alpaca.markets"
    alpha_vantage: YOUR_API_KEY
```

## 5. 安装依赖

```bash
pip install alpaca-trade-api
```

## 6. 测试连接

```bash
cd /path/to/quant-bot
python examples/alpaca_adapter.py
```

## 7. 优势对比

| 功能 | Interactive Brokers | Alpaca |
|------|-------------------|---------|
| 佣金 | $0.005/股 | 免佣金 |
| API稳定性 | 复杂，经常断连 | 简单稳定 |
| 设置难度 | 困难 | 简单 |
| 模拟交易 | 支持 | 支持 |
| 市场数据 | 丰富 | 充足 |
| 算法交易 | 支持 | 专为此设计 |

## 8. 注意事项

- 先使用Paper Trading测试策略
- Alpaca主要支持美股交易
- 确保遵守PDT规则（日内交易规则）
- 定期检查API限制和费用

## 9. 故障排除

### 常见问题：
1. **401 Unauthorized**: 检查API密钥是否正确
2. **403 Forbidden**: 检查账户状态和权限
3. **连接超时**: 检查网络连接

### 解决方案：
```python
# 测试API连接
import requests

def test_alpaca_api(api_key, secret_key, base_url):
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key
    }
    
    response = requests.get(f"{base_url}/v2/account", headers=headers)
    
    if response.status_code == 200:
        print("✅ API连接成功")
        return True
    else:
        print(f"❌ API连接失败: {response.status_code} - {response.text}")
        return False
```