# Interactive Brokers API 设置指南

## 1. TWS API 配置步骤

### 登录后配置API
1. 在TWS主界面，点击 **File** → **Global Configuration**
2. 在左侧菜单中选择 **API** → **Settings**
3. 启用以下选项：
   - ✅ **Enable ActiveX and Socket Clients**
   - ✅ **Allow connections from localhost only** (推荐用于开发)
   - Socket Port: **7497** (实时账户) 或 **7496** (模拟账户)
   - Master API client ID: **0** (默认)

### 安全设置
4. 在 **Trusted IPs** 中添加：
   - `127.0.0.1` (本地连接)
   - 如需远程连接，添加相应IP地址

5. 点击 **OK** 保存设置
6. **重启TWS** 使设置生效

## 2. 验证API连接

### 检查端口监听
```bash
# 检查TWS是否在监听API端口
lsof -i :7497  # 实时账户
lsof -i :7496  # 模拟账户
```

### 测试连接
```python
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import threading
import time

class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        
    def nextValidId(self, orderId: int):
        print(f"连接成功！下一个有效订单ID: {orderId}")
        self.disconnect()

# 测试连接
app = TestApp()
app.connect("127.0.0.1", 7497, 0)  # host, port, clientId

# 运行消息循环
api_thread = threading.Thread(target=app.run)
api_thread.start()

time.sleep(2)  # 等待连接
```

## 3. 常见问题解决

### 连接被拒绝
- 确认TWS已启动并登录
- 检查API设置是否正确启用
- 验证端口号是否正确
- 确认IP地址在可信列表中

### 端口冲突
- 实时账户默认端口：7497
- 模拟账户默认端口：7496
- 可在API设置中修改端口号

### 权限问题
- 确保账户有API访问权限
- 免费试用账户通常包含API访问权限

## 4. 生产环境注意事项

### 账户类型确认
- 免费试用账户：有限功能，适合测试
- 模拟账户：需要真实账户，完整功能测试
- 实时账户：真实交易，需要资金

### 安全建议
- 仅允许必要的IP地址连接
- 使用强密码和双因素认证
- 定期更新API密钥（如适用）
- 监控API使用情况

## 5. 配置文件设置

更新 `config.py` 中的IB配置：
```python
# Interactive Brokers配置
IB_HOST = "127.0.0.1"
IB_PORT = 7497  # 实时账户端口
IB_CLIENT_ID = 1
IB_ENABLED = True
```

## 6. 下一步
1. 等待免费试用账户激活
2. 登录TWS并配置API设置
3. 运行连接测试
4. 开始API开发和测试