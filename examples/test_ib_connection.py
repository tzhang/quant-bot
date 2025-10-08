#!/usr/bin/env python3
"""
Interactive Brokers API 连接测试脚本
用于验证TWS API连接是否正常工作
"""

import sys
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBTestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.next_order_id = None
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """处理错误消息"""
        print(f"错误 - ID: {reqId}, 代码: {errorCode}, 消息: {errorString}")
        
    def connectAck(self):
        """连接确认"""
        print("✅ 连接确认收到")
        
    def nextValidId(self, orderId: int):
        """接收下一个有效订单ID"""
        print(f"✅ 连接成功！下一个有效订单ID: {orderId}")
        self.connected = True
        self.next_order_id = orderId
        
        # 请求账户信息
        self.reqAccountSummary(1, "All", "TotalCashValue,NetLiquidation")
        
        # 请求市场数据测试
        self.test_market_data()
        
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """账户摘要信息"""
        print(f"📊 账户信息 - {tag}: {value} {currency}")
        
    def accountSummaryEnd(self, reqId: int):
        """账户摘要结束"""
        print("✅ 账户信息获取完成")
        
    def tickPrice(self, reqId, tickType, price, attrib):
        """市场数据回调"""
        tick_types = {1: "买价", 2: "卖价", 4: "最新价", 6: "最高价", 7: "最低价", 9: "收盘价"}
        tick_name = tick_types.get(tickType, f"类型{tickType}")
        print(f"📈 市场数据 - {tick_name}: ${price}")
        
    def test_market_data(self):
        """测试市场数据请求"""
        print("\n🔍 测试市场数据请求...")
        
        # 创建AAPL合约
        contract = Contract()
        contract.symbol = "AAPL"
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        # 请求市场数据
        self.reqMktData(1, contract, "", False, False, [])
        
def test_connection():
    """测试IB API连接"""
    print("🚀 开始测试 Interactive Brokers API 连接...")
    print("=" * 50)
    
    # 创建应用实例
    app = IBTestApp()
    
    # 连接参数
    host = "127.0.0.1"
    port = 7497  # 实时账户端口
    client_id = 1
    
    print(f"📡 尝试连接到 {host}:{port} (客户端ID: {client_id})")
    
    try:
        # 连接到TWS
        app.connect(host, port, client_id)
        
        # 启动消息循环线程
        api_thread = threading.Thread(target=app.run, daemon=True)
        api_thread.start()
        
        # 等待连接建立
        timeout = 10
        start_time = time.time()
        
        while not app.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if app.connected:
            print("\n✅ 连接测试成功！")
            print("🎉 TWS API 配置正确，可以开始开发了！")
            
            # 保持连接一段时间以接收数据
            time.sleep(5)
            
        else:
            print("\n❌ 连接超时")
            print("请检查：")
            print("1. TWS是否已启动并登录")
            print("2. API设置是否已启用")
            print("3. 端口号是否正确 (7497)")
            print("4. IP地址是否在可信列表中")
            
    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        print("请检查：")
        print("1. TWS是否已启动")
        print("2. 网络连接是否正常")
        print("3. 防火墙设置")
        
    finally:
        if app.isConnected():
            app.disconnect()
            print("\n🔌 连接已断开")

if __name__ == "__main__":
    # 检查是否安装了ibapi
    try:
        import ibapi
        print(f"📦 IB API 版本: {ibapi.__version__ if hasattr(ibapi, '__version__') else '未知'}")
    except ImportError:
        print("❌ 未安装 ibapi 库")
        print("请运行: pip install ibapi")
        sys.exit(1)
        
    test_connection()