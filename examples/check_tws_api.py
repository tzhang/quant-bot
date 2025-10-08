#!/usr/bin/env python3
"""
检查 TWS API 设置和连接状态
"""

import socket
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.common import OrderId

class DetailedTester(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.connection_events = []
        
    def log_event(self, event):
        timestamp = time.strftime("%H:%M:%S")
        self.connection_events.append(f"[{timestamp}] {event}")
        print(f"[{timestamp}] {event}")
        
    def connectAck(self):
        self.log_event("✅ 收到连接确认 (connectAck)")
        
    def nextValidId(self, orderId: OrderId):
        self.connected = True
        self.log_event(f"✅ 连接完全建立！下一个订单ID: {orderId}")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode == 502:
            self.log_event(f"❌ 连接被拒绝: {errorString}")
            self.log_event("💡 可能原因: TWS API 未启用或客户端ID冲突")
        elif errorCode == 504:
            self.log_event(f"❌ 连接超时: {errorString}")
            self.log_event("💡 可能原因: TWS 未响应或网络问题")
        elif errorCode in [2104, 2106, 2158]:
            self.log_event(f"ℹ️  信息: {errorString}")
        else:
            self.log_event(f"❌ 错误 [{errorCode}]: {errorString}")
    
    def connectionClosed(self):
        self.log_event("🔌 连接已关闭")

def detailed_connection_test():
    """详细的连接测试"""
    print("🔍 详细 IB TWS API 连接测试")
    print("=" * 60)
    
    host = "127.0.0.1"
    port = 7497
    client_id = 1
    
    # 测试端口连通性
    print(f"\n1️⃣ 测试端口连通性 {host}:{port}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("✅ 端口可达 - TWS 正在监听此端口")
        else:
            print("❌ 端口不可达 - TWS 可能未启动或端口错误")
            return
    except Exception as e:
        print(f"❌ 端口测试失败: {e}")
        return
    
    # 测试 API 连接
    print(f"\n2️⃣ 测试 IB API 连接 (客户端ID: {client_id})")
    tester = DetailedTester()
    
    try:
        tester.log_event(f"🔄 尝试连接到 {host}:{port}")
        tester.connect(host, port, client_id)
        
        # 启动消息循环
        tester.log_event("🔄 启动消息循环线程")
        api_thread = threading.Thread(target=tester.run, daemon=True)
        api_thread.start()
        
        # 等待连接，增加超时时间
        timeout = 15
        start_time = time.time()
        tester.log_event(f"⏳ 等待连接建立 (超时: {timeout}秒)")
        
        while not tester.connected and (time.time() - start_time) < timeout:
            time.sleep(0.5)
            
        if tester.connected:
            tester.log_event("🎉 连接测试成功！")
            time.sleep(2)  # 保持连接一会儿
            tester.disconnect()
        else:
            tester.log_event("❌ 连接测试失败 - 超时")
            tester.disconnect()
            
    except Exception as e:
        tester.log_event(f"❌ 连接异常: {e}")
    
    print("\n" + "=" * 60)
    print("📋 连接事件日志:")
    for event in tester.connection_events:
        print(f"   {event}")
    
    print("\n💡 故障排除建议:")
    print("   1. 确保 IB TWS 已启动并完全加载")
    print("   2. 在 TWS 中启用 API:")
    print("      - 菜单: Configure → Global Configuration → API → Settings")
    print("      - 勾选 'Enable ActiveX and Socket Clients'")
    print("      - 设置 Socket port: 7497 (模拟交易)")
    print("      - 取消勾选 'Read-Only API'")
    print("   3. 检查客户端ID是否冲突 (尝试不同的ID)")
    print("   4. 重启 TWS 并重新测试")

if __name__ == "__main__":
    detailed_connection_test()