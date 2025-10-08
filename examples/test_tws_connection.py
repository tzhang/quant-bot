#!/usr/bin/env python3
"""
专门测试 TWS 连接的工具
TWS 已确认在端口 7497 上运行，但连接失败
"""

import socket
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.common import OrderId

class TWSTester(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.connection_established = False
        self.events = []
        
    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S.%f")[:-3]
        event = f"[{timestamp}] {msg}"
        self.events.append(event)
        print(event)
        
    def connectAck(self):
        self.log("✅ connectAck - 连接确认")
        
    def nextValidId(self, orderId: OrderId):
        self.connected = True
        self.connection_established = True
        self.log(f"✅ nextValidId - 连接完全建立！订单ID: {orderId}")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode == 502:
            self.log(f"❌ 错误 502 - 连接被拒绝: {errorString}")
            self.log("💡 可能原因: API 未启用、客户端ID冲突或权限问题")
        elif errorCode == 504:
            self.log(f"❌ 错误 504 - 连接超时: {errorString}")
        elif errorCode == 1100:
            self.log(f"⚠️  错误 1100 - 连接丢失: {errorString}")
        elif errorCode == 1101:
            self.log(f"✅ 错误 1101 - 连接恢复: {errorString}")
        elif errorCode == 1102:
            self.log(f"⚠️  错误 1102 - 连接断开: {errorString}")
        elif errorCode in [2104, 2106, 2158]:
            self.log(f"ℹ️  信息 {errorCode}: {errorString}")
        else:
            self.log(f"❓ 错误 {errorCode}: {errorString}")
    
    def connectionClosed(self):
        self.log("🔌 connectionClosed - 连接已关闭")
        
    def managedAccounts(self, accountsList: str):
        self.log(f"📊 managedAccounts - 账户列表: {accountsList}")

def test_socket_connection():
    """测试原始 socket 连接"""
    print("🔍 测试原始 socket 连接到 TWS")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        
        print(f"🔄 连接到 127.0.0.1:7497...")
        result = sock.connect_ex(("127.0.0.1", 7497))
        
        if result == 0:
            print("✅ Socket 连接成功")
            # 尝试发送一些数据看看响应
            try:
                sock.send(b"test")
                print("✅ 数据发送成功")
            except Exception as e:
                print(f"⚠️  数据发送失败: {e}")
        else:
            print(f"❌ Socket 连接失败，错误码: {result}")
            
        sock.close()
        return result == 0
        
    except Exception as e:
        print(f"❌ Socket 测试异常: {e}")
        return False

def test_ib_api_connection():
    """测试 IB API 连接"""
    print("\n🔍 测试 IB API 连接")
    
    for client_id in [1, 2, 0, 10]:
        print(f"\n📋 尝试客户端ID: {client_id}")
        
        tester = TWSTester()
        
        try:
            tester.log(f"🔄 连接到 127.0.0.1:7497 (客户端ID: {client_id})")
            tester.connect("127.0.0.1", 7497, client_id)
            
            # 启动消息循环
            api_thread = threading.Thread(target=tester.run, daemon=True)
            api_thread.start()
            
            # 等待连接
            timeout = 10
            start_time = time.time()
            
            while not tester.connection_established and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if tester.connection_established:
                tester.log("🎉 连接测试成功！")
                time.sleep(1)
                tester.disconnect()
                return True
            else:
                tester.log("❌ 连接超时")
                tester.disconnect()
                
        except Exception as e:
            tester.log(f"❌ 连接异常: {e}")
            
        time.sleep(2)  # 等待一下再尝试下一个客户端ID
    
    return False

def main():
    print("🚀 TWS 连接专项测试")
    print("=" * 50)
    print("📍 已确认 TWS 在端口 7497 上运行")
    
    # 测试原始 socket 连接
    socket_ok = test_socket_connection()
    
    if not socket_ok:
        print("\n❌ 原始 socket 连接失败，TWS 可能未正确启动")
        return
    
    # 测试 IB API 连接
    api_ok = test_ib_api_connection()
    
    print("\n" + "=" * 50)
    print("📊 测试结果:")
    print(f"   Socket 连接: {'✅ 成功' if socket_ok else '❌ 失败'}")
    print(f"   IB API 连接: {'✅ 成功' if api_ok else '❌ 失败'}")
    
    if socket_ok and not api_ok:
        print("\n💡 故障排除建议:")
        print("   1. 在 TWS 中检查 API 设置:")
        print("      - File → Global Configuration → API → Settings")
        print("      - 确保 'Enable ActiveX and Socket Clients' 已勾选")
        print("      - Socket port 设置为 7497")
        print("      - 取消勾选 'Read-Only API' (如果需要交易)")
        print("      - 检查 'Trusted IPs' 是否包含 127.0.0.1")
        print("   2. 重启 TWS 并重新测试")
        print("   3. 检查 TWS 日志文件中的错误信息")

if __name__ == "__main__":
    main()