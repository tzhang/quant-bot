#!/usr/bin/env python3
"""
简单的TWS API连接测试
用于诊断TWS API配置问题
"""

import socket
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class SimpleTWSTest(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.connection_error = None
        
    def connectAck(self):
        """连接确认回调"""
        print("✅ API连接成功建立！")
        self.connected = True
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """错误回调"""
        print(f"❌ 错误 - ID: {reqId}, 代码: {errorCode}, 消息: {errorString}")
        if errorCode in [502, 504]:  # 连接错误
            self.connection_error = f"连接错误 {errorCode}: {errorString}"

def test_socket_connection(host='127.0.0.1', port=7497, timeout=5):
    """测试基础socket连接"""
    print(f"🔍 测试Socket连接到 {host}:{port}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("✅ Socket连接成功")
            return True
        else:
            print(f"❌ Socket连接失败，错误代码: {result}")
            return False
    except Exception as e:
        print(f"❌ Socket连接异常: {e}")
        return False

def test_api_connection(host='127.0.0.1', port=7497, client_id=1, timeout=10):
    """测试API连接"""
    print(f"\n🔍 测试API连接到 {host}:{port} (客户端ID: {client_id})")
    
    app = SimpleTWSTest()
    
    def run_connection():
        try:
            app.connect(host, port, client_id)
            app.run()
        except Exception as e:
            app.connection_error = str(e)
    
    # 在单独线程中运行连接
    thread = threading.Thread(target=run_connection, daemon=True)
    thread.start()
    
    # 等待连接结果
    start_time = time.time()
    while time.time() - start_time < timeout:
        if app.connected:
            print("✅ API连接测试成功！")
            app.disconnect()
            return True
        elif app.connection_error:
            print(f"❌ API连接失败: {app.connection_error}")
            return False
        time.sleep(0.1)
    
    print("❌ API连接超时")
    app.disconnect()
    return False

def main():
    print("🚀 简单TWS API连接测试")
    print("=" * 50)
    
    # 1. 测试Socket连接
    socket_ok = test_socket_connection()
    
    if not socket_ok:
        print("\n❌ Socket连接失败，请检查：")
        print("   - TWS是否正在运行")
        print("   - 端口7497是否正确")
        return
    
    # 2. 测试API连接
    api_ok = test_api_connection()
    
    if not api_ok:
        print("\n🔧 API连接失败，请检查TWS配置：")
        print("   1. 在TWS中打开: Configure → Global Configuration → API → Settings")
        print("   2. ✅ 勾选 'Enable ActiveX and Socket Clients'")
        print("   3. ❌ 取消勾选 'Read-Only API' (重要！)")
        print("   4. 设置 Socket Port: 7497")
        print("   5. 添加 Trusted IP: 127.0.0.1")
        print("   6. 点击 Apply 并重启TWS")
        print("\n⚠️  特别注意：Read-Only API 必须取消勾选！")
    else:
        print("\n🎉 TWS API配置正确，连接成功！")

if __name__ == "__main__":
    main()