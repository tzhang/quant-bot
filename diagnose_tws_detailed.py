#!/usr/bin/env python3
"""
详细的 TWS API 诊断工具
"""

import socket
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class DiagnosticWrapper(EWrapper):
    def __init__(self):
        EWrapper.__init__(self)
        self.connected = False
        self.connection_time = None
        self.error_messages = []
        self.next_valid_id = None
        
    def connectAck(self):
        print("✅ [API] 连接确认收到")
        self.connected = True
        self.connection_time = time.time()
        
    def nextValidId(self, orderId: int):
        print(f"✅ [API] 收到下一个有效订单ID: {orderId}")
        self.next_valid_id = orderId
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        error_msg = f"❌ [API错误] ID:{reqId}, 代码:{errorCode}, 消息:{errorString}"
        print(error_msg)
        self.error_messages.append(error_msg)
        
        # 特定错误代码的解释
        if errorCode == 502:
            print("   💡 错误502: 无法连接到TWS - 检查TWS是否启动并启用API")
        elif errorCode == 504:
            print("   💡 错误504: 未连接 - API连接已断开")
        elif errorCode == 1100:
            print("   💡 错误1100: 连接丢失")
        elif errorCode == 2104:
            print("   💡 错误2104: 市场数据农场连接正常")
        elif errorCode == 2106:
            print("   💡 错误2106: HMDS数据农场连接正常")
        elif errorCode == 2158:
            print("   💡 错误2158: 安全定义选择服务连接正常")

class DiagnosticClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

def test_socket_connection(host, port):
    """测试基础socket连接"""
    print(f"🔌 测试Socket连接 {host}:{port}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"   ✅ Socket连接成功")
            return True
        else:
            print(f"   ❌ Socket连接失败: {result}")
            return False
    except Exception as e:
        print(f"   ❌ Socket连接异常: {e}")
        return False

def test_api_handshake(host, port, client_id):
    """测试API握手过程"""
    print(f"\n🤝 测试API握手 {host}:{port} (客户端ID: {client_id})")
    
    wrapper = DiagnosticWrapper()
    client = DiagnosticClient(wrapper)
    
    try:
        print("   🔄 开始连接...")
        client.connect(host, port, client_id)
        
        # 启动消息循环
        print("   🔄 启动消息循环...")
        api_thread = threading.Thread(target=client.run, daemon=True)
        api_thread.start()
        
        # 等待连接建立
        timeout = 10
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if wrapper.connected and wrapper.next_valid_id is not None:
                print(f"   ✅ API握手成功! 耗时: {time.time() - start_time:.2f}秒")
                
                # 测试简单的API调用
                print("   🔄 测试API调用...")
                try:
                    client.reqCurrentTime()
                    time.sleep(2)
                    print("   ✅ API调用测试完成")
                except Exception as e:
                    print(f"   ⚠️ API调用测试失败: {e}")
                
                client.disconnect()
                return True
                
            time.sleep(0.1)
        
        print(f"   ❌ API握手超时 ({timeout}秒)")
        print(f"   📊 连接状态: connected={wrapper.connected}, next_valid_id={wrapper.next_valid_id}")
        
        if wrapper.error_messages:
            print("   📋 错误消息:")
            for msg in wrapper.error_messages:
                print(f"      {msg}")
        
        client.disconnect()
        return False
        
    except Exception as e:
        print(f"   ❌ API握手异常: {e}")
        client.disconnect()
        return False

def main():
    print("🔍 TWS API 详细诊断工具")
    print("=" * 60)
    
    host = "127.0.0.1"
    ports = [7497, 7496]
    
    for port in ports:
        print(f"\n📡 测试端口 {port}")
        print("-" * 40)
        
        # 1. 测试Socket连接
        if not test_socket_connection(host, port):
            print(f"   ⏭️ 跳过端口 {port} (Socket连接失败)")
            continue
        
        # 2. 测试API握手
        success = test_api_handshake(host, port, 1)
        
        if success:
            print(f"   🎉 端口 {port} 测试成功!")
            break
        else:
            print(f"   ❌ 端口 {port} 测试失败")
    
    print("\n" + "=" * 60)
    print("🔧 故障排除建议:")
    print("1. 确保TWS完全启动并登录")
    print("2. 在TWS中: Configure → Global Configuration → API → Settings")
    print("3. 勾选 'Enable ActiveX and Socket Clients'")
    print("4. 取消勾选 'Read-Only API'")
    print("5. 设置Socket Port为7497")
    print("6. 添加127.0.0.1到Trusted IPs")
    print("7. 点击Apply并重启TWS")
    print("8. 确保没有其他程序使用相同的客户端ID")

if __name__ == "__main__":
    main()