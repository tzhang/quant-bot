#!/usr/bin/env python3
"""
测试不同客户端ID的TWS连接
"""

import socket
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class TestWrapper(EWrapper):
    def __init__(self, client_id):
        EWrapper.__init__(self)
        self.client_id = client_id
        self.connected = False
        self.next_valid_id = None
        self.connection_time = None
        
    def connectAck(self):
        print(f"✅ [客户端{self.client_id}] 连接确认")
        self.connected = True
        self.connection_time = time.time()
        
    def nextValidId(self, orderId: int):
        print(f"✅ [客户端{self.client_id}] 收到有效订单ID: {orderId}")
        self.next_valid_id = orderId
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"❌ [客户端{self.client_id}] 错误 {errorCode}: {errorString}")

class TestClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

def test_client_id(client_id, timeout=8):
    """测试特定客户端ID"""
    print(f"\n🔄 测试客户端ID: {client_id}")
    
    wrapper = TestWrapper(client_id)
    client = TestClient(wrapper)
    
    try:
        client.connect("127.0.0.1", 7497, client_id)
        
        # 启动消息循环
        api_thread = threading.Thread(target=client.run, daemon=True)
        api_thread.start()
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if wrapper.connected and wrapper.next_valid_id is not None:
                elapsed = time.time() - start_time
                print(f"   ✅ 成功! 耗时: {elapsed:.2f}秒")
                client.disconnect()
                return True
            time.sleep(0.1)
        
        print(f"   ❌ 超时 ({timeout}秒)")
        client.disconnect()
        return False
        
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        try:
            client.disconnect()
        except:
            pass
        return False

def main():
    print("🔍 测试不同客户端ID的TWS连接")
    print("=" * 50)
    
    # 测试多个客户端ID
    client_ids = [1, 2, 3, 10, 100]
    
    for client_id in client_ids:
        success = test_client_id(client_id)
        if success:
            print(f"\n🎉 客户端ID {client_id} 连接成功!")
            break
        time.sleep(1)  # 短暂等待避免连接冲突
    else:
        print("\n❌ 所有客户端ID都连接失败")
        print("\n💡 建议检查:")
        print("1. TWS是否完全启动并登录")
        print("2. API设置中是否取消勾选'Read-Only API'")
        print("3. 是否有其他程序占用API连接")

if __name__ == "__main__":
    main()