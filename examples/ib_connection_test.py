#!/usr/bin/env python3
"""
IB TWS 连接诊断工具
用于测试不同端口和客户端ID的连接情况
"""

import socket
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.common import OrderId

class ConnectionTester(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.connection_time = None
        
    def connectAck(self):
        print("✅ 连接确认收到")
        
    def nextValidId(self, orderId: OrderId):
        self.connected = True
        self.connection_time = time.time()
        print(f"✅ 连接成功！下一个订单ID: {orderId}")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158]:
            print(f"ℹ️  信息: {errorString}")
        else:
            print(f"❌ 错误 [{errorCode}]: {errorString}")

def test_port_connectivity(host, port):
    """测试端口连通性"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        return False

def test_ib_connection(host, port, client_id, timeout=10):
    """测试 IB 连接"""
    print(f"\n🔍 测试连接: {host}:{port} (客户端ID: {client_id})")
    
    # 首先测试端口连通性
    if not test_port_connectivity(host, port):
        print(f"❌ 端口 {port} 不可达")
        return False
    
    print(f"✅ 端口 {port} 可达")
    
    # 测试 IB API 连接
    tester = ConnectionTester()
    
    try:
        tester.connect(host, port, client_id)
        
        # 启动消息循环
        api_thread = threading.Thread(target=tester.run, daemon=True)
        api_thread.start()
        
        # 等待连接
        start_time = time.time()
        while not tester.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if tester.connected:
            print(f"✅ IB API 连接成功 (耗时: {tester.connection_time - start_time:.2f}秒)")
            tester.disconnect()
            return True
        else:
            print(f"❌ IB API 连接超时 ({timeout}秒)")
            tester.disconnect()
            return False
            
    except Exception as e:
        print(f"❌ 连接异常: {e}")
        return False

def main():
    """主函数"""
    print("🚀 IB TWS 连接诊断工具")
    print("=" * 50)
    
    host = "127.0.0.1"
    
    # 常见的 IB TWS 端口
    test_configs = [
        (7497, 1, "TWS 模拟交易"),
        (7496, 1, "TWS 真实交易"), 
        (4001, 1, "IB Gateway 模拟交易"),
        (4000, 1, "IB Gateway 真实交易"),
        (7497, 2, "TWS 模拟交易 (客户端ID: 2)"),
        (7497, 0, "TWS 模拟交易 (客户端ID: 0)"),
    ]
    
    successful_connections = []
    
    for port, client_id, description in test_configs:
        print(f"\n📋 {description}")
        if test_ib_connection(host, port, client_id):
            successful_connections.append((port, client_id, description))
        time.sleep(1)  # 避免连接过于频繁
    
    print("\n" + "=" * 50)
    print("📊 诊断结果:")
    
    if successful_connections:
        print("✅ 发现可用连接:")
        for port, client_id, desc in successful_connections:
            print(f"   - 端口 {port}, 客户端ID {client_id}: {desc}")
        
        print("\n💡 建议配置:")
        port, client_id, _ = successful_connections[0]
        print(f"   'ib_host': '{host}',")
        print(f"   'ib_port': {port},")
        print(f"   'client_id': {client_id},")
        
    else:
        print("❌ 未发现可用连接")
        print("\n🔧 请检查:")
        print("   1. IB TWS 或 IB Gateway 是否已启动")
        print("   2. TWS 中的 API 设置是否已启用")
        print("   3. 防火墙是否阻止了连接")
        print("   4. 端口是否被其他程序占用")

if __name__ == "__main__":
    main()