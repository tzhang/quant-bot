#!/usr/bin/env python3
"""
综合TWS API连接测试工具
尝试不同端口、客户端ID和超时设置
"""

import socket
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading

class TestWrapper(EWrapper):
    def __init__(self):
        EWrapper.__init__(self)
        self.connected = False
        self.connection_time = None
        self.account_info = None
        self.positions = []
        
    def connectAck(self):
        print("✅ API连接确认收到")
        self.connected = True
        self.connection_time = time.time()
        
    def managedAccounts(self, accountsList):
        print(f"✅ 账户列表: {accountsList}")
        self.account_info = accountsList
        
    def position(self, account, contract, position, avgCost):
        print(f"📊 持仓: {contract.symbol} - {position} @ {avgCost}")
        self.positions.append({
            'symbol': contract.symbol,
            'position': position,
            'avgCost': avgCost
        })
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158]:  # 这些是信息性消息，不是错误
            print(f"ℹ️  信息: {errorString}")
        else:
            print(f"❌ 错误 {errorCode}: {errorString}")

class TestClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

def test_socket_connection(host, port):
    """测试Socket连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        return False

def test_api_connection(host, port, client_id, timeout=30):
    """测试API连接"""
    print(f"\n🔄 测试 {host}:{port} (客户端ID: {client_id}, 超时: {timeout}秒)")
    
    wrapper = TestWrapper()
    client = TestClient(wrapper)
    
    try:
        # 连接
        client.connect(host, port, client_id)
        
        # 启动消息循环
        api_thread = threading.Thread(target=client.run, daemon=True)
        api_thread.start()
        
        # 等待连接
        start_time = time.time()
        while not wrapper.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if wrapper.connected:
            print(f"✅ API连接成功！连接时间: {wrapper.connection_time - start_time:.2f}秒")
            
            # 请求账户信息
            client.reqManagedAccts()
            time.sleep(2)
            
            # 请求持仓信息
            client.reqPositions()
            time.sleep(3)
            
            # 断开连接
            client.disconnect()
            return True
        else:
            print(f"❌ API连接超时 ({timeout}秒)")
            client.disconnect()
            return False
            
    except Exception as e:
        print(f"❌ 连接异常: {e}")
        try:
            client.disconnect()
        except:
            pass
        return False

def main():
    print("🚀 综合TWS API连接测试")
    print("=" * 60)
    
    # 测试配置
    host = "127.0.0.1"
    ports = [7497, 7496]  # 尝试两个常用端口
    client_ids = [1, 2, 0, 10, 100]  # 尝试多个客户端ID
    timeout = 30  # 更长的超时时间
    
    success = False
    
    for port in ports:
        print(f"\n🔍 测试端口 {port}")
        print("-" * 30)
        
        # 首先测试Socket连接
        if test_socket_connection(host, port):
            print(f"✅ Socket连接到 {host}:{port} 成功")
            
            # 测试API连接
            for client_id in client_ids:
                if test_api_connection(host, port, client_id, timeout):
                    success = True
                    print(f"\n🎉 成功！使用配置: {host}:{port}, 客户端ID: {client_id}")
                    break
                time.sleep(2)  # 客户端ID之间的间隔
                
            if success:
                break
        else:
            print(f"❌ Socket连接到 {host}:{port} 失败")
    
    if not success:
        print("\n" + "=" * 60)
        print("❌ 所有连接尝试都失败了")
        print("\n🔧 故障排除建议:")
        print("1. 确保TWS完全登录并显示主界面")
        print("2. 检查TWS API配置:")
        print("   - Configure → Global Configuration → API → Settings")
        print("   - ✅ 勾选 'Enable ActiveX and Socket Clients'")
        print("   - ❌ 取消勾选 'Read-Only API'")
        print("   - 设置正确的端口 (7497 或 7496)")
        print("   - 添加 127.0.0.1 到 Trusted IPs")
        print("3. 完全重启TWS")
        print("4. 检查防火墙设置")
        print("5. 尝试纸上交易模式")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()