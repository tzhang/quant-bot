#!/usr/bin/env python3
"""
简单的IB连接测试脚本
"""
import socket
import time
import sys

def test_socket_connection(host, port, timeout=5):
    """测试socket连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Socket连接异常: {e}")
        return False

def test_ib_handshake(host, port, client_id=1, timeout=10):
    """测试IB API握手"""
    try:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
        import threading
        
        class TestWrapper(EWrapper):
            def __init__(self):
                self.connected = False
                self.error_occurred = False
                self.error_msg = ""
                
            def connectAck(self):
                print("✅ IB API握手成功")
                self.connected = True
                
            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
                print(f"❌ IB API错误: {errorCode} - {errorString}")
                self.error_occurred = True
                self.error_msg = f"{errorCode}: {errorString}"
        
        wrapper = TestWrapper()
        client = EClient(wrapper)
        
        print(f"尝试连接到 {host}:{port} (客户端ID: {client_id})")
        client.connect(host, port, client_id)
        
        # 启动API线程
        api_thread = threading.Thread(target=client.run, daemon=True)
        api_thread.start()
        
        # 等待连接结果
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if wrapper.connected:
                client.disconnect()
                return True, "连接成功"
            if wrapper.error_occurred:
                client.disconnect()
                return False, wrapper.error_msg
            time.sleep(0.1)
        
        client.disconnect()
        return False, "连接超时"
        
    except ImportError:
        return False, "ibapi模块未安装"
    except Exception as e:
        return False, f"连接异常: {e}"

if __name__ == "__main__":
    host = "127.0.0.1"
    ports = [4001, 4000, 7497, 7496]
    
    print("=== IB连接诊断 ===")
    
    for port in ports:
        print(f"\n测试端口 {port}:")
        
        # 1. Socket连接测试
        socket_ok = test_socket_connection(host, port)
        print(f"  Socket连接: {'✅ 成功' if socket_ok else '❌ 失败'}")
        
        if socket_ok:
            # 2. IB API握手测试
            api_ok, msg = test_ib_handshake(host, port)
            print(f"  IB API握手: {'✅ 成功' if api_ok else '❌ 失败'} - {msg}")
        else:
            print(f"  IB API握手: ⏭️ 跳过 (Socket连接失败)")
    
    print("\n=== 诊断完成 ===")
