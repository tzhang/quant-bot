#!/usr/bin/env python3
"""
最小化TWS API连接测试
专注于基本连接诊断
"""

import socket
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import threading

class MinimalWrapper(EWrapper):
    def __init__(self):
        EWrapper.__init__(self)
        self.connected = False
        self.error_messages = []
        self.next_valid_id = None
        # 新增：记录connectAck是否到达
        self.connect_ack = False
        
    def connectAck(self):
        print("✅ 收到connectAck（握手开始）")
        # 标记connectAck
        self.connect_ack = True
        
    def nextValidId(self, orderId: int):
        self.next_valid_id = orderId
        self.connected = True
        print(f"✅ nextValidId回调收到，握手完成，orderId={orderId}")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        self.error_messages.append(f"错误 {errorCode}: {errorString}")
        if errorCode in [2104, 2106, 2158, 2110]:
            print(f"ℹ️  {errorString}")
        else:
            print(f"❌ 错误 {errorCode}: {errorString}")

class MinimalClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)


def test_basic_connection(port=7497, client_id=1, timeout=25):
    print("🚀 最小化TWS API连接测试")
    print("=" * 50)
    
    print("\n1️⃣ 测试Socket连接...")
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        if result == 0:
            print("✅ Socket连接成功")
        else:
            print("❌ Socket连接失败")
            return False
    except Exception as e:
        print(f"❌ Socket连接异常: {e}")
        return False
    
    print("\n2️⃣ 测试API握手...")
    wrapper = MinimalWrapper()
    client = MinimalClient(wrapper)
    try:
        print(f"🔄 连接到 127.0.0.1:{port} (客户端ID: {client_id})")
        # 连接选项：限制超时并关闭Nagle，避免堆积
        try:
            client.setConnectOptions("ConnectTimeout=6000;UseNagleAlgorithm=0")
        except Exception:
            pass
        client.connect("127.0.0.1", port, client_id)
        
        api_thread = threading.Thread(target=client.run, daemon=True)
        api_thread.start()
        
        # 仅在收到connectAck后，才尝试startApi/reqIds，避免部分版本崩溃
        start_time = time.time()
        connect_ack_timeout = min(5, timeout)
        while not wrapper.connect_ack and (time.time() - start_time) < connect_ack_timeout:
            time.sleep(0.2)
        
        if wrapper.connect_ack:
            # 显式发送 START_API（有些配置需要）
            try:
                client.startApi()
                print("📨 已发送 startApi() 启动指令（在connectAck后）")
            except Exception as e:
                print(f"⚠️  发送 startApi 失败: {e}")
            
            # 触发 nextValidId
            try:
                client.reqIds(1)
                print("📨 已发送 reqIds(1) 请求，等待 nextValidId...")
            except Exception as e:
                print(f"⚠️  发送 reqIds 失败: {e}")
        else:
            print("⏳ 未收到 connectAck，跳过 startApi/reqIds 以避免Gateway崩溃")
        
        # 等待 nextValidId
        wait_start = time.time()
        while wrapper.next_valid_id is None and (time.time() - wait_start) < timeout:
            time.sleep(0.2)
        
        if wrapper.next_valid_id is not None:
            print("✅ API握手成功 (nextValidId 收到)")
            time.sleep(1)
            client.disconnect()
            return True
        else:
            print(f"❌ API握手超时 ({timeout}秒)，未收到nextValidId")
            if wrapper.error_messages:
                print("\n收到的错误消息:")
                for msg in wrapper.error_messages:
                    print(f"  - {msg}")
            client.disconnect()
            return False
    except Exception as e:
        print(f"❌ API连接异常: {e}")
        try:
            client.disconnect()
        except:
            pass
        return False

def check_tws_status():
    print("\n3️⃣ 检查TWS状态...")
    
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'JavaApplicationStub' in result.stdout or 'Trader Workstation' in result.stdout:
            print("✅ TWS进程正在运行")
        else:
            print("❌ 未找到TWS进程")
    except:
        print("⚠️  无法检查进程状态")
    
    try:
        result = subprocess.run(['lsof', '-i', ':7497'], capture_output=True, text=True)
        if result.stdout:
            print("✅ 端口7497正在监听")
            print(f"   详情: {result.stdout.split()[0]} (PID: {result.stdout.split()[1]})")
        else:
            print("❌ 端口7497未监听")
    except:
        print("⚠️  无法检查端口状态")

def main():
    check_tws_status()
    
    success = test_basic_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TWS API连接测试成功！")
    else:
        print("❌ TWS API连接测试失败")
        print("\n🔧 建议检查:")
        print("1. TWS是否完全登录（显示主界面，不是登录界面）")
        print("2. API设置是否正确:")
        print("   - Configure → Global Configuration → API → Settings")
        print("   - ✅ Enable ActiveX and Socket Clients")
        print("   - ❌ Read-Only API (必须取消勾选)")
        print("   - Socket Port: 7497")
        print("   - Trusted IPs: 127.0.0.1")
        print("3. 是否点击了Apply按钮")
        print("4. 是否完全重启了TWS")

    print("\n3️⃣ 检查TWS状态...")
    import subprocess
    try:
        result = subprocess.run(['lsof', '-i', ':7497'], capture_output=True, text=True)
        if result.stdout:
            print("✅ 端口7497正在监听")
        else:
            print("❌ 端口7497未监听")
    except:
        pass
    
    for cid in [1, 2, 3, 11, 99]:
        ok = test_basic_connection(port=7497, client_id=cid, timeout=25)
        if ok:
            print(f"🎉 使用 clientId={cid} 连接成功！")
            return
        else:
            print(f"➡️ 使用 clientId={cid} 失败，尝试下一个…")
    
    print("\n❌ 所有clientId均失败。请在TWS中：")
    print("- 检查是否弹出授权窗口并点击允许")
    print("- 在 Trusted IPs 中添加 127.0.0.1 和 ::1")
    print("- Paper账户请选择端口 7497；如是Live，尝试 7496")

if __name__ == "__main__":
    main()