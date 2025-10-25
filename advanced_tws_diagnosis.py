#!/usr/bin/env python3
"""
高级TWS API诊断工具
用于深度诊断TWS API配置问题
"""

import socket
import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class AdvancedTWSTest(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.connection_error = None
        self.error_messages = []
        self.connection_time = None
        
    def connectAck(self):
        """连接确认回调"""
        self.connected = True
        self.connection_time = time.time()
        print("✅ API连接成功建立！")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """错误回调"""
        error_msg = f"错误 - ID: {reqId}, 代码: {errorCode}, 消息: {errorString}"
        print(f"❌ {error_msg}")
        self.error_messages.append((errorCode, errorString))
        
        # 特定错误代码处理
        if errorCode == 502:
            self.connection_error = "无法连接到TWS - 请检查TWS是否运行"
        elif errorCode == 504:
            self.connection_error = "不是有效的客户端ID"
        elif errorCode == 1100:
            self.connection_error = "连接丢失"
        elif errorCode == 2104:
            print("ℹ️  市场数据农场连接正常")
        elif errorCode == 2106:
            print("ℹ️  HMDS数据农场连接正常")

def test_multiple_client_ids(host='127.0.0.1', port=7497, timeout=8):
    """测试多个客户端ID"""
    print(f"\n🔍 测试多个客户端ID连接到 {host}:{port}")
    print("=" * 50)
    
    client_ids = [1, 2, 3, 0, 10]  # 常用的客户端ID
    
    for client_id in client_ids:
        print(f"\n🔄 测试客户端ID: {client_id}")
        
        app = AdvancedTWSTest()
        
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
                print(f"✅ 客户端ID {client_id} 连接成功！")
                app.disconnect()
                return client_id
            elif app.connection_error:
                print(f"❌ 客户端ID {client_id} 连接失败: {app.connection_error}")
                break
            time.sleep(0.1)
        
        if not app.connected and not app.connection_error:
            print(f"⏰ 客户端ID {client_id} 连接超时")
        
        try:
            app.disconnect()
        except:
            pass
        
        time.sleep(1)  # 等待清理
    
    return None

def check_tws_status():
    """检查TWS状态"""
    print("\n🔍 检查TWS状态")
    print("=" * 30)
    
    # 检查进程
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        tws_processes = [line for line in result.stdout.split('\n') 
                        if 'trader workstation' in line.lower() or 'tws' in line.lower()]
        
        if tws_processes:
            print("✅ TWS进程正在运行")
            for proc in tws_processes[:2]:  # 只显示前2个
                print(f"   {proc.split()[1]} - {proc.split()[10] if len(proc.split()) > 10 else 'TWS'}")
        else:
            print("❌ 未找到TWS进程")
            return False
    except Exception as e:
        print(f"⚠️  无法检查进程状态: {e}")
    
    # 检查端口
    try:
        result = subprocess.run(['lsof', '-i', ':7497'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            print("✅ 端口7497正在监听")
            lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    print(f"   进程: {parts[0]} (PID: {parts[1]})")
        else:
            print("❌ 端口7497未监听")
            return False
    except Exception as e:
        print(f"⚠️  无法检查端口状态: {e}")
    
    return True

def provide_detailed_guidance():
    """提供详细的配置指导"""
    print("\n🔧 详细配置指导")
    print("=" * 40)
    print("如果连接仍然失败，请按照以下步骤操作：")
    print()
    print("1️⃣ 检查TWS登录状态：")
    print("   - 确保TWS完全登录（不在登录界面）")
    print("   - 确保账户状态正常")
    print("   - 如果是纸上交易，确保已切换到纸上交易模式")
    print()
    print("2️⃣ 重新配置API设置：")
    print("   - 在TWS中：Configure → Global Configuration → API → Settings")
    print("   - ✅ 勾选 'Enable ActiveX and Socket Clients'")
    print("   - ❌ 取消勾选 'Read-Only API' (最重要！)")
    print("   - 设置 'Socket Port': 7497")
    print("   - 在 'Trusted IPs' 添加: 127.0.0.1")
    print("   - 点击 'Apply' 按钮")
    print()
    print("3️⃣ 完全重启TWS：")
    print("   - 完全关闭TWS应用程序")
    print("   - 等待5-10秒")
    print("   - 重新启动TWS")
    print("   - 等待完全加载")
    print()
    print("4️⃣ 其他可能的解决方案：")
    print("   - 尝试使用端口7496而不是7497")
    print("   - 检查防火墙设置")
    print("   - 确保没有其他程序占用相同的客户端ID")
    print("   - 考虑更新TWS到最新版本")
    print()
    print("⚠️  最常见的问题是 'Read-Only API' 仍然启用！")

def main():
    print("🚀 高级TWS API诊断工具")
    print("=" * 60)
    
    # 1. 检查TWS状态
    if not check_tws_status():
        print("\n❌ TWS状态检查失败，请先启动TWS")
        return
    
    # 2. 测试Socket连接
    print(f"\n🔍 测试Socket连接到 127.0.0.1:7497")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 7497))
        sock.close()
        
        if result == 0:
            print("✅ Socket连接成功")
        else:
            print(f"❌ Socket连接失败，错误代码: {result}")
            return
    except Exception as e:
        print(f"❌ Socket连接异常: {e}")
        return
    
    # 3. 测试多个客户端ID
    successful_client_id = test_multiple_client_ids()
    
    if successful_client_id is not None:
        print(f"\n🎉 成功！客户端ID {successful_client_id} 可以连接TWS API")
        print("✅ TWS API配置正确")
    else:
        print(f"\n❌ 所有客户端ID都无法连接")
        provide_detailed_guidance()

if __name__ == "__main__":
    main()