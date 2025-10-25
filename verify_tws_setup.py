#!/usr/bin/env python3
"""
TWS设置验证脚本
用于检查TWS是否正确启动和配置
"""

import socket
import subprocess
import sys
from typing import Tuple, List

def check_port_connection(host: str = "127.0.0.1", port: int = 7497) -> bool:
    """检查端口连通性"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def check_tws_process() -> List[str]:
    """检查TWS相关进程"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        tws_processes = [line for line in lines if any(keyword in line.lower() 
                        for keyword in ['tws', 'trader workstation', 'interactive brokers'])]
        return tws_processes
    except Exception:
        return []

def check_java_processes() -> List[str]:
    """检查Java进程（TWS通常是Java应用）"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        java_processes = [line for line in lines if 'java' in line.lower() and 
                         any(keyword in line.lower() for keyword in ['tws', 'ib', 'trader'])]
        return java_processes
    except Exception:
        return []

def main():
    print("🔍 TWS设置验证工具")
    print("=" * 50)
    
    # 检查常用端口
    ports_to_check = [7497, 7496]  # 实时账户和模拟账户端口
    
    print("\n1️⃣ 检查API端口连通性")
    for port in ports_to_check:
        is_connected = check_port_connection(port=port)
        account_type = "实时账户" if port == 7497 else "模拟账户"
        status = "✅ 可连接" if is_connected else "❌ 不可达"
        print(f"   端口 {port} ({account_type}): {status}")
    
    print("\n2️⃣ 检查TWS进程")
    tws_processes = check_tws_process()
    if tws_processes:
        print("   ✅ 发现TWS相关进程:")
        for process in tws_processes[:3]:  # 只显示前3个
            print(f"   {process[:100]}...")
    else:
        print("   ❌ 未发现TWS进程")
    
    print("\n3️⃣ 检查Java进程")
    java_processes = check_java_processes()
    if java_processes:
        print("   ✅ 发现相关Java进程:")
        for process in java_processes[:2]:  # 只显示前2个
            print(f"   {process[:100]}...")
    else:
        print("   ❌ 未发现相关Java进程")
    
    # 综合判断
    print("\n📊 综合诊断结果")
    any_port_open = any(check_port_connection(port=port) for port in ports_to_check)
    
    if any_port_open:
        print("   ✅ TWS API已启动并可连接")
        print("   💡 建议：运行 python examples/check_tws_api.py 进行详细测试")
    else:
        print("   ❌ TWS API未启动或配置有误")
        print("\n🔧 解决建议：")
        print("   1. 确保TWS已启动并登录")
        print("   2. 检查API设置：File → Global Configuration → API → Settings")
        print("   3. 启用 'Enable ActiveX and Socket Clients'")
        print("   4. 添加 127.0.0.1 到可信IP列表")
        print("   5. 重启TWS使设置生效")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()