#!/usr/bin/env python3
"""
IB Gateway 状态检查和诊断工具
帮助用户诊断IB Gateway连接问题
"""

import socket
import subprocess
import platform
import psutil
import time
from typing import List, Tuple, Dict

def check_port_detailed(host: str, port: int, timeout: int = 5) -> Dict:
    """详细检查端口状态"""
    result = {
        'port': port,
        'host': host,
        'status': 'unknown',
        'error': None,
        'response_time': None
    }
    
    try:
        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        connection_result = sock.connect_ex((host, port))
        response_time = time.time() - start_time
        
        sock.close()
        
        result['response_time'] = round(response_time * 1000, 2)  # ms
        
        if connection_result == 0:
            result['status'] = 'open'
        else:
            result['status'] = 'closed'
            result['error'] = f"连接错误码: {connection_result}"
            
    except socket.timeout:
        result['status'] = 'timeout'
        result['error'] = f"连接超时 ({timeout}秒)"
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result

def find_ib_processes() -> List[Dict]:
    """查找IB相关进程"""
    ib_processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
            try:
                proc_info = proc.info
                name = proc_info['name'].lower()
                cmdline = ' '.join(proc_info['cmdline'] or []).lower()
                
                # 查找IB相关进程
                if any(keyword in name or keyword in cmdline for keyword in [
                    'tws', 'gateway', 'ibgateway', 'interactive', 'brokers'
                ]):
                    ib_processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'cmdline': proc_info['cmdline'],
                        'status': proc_info['status']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
    except Exception as e:
        print(f"查找进程时出错: {e}")
    
    return ib_processes

def check_network_connectivity() -> Dict:
    """检查网络连接"""
    result = {
        'localhost_ping': False,
        'external_ping': False,
        'dns_resolution': False
    }
    
    try:
        # 检查localhost连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        localhost_result = sock.connect_ex(('127.0.0.1', 80))
        sock.close()
        result['localhost_ping'] = localhost_result == 0
        
        # 检查外部连接 (Google DNS)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        external_result = sock.connect_ex(('8.8.8.8', 53))
        sock.close()
        result['external_ping'] = external_result == 0
        
        # 检查DNS解析
        try:
            socket.gethostbyname('google.com')
            result['dns_resolution'] = True
        except:
            result['dns_resolution'] = False
            
    except Exception as e:
        print(f"网络检查出错: {e}")
    
    return result

def check_firewall_status() -> str:
    """检查防火墙状态 (macOS)"""
    try:
        if platform.system() == 'Darwin':  # macOS
            result = subprocess.run(['sudo', 'pfctl', '-s', 'info'], 
                                  capture_output=True, text=True, timeout=5)
            if 'Status: Enabled' in result.stdout:
                return "已启用"
            elif 'Status: Disabled' in result.stdout:
                return "已禁用"
            else:
                return "未知状态"
        else:
            return "非macOS系统，无法检查"
    except Exception as e:
        return f"检查失败: {str(e)}"

def get_listening_ports() -> List[Tuple[int, str]]:
    """获取正在监听的端口"""
    listening_ports = []
    
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == psutil.CONN_LISTEN and conn.laddr:
                port = conn.laddr.port
                try:
                    proc = psutil.Process(conn.pid) if conn.pid else None
                    proc_name = proc.name() if proc else "未知"
                except:
                    proc_name = "未知"
                
                listening_ports.append((port, proc_name))
    except Exception as e:
        print(f"获取监听端口失败: {e}")
    
    return sorted(listening_ports)

def main():
    """主诊断函数"""
    print("🔍 IB Gateway 详细状态检查")
    print("=" * 60)
    
    # 1. 检查IB相关进程
    print("\n📋 1. 检查IB相关进程")
    print("-" * 30)
    ib_processes = find_ib_processes()
    
    if ib_processes:
        print(f"✅ 发现 {len(ib_processes)} 个IB相关进程:")
        for proc in ib_processes:
            print(f"   PID: {proc['pid']}, 名称: {proc['name']}, 状态: {proc['status']}")
            if proc['cmdline']:
                print(f"   命令行: {' '.join(proc['cmdline'][:3])}...")
    else:
        print("❌ 未发现IB相关进程")
        print("   请确认IB Gateway或TWS已启动")
    
    # 2. 检查端口状态
    print("\n🔌 2. 详细端口检查")
    print("-" * 30)
    
    ports_to_check = [
        (4001, "IB Gateway 模拟交易"),
        (4000, "IB Gateway 真实交易"), 
        (7497, "TWS 模拟交易"),
        (7496, "TWS 真实交易")
    ]
    
    open_ports = []
    for port, description in ports_to_check:
        result = check_port_detailed('127.0.0.1', port)
        status_icon = "✅" if result['status'] == 'open' else "❌"
        
        print(f"{status_icon} 端口 {port} ({description}): {result['status']}")
        if result['response_time']:
            print(f"   响应时间: {result['response_time']}ms")
        if result['error']:
            print(f"   错误: {result['error']}")
        
        if result['status'] == 'open':
            open_ports.append((port, description))
    
    # 3. 检查所有监听端口
    print("\n👂 3. 系统监听端口 (4000-4010, 7490-7500)")
    print("-" * 30)
    
    listening_ports = get_listening_ports()
    relevant_ports = [(port, proc) for port, proc in listening_ports 
                     if (4000 <= port <= 4010) or (7490 <= port <= 7500)]
    
    if relevant_ports:
        print("发现相关端口:")
        for port, proc_name in relevant_ports:
            print(f"   端口 {port}: {proc_name}")
    else:
        print("❌ 未发现IB相关端口在监听")
    
    # 4. 网络连接检查
    print("\n🌐 4. 网络连接检查")
    print("-" * 30)
    
    network_status = check_network_connectivity()
    
    localhost_icon = "✅" if network_status['localhost_ping'] else "❌"
    print(f"{localhost_icon} 本地网络连接")
    
    external_icon = "✅" if network_status['external_ping'] else "❌"
    print(f"{external_icon} 外部网络连接")
    
    dns_icon = "✅" if network_status['dns_resolution'] else "❌"
    print(f"{dns_icon} DNS解析")
    
    # 5. 防火墙检查
    print("\n🛡️  5. 防火墙状态")
    print("-" * 30)
    firewall_status = check_firewall_status()
    print(f"防火墙状态: {firewall_status}")
    
    # 6. 诊断建议
    print("\n💡 6. 诊断建议")
    print("-" * 30)
    
    if not ib_processes:
        print("🔧 IB Gateway未运行:")
        print("   1. 启动IB Gateway应用程序")
        print("   2. 使用您的IB账户登录")
        print("   3. 选择交易模式 (模拟/真实)")
    
    if not open_ports:
        print("🔧 API端口未开放:")
        print("   1. 在IB Gateway中启用API设置")
        print("   2. 勾选 'Enable ActiveX and Socket Clients'")
        print("   3. 设置正确的端口号")
        print("   4. 添加 127.0.0.1 到信任IP列表")
        print("   5. 重启IB Gateway")
    
    if not network_status['localhost_ping']:
        print("🔧 本地网络问题:")
        print("   1. 检查网络配置")
        print("   2. 重启网络服务")
    
    if firewall_status == "已启用":
        print("🔧 防火墙可能阻止连接:")
        print("   1. 在防火墙中允许IB Gateway")
        print("   2. 或临时禁用防火墙进行测试")
    
    # 7. 总结
    print("\n📊 7. 状态总结")
    print("-" * 30)
    
    if open_ports:
        print(f"✅ 发现 {len(open_ports)} 个可用端口:")
        for port, desc in open_ports:
            print(f"   端口 {port}: {desc}")
        print("\n🎯 建议使用第一个可用端口进行连接")
    else:
        print("❌ 未发现可用的IB API端口")
        print("🔧 请按照上述建议检查IB Gateway配置")
    
    return len(open_ports) > 0

if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 60)
    if success:
        print("✅ 诊断完成 - 发现可用连接")
    else:
        print("❌ 诊断完成 - 需要解决连接问题")
    
    exit(0 if success else 1)