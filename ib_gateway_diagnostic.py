#!/usr/bin/env python3
"""
IB Gateway 详细诊断工具
帮助用户检查和配置IB Gateway连接
"""

import socket
import subprocess
import sys
import time
import psutil
from datetime import datetime

class IBGatewayDiagnostic:
    """IB Gateway 诊断工具"""
    
    def __init__(self):
        self.ib_ports = {
            4001: "IB Gateway 模拟交易",
            4000: "IB Gateway 真实交易", 
            7497: "TWS 模拟交易",
            7496: "TWS 真实交易"
        }
        
    def print_header(self, title):
        """打印标题"""
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
        
    def check_port_status(self, host, port, timeout=3):
        """检查端口状态"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            return False
            
    def find_ib_processes(self):
        """查找IB相关进程"""
        ib_processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    name = proc.info['name'].lower()
                    cmdline = ' '.join(proc.info['cmdline']).lower() if proc.info['cmdline'] else ''
                    
                    if any(keyword in name or keyword in cmdline for keyword in 
                          ['gateway', 'tws', 'interactive', 'brokers', 'ibgateway']):
                        ib_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"❌ 查找进程时出错: {e}")
            
        return ib_processes
        
    def get_listening_ports(self):
        """获取系统监听的端口"""
        listening_ports = []
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == psutil.CONN_LISTEN:
                    listening_ports.append(conn.laddr.port)
        except Exception as e:
            print(f"❌ 获取监听端口时出错: {e}")
            
        return sorted(set(listening_ports))
        
    def check_network_connectivity(self):
        """检查网络连通性"""
        try:
            # 测试本地回环
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 80))
            sock.close()
            return True
        except:
            return False
            
    def run_diagnostic(self):
        """运行完整诊断"""
        print("🚀 IB Gateway 详细诊断工具")
        print(f"⏰ 诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 检查端口状态
        self.print_header("端口连接状态检查")
        available_ports = []
        for port, description in self.ib_ports.items():
            is_available = self.check_port_status('127.0.0.1', port)
            status = "✅ 可用" if is_available else "❌ 不可用"
            print(f"{status} - 端口 {port}: {description}")
            if is_available:
                available_ports.append(port)
                
        # 2. 查找IB相关进程
        self.print_header("IB相关进程检查")
        ib_processes = self.find_ib_processes()
        if ib_processes:
            print("✅ 发现以下IB相关进程:")
            for proc in ib_processes:
                print(f"   PID: {proc['pid']}, 名称: {proc['name']}")
                if proc['cmdline']:
                    print(f"   命令行: {proc['cmdline'][:100]}...")
        else:
            print("❌ 未发现IB相关进程")
            
        # 3. 检查系统监听端口
        self.print_header("系统监听端口检查")
        listening_ports = self.get_listening_ports()
        ib_related_ports = [p for p in listening_ports if p in self.ib_ports.keys()]
        
        if ib_related_ports:
            print("✅ 发现以下IB相关监听端口:")
            for port in ib_related_ports:
                print(f"   端口 {port}: {self.ib_ports[port]}")
        else:
            print("❌ 未发现IB相关监听端口")
            print("📋 当前系统监听的端口 (前20个):")
            for port in listening_ports[:20]:
                print(f"   {port}")
                
        # 4. 网络连通性检查
        self.print_header("网络连通性检查")
        network_ok = self.check_network_connectivity()
        if network_ok:
            print("✅ 本地网络连通性正常")
        else:
            print("❌ 本地网络连通性异常")
            
        # 5. 生成诊断报告和建议
        self.print_header("诊断结果和建议")
        
        if available_ports:
            print(f"🎉 发现可用端口: {available_ports}")
            print("✅ IB Gateway/TWS 已正确配置并运行")
            print("💡 建议: 在监控面板中使用这些端口连接")
        else:
            print("❌ 未发现可用的IB端口")
            print("\n🔧 故障排除步骤:")
            
            if not ib_processes:
                print("1. ❗ 启动IB Gateway:")
                print("   - 双击IB Gateway应用程序")
                print("   - 使用您的IB账户登录")
                
            print("2. ❗ 配置API设置:")
            print("   - 在IB Gateway中，点击 'Configure' -> 'Settings'")
            print("   - 选择 'API' 标签页")
            print("   - 勾选 'Enable ActiveX and Socket Clients'")
            print("   - 设置端口号 (模拟: 4001, 真实: 4000)")
            print("   - 在 'Trusted IPs' 中添加 '127.0.0.1'")
            print("   - 点击 'Apply' 并重启Gateway")
            
            print("3. ❗ 检查防火墙设置:")
            print("   - 确保防火墙允许IB Gateway访问网络")
            print("   - 允许端口4000和4001的入站连接")
            
            print("4. ❗ 重启Gateway:")
            print("   - 完全关闭IB Gateway")
            print("   - 等待10秒后重新启动")
            print("   - 重新登录并检查API设置")
            
        print(f"\n{'='*60}")
        print("📞 如需更多帮助，请参考IB官方文档或联系技术支持")
        print(f"{'='*60}")

def main():
    """主函数"""
    diagnostic = IBGatewayDiagnostic()
    diagnostic.run_diagnostic()

if __name__ == "__main__":
    main()