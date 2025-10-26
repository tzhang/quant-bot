#!/usr/bin/env python3
"""
IB Gateway API配置助手
帮助用户正确配置IB Gateway的API设置
"""

import socket
import time
import subprocess
import sys
from typing import List, Dict, Any

class IBAPIConfigHelper:
    """IB Gateway API配置助手"""
    
    def __init__(self):
        self.ib_ports = {
            4001: "IB Gateway 模拟交易",
            4000: "IB Gateway 真实交易", 
            7497: "TWS 模拟交易",
            7496: "TWS 真实交易"
        }
        
    def check_port_status(self, host: str = "127.0.0.1", port: int = 4001, timeout: float = 2.0) -> Dict[str, Any]:
        """检查端口状态"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return {"status": "open", "message": "端口开放"}
            else:
                return {"status": "closed", "message": f"端口关闭 (错误码: {result})"}
                
        except Exception as e:
            return {"status": "error", "message": f"检查失败: {str(e)}"}
    
    def find_ib_processes(self) -> List[Dict[str, str]]:
        """查找IB相关进程"""
        processes = []
        try:
            # 查找IB Gateway进程
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['gateway', 'tws', 'interactive', 'ib']):
                    if 'grep' not in line and line.strip():
                        parts = line.split()
                        if len(parts) >= 11:
                            processes.append({
                                'pid': parts[1],
                                'name': parts[10],
                                'command': ' '.join(parts[10:])
                            })
        except Exception as e:
            print(f"查找进程失败: {e}")
            
        return processes
    
    def print_configuration_guide(self):
        """打印配置指南"""
        print("\n" + "="*60)
        print("🔧 IB Gateway API 配置指南")
        print("="*60)
        
        print("\n📋 步骤1: 启动IB Gateway")
        print("   • 确保IB Gateway已经启动并登录")
        print("   • 登录后不要关闭Gateway窗口")
        
        print("\n📋 步骤2: 配置API设置")
        print("   • 在IB Gateway菜单中选择 'Configure' -> 'Settings'")
        print("   • 或者点击齿轮图标 ⚙️")
        
        print("\n📋 步骤3: API配置")
        print("   • 找到 'API' 选项卡")
        print("   • ✅ 勾选 'Enable ActiveX and Socket Clients'")
        print("   • 设置端口号:")
        print("     - 模拟交易: 4001")
        print("     - 真实交易: 4000")
        print("   • Socket端口: 4001 (模拟) 或 4000 (真实)")
        
        print("\n📋 步骤4: 信任IP设置")
        print("   • 在 'Trusted IPs' 中添加: 127.0.0.1")
        print("   • 确保勾选 'Create API message log file'")
        
        print("\n📋 步骤5: 应用设置")
        print("   • 点击 'OK' 保存设置")
        print("   • 重启IB Gateway使设置生效")
        
        print("\n⚠️  重要提醒:")
        print("   • 首次使用建议选择模拟交易模式")
        print("   • 确保防火墙允许端口访问")
        print("   • 保持Gateway窗口打开状态")
        
    def test_api_connection(self, port: int = 4001) -> bool:
        """测试API连接"""
        print(f"\n🔌 测试端口 {port} 连接...")
        
        try:
            # 尝试建立socket连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex(("127.0.0.1", port))
            
            if result == 0:
                print(f"✅ 端口 {port} 连接成功!")
                
                # 尝试发送简单的API握手
                try:
                    # IB API握手消息
                    handshake = b"API\0\0\0\x00v100..20220429 16:33:41 EST\0"
                    sock.send(handshake)
                    time.sleep(1)
                    
                    # 接收响应
                    response = sock.recv(1024)
                    if response:
                        print(f"✅ API响应接收成功: {len(response)} 字节")
                        return True
                    else:
                        print("⚠️  未收到API响应")
                        
                except Exception as e:
                    print(f"⚠️  API握手失败: {e}")
                    
            else:
                print(f"❌ 端口 {port} 连接失败 (错误码: {result})")
                
            sock.close()
            return False
            
        except Exception as e:
            print(f"❌ 连接测试失败: {e}")
            return False
    
    def run_diagnosis(self):
        """运行完整诊断"""
        print("🔍 IB Gateway API 连接诊断")
        print("="*50)
        
        # 1. 检查进程
        print("\n1️⃣ 检查IB进程状态:")
        processes = self.find_ib_processes()
        if processes:
            for proc in processes:
                print(f"   ✅ 进程: {proc['name']} (PID: {proc['pid']})")
        else:
            print("   ❌ 未发现IB相关进程")
            print("   💡 请先启动IB Gateway")
            return
        
        # 2. 检查端口状态
        print("\n2️⃣ 检查API端口状态:")
        available_ports = []
        for port, desc in self.ib_ports.items():
            status = self.check_port_status(port=port)
            if status["status"] == "open":
                print(f"   ✅ {desc} (端口 {port}): 可用")
                available_ports.append(port)
            else:
                print(f"   ❌ {desc} (端口 {port}): {status['message']}")
        
        # 3. 测试连接
        if available_ports:
            print(f"\n3️⃣ 测试API连接:")
            for port in available_ports:
                if self.test_api_connection(port):
                    print(f"   🎉 端口 {port} API连接成功!")
                    return True
        else:
            print("\n❌ 没有可用的API端口")
            self.print_configuration_guide()
            return False
        
        return False

def main():
    """主函数"""
    helper = IBAPIConfigHelper()
    
    print("🚀 IB Gateway API 配置助手")
    print("="*50)
    
    # 运行诊断
    success = helper.run_diagnosis()
    
    if not success:
        print("\n" + "="*50)
        print("❌ API连接未成功建立")
        print("📖 请按照上述配置指南设置IB Gateway")
        print("🔄 配置完成后重新运行此脚本")
        sys.exit(1)
    else:
        print("\n" + "="*50)
        print("🎉 IB Gateway API 配置成功!")
        print("✅ 现在可以使用真实数据了")

if __name__ == "__main__":
    main()