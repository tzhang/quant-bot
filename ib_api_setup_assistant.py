#!/usr/bin/env python3
"""
IB Gateway API 设置助手
提供逐步指导来配置IB Gateway的API设置
"""

import socket
import time
import sys
from datetime import datetime

class IBAPISetupAssistant:
    """IB API设置助手"""
    
    def __init__(self):
        self.test_ports = [4001, 4000, 7497, 7496]
        self.port_descriptions = {
            4001: "IB Gateway 模拟交易",
            4000: "IB Gateway 真实交易", 
            7497: "TWS 模拟交易",
            7496: "TWS 真实交易"
        }
        
    def print_step(self, step_num, title, content):
        """打印配置步骤"""
        print(f"\n{'='*60}")
        print(f"📋 步骤 {step_num}: {title}")
        print(f"{'='*60}")
        print(content)
        
    def test_port(self, port, timeout=2):
        """测试端口连接"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            return result == 0
        except:
            return False
            
    def wait_for_user_input(self, message="按回车键继续..."):
        """等待用户输入"""
        input(f"\n💡 {message}")
        
    def test_all_ports(self):
        """测试所有端口"""
        print("\n🔍 正在测试端口连接...")
        available_ports = []
        
        for port in self.test_ports:
            is_available = self.test_port(port)
            status = "✅" if is_available else "❌"
            description = self.port_descriptions.get(port, f"端口 {port}")
            print(f"{status} {description} (端口 {port})")
            
            if is_available:
                available_ports.append(port)
                
        return available_ports
        
    def run_setup_guide(self):
        """运行设置指南"""
        print("🚀 IB Gateway API 设置助手")
        print(f"⏰ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n本助手将指导您完成IB Gateway API的配置")
        
        # 初始端口测试
        available_ports = self.test_all_ports()
        
        if available_ports:
            print(f"\n🎉 太好了！发现可用端口: {available_ports}")
            print("✅ 您的IB Gateway API已经配置成功！")
            print("\n💡 您现在可以在监控面板中连接到IB Gateway")
            return True
            
        print("\n❌ 未发现可用端口，需要配置API设置")
        
        # 步骤1: 确认IB Gateway已启动
        step1_content = """请确保：
✓ IB Gateway应用程序已经启动
✓ 您已经使用IB账户成功登录
✓ 看到Gateway的主界面

如果还没有启动，请：
1. 双击IB Gateway应用程序
2. 输入您的用户名和密码
3. 选择交易模式（建议先选择"模拟交易"进行测试）
4. 点击登录"""
        
        self.print_step(1, "确认IB Gateway已启动", step1_content)
        self.wait_for_user_input("确认IB Gateway已启动并登录后，按回车键继续...")
        
        # 步骤2: 打开API设置
        step2_content = """在IB Gateway中：
1. 点击顶部菜单栏的 "Configure" (配置)
2. 选择 "Settings" (设置)
3. 在弹出的设置窗口中，点击左侧的 "API" 标签页

📝 注意：如果您使用的是中文版本，菜单可能显示为"配置" -> "设置" -> "API" """
        
        self.print_step(2, "打开API设置", step2_content)
        self.wait_for_user_input("打开API设置页面后，按回车键继续...")
        
        # 步骤3: 启用API
        step3_content = """在API设置页面中：
1. ✅ 勾选 "Enable ActiveX and Socket Clients" 
   (启用ActiveX和Socket客户端)
   
2. 📝 确认端口设置：
   - Socket port: 4001 (模拟交易) 或 4000 (真实交易)
   - 如果您当前是模拟交易模式，使用4001
   - 如果您当前是真实交易模式，使用4000
   
3. ✅ 勾选 "Read-Only API" (只读API) - 推荐用于数据监控"""
        
        self.print_step(3, "启用API功能", step3_content)
        self.wait_for_user_input("完成API启用设置后，按回车键继续...")
        
        # 步骤4: 配置信任IP
        step4_content = """在同一个API设置页面中：
1. 找到 "Trusted IPs" (信任IP) 部分
2. 点击 "+" 按钮添加新的IP地址
3. 输入: 127.0.0.1
4. 点击 "OK" 确认

📝 127.0.0.1 是本地回环地址，允许本机程序连接到API"""
        
        self.print_step(4, "配置信任IP地址", step4_content)
        self.wait_for_user_input("完成信任IP配置后，按回车键继续...")
        
        # 步骤5: 应用设置
        step5_content = """保存并应用设置：
1. 点击设置窗口底部的 "Apply" (应用) 按钮
2. 点击 "OK" 关闭设置窗口
3. 重启IB Gateway (完全关闭后重新启动)
4. 重新登录您的账户

⚠️  重要：必须重启Gateway才能使API设置生效！"""
        
        self.print_step(5, "应用设置", step5_content)
        self.wait_for_user_input("完成设置应用和Gateway重启后，按回车键进行测试...")
        
        # 步骤6: 测试连接
        self.print_step(6, "测试API连接", "正在测试API连接...")
        
        # 等待一下让用户有时间重启
        print("⏳ 等待5秒让Gateway完全启动...")
        time.sleep(5)
        
        # 重新测试端口
        available_ports = self.test_all_ports()
        
        if available_ports:
            print(f"\n🎉 恭喜！API配置成功！")
            print(f"✅ 可用端口: {available_ports}")
            print("\n🚀 下一步：")
            print("1. 打开监控面板 (http://localhost:8502)")
            print("2. 在面板中选择 'IB Gateway API' 数据源")
            print("3. 点击 '连接IB' 按钮")
            print("4. 开始监控真实市场数据！")
            return True
        else:
            print(f"\n❌ 仍然无法连接到API端口")
            print("\n🔧 请检查：")
            print("1. IB Gateway是否已完全重启")
            print("2. API设置是否正确保存")
            print("3. 防火墙是否阻止了连接")
            print("4. 是否选择了正确的交易模式")
            
            print(f"\n📞 如果问题持续存在，建议：")
            print("1. 完全卸载并重新安装IB Gateway")
            print("2. 联系IB技术支持")
            print("3. 查看IB官方API文档")
            return False

def main():
    """主函数"""
    assistant = IBAPISetupAssistant()
    success = assistant.run_setup_guide()
    
    if success:
        print(f"\n{'='*60}")
        print("🎊 设置完成！您现在可以使用真实数据了！")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ 设置未完成，请按照建议进行故障排除")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()