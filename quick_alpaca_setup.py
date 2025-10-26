#!/usr/bin/env python3
"""
Alpaca 快速设置脚本
帮助用户快速配置和测试Alpaca连接
"""

import os
import sys
import yaml
from pathlib import Path

def setup_alpaca():
    """设置Alpaca配置"""
    print("🚀 Alpaca 快速设置向导")
    print("=" * 50)
    
    # 获取API密钥
    print("\n📋 请提供您的Alpaca API信息:")
    print("(可以从 https://app.alpaca.markets/paper/dashboard/overview 获取)")
    
    api_key = input("API Key ID: ").strip()
    secret_key = input("Secret Key: ").strip()
    
    # 选择环境
    print("\n🌍 选择交易环境:")
    print("1. Paper Trading (模拟交易) - 推荐")
    print("2. Live Trading (实盘交易)")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "2":
        base_url = "https://api.alpaca.markets"
        print("⚠️  您选择了实盘交易环境，请确保您已准备好真实资金!")
    else:
        base_url = "https://paper-api.alpaca.markets"
        print("✅ 您选择了模拟交易环境，这是安全的测试环境")
    
    # 更新配置文件
    config_path = Path("trading_config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 更新Alpaca配置
        if 'data_sources' not in config:
            config['data_sources'] = {}
        if 'api_keys' not in config['data_sources']:
            config['data_sources']['api_keys'] = {}
        
        config['data_sources']['api_keys']['alpaca'] = {
            'api_key': api_key,
            'secret_key': secret_key,
            'base_url': base_url
        }
        config['data_sources']['primary'] = 'alpaca'
        
        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n✅ 配置已保存到 {config_path}")
    else:
        print(f"\n⚠️  未找到配置文件 {config_path}")
    
    # 设置环境变量
    print("\n🔧 设置环境变量...")
    os.environ['ALPACA_API_KEY'] = api_key
    os.environ['ALPACA_SECRET_KEY'] = secret_key
    os.environ['ALPACA_BASE_URL'] = base_url
    
    # 测试连接
    print("\n🔍 测试Alpaca连接...")
    try:
        # 尝试导入并测试
        import requests
        
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key
        }
        
        response = requests.get(f"{base_url}/v2/account", headers=headers)
        
        if response.status_code == 200:
            account_data = response.json()
            print("✅ Alpaca连接成功!")
            print(f"📊 账户信息:")
            print(f"   - 账户ID: {account_data.get('id', 'N/A')}")
            print(f"   - 现金余额: ${account_data.get('cash', 0)}")
            print(f"   - 购买力: ${account_data.get('buying_power', 0)}")
            print(f"   - 账户状态: {account_data.get('status', 'N/A')}")
            
            return True
        else:
            print(f"❌ 连接失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except ImportError:
        print("⚠️  需要安装 requests 库: pip install requests")
        return False
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False

def install_dependencies():
    """安装必要的依赖"""
    print("\n📦 安装Alpaca依赖...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "alpaca-trade-api", "requests"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 依赖安装成功")
            return True
        else:
            print(f"❌ 依赖安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 安装过程出错: {e}")
        return False

def main():
    """主函数"""
    print("欢迎使用Alpaca快速设置工具!")
    print("这将帮助您配置Alpaca作为IB的替代方案\n")
    
    # 检查并安装依赖
    try:
        import alpaca_trade_api
        print("✅ alpaca-trade-api 已安装")
    except ImportError:
        print("📦 需要安装 alpaca-trade-api...")
        if not install_dependencies():
            print("❌ 无法安装依赖，请手动运行: pip install alpaca-trade-api requests")
            return
    
    # 运行设置
    if setup_alpaca():
        print("\n🎉 设置完成!")
        print("\n📚 接下来的步骤:")
        print("1. 查看 setup_alpaca_guide.md 获取详细指南")
        print("2. 运行 python examples/alpaca_adapter.py 测试适配器")
        print("3. 开始使用您的量化交易系统!")
        
        print("\n💡 提示:")
        print("- 建议先在模拟环境中测试您的策略")
        print("- 确保遵守相关的交易规则和风险管理")
        print("- 定期检查API使用限制")
    else:
        print("\n❌ 设置失败，请检查您的API密钥和网络连接")
        print("💡 获取帮助:")
        print("1. 确认API密钥正确")
        print("2. 检查网络连接")
        print("3. 查看 setup_alpaca_guide.md 获取详细指南")

if __name__ == "__main__":
    main()