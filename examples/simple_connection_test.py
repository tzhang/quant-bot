#!/usr/bin/env python3
"""
简化的 Firstrade 连接测试脚本
用于诊断网络连接和 API 问题
"""

import os
import sys
import requests
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_network_connectivity():
    """测试基本网络连接"""
    print("🌐 测试网络连接...")
    
    test_urls = [
        "https://www.google.com",
        "https://api3x.firstrade.com",
        "https://investor.firstrade.com"
    ]
    
    for url in test_urls:
        try:
            print(f"   测试连接到 {url}...")
            response = requests.get(url, timeout=10)
            print(f"   ✅ {url} - 状态码: {response.status_code}")
        except requests.exceptions.ConnectTimeout:
            print(f"   ❌ {url} - 连接超时")
        except requests.exceptions.ConnectionError as e:
            print(f"   ❌ {url} - 连接错误: {e}")
        except Exception as e:
            print(f"   ⚠️ {url} - 其他错误: {e}")
        
        time.sleep(1)  # 避免请求过于频繁

def test_firstrade_import():
    """测试 Firstrade API 导入"""
    print("\n📦 测试 Firstrade API 导入...")
    
    try:
        from firstrade.account import FTSession, FTAccountData
        from firstrade import symbols
        print("   ✅ Firstrade API 导入成功")
        return True
    except ImportError as e:
        print(f"   ❌ Firstrade API 导入失败: {e}")
        return False

def test_simple_session_creation():
    """测试简单的会话创建（不进行实际登录）"""
    print("\n🔧 测试会话创建...")
    
    try:
        from firstrade.account import FTSession
        
        # 使用虚拟凭据测试会话创建
        session = FTSession(
            username="test_user",
            password="test_pass",
            pin="1234"
        )
        print("   ✅ FTSession 对象创建成功")
        print(f"   📋 Session 对象类型: {type(session)}")
        print(f"   📋 Session 对象属性: {dir(session)}")
        return True
        
    except Exception as e:
        print(f"   ❌ 会话创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 Firstrade 连接诊断工具")
    print("=" * 60)
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试网络连接
    test_network_connectivity()
    
    # 测试 API 导入
    if not test_firstrade_import():
        print("\n❌ API 导入失败，请先安装 firstrade 包:")
        print("   pip install firstrade")
        return
    
    # 测试会话创建
    test_simple_session_creation()
    
    print("\n" + "=" * 60)
    print("📋 诊断完成")
    print("=" * 60)
    
    print("\n💡 如果网络连接正常但 Firstrade API 仍然超时，可能的原因:")
    print("   1. Firstrade 服务器暂时不可用")
    print("   2. 需要 VPN 或代理连接")
    print("   3. 防火墙阻止了连接")
    print("   4. API 版本不兼容")

if __name__ == "__main__":
    main()