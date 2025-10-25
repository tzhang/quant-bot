#!/usr/bin/env python3
"""
快速TWS API连接测试
"""
import socket
import time

def test_api_connection():
    """测试TWS API连接"""
    print("🔍 快速TWS API连接测试")
    print("=" * 40)
    
    ports = [7497, 7496]  # 实时账户和模拟账户端口
    
    for port in ports:
        print(f"\n📡 测试端口 {port}...")
        
        try:
            # 创建socket连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)  # 3秒超时
            
            result = sock.connect_ex(('127.0.0.1', port))
            
            if result == 0:
                print(f"   ✅ 端口 {port} 连接成功！")
                
                # 尝试发送简单的API握手
                try:
                    # IB API握手消息
                    handshake = b"API\x00\x00\x00\x09\x00\x00\x00\x01"
                    sock.send(handshake)
                    
                    # 等待响应
                    sock.settimeout(2)
                    response = sock.recv(1024)
                    
                    if response:
                        print(f"   ✅ API握手成功，收到响应: {len(response)} 字节")
                        return True
                    else:
                        print(f"   ⚠️  端口开放但API握手失败")
                        
                except Exception as e:
                    print(f"   ⚠️  API握手失败: {e}")
                    
            else:
                print(f"   ❌ 端口 {port} 连接失败")
                
        except Exception as e:
            print(f"   ❌ 连接错误: {e}")
            
        finally:
            sock.close()
    
    print(f"\n❌ 所有端口连接失败")
    print("\n🔧 请按照以下步骤配置TWS API:")
    print("   1. 在TWS中: File → Global Configuration → API → Settings")
    print("   2. 勾选 'Enable ActiveX and Socket Clients'")
    print("   3. 设置Socket Port为7497或7496")
    print("   4. 添加127.0.0.1到Trusted IPs")
    print("   5. 点击Apply并重启TWS")
    
    return False

if __name__ == "__main__":
    success = test_api_connection()
    
    if success:
        print("\n🎉 TWS API配置成功！可以开始使用量化交易系统了。")
    else:
        print("\n⚠️  请完成API配置后重新运行此脚本验证。")