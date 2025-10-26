#!/usr/bin/env python3
"""
IB Gateway 连接测试脚本
测试IB Gateway API连接和数据获取功能
"""

import socket
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import TickerId

from ib_gateway_config import get_config, IBGatewayConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBGatewayTester(EWrapper, EClient):
    """IB Gateway 连接测试器"""
    
    def __init__(self, config: IBGatewayConfig):
        EClient.__init__(self, self)
        self.config = config
        self.connected = False
        self.next_order_id = None
        self.account_info = {}
        self.market_data = {}
        self.connection_time = None
        
    def test_socket_connection(self, host: str, port: int) -> bool:
        """测试Socket连接"""
        try:
            logger.info(f"🔍 测试Socket连接到 {host}:{port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"✅ Socket连接到 {host}:{port} 成功")
                return True
            else:
                logger.error(f"❌ Socket连接到 {host}:{port} 失败 (错误码: {result})")
                return False
                
        except Exception as e:
            logger.error(f"❌ Socket连接测试异常: {e}")
            return False
    
    def connect_to_gateway(self) -> bool:
        """连接到IB Gateway"""
        try:
            logger.info(f"🚀 正在连接到IB Gateway: {self.config.get_connection_string()}")
            
            # 先测试Socket连接
            if not self.test_socket_connection(self.config.host, self.config.port):
                return False
            
            # 建立API连接
            self.connect(self.config.host, self.config.port, self.config.client_id)
            
            # 启动消息循环线程
            api_thread = threading.Thread(target=self.run, daemon=True)
            api_thread.start()
            
            # 等待连接建立
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < self.config.timeout:
                time.sleep(0.1)
            
            if self.connected:
                self.connection_time = datetime.now()
                logger.info(f"✅ IB Gateway API连接成功")
                return True
            else:
                logger.error(f"❌ IB Gateway API连接超时 ({self.config.timeout}秒)")
                return False
                
        except Exception as e:
            logger.error(f"❌ IB Gateway连接失败: {e}")
            return False
    
    def disconnect_from_gateway(self):
        """断开Gateway连接"""
        if self.isConnected():
            self.disconnect()
            self.connected = False
            logger.info("🔌 IB Gateway连接已断开")
    
    # ========== IB API 回调函数 ==========
    
    def connectAck(self):
        """连接确认"""
        logger.info("📡 IB Gateway连接确认收到")
    
    def nextValidId(self, orderId: int):
        """接收下一个有效订单ID"""
        self.connected = True
        self.next_order_id = orderId
        logger.info(f"✅ IB Gateway连接成功，下一个订单ID: {orderId}")
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """错误处理"""
        # 过滤信息性消息
        if errorCode in [2104, 2106, 2158, 2119]:  # 连接状态信息
            logger.debug(f"IB 信息 [{errorCode}]: {errorString}")
        elif errorCode == 10089:  # 市场数据订阅
            logger.warning(f"市场数据需要订阅 [{errorCode}]: {errorString}")
        elif errorCode == 502:  # 无法连接到TWS
            logger.error(f"无法连接到IB Gateway [{errorCode}]: {errorString}")
        else:
            logger.error(f"IB 错误 [{errorCode}]: {errorString}")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """账户摘要信息"""
        self.account_info[tag] = {
            'value': value,
            'currency': currency,
            'account': account
        }
        logger.info(f"📊 账户信息 - {tag}: {value} {currency}")
    
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """市场数据价格"""
        tick_types = {1: "bid", 2: "ask", 4: "last", 6: "high", 7: "low", 9: "close"}
        tick_name = tick_types.get(tickType, f"tick_{tickType}")
        
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        
        self.market_data[reqId][tick_name] = price
        logger.info(f"📈 市场数据 - 请求ID {reqId}, {tick_name}: ${price}")

def test_gateway_ports():
    """测试常见的IB Gateway端口"""
    print("🔍 测试IB Gateway端口连接")
    print("=" * 50)
    
    # IB Gateway 端口配置
    gateway_ports = [
        (4001, "IB Gateway 模拟交易"),
        (4000, "IB Gateway 真实交易"),
        (7497, "TWS 模拟交易"),
        (7496, "TWS 真实交易"),
    ]
    
    available_ports = []
    
    for port, description in gateway_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            
            if result == 0:
                print(f"✅ 端口 {port} 可用 - {description}")
                available_ports.append((port, description))
            else:
                print(f"❌ 端口 {port} 不可用 - {description}")
                
        except Exception as e:
            print(f"❌ 端口 {port} 测试异常 - {e}")
    
    return available_ports

def test_api_connection(config: IBGatewayConfig) -> bool:
    """测试API连接"""
    print(f"\n🚀 测试IB Gateway API连接")
    print("=" * 50)
    
    tester = IBGatewayTester(config)
    
    # 连接测试
    if not tester.connect_to_gateway():
        return False
    
    try:
        # 请求账户信息
        logger.info("📋 请求账户摘要信息...")
        tester.reqAccountSummary(9001, "All", "$LEDGER")
        
        # 订阅市场数据测试
        logger.info("📊 测试市场数据订阅...")
        contract = Contract()
        contract.symbol = "AAPL"
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        tester.reqMktData(1001, contract, "", False, False, [])
        
        # 等待数据
        time.sleep(5)
        
        # 显示结果
        print(f"\n📊 测试结果:")
        print(f"连接时间: {tester.connection_time}")
        print(f"下一个订单ID: {tester.next_order_id}")
        print(f"账户信息条目: {len(tester.account_info)}")
        print(f"市场数据条目: {len(tester.market_data)}")
        
        if tester.account_info:
            print(f"\n💰 账户信息:")
            for tag, info in tester.account_info.items():
                print(f"  {tag}: {info['value']} {info['currency']}")
        
        if tester.market_data:
            print(f"\n📈 市场数据:")
            for req_id, data in tester.market_data.items():
                print(f"  请求ID {req_id}: {data}")
        
        return True
        
    except Exception as e:
        logger.error(f"API测试异常: {e}")
        return False
    
    finally:
        tester.disconnect_from_gateway()

def main():
    """主函数"""
    print("🚀 IB Gateway 连接测试工具")
    print("=" * 60)
    
    # 1. 测试端口连接
    available_ports = test_gateway_ports()
    
    if not available_ports:
        print("\n❌ 未发现可用的IB Gateway端口")
        print("\n🔧 请检查:")
        print("   1. IB Gateway 是否已启动并登录")
        print("   2. Gateway API设置是否已启用")
        print("   3. 防火墙是否阻止了连接")
        print("   4. 端口是否被其他程序占用")
        return False
    
    # 2. 加载配置
    config = get_config()
    
    # 3. 如果配置的端口不可用，使用第一个可用端口
    config_port_available = any(port == config.port for port, _ in available_ports)
    if not config_port_available and available_ports:
        new_port, description = available_ports[0]
        print(f"\n⚠️  配置端口 {config.port} 不可用，使用 {new_port} ({description})")
        config.port = new_port
    
    # 4. 测试API连接
    success = test_api_connection(config)
    
    # 5. 显示结果
    print("\n" + "=" * 60)
    if success:
        print("✅ IB Gateway 连接测试成功！")
        print(f"\n💡 推荐配置:")
        print(f"   IB_HOST='{config.host}'")
        print(f"   IB_PORT={config.port}")
        print(f"   IB_CLIENT_ID={config.client_id}")
        print(f"\n🎯 可以开始使用IB Gateway API进行真实数据获取")
    else:
        print("❌ IB Gateway 连接测试失败")
        print("\n🔧 请检查IB Gateway设置和网络连接")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)