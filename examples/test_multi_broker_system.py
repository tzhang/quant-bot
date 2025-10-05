"""
多券商系统测试
测试修复后的实盘交易接口，包括Firstrade、Interactive Brokers和Alpaca
"""

import logging
import time
from typing import Dict, Any, List
from broker_factory import BrokerFactory, MultiBrokerManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MultiBrokerTester:
    """多券商系统测试器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.manager = MultiBrokerManager()
        self.test_results = {}
        
    def setup_brokers(self):
        """设置券商连接"""
        print("🔧 设置券商连接...")
        
        try:
            # 1. Firstrade (使用模拟参数)
            print("  添加 Firstrade...")
            firstrade = BrokerFactory.create_broker(
                'firstrade',
                username='demo_user',
                password='demo_pass',
                pin='1234'
            )
            self.manager.add_broker('firstrade', firstrade)
            
            # 2. Interactive Brokers
            print("  添加 Interactive Brokers...")
            ib = BrokerFactory.create_broker('ib')
            self.manager.add_broker('ib', ib)
            
            # 3. Alpaca
            print("  添加 Alpaca...")
            alpaca = BrokerFactory.create_broker('alpaca')
            self.manager.add_broker('alpaca', alpaca)
            
            print(f"✅ 已添加券商: {list(self.manager.brokers.keys())}")
            return True
            
        except Exception as e:
            print(f"❌ 设置券商失败: {str(e)}")
            return False
            
    def test_connections(self):
        """测试连接"""
        print("\n🔗 测试券商连接...")
        
        connection_results = self.manager.connect_all()
        self.test_results['connections'] = connection_results
        
        for broker_name, connected in connection_results.items():
            status = "✅ 成功" if connected else "❌ 失败"
            print(f"  {broker_name}: {status}")
            
        return connection_results
        
    def test_account_info(self):
        """测试账户信息获取"""
        print("\n💰 测试账户信息获取...")
        
        all_accounts = self.manager.get_all_account_info()
        self.test_results['account_info'] = all_accounts
        
        for broker_name, account_info in all_accounts.items():
            print(f"\n  {broker_name} 账户信息:")
            if 'error' in account_info:
                print(f"    ❌ 错误: {account_info['error']}")
            else:
                for key, value in account_info.items():
                    print(f"    {key}: {value}")
                    
        return all_accounts
        
    def test_positions(self):
        """测试持仓信息获取"""
        print("\n📊 测试持仓信息获取...")
        
        all_positions = self.manager.get_all_positions()
        self.test_results['positions'] = all_positions
        
        for broker_name, positions in all_positions.items():
            print(f"\n  {broker_name} 持仓信息:")
            if not positions:
                print("    无持仓")
            else:
                for pos in positions:
                    print(f"    {pos.get('symbol', 'N/A')}: {pos.get('quantity', 0)} 股")
                    
        return all_positions
        
    def test_market_data(self):
        """测试市场数据获取"""
        print("\n📈 测试市场数据获取...")
        
        test_symbols = ['AAPL', 'TSLA', 'MSFT']
        market_data_results = {}
        
        for broker_name, broker in self.manager.brokers.items():
            market_data_results[broker_name] = {}
            print(f"\n  {broker_name} 市场数据:")
            
            for symbol in test_symbols:
                try:
                    data = broker.get_market_data(symbol)
                    market_data_results[broker_name][symbol] = data
                    
                    if 'error' in data:
                        print(f"    {symbol}: ❌ {data['error']}")
                    else:
                        price_info = f"价格: {data.get('price', data.get('bid', 'N/A'))}"
                        print(f"    {symbol}: ✅ {price_info}")
                        
                except Exception as e:
                    print(f"    {symbol}: ❌ 异常: {str(e)}")
                    market_data_results[broker_name][symbol] = {"error": str(e)}
                    
        self.test_results['market_data'] = market_data_results
        return market_data_results
        
    def test_order_placement(self):
        """测试下单功能（模拟）"""
        print("\n📝 测试下单功能（模拟）...")
        
        test_orders = [
            {"symbol": "AAPL", "action": "buy", "quantity": 1, "order_type": "market"},
            {"symbol": "TSLA", "action": "buy", "quantity": 2, "order_type": "limit", "price": 200.0}
        ]
        
        order_results = {}
        
        for broker_name, broker in self.manager.brokers.items():
            order_results[broker_name] = []
            print(f"\n  {broker_name} 下单测试:")
            
            for order in test_orders:
                try:
                    result = broker.place_order(**order)
                    order_results[broker_name].append(result)
                    
                    if result.get('success', False):
                        print(f"    ✅ {order['action']} {order['quantity']} {order['symbol']}: 订单ID {result.get('order_id', 'N/A')}")
                    else:
                        print(f"    ❌ {order['action']} {order['quantity']} {order['symbol']}: {result.get('error', '未知错误')}")
                        
                except Exception as e:
                    print(f"    ❌ {order['action']} {order['quantity']} {order['symbol']}: 异常 {str(e)}")
                    order_results[broker_name].append({"success": False, "error": str(e)})
                    
        self.test_results['orders'] = order_results
        return order_results
        
    def test_error_handling(self):
        """测试错误处理"""
        print("\n🚨 测试错误处理...")
        
        error_test_results = {}
        
        for broker_name, broker in self.manager.brokers.items():
            error_test_results[broker_name] = {}
            print(f"\n  {broker_name} 错误处理测试:")
            
            # 测试无效股票代码
            try:
                invalid_data = broker.get_market_data("INVALID_SYMBOL_12345")
                error_test_results[broker_name]['invalid_symbol'] = invalid_data
                print(f"    无效股票代码: {'✅ 正确处理' if 'error' in invalid_data else '❌ 未正确处理'}")
            except Exception as e:
                error_test_results[broker_name]['invalid_symbol'] = {"error": str(e)}
                print(f"    无效股票代码: ✅ 异常处理 - {str(e)}")
                
            # 测试无效订单
            try:
                invalid_order = broker.place_order("AAPL", "invalid_action", -1)
                error_test_results[broker_name]['invalid_order'] = invalid_order
                print(f"    无效订单: {'✅ 正确拒绝' if not invalid_order.get('success', True) else '❌ 未正确拒绝'}")
            except Exception as e:
                error_test_results[broker_name]['invalid_order'] = {"error": str(e)}
                print(f"    无效订单: ✅ 异常处理 - {str(e)}")
                
        self.test_results['error_handling'] = error_test_results
        return error_test_results
        
    def generate_test_report(self):
        """生成测试报告"""
        print("\n📋 生成测试报告...")
        
        report = {
            "测试时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "测试券商": list(self.manager.brokers.keys()),
            "测试结果": self.test_results
        }
        
        # 计算成功率
        connection_success = sum(1 for connected in self.test_results.get('connections', {}).values() if connected)
        total_brokers = len(self.manager.brokers)
        success_rate = (connection_success / total_brokers * 100) if total_brokers > 0 else 0
        
        print(f"\n📊 测试总结:")
        print(f"  测试券商数量: {total_brokers}")
        print(f"  连接成功数量: {connection_success}")
        print(f"  连接成功率: {success_rate:.1f}%")
        
        # 保存报告到文件
        try:
            import json
            with open('multi_broker_test_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"  ✅ 测试报告已保存到: multi_broker_test_report.json")
        except Exception as e:
            print(f"  ❌ 保存报告失败: {str(e)}")
            
        return report
        
    def cleanup(self):
        """清理资源"""
        print("\n🧹 清理资源...")
        self.manager.disconnect_all()
        print("✅ 清理完成")
        
    def run_full_test(self):
        """运行完整测试"""
        print("🚀 开始多券商系统测试")
        print("=" * 50)
        
        try:
            # 1. 设置券商
            if not self.setup_brokers():
                return False
                
            # 2. 测试连接
            self.test_connections()
            
            # 3. 测试账户信息
            self.test_account_info()
            
            # 4. 测试持仓信息
            self.test_positions()
            
            # 5. 测试市场数据
            self.test_market_data()
            
            # 6. 测试下单功能
            self.test_order_placement()
            
            # 7. 测试错误处理
            self.test_error_handling()
            
            # 8. 生成报告
            self.generate_test_report()
            
            print("\n🎉 测试完成！")
            return True
            
        except Exception as e:
            print(f"\n💥 测试过程中发生异常: {str(e)}")
            return False
            
        finally:
            # 9. 清理资源
            self.cleanup()

def main():
    """主函数"""
    tester = MultiBrokerTester()
    success = tester.run_full_test()
    
    if success:
        print("\n✅ 多券商系统测试成功完成")
    else:
        print("\n❌ 多券商系统测试失败")
        
    return success

if __name__ == "__main__":
    main()