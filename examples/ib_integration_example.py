#!/usr/bin/env python3
"""
Interactive Brokers 集成示例
演示如何使用 IB 适配器进行量化交易
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.config import Config
from examples.ib_adapter import IBAdapter

async def main():
    """主函数 - 演示 IB 适配器的使用"""
    print("🚀 Interactive Brokers 集成示例")
    print("=" * 50)
    
    # 加载配置
    config = Config()
    
    # 创建 IB 适配器实例
    ib_adapter = IBAdapter(config.interactive_brokers)
    
    try:
        # 1. 连接到 IB
        print("📡 连接到 Interactive Brokers...")
        await ib_adapter.connect()
        
        # 等待连接稳定
        await asyncio.sleep(2)
        
        # 2. 获取账户信息
        print("\n💰 获取账户信息...")
        account_info = await ib_adapter.get_account_info()
        if account_info:
            print(f"账户净值: ${account_info.net_liquidation:,.2f}")
            print(f"现金余额: ${account_info.total_cash_value:,.2f}")
            print(f"购买力: ${account_info.buying_power:,.2f}")
        
        # 3. 获取持仓信息
        print("\n📊 获取持仓信息...")
        positions = await ib_adapter.get_positions()
        if positions:
            print(f"当前持仓数量: {len(positions)}")
            for position in positions:
                print(f"  {position.symbol}: {position.quantity} 股 @ ${position.avg_cost:.2f}")
        else:
            print("当前无持仓")
        
        # 4. 订阅市场数据（演示用）
        print("\n📈 订阅市场数据...")
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in symbols:
            try:
                await ib_adapter.subscribe_market_data(symbol)
                print(f"✅ 已订阅 {symbol} 市场数据")
            except Exception as e:
                print(f"⚠️  订阅 {symbol} 失败: {e}")
        
        # 等待一段时间接收市场数据
        print("\n⏳ 等待市场数据...")
        await asyncio.sleep(5)
        
        # 5. 获取最新市场数据
        print("\n💹 最新市场数据:")
        for symbol in symbols:
            market_data = await ib_adapter.get_market_data(symbol)
            if market_data:
                print(f"  {symbol}: ${market_data.last_price:.2f} "
                      f"(买: ${market_data.bid:.2f}, 卖: ${market_data.ask:.2f})")
        
        # 6. 模拟下单（仅在模拟环境中）
        if config.interactive_brokers.dry_run:
            print("\n📝 模拟下单示例...")
            try:
                order_id = await ib_adapter.place_order(
                    symbol="AAPL",
                    quantity=10,
                    order_type="MKT",
                    action="BUY"
                )
                print(f"✅ 模拟订单已提交，订单ID: {order_id}")
            except Exception as e:
                print(f"❌ 下单失败: {e}")
        else:
            print("\n⚠️  当前为实盘环境，跳过下单示例")
        
        print("\n🎉 集成示例完成！")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 断开连接
        print("\n🔌 断开连接...")
        await ib_adapter.disconnect()

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())