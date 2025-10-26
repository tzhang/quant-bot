#!/usr/bin/env python3
"""
综合数据源测试脚本
测试修复后的数据获取功能和系统稳定性
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.data.data_adapter import DataAdapter
    from config.settings import DataSourceConfig
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_source_priority():
    """测试数据源优先级配置"""
    print("🔧 测试数据源优先级配置")
    print("=" * 60)
    
    config = DataSourceConfig()
    
    print(f"✅ 主要数据源: {config.primary_sources}")
    print(f"✅ 备用数据源: {config.fallback_sources}")
    print(f"✅ IB Gateway配置: {config.ib_host}:{config.ib_port}")
    print(f"✅ Yahoo频率限制: {config.yahoo_rate_limit}秒")
    print(f"✅ Yahoo重试次数: {config.yahoo_max_retries}")
    print()

def test_data_adapter_initialization():
    """测试DataAdapter初始化"""
    print("🚀 测试DataAdapter初始化")
    print("=" * 60)
    
    try:
        adapter = DataAdapter(
            prefer_qlib=True,
            enable_openbb=True,
            enable_ib=True,
            fallback_to_yfinance=True
        )
        print("✅ DataAdapter初始化成功")
        
        # 检查各数据源状态
        print(f"📊 Qlib可用: {adapter.qlib_provider is not None}")
        print(f"📊 OpenBB可用: {adapter.openbb_provider is not None}")
        print(f"📊 IB可用: {adapter.ib_provider is not None}")
        print(f"📊 yfinance可用: {adapter.fallback_to_yfinance}")
        print()
        
        return adapter
        
    except Exception as e:
        print(f"❌ DataAdapter初始化失败: {e}")
        return None

def test_yfinance_rate_limiting():
    """测试yfinance频率控制机制"""
    print("⏱️  测试yfinance频率控制机制")
    print("=" * 60)
    
    try:
        adapter = DataAdapter(
            prefer_qlib=False,  # 禁用其他数据源
            enable_openbb=False,
            enable_ib=False,
            fallback_to_yfinance=True
        )
        
        # 测试连续请求的频率控制
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        print(f"📈 测试连续请求频率控制 ({len(symbols)}个股票)")
        
        start_time = time.time()
        for i, symbol in enumerate(symbols):
            print(f"  请求 {i+1}/{len(symbols)}: {symbol}")
            request_start = time.time()
            
            try:
                data = adapter._get_yfinance_data(symbol, start_date, end_date)
                request_time = time.time() - request_start
                
                if not data.empty:
                    print(f"    ✅ 成功获取 {len(data)} 条记录 (耗时: {request_time:.2f}秒)")
                else:
                    print(f"    ⚠️  无数据返回 (耗时: {request_time:.2f}秒)")
                    
            except Exception as e:
                request_time = time.time() - request_start
                print(f"    ❌ 请求失败: {e} (耗时: {request_time:.2f}秒)")
        
        total_time = time.time() - start_time
        print(f"📊 总耗时: {total_time:.2f}秒")
        print(f"📊 平均每请求: {total_time/len(symbols):.2f}秒")
        print()
        
    except Exception as e:
        print(f"❌ 频率控制测试失败: {e}")
        print()

def test_data_source_fallback():
    """测试数据源回退机制"""
    print("🔄 测试数据源回退机制")
    print("=" * 60)
    
    try:
        # 创建适配器，启用所有数据源
        adapter = DataAdapter(
            prefer_qlib=True,
            enable_openbb=True,
            enable_ib=True,
            fallback_to_yfinance=True
        )
        
        symbol = 'AAPL'
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"📈 测试股票: {symbol}")
        print(f"📅 日期范围: {start_date} 到 {end_date}")
        
        start_time = time.time()
        data = adapter.get_stock_data(symbol, start_date, end_date)
        total_time = time.time() - start_time
        
        if not data.empty:
            print(f"✅ 成功获取数据: {len(data)} 条记录")
            print(f"📊 数据列: {list(data.columns)}")
            print(f"📊 日期范围: {data.index.min()} 到 {data.index.max()}")
            print(f"⏱️  耗时: {total_time:.2f}秒")
            
            # 显示前几行数据
            print("\n📋 数据样本:")
            print(data.head(3).to_string())
        else:
            print(f"❌ 未获取到数据 (耗时: {total_time:.2f}秒)")
        
        print()
        
    except Exception as e:
        print(f"❌ 回退机制测试失败: {e}")
        print()

def test_multiple_symbols():
    """测试多个股票的数据获取"""
    print("📊 测试多个股票数据获取")
    print("=" * 60)
    
    try:
        adapter = DataAdapter(
            prefer_qlib=True,
            enable_openbb=True,
            enable_ib=True,
            fallback_to_yfinance=True
        )
        
        symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        print(f"📈 测试股票: {symbols}")
        print(f"📅 日期范围: {start_date} 到 {end_date}")
        
        results = {}
        total_start_time = time.time()
        
        for symbol in symbols:
            print(f"\n  处理 {symbol}...")
            start_time = time.time()
            
            try:
                data = adapter.get_stock_data(symbol, start_date, end_date)
                elapsed_time = time.time() - start_time
                
                if not data.empty:
                    results[symbol] = {
                        'success': True,
                        'records': len(data),
                        'time': elapsed_time
                    }
                    print(f"    ✅ 成功: {len(data)} 条记录 ({elapsed_time:.2f}秒)")
                else:
                    results[symbol] = {
                        'success': False,
                        'records': 0,
                        'time': elapsed_time
                    }
                    print(f"    ⚠️  无数据 ({elapsed_time:.2f}秒)")
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                results[symbol] = {
                    'success': False,
                    'records': 0,
                    'time': elapsed_time,
                    'error': str(e)
                }
                print(f"    ❌ 失败: {e} ({elapsed_time:.2f}秒)")
        
        total_time = time.time() - total_start_time
        
        # 统计结果
        successful = sum(1 for r in results.values() if r['success'])
        total_records = sum(r['records'] for r in results.values())
        avg_time = sum(r['time'] for r in results.values()) / len(results)
        
        print(f"\n📊 测试总结:")
        print(f"  成功率: {successful}/{len(symbols)} ({successful/len(symbols)*100:.1f}%)")
        print(f"  总记录数: {total_records}")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  平均每股票: {avg_time:.2f}秒")
        print()
        
    except Exception as e:
        print(f"❌ 多股票测试失败: {e}")
        print()

def main():
    """主测试函数"""
    print("🧪 综合数据源测试")
    print("=" * 80)
    print(f"⏰ 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 测试配置
    test_data_source_priority()
    
    # 2. 测试初始化
    adapter = test_data_adapter_initialization()
    if not adapter:
        print("❌ 无法继续测试，DataAdapter初始化失败")
        return
    
    # 3. 测试频率控制（仅在yfinance可用时）
    if adapter.fallback_to_yfinance:
        test_yfinance_rate_limiting()
    
    # 4. 测试回退机制
    test_data_source_fallback()
    
    # 5. 测试多股票
    test_multiple_symbols()
    
    print("🎉 综合测试完成")
    print("=" * 80)

if __name__ == "__main__":
    main()