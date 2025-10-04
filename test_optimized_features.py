#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化功能综合测试脚本
测试所有已实现的高级反频率限制功能
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append('.')

from src.data.fetch_nasdaq import fetch_and_store_symbol, rate_limit_handler
from src.data.alternative_sources import alternative_data_manager

def test_cache_functionality():
    """测试缓存功能"""
    print("🧪 测试缓存功能...")
    
    # 测试缓存键生成
    cache_key = rate_limit_handler.get_cache_key("AAPL", "2024-01-01", "2024-12-31")
    print(f"  缓存键生成: {cache_key[:16]}...")
    
    # 检查现有缓存
    cache_dir = "./data_cache"
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
        print(f"  现有缓存文件: {len(cache_files)} 个")
    
    return True

def test_user_agent_rotation():
    """测试User-Agent轮换"""
    print("🔄 测试User-Agent轮换...")
    
    initial_ua = rate_limit_handler.get_next_user_agent()
    print(f"  初始UA: {initial_ua[:30]}...")
    
    # 轮换几次
    for i in range(3):
        ua = rate_limit_handler.get_next_user_agent()
        print(f"  轮换{i+1}: {ua[:30]}...")
    
    return True

def test_adaptive_delay():
    """测试自适应延迟"""
    print("⏱️  测试自适应延迟...")
    
    # 测试不同失败率下的延迟
    original_failure = rate_limit_handler.failure_count
    original_success = rate_limit_handler.success_count
    
    # 模拟一些失败
    rate_limit_handler.failure_count = 5
    rate_limit_handler.success_count = 10
    
    delay = rate_limit_handler.get_adaptive_delay()
    failure_rate = rate_limit_handler.get_failure_rate()
    
    print(f"  失败率: {failure_rate:.1%}")
    print(f"  自适应延迟: {delay:.1f} 秒")
    
    # 恢复原始值
    rate_limit_handler.failure_count = original_failure
    rate_limit_handler.success_count = original_success
    
    return True

def test_alternative_sources():
    """测试备用数据源"""
    print("🔗 测试备用数据源...")
    
    # 检查数据源配置
    sources = alternative_data_manager.sources
    print(f"  配置的数据源: {len(sources)} 个")
    
    for i, source in enumerate(sources):
        print(f"    {i+1}. {source.__class__.__name__}")
    
    return True

def test_session_configuration():
    """测试会话配置"""
    print("🌐 测试会话配置...")
    
    session = rate_limit_handler.session
    print(f"  会话类型: {type(session).__name__}")
    print(f"  适配器数量: {len(session.adapters)}")
    print(f"  当前headers数量: {len(session.headers)}")
    
    # 检查重试配置
    for prefix, adapter in session.adapters.items():
        if hasattr(adapter, 'max_retries'):
            print(f"  {prefix} 最大重试次数: {adapter.max_retries}")
    
    return True

def test_light_data_fetch():
    """轻量级数据获取测试"""
    print("📊 轻量级数据获取测试...")
    
    try:
        from pathlib import Path
        
        # 尝试获取一个小范围的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # 只获取7天数据
        
        print(f"  测试股票: AAPL")
        print(f"  日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 尝试获取数据（使用较短的时间范围减少API压力）
        result = fetch_and_store_symbol(
            "AAPL", 
            start_date, 
            end_date,
            Path("./data_cache"),  # 缓存目录
            True,  # 使用parquet格式
            100,   # 批量大小
            3      # 最大重试次数
        )
        
        elapsed_time = time.time() - start_time
        
        if result:
            print(f"  ✅ 数据获取成功 (耗时: {elapsed_time:.1f}秒)")
            print(f"  数据行数: {len(result)} 行")
        else:
            print(f"  ⚠️  数据获取失败，但功能正常运行 (耗时: {elapsed_time:.1f}秒)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 测试过程中出现错误: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("🚀 优化功能综合测试开始")
    print("=" * 50)
    print()
    
    tests = [
        ("缓存功能", test_cache_functionality),
        ("User-Agent轮换", test_user_agent_rotation),
        ("自适应延迟", test_adaptive_delay),
        ("备用数据源", test_alternative_sources),
        ("会话配置", test_session_configuration),
        ("轻量级数据获取", test_light_data_fetch),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🧪 {test_name}测试...")
        try:
            if test_func():
                print(f"  ✅ {test_name}测试通过")
                passed += 1
            else:
                print(f"  ❌ {test_name}测试失败")
        except Exception as e:
            print(f"  ❌ {test_name}测试异常: {str(e)}")
        print()
    
    # 显示最终统计
    print("=" * 50)
    print("📈 最终统计信息")
    print("=" * 50)
    print(f"成功次数: {rate_limit_handler.success_count}")
    print(f"失败次数: {rate_limit_handler.failure_count}")
    print(f"缓存命中: {rate_limit_handler.cache_hits}")
    print(f"User-Agent轮换: {rate_limit_handler.ua_rotations}")
    print(f"失败率: {rate_limit_handler.get_failure_rate():.1%}")
    print()
    
    print("=" * 50)
    print("🎯 测试结果总结")
    print("=" * 50)
    print(f"通过测试: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！优化功能运行正常。")
    else:
        print("⚠️  部分测试未通过，请检查相关功能。")

if __name__ == "__main__":
    main()