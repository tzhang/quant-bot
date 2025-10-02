#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发环境测试脚本

用于验证 Python 开发环境是否正确配置
"""

import sys
import importlib
from typing import List, Tuple

def test_python_version() -> bool:
    """测试 Python 版本"""
    print(f"Python 版本: {sys.version}")
    return sys.version_info >= (3, 8)

def test_imports() -> List[Tuple[str, bool, str]]:
    """测试核心库导入"""
    libraries = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('yfinance', 'yf'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('plotly.graph_objects', 'go'),
        ('sklearn', None),
        ('statsmodels.api', 'sm'),
        ('pandas_ta', 'ta'),
        ('fastapi', None),
        ('streamlit', 'st'),
        ('sqlalchemy', None),
        ('redis', None),
        ('requests', None),
        ('pydantic', None),
        ('pytest', None),
        ('loguru', None),
    ]
    
    results = []
    for lib_name, alias in libraries:
        try:
            lib = importlib.import_module(lib_name)
            version = getattr(lib, '__version__', 'Unknown')
            results.append((lib_name, True, version))
            print(f"✅ {lib_name}: {version}")
        except ImportError as e:
            results.append((lib_name, False, str(e)))
            print(f"❌ {lib_name}: 导入失败 - {e}")
    
    return results

def test_data_fetch() -> bool:
    """测试数据获取功能"""
    try:
        import yfinance as yf
        import pandas as pd
        
        print("\n测试数据获取...")
        # 获取苹果股票最近5天的数据
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        
        if len(data) > 0:
            print(f"✅ 成功获取 AAPL 数据，共 {len(data)} 条记录")
            print(f"   最新收盘价: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ 获取数据为空")
            return False
            
    except Exception as e:
        print(f"❌ 数据获取测试失败: {e}")
        return False

def test_technical_analysis() -> bool:
    """测试技术分析功能"""
    try:
        import pandas as pd
        import pandas_ta as ta
        import numpy as np
        
        print("\n测试技术分析...")
        # 创建模拟数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'Close': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # 计算技术指标
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        if not df['SMA_20'].isna().all() and not df['RSI'].isna().all():
            print("✅ 技术分析功能正常")
            print(f"   SMA(20) 最新值: {df['SMA_20'].iloc[-1]:.2f}")
            print(f"   RSI 最新值: {df['RSI'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ 技术指标计算失败")
            return False
            
    except Exception as e:
        print(f"❌ 技术分析测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("量化交易系统开发环境测试")
    print("=" * 50)
    
    # 测试 Python 版本
    python_ok = test_python_version()
    
    # 测试库导入
    print("\n测试库导入...")
    import_results = test_imports()
    
    # 统计导入成功的库
    successful_imports = sum(1 for _, success, _ in import_results if success)
    total_imports = len(import_results)
    
    print(f"\n导入统计: {successful_imports}/{total_imports} 个库导入成功")
    
    # 测试数据获取
    data_ok = test_data_fetch()
    
    # 测试技术分析
    ta_ok = test_technical_analysis()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"✅ Python 版本: {'通过' if python_ok else '失败'}")
    print(f"✅ 库导入: {successful_imports}/{total_imports} 成功")
    print(f"✅ 数据获取: {'通过' if data_ok else '失败'}")
    print(f"✅ 技术分析: {'通过' if ta_ok else '失败'}")
    
    if python_ok and successful_imports >= total_imports * 0.8 and data_ok and ta_ok:
        print("\n🎉 开发环境配置成功！可以开始量化交易系统开发。")
        return True
    else:
        print("\n⚠️  开发环境存在问题，请检查上述失败项。")
        return False

if __name__ == "__main__":
    main()