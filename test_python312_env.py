#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Python 3.12环境下的依赖库功能
验证所有关键量化分析库是否正常工作
"""

import sys
import traceback
from datetime import datetime

def test_import(module_name, description=""):
    """
    测试模块导入
    
    Args:
        module_name: 模块名称
        description: 模块描述
    
    Returns:
        bool: 导入是否成功
    """
    try:
        __import__(module_name)
        print(f"✅ {module_name:<20} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name:<20} - {description} (导入失败: {e})")
        return False
    except Exception as e:
        print(f"⚠️  {module_name:<20} - {description} (其他错误: {e})")
        return False

def test_functionality():
    """
    测试关键库的基本功能
    """
    print("\n=== 功能测试 ===")
    
    # 测试numpy和pandas基础功能
    try:
        import numpy as np
        import pandas as pd
        
        # 创建测试数据
        data = np.random.randn(100, 4)
        df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
        print(f"✅ NumPy/Pandas基础功能正常 - 创建了{df.shape}的DataFrame")
    except Exception as e:
        print(f"❌ NumPy/Pandas基础功能测试失败: {e}")
    
    # 测试TA-Lib
    try:
        import talib
        import numpy as np
        
        # 创建测试价格数据
        close_prices = np.random.uniform(100, 200, 100)
        sma = talib.SMA(close_prices, timeperiod=20)
        print(f"✅ TA-Lib功能正常 - 计算了SMA指标，结果长度: {len(sma[~np.isnan(sma)])}")
    except Exception as e:
        print(f"❌ TA-Lib功能测试失败: {e}")
    
    # 测试numba JIT编译
    try:
        from numba import jit
        
        @jit
        def test_function(x):
            return x * 2 + 1
        
        result = test_function(5)
        print(f"✅ Numba JIT编译功能正常 - 测试结果: {result}")
    except Exception as e:
        print(f"❌ Numba JIT编译测试失败: {e}")
    
    # 测试QuantLib
    try:
        import QuantLib as ql
        
        # 创建简单的日期
        date = ql.Date(15, 1, 2024)
        print(f"✅ QuantLib功能正常 - 创建日期: {date}")
    except Exception as e:
        print(f"❌ QuantLib功能测试失败: {e}")
    
    # 测试zipline-reloaded
    try:
        import zipline
        print(f"✅ Zipline-reloaded导入成功 - 版本: {zipline.__version__}")
    except Exception as e:
        print(f"❌ Zipline-reloaded测试失败: {e}")
    
    # 测试riskfolio-lib
    try:
        import riskfolio as rp
        print(f"✅ Riskfolio-lib导入成功")
    except Exception as e:
        print(f"❌ Riskfolio-lib测试失败: {e}")

def main():
    """
    主测试函数
    """
    print(f"Python 3.12环境测试报告")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version}")
    print("=" * 60)
    
    # 核心数据处理库
    print("\n=== 核心数据处理库 ===")
    success_count = 0
    total_count = 0
    
    modules_to_test = [
        ("numpy", "数值计算基础库"),
        ("pandas", "数据分析库"),
        ("scipy", "科学计算库"),
        ("matplotlib", "绘图库"),
        ("seaborn", "统计绘图库"),
        ("plotly", "交互式绘图库"),
    ]
    
    for module, desc in modules_to_test:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    # 数据获取库
    print("\n=== 数据获取库 ===")
    data_modules = [
        ("yfinance", "Yahoo Finance数据"),
        ("requests", "HTTP请求库"),
        ("aiohttp", "异步HTTP库"),
    ]
    
    for module, desc in data_modules:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    # 机器学习库
    print("\n=== 机器学习库 ===")
    ml_modules = [
        ("sklearn", "机器学习库"),
        ("numba", "JIT编译加速"),
    ]
    
    for module, desc in ml_modules:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    # 高级量化分析库
    print("\n=== 高级量化分析库 ===")
    quant_modules = [
        ("talib", "技术分析库"),
        ("QuantLib", "金融数学库"),
    ]
    
    for module, desc in quant_modules:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    # 量化回测框架
    print("\n=== 量化回测框架 ===")
    backtest_modules = [
        ("zipline", "Zipline回测框架"),
        ("empyrical", "性能分析库"),
        ("pyfolio", "投资组合分析"),
    ]
    
    for module, desc in backtest_modules:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    # 投资组合优化
    print("\n=== 投资组合优化 ===")
    portfolio_modules = [
        ("riskfolio", "风险组合优化"),
        ("cvxpy", "凸优化库"),
    ]
    
    for module, desc in portfolio_modules:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    # Web框架
    print("\n=== Web框架 ===")
    web_modules = [
        ("fastapi", "FastAPI框架"),
        ("streamlit", "Streamlit应用框架"),
        ("uvicorn", "ASGI服务器"),
    ]
    
    for module, desc in web_modules:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    # 数据库
    print("\n=== 数据库 ===")
    db_modules = [
        ("sqlite3", "SQLite数据库"),
        ("redis", "Redis缓存"),
        ("peewee", "ORM框架"),
    ]
    
    for module, desc in db_modules:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    # 执行功能测试
    test_functionality()
    
    # 总结
    print("\n" + "=" * 60)
    print(f"测试总结: {success_count}/{total_count} 个模块导入成功")
    success_rate = (success_count / total_count) * 100
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 Python 3.12环境配置优秀！")
    elif success_rate >= 80:
        print("✅ Python 3.12环境配置良好！")
    elif success_rate >= 70:
        print("⚠️  Python 3.12环境配置一般，建议检查失败的模块")
    else:
        print("❌ Python 3.12环境配置需要改进")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()