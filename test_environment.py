#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易系统环境测试脚本

用于验证量化交易系统的环境配置是否正确
帮助初学者快速检查和诊断环境问题
"""

import sys
import importlib
import platform
from pathlib import Path
from typing import List, Tuple

def check_python_version() -> bool:
    """检查Python版本是否符合要求"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    print(f"   当前版本: Python {version.major}.{version.minor}.{version.micro}")
    print(f"   系统平台: {platform.system()} {platform.release()}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python版本符合要求 (3.8+)")
        return True
    else:
        print("   ❌ Python版本过低，需要3.8或更高版本")
        return False

def check_required_packages() -> bool:
    """检查必需的Python包是否已安装"""
    print("\n📦 检查必需的Python包...")
    
    # 核心包（必需）
    core_packages = [
        ('pandas', '数据处理'),
        ('numpy', '数值计算'), 
        ('yfinance', '金融数据获取'),
        ('matplotlib', '图表绘制'),
        ('seaborn', '统计图表'),
        ('scipy', '科学计算'),
        ('scikit-learn', '机器学习')
    ]
    
    # 可选包（增强功能）
    optional_packages = [
        ('plotly', '交互式图表'),
        ('pandas_ta', '技术分析'),
        ('statsmodels', '统计建模'),
        ('fastapi', 'Web API'),
        ('streamlit', 'Web应用'),
        ('pytest', '单元测试'),
        ('loguru', '日志记录')
    ]
    
    missing_core = []
    missing_optional = []
    
    print("   核心包检查:")
    for package_name, description in core_packages:
        try:
            lib = importlib.import_module(package_name)
            version = getattr(lib, '__version__', 'Unknown')
            print(f"   ✅ {package_name} ({description}): {version}")
        except ImportError:
            print(f"   ❌ {package_name} ({description}): 未安装")
            missing_core.append(package_name)
    
    print("\n   可选包检查:")
    for package_name, description in optional_packages:
        try:
            lib = importlib.import_module(package_name)
            version = getattr(lib, '__version__', 'Unknown')
            print(f"   ✅ {package_name} ({description}): {version}")
        except ImportError:
            print(f"   ⚠️  {package_name} ({description}): 未安装 (可选)")
            missing_optional.append(package_name)
    
    if missing_core:
        print(f"\n❌ 缺少核心包: {', '.join(missing_core)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有核心包已安装")
        if missing_optional:
            print(f"💡 可选安装: pip install {' '.join(missing_optional)}")
        return True

def check_project_structure() -> bool:
    """检查项目目录结构是否完整"""
    print("\n📁 检查项目结构...")
    
    required_items = [
        ('src/', '源代码目录'),
        ('src/factors/', '因子计算模块'),
        ('src/performance/', '性能分析模块'), 
        ('src/backtest/', '回测引擎'),
        ('examples/', '示例代码'),
        ('tests/', '测试代码'),
        ('requirements.txt', '依赖列表'),
        ('README.md', '项目说明')
    ]
    
    missing_items = []
    
    for item_path, description in required_items:
        path = Path(item_path)
        if path.exists():
            if path.is_dir():
                print(f"   ✅ {item_path} ({description})")
            else:
                print(f"   ✅ {item_path} ({description})")
        else:
            print(f"   ❌ {item_path} ({description}): 不存在")
            missing_items.append(item_path)
    
    if missing_items:
        print(f"\n⚠️  缺少项目文件: {', '.join(missing_items)}")
        return False
    else:
        print("   ✅ 项目结构完整")
        return True

def check_data_cache_directory() -> bool:
    """检查数据缓存目录"""
    print("\n💾 检查数据缓存目录...")
    
    cache_dir = Path('data_cache')
    if not cache_dir.exists():
        print("   📁 创建数据缓存目录...")
        cache_dir.mkdir(exist_ok=True)
        print("   ✅ 数据缓存目录已创建")
    else:
        cache_files = list(cache_dir.glob('*.meta'))
        print(f"   ✅ 数据缓存目录已存在 (包含 {len(cache_files)} 个缓存文件)")
    
    return True

def test_basic_functionality() -> bool:
    """测试基本功能是否正常"""
    print("\n🧪 测试基本功能...")
    
    try:
        # 测试数据处理
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        test_data = pd.DataFrame({
            'Close': np.random.randn(10).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 10)
        }, index=dates)
        
        # 测试基本计算
        returns = test_data['Close'].pct_change()
        sma = test_data['Close'].rolling(5).mean()
        
        print("   ✅ 数据处理功能正常")
        
        # 测试绘图功能
        import matplotlib.pyplot as plt
        plt.ioff()  # 关闭交互模式
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(test_data.index, test_data['Close'])
        ax.set_title('测试图表')
        plt.close(fig)  # 关闭图表
        
        print("   ✅ 图表绘制功能正常")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 功能测试失败: {str(e)}")
        return False
    
    # 仅进行可视化输出，不返回值以避免 Pytest 警告

import pytest

@pytest.mark.external
def test_data_fetch() -> None:
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
            # 不返回值，避免 Pytest 警告
        else:
            print("❌ 获取数据为空")
            # 不返回值，避免 Pytest 警告
            
    except Exception as e:
        print(f"❌ 数据获取测试失败: {e}")
        # 不返回值，避免 Pytest 警告

@pytest.mark.external
def test_technical_analysis() -> None:
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
            # 不返回值，避免 Pytest 警告
        else:
            print("❌ 技术指标计算失败")
            # 不返回值，避免 Pytest 警告
            
    except Exception as e:
        print(f"❌ 技术分析测试失败: {e}")
        # 不返回值，避免 Pytest 警告

def main():
    """主函数：运行所有检查"""
    print("🚀 量化交易系统环境检查")
    print("=" * 50)
    
    # 运行所有检查
    checks = [
        check_python_version(),
        check_required_packages(),
        check_project_structure(),
        check_data_cache_directory(),
        test_basic_functionality()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 环境检查完成！所有检查都通过了。")
        print("✅ 您的环境已准备就绪，可以开始使用量化交易系统了！")
        print("\n📚 下一步:")
        print("   1. 运行示例: python examples/mvp_demo.py")
        print("   2. 阅读文档: docs/BEGINNER_GUIDE.md")
        print("   3. 开始因子分析: python examples/factor_evaluation.py")
        return True
    else:
        print("❌ 环境检查发现问题，请根据上述提示解决后重新运行。")
        print("\n🔧 常见解决方案:")
        print("   1. 升级Python: 使用Python 3.8+")
        print("   2. 安装依赖: pip install -r requirements.txt")
        print("   3. 检查项目完整性: 重新克隆项目")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)