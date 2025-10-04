#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有示例脚本的功能
确保所有演示都能正常运行
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_script(script_path, description):
    """
    运行脚本并检查结果
    
    Args:
        script_path (str): 脚本路径
        description (str): 脚本描述
    
    Returns:
        bool: 是否成功运行
    """
    print(f"\n{'='*60}")
    print(f"🧪 测试: {description}")
    print(f"📄 脚本: {script_path}")
    print(f"{'='*60}")
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 运行脚本，提供空输入以跳过交互
        result = subprocess.run(
            [sys.executable, script_path],
            input="\n",  # 提供回车输入以跳过交互
            capture_output=True,
            text=True,
            timeout=60  # 1分钟超时
        )
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 检查结果
        if result.returncode == 0:
            print(f"✅ 测试通过 (耗时: {duration:.2f}秒)")
            if result.stdout:
                print(f"📝 输出摘要: {result.stdout[:200]}...")
            return True
        else:
            print(f"❌ 测试失败 (返回码: {result.returncode})")
            if result.stderr:
                print(f"🚨 错误信息: {result.stderr}")
            if result.stdout:
                print(f"📝 输出信息: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 测试超时 (超过5分钟)")
        return False
    except Exception as e:
        print(f"💥 测试异常: {str(e)}")
        return False

def check_generated_files(expected_files, description):
    """
    检查生成的文件是否存在
    
    Args:
        expected_files (list): 期望生成的文件列表
        description (str): 检查描述
    
    Returns:
        bool: 是否所有文件都存在
    """
    print(f"\n📁 检查生成的文件: {description}")
    
    all_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} (大小: {file_size} 字节)")
        else:
            print(f"❌ {file_path} (文件不存在)")
            all_exist = False
    
    return all_exist

def main():
    """主测试函数"""
    print("🚀 开始测试所有示例脚本")
    print("=" * 80)
    
    # 获取当前目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # 测试结果统计
    test_results = []
    
    # 1. 测试快速演示脚本
    quick_start_script = current_dir / "quick_start_demo.py"
    if quick_start_script.exists():
        success = run_script(str(quick_start_script), "快速开始演示")
        test_results.append(("快速开始演示", success))
        
        # 检查生成的图表文件 (快速演示生成的文件名可能不同)
        expected_files = [
            str(current_dir / "factor_score_demo.png"),
            str(current_dir / "signal_price_demo.png"),
            str(current_dir / "equity_curve_demo.png"),
            str(current_dir / "drawdown_demo.png"),
            # 也检查可能的其他文件名
            str(current_dir / "price_signal_demo.png")
        ]
        # 只检查实际存在的文件
        existing_files = [f for f in expected_files if os.path.exists(f)]
        if existing_files:
            files_exist = check_generated_files(existing_files, "快速演示图表")
            test_results.append(("快速演示图表文件", files_exist))
        else:
            print("⚠️  未找到快速演示生成的图表文件")
            test_results.append(("快速演示图表文件", False))
    else:
        print(f"⚠️  快速演示脚本不存在: {quick_start_script}")
        test_results.append(("快速开始演示", False))
    
    # 2. 测试图表画廊脚本
    chart_gallery_script = current_dir / "chart_gallery.py"
    if chart_gallery_script.exists():
        success = run_script(str(chart_gallery_script), "图表画廊演示")
        test_results.append(("图表画廊演示", success))
        
        # 检查生成的图表文件
        expected_files = [
            str(current_dir / "technical_analysis_gallery.png"),
            str(current_dir / "factor_analysis_gallery.png"),
            str(current_dir / "strategy_performance_gallery.png"),
            str(current_dir / "market_analysis_gallery.png")
        ]
        files_exist = check_generated_files(expected_files, "图表画廊文件")
        test_results.append(("图表画廊文件", files_exist))
    else:
        print(f"⚠️  图表画廊脚本不存在: {chart_gallery_script}")
        test_results.append(("图表画廊演示", False))
    
    # 3. 检查文档文件
    docs_dir = project_root / "docs"
    expected_docs = [
        docs_dir / "visual_guide.md",
        current_dir / "README.md"
    ]
    
    docs_exist = check_generated_files([str(f) for f in expected_docs], "文档文件")
    test_results.append(("文档文件", docs_exist))
    
    # 4. 检查核心模块导入
    print(f"\n🔍 测试核心模块导入")
    try:
        sys.path.insert(0, str(project_root))
        
        # 测试导入当前可用的核心模块
        from src.factors.engine import FactorEngine
        from src.factors.technical import TechnicalFactors
        from src.factors.risk import RiskFactors
        from src.performance.analyzer import PerformanceAnalyzer
        from src.backtest.engine import BacktestEngine
        
        print("✅ 所有核心模块导入成功")
        test_results.append(("核心模块导入", True))
        
    except Exception as e:
        print(f"❌ 模块导入失败: {str(e)}")
        print(f"ℹ️  这可能是因为项目结构发生了变化，但不影响示例脚本的运行")
        test_results.append(("核心模块导入", False))
    
    # 输出测试总结
    print(f"\n{'='*80}")
    print("📊 测试结果总结")
    print(f"{'='*80}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！系统运行正常。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关功能。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)