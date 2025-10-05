#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有新券商功能测试的主脚本
"""

import sys
import os
import logging
import subprocess
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test_script(script_name):
    """运行测试脚本"""
    logger.info(f"运行测试脚本: {script_name}")
    
    try:
        # 获取脚本路径
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            logger.error(f"测试脚本不存在: {script_path}")
            return False
        
        # 运行脚本
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        # 输出结果
        if result.stdout:
            logger.info(f"{script_name} 输出:")
            print(result.stdout)
        
        if result.stderr:
            logger.warning(f"{script_name} 错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            logger.info(f"✓ {script_name} 执行成功")
            return True
        else:
            logger.error(f"✗ {script_name} 执行失败 (返回码: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {script_name} 执行超时")
        return False
    except Exception as e:
        logger.error(f"✗ {script_name} 执行异常: {e}")
        return False

def check_dependencies():
    """检查依赖"""
    logger.info("检查依赖...")
    
    required_files = [
        'config.py',
        'broker_factory.py',
        'td_ameritrade_adapter.py',
        'charles_schwab_adapter.py',
        'etrade_adapter.py',
        'robinhood_adapter.py'
    ]
    
    missing_files = []
    current_dir = Path(__file__).parent
    
    for file_name in required_files:
        file_path = current_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"缺少必要文件: {', '.join(missing_files)}")
        return False
    
    logger.info("✓ 所有依赖文件存在")
    return True

def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("新券商功能完整测试套件")
    logger.info("=" * 80)
    
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败，退出测试")
        sys.exit(1)
    
    # 测试脚本列表
    test_scripts = [
        'test_new_brokers.py',
        'test_config_integration.py'
    ]
    
    # 运行测试统计
    total_tests = len(test_scripts)
    passed_tests = 0
    failed_tests = 0
    
    # 运行所有测试
    for script in test_scripts:
        logger.info("-" * 60)
        
        if run_test_script(script):
            passed_tests += 1
        else:
            failed_tests += 1
        
        logger.info("-" * 60)
        print()  # 空行分隔
    
    # 输出测试总结
    logger.info("=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"通过测试: {passed_tests}")
    logger.info(f"失败测试: {failed_tests}")
    
    if failed_tests == 0:
        logger.info("🎉 所有测试通过!")
        sys.exit(0)
    else:
        logger.error(f"❌ {failed_tests} 个测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()