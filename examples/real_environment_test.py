#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Firstrade真实环境验证脚本
用于安全地测试Firstrade API连接和基本功能

注意：
1. 此脚本仅用于验证连接，不会执行任何实际交易
2. 所有操作都在只读模式下进行
3. 用户凭据通过环境变量或交互式输入获取，不会保存到文件
"""

import os
import sys
import getpass
import logging
from datetime import datetime
from typing import Dict, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from firstrade.account import FTSession
    HAS_FIRSTRADE_API = True
    print("✅ Firstrade API已成功导入")
except ImportError as e:
    print(f"错误: Firstrade API导入失败: {e}")
    print("请运行: pip install firstrade")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'firstrade_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class FirstradeRealEnvironmentTester:
    """Firstrade真实环境测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.ft = None
        self.username = None
        self.password = None
        self.pin = None
        self.test_results = {}
        
    def get_credentials(self) -> bool:
        """
        安全获取用户凭据
        优先级：环境变量 > 交互式输入
        """
        try:
            # 尝试从环境变量获取
            self.username = os.getenv('FIRSTRADE_USERNAME')
            self.password = os.getenv('FIRSTRADE_PASSWORD')
            self.pin = os.getenv('FIRSTRADE_PIN')
            
            # 如果环境变量不存在，则交互式输入
            if not self.username:
                print("\n=== Firstrade 凭据配置 ===")
                print("提示：您可以设置环境变量 FIRSTRADE_USERNAME, FIRSTRADE_PASSWORD, FIRSTRADE_PIN")
                print("      来避免每次手动输入")
                print()
                
                self.username = input("请输入Firstrade用户名: ").strip()
                if not self.username:
                    logger.error("用户名不能为空")
                    return False
                    
            if not self.password:
                self.password = getpass.getpass("请输入Firstrade密码: ")
                if not self.password:
                    logger.error("密码不能为空")
                    return False
                    
            if not self.pin:
                pin_input = getpass.getpass("请输入PIN码（如果没有请直接回车）: ")
                self.pin = pin_input if pin_input.strip() else None
                
            logger.info("凭据获取成功")
            return True
            
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return False
        except Exception as e:
            logger.error(f"获取凭据时发生错误: {e}")
            return False
    
    def test_connection(self) -> bool:
        """测试Firstrade连接"""
        try:
            logger.info("开始测试Firstrade连接...")
            
            # 创建会话
            self.ft = FTSession()
            
            # 尝试登录
            logger.info("正在登录...")
            login_result = self.ft.login(
                username=self.username,
                password=self.password,
                pin=self.pin
            )
            
            if login_result:
                logger.info("✅ 登录成功！")
                self.test_results['login'] = {'status': 'success', 'message': '登录成功'}
                return True
            else:
                logger.error("❌ 登录失败")
                self.test_results['login'] = {'status': 'failed', 'message': '登录失败'}
                return False
                
        except Exception as e:
            logger.error(f"❌ 连接测试失败: {e}")
            self.test_results['login'] = {'status': 'error', 'message': str(e)}
            return False
    
    def test_account_info(self) -> bool:
        """测试账户信息获取"""
        try:
            logger.info("正在获取账户信息...")
            
            # 获取账户数据
            account_data = self.ft.get_account()
            
            if account_data:
                logger.info("✅ 账户信息获取成功")
                
                # 安全地显示账户信息（隐藏敏感数据）
                safe_info = {}
                for key, value in account_data.items():
                    if 'account' in key.lower() and isinstance(value, (int, float)):
                        safe_info[key] = f"${value:,.2f}" if value else "N/A"
                    elif 'number' in key.lower() or 'id' in key.lower():
                        safe_info[key] = "***" + str(value)[-4:] if value else "N/A"
                    else:
                        safe_info[key] = value
                
                logger.info(f"账户信息: {safe_info}")
                self.test_results['account_info'] = {'status': 'success', 'data': safe_info}
                return True
            else:
                logger.warning("⚠️ 未获取到账户信息")
                self.test_results['account_info'] = {'status': 'warning', 'message': '未获取到账户信息'}
                return False
                
        except Exception as e:
            logger.error(f"❌ 账户信息获取失败: {e}")
            self.test_results['account_info'] = {'status': 'error', 'message': str(e)}
            return False
    
    def test_positions(self) -> bool:
        """测试持仓信息获取"""
        try:
            logger.info("正在获取持仓信息...")
            
            # 获取持仓数据
            positions = self.ft.get_positions()
            
            if positions is not None:
                logger.info(f"✅ 持仓信息获取成功，共有 {len(positions)} 个持仓")
                
                # 显示持仓概要（不显示具体数量和金额）
                if positions:
                    symbols = [pos.get('symbol', 'Unknown') for pos in positions[:5]]  # 只显示前5个
                    logger.info(f"持仓股票（前5个）: {', '.join(symbols)}")
                else:
                    logger.info("当前无持仓")
                
                self.test_results['positions'] = {
                    'status': 'success', 
                    'count': len(positions),
                    'message': f'获取到 {len(positions)} 个持仓'
                }
                return True
            else:
                logger.warning("⚠️ 未获取到持仓信息")
                self.test_results['positions'] = {'status': 'warning', 'message': '未获取到持仓信息'}
                return False
                
        except Exception as e:
            logger.error(f"❌ 持仓信息获取失败: {e}")
            self.test_results['positions'] = {'status': 'error', 'message': str(e)}
            return False
    
    def test_quote(self, symbol: str = "AAPL") -> bool:
        """测试股票报价获取"""
        try:
            logger.info(f"正在获取 {symbol} 的报价信息...")
            
            # 获取股票报价
            quote = self.ft.get_quote(symbol)
            
            if quote:
                logger.info(f"✅ {symbol} 报价获取成功")
                
                # 显示报价信息
                price = quote.get('last_price', quote.get('price', 'N/A'))
                change = quote.get('change', 'N/A')
                change_percent = quote.get('change_percent', 'N/A')
                volume = quote.get('volume', 'N/A')
                
                logger.info(f"{symbol} 价格: ${price}, 变化: {change} ({change_percent}%), 成交量: {volume}")
                
                self.test_results['quote'] = {
                    'status': 'success',
                    'symbol': symbol,
                    'data': {
                        'price': price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': volume
                    }
                }
                return True
            else:
                logger.warning(f"⚠️ 未获取到 {symbol} 的报价信息")
                self.test_results['quote'] = {'status': 'warning', 'message': f'未获取到 {symbol} 报价'}
                return False
                
        except Exception as e:
            logger.error(f"❌ {symbol} 报价获取失败: {e}")
            self.test_results['quote'] = {'status': 'error', 'message': str(e)}
            return False
    
    def run_comprehensive_test(self) -> Dict:
        """运行综合测试"""
        print("\n" + "="*60)
        print("🚀 Firstrade 真实环境验证测试")
        print("="*60)
        print("注意：此测试仅验证连接和数据获取，不会执行任何交易操作")
        print()
        
        # 获取凭据
        if not self.get_credentials():
            return {'status': 'failed', 'message': '凭据获取失败'}
        
        # 测试步骤
        test_steps = [
            ('连接测试', self.test_connection),
            ('账户信息', self.test_account_info),
            ('持仓信息', self.test_positions),
            ('报价信息', self.test_quote),
        ]
        
        success_count = 0
        total_tests = len(test_steps)
        
        for step_name, test_func in test_steps:
            print(f"\n📋 {step_name}测试...")
            try:
                if test_func():
                    success_count += 1
                    print(f"✅ {step_name}测试通过")
                else:
                    print(f"⚠️ {step_name}测试未完全成功")
            except Exception as e:
                print(f"❌ {step_name}测试失败: {e}")
        
        # 生成测试报告
        print("\n" + "="*60)
        print("📊 测试结果汇总")
        print("="*60)
        print(f"总测试数: {total_tests}")
        print(f"成功数: {success_count}")
        print(f"成功率: {success_count/total_tests*100:.1f}%")
        
        if success_count >= 2:  # 至少连接和账户信息成功
            print("\n🎉 基本功能验证成功！您的Firstrade账户可以正常连接。")
            status = 'success'
        elif success_count >= 1:  # 至少连接成功
            print("\n⚠️ 部分功能验证成功，建议检查账户权限设置。")
            status = 'partial'
        else:
            print("\n❌ 验证失败，请检查凭据和网络连接。")
            status = 'failed'
        
        # 清理敏感信息
        self.username = None
        self.password = None
        self.pin = None
        
        return {
            'status': status,
            'success_count': success_count,
            'total_tests': total_tests,
            'results': self.test_results
        }

def main():
    """主函数"""
    try:
        tester = FirstradeRealEnvironmentTester()
        result = tester.run_comprehensive_test()
        
        # 根据结果设置退出码
        if result['status'] == 'success':
            sys.exit(0)
        elif result['status'] == 'partial':
            sys.exit(1)
        else:
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
        sys.exit(130)
    except Exception as e:
        logger.error(f"测试过程中发生未预期错误: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()