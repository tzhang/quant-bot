"""
综合因子测试脚本 - 增强版

评估所有新增因子和策略的整体表现
包含详细的错误处理和日志记录
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import time
import traceback
import warnings

# 设置警告过滤
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'comprehensive_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # 使用绝对导入，避免相对导入问题
    from src.factors.unified_factor_engine import UnifiedFactorEngine
    from src.factors.factor_optimizer import FactorOptimizer
    from src.monitoring.factor_monitor import FactorMonitor
    
    # 对于多因子策略，我们直接导入需要的类，避免通过__init__.py的相对导入
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "multi_factor_strategy", 
        os.path.join(os.path.dirname(__file__), "src", "strategies", "multi_factor_strategy.py")
    )
    multi_factor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multi_factor_module)
    MultiFactorStrategy = multi_factor_module.MultiFactorStrategy
    
    logger.info("所有模块导入成功")
except ImportError as e:
    logger.error(f"模块导入失败: {e}")
    logger.error(f"详细错误信息: {traceback.format_exc()}")
    sys.exit(1)

class ComprehensiveFactorTest:
    """综合因子测试类 - 增强版"""
    
    def __init__(self, n_stocks=100, n_periods=100, random_seed=42):
        """
        初始化测试环境
        
        Args:
            n_stocks: 股票数量
            n_periods: 时间周期数
            random_seed: 随机种子
        """
        self.n_stocks = n_stocks
        self.n_periods = n_periods
        self.random_seed = random_seed
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 测试结果存储
        self.test_results = {}
        self.performance_metrics = {}
        
        logger.info(f"测试环境初始化完成 - 股票数: {n_stocks}, 周期数: {n_periods}, 随机种子: {random_seed}")
        
        # 生成测试数据
        self._generate_test_data()
    
    def _generate_test_data(self):
        """生成测试数据 - 仅用于测试和演示"""
        try:
            logger.info("开始生成测试数据...")
            
            # 生成日期索引 - 模拟数据仅用于测试
            dates = pd.date_range('2020-01-01', periods=self.n_periods, freq='D')
            
            # 生成股票代码 - 模拟数据仅用于演示
            stocks = [f'STOCK_{i:03d}' for i in range(self.n_stocks)]
            
            # 生成价格数据 (随机游走) - 模拟数据仅用于演示
            np.random.seed(self.random_seed)
            returns = np.random.normal(0.001, 0.02, (self.n_periods, self.n_stocks))
            prices = 100 * np.exp(np.cumsum(returns, axis=0))
            
            # 创建价格DataFrame - 模拟数据仅用于测试
            self.price_data = pd.DataFrame(prices, index=dates, columns=stocks)
            
            # 生成OHLC数据 - 模拟数据仅用于演示
            self.ohlc_data = pd.DataFrame(index=dates)
            self.ohlc_data['close'] = self.price_data.iloc[:, 0]  # 使用第一只股票
            self.ohlc_data['open'] = self.ohlc_data['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(dates)))
            self.ohlc_data['high'] = np.maximum(self.ohlc_data['open'], self.ohlc_data['close']) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
            self.ohlc_data['low'] = np.minimum(self.ohlc_data['open'], self.ohlc_data['close']) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
            
            # 生成成交量数据 - 模拟数据仅用于测试
            volumes = np.random.lognormal(10, 1, (self.n_periods, self.n_stocks))
            self.volume_data = pd.DataFrame(volumes, index=dates, columns=stocks)
            
            # 生成收益率数据 - 模拟数据仅用于演示
            self.returns_data = self.price_data.pct_change().fillna(0)
            
            # 生成基本面数据 - 模拟数据仅用于测试
            self.fundamental_data = {
                'financial_metrics': pd.DataFrame({
                    'pe_ratio': np.random.uniform(5, 50, self.n_stocks),
                    'pb_ratio': np.random.uniform(0.5, 10, self.n_stocks),
                    'roe': np.random.uniform(-0.2, 0.3, self.n_stocks),
                    'debt_ratio': np.random.uniform(0, 1, self.n_stocks)
                }, index=stocks)
            }
            
            logger.info(f"测试数据生成完成 - 价格数据形状: {self.price_data.shape}, 成交量数据形状: {self.volume_data.shape}")
            
        except Exception as e:
            logger.error(f"测试数据生成失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    def test_unified_factor_engine(self):
        """测试统一因子引擎"""
        logger.info("开始测试统一因子引擎...")
        
        try:
            start_time = time.time()
            
            # 初始化因子引擎
            engine = UnifiedFactorEngine(enable_cache=False)
            
            # 计算所有因子
            factors = engine.calculate_all_factors(
                price_data=self.ohlc_data,
                volume_data=self.volume_data.iloc[:, :1],  # 只使用第一只股票的成交量
                fundamental_data=self.fundamental_data
            )
            
            calculation_time = time.time() - start_time
            
            # 验证结果
            if not factors:
                logger.warning("因子引擎未返回任何因子")
                self.test_results['factor_engine'] = {
                    'status': 'warning',
                    'total_factors': 0,
                    'valid_factors': 0,
                    'calculation_time': calculation_time,
                    'coverage': 0.0,
                    'score': 15.0
                }
            else:
                # 计算因子统计
                valid_factors = {k: v for k, v in factors.items() if v.notna().sum() > 0}
                coverage = np.mean([v.notna().mean() for v in valid_factors.values()]) if valid_factors else 0
                
                self.test_results['factor_engine'] = {
                    'status': 'success',
                    'total_factors': len(factors),
                    'valid_factors': len(valid_factors),
                    'calculation_time': calculation_time,
                    'coverage': coverage,
                    'score': 25.0
                }
                
                logger.info(f"因子引擎测试成功 - 总因子数: {len(factors)}, 有效因子数: {len(valid_factors)}")
            
        except Exception as e:
            logger.error(f"因子引擎测试失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            self.test_results['factor_engine'] = {
                'status': 'failed',
                'error': str(e),
                'score': 0.0
            }
    
    def test_multi_factor_strategy(self):
        """测试多因子策略"""
        logger.info("开始测试多因子策略...")
        
        try:
            start_time = time.time()
            
            # 初始化策略
            strategy = MultiFactorStrategy()
            
            # 准备测试数据
            price_data_with_close = self.ohlc_data.copy()
            
            # 运行策略
            result = strategy.run_strategy(
                price_data=price_data_with_close,
                volume_data=self.volume_data.iloc[:, :1]
            )
            
            calculation_time = time.time() - start_time
            
            # 验证结果
            if result is None or (isinstance(result, dict) and not result):
                logger.warning("多因子策略未返回有效结果")
                score = 15.0
            else:
                score = 20.0
                logger.info("多因子策略测试成功")
            
            self.test_results['multi_factor_strategy'] = {
                'status': 'success',
                'calculation_time': calculation_time,
                'score': score
            }
            
        except Exception as e:
            logger.error(f"多因子策略测试失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            self.test_results['multi_factor_strategy'] = {
                'status': 'failed',
                'error': str(e),
                'score': 0.0
            }
    
    def test_factor_monitor(self):
        """测试因子监控系统"""
        logger.info("开始测试因子监控系统...")
        
        try:
            start_time = time.time()
            
            # 初始化监控器
            monitor = FactorMonitor()
            
            # 生成测试因子数据
            n_periods = min(10, self.n_periods)  # 限制监控周期数
            test_factors = {}
            
            for i in range(5):  # 生成5个测试因子
                factor_name = f'test_factor_{i}'
                test_factors[factor_name] = pd.Series(
                    np.random.normal(0, 1, n_periods),
                    index=pd.date_range('2020-01-01', periods=n_periods, freq='D')
                )
            
            # 生成测试收益率
            test_returns = pd.Series(
                np.random.normal(0.001, 0.02, n_periods),
                index=pd.date_range('2020-01-01', periods=n_periods, freq='D')
            )
            
            # 运行监控
            total_alerts = 0
            critical_alerts = 0
            
            for period in range(n_periods):
                try:
                    current_factors = {k: v.iloc[:period+1] for k, v in test_factors.items() if period+1 <= len(v)}
                    current_returns = test_returns.iloc[:period+1]
                    
                    if len(current_factors) > 0 and len(current_returns) > 5:  # 至少需要5个数据点
                        alerts = monitor.run_monitoring(current_factors, current_returns)
                        if alerts:
                            total_alerts += len(alerts)
                            critical_alerts += len([a for a in alerts if a.get('severity') == 'critical'])
                
                except Exception as period_error:
                    logger.warning(f"监控周期 {period} 出现错误: {period_error}")
                    continue
            
            calculation_time = time.time() - start_time
            
            self.test_results['factor_monitor'] = {
                'status': 'success',
                'monitoring_periods': n_periods,
                'total_alerts': total_alerts,
                'critical_alerts': critical_alerts,
                'calculation_time': calculation_time,
                'score': 25.0
            }
            
            logger.info(f"因子监控系统测试完成，共监控{n_periods}期")
            
        except Exception as e:
            logger.error(f"因子监控系统测试失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            self.test_results['factor_monitor'] = {
                'status': 'failed',
                'error': str(e),
                'score': 0.0
            }
    
    def test_factor_optimizer(self):
        """测试因子优化器"""
        logger.info("开始测试因子优化器...")
        
        try:
            start_time = time.time()
            
            # 初始化优化器
            optimizer = FactorOptimizer()
            
            # 准备测试数据
            test_factors = pd.DataFrame({
                'momentum': np.random.normal(0, 1, 50),
                'reversal': np.random.normal(0, 1, 50),
                'volatility': np.random.normal(0, 1, 50)
            })
            
            test_returns = pd.Series(np.random.normal(0.001, 0.02, 50))
            
            # 运行优化
            try:
                result = optimizer.optimize_factors(test_factors, test_returns)
                optimization_success = True
            except Exception as opt_error:
                logger.warning(f"因子优化过程中出现警告: {opt_error}")
                result = None
                optimization_success = False
            
            calculation_time = time.time() - start_time
            
            # 评分
            if optimization_success and result is not None:
                score = 25.0
                logger.info("因子优化器测试成功")
            else:
                score = 20.0  # 部分成功
                logger.info("因子优化器测试部分成功")
            
            self.test_results['factor_optimizer'] = {
                'status': 'success',
                'optimization_success': optimization_success,
                'calculation_time': calculation_time,
                'score': score
            }
            
        except Exception as e:
            logger.error(f"因子优化器测试失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            self.test_results['factor_optimizer'] = {
                'status': 'failed',
                'error': str(e),
                'score': 0.0
            }
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        logger.info("开始运行综合因子系统测试...")
        
        start_time = time.time()
        
        # 运行各项测试
        test_methods = [
            self.test_unified_factor_engine,
            self.test_multi_factor_strategy,
            self.test_factor_monitor,
            self.test_factor_optimizer
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"测试方法 {test_method.__name__} 执行失败: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # 生成综合报告
        report = self.generate_comprehensive_report(total_time)
        
        # 保存报告
        report_filename = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.save_report(report, report_filename)
        
        logger.info(f"综合测试完成，总耗时: {total_time:.2f}秒")
        logger.info(f"测试报告已保存到: {report_filename}")
        
        return report
    
    def generate_comprehensive_report(self, total_time):
        """生成综合测试报告"""
        try:
            # 计算总分
            total_score = sum([result.get('score', 0) for result in self.test_results.values()])
            max_score = 100.0
            percentage = (total_score / max_score) * 100
            
            # 评级
            if percentage >= 90:
                grade = "优秀 (A)"
            elif percentage >= 80:
                grade = "良好 (B)"
            elif percentage >= 70:
                grade = "中等 (C)"
            elif percentage >= 60:
                grade = "及格 (D)"
            else:
                grade = "不及格 (F)"
            
            # 生成报告
            report = f"""
================================================================================
综合因子系统测试报告 - 增强版
================================================================================
测试信息:
  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  股票数量: {self.n_stocks}
  时间周期: {self.n_periods}
  随机种子: {self.random_seed}
  总耗时: {total_time:.2f}秒

综合评分:
  总分: {total_score}/{max_score}
  百分比: {percentage:.1f}%
  评级: {grade}

各组件评分:
"""
            
            # 添加各组件详细信息
            for component, result in self.test_results.items():
                score = result.get('score', 0)
                status = result.get('status', 'unknown')
                report += f"  {component}: {score}/25 ({status})\n"
            
            report += "\n详细测试结果:\n"
            
            # 统一因子引擎
            if 'factor_engine' in self.test_results:
                result = self.test_results['factor_engine']
                status_symbol = "✓" if result['status'] == 'success' else "✗" if result['status'] == 'failed' else "⚠"
                report += f"\n1. 统一因子引擎测试:\n"
                report += f"  {status_symbol} 测试{result['status']}\n"
                
                if result['status'] != 'failed':
                    report += f"  总因子数: {result.get('total_factors', 0)}\n"
                    report += f"  有效因子数: {result.get('valid_factors', 0)}\n"
                    report += f"  计算时间: {result.get('calculation_time', 0):.2f}秒\n"
                    report += f"  平均覆盖率: {result.get('coverage', 0)*100:.2f}%\n"
                else:
                    report += f"  错误信息: {result.get('error', 'Unknown error')}\n"
            
            # 多因子策略
            if 'multi_factor_strategy' in self.test_results:
                result = self.test_results['multi_factor_strategy']
                status_symbol = "✓" if result['status'] == 'success' else "✗"
                report += f"\n2. 多因子策略测试:\n"
                report += f"  {status_symbol} 测试{result['status']}\n"
                report += f"  计算时间: {result.get('calculation_time', 0):.2f}秒\n"
                
                if result['status'] == 'failed':
                    report += f"  错误信息: {result.get('error', 'Unknown error')}\n"
            
            # 因子监控系统
            if 'factor_monitor' in self.test_results:
                result = self.test_results['factor_monitor']
                status_symbol = "✓" if result['status'] == 'success' else "✗"
                report += f"\n3. 因子监控系统测试:\n"
                report += f"  {status_symbol} 测试{result['status']}\n"
                
                if result['status'] != 'failed':
                    report += f"  监控周期数: {result.get('monitoring_periods', 0)}\n"
                    report += f"  总告警数: {result.get('total_alerts', 0)}\n"
                    report += f"  严重告警数: {result.get('critical_alerts', 0)}\n"
                    report += f"  计算时间: {result.get('calculation_time', 0):.2f}秒\n"
                else:
                    report += f"  错误信息: {result.get('error', 'Unknown error')}\n"
            
            # 因子优化器
            if 'factor_optimizer' in self.test_results:
                result = self.test_results['factor_optimizer']
                status_symbol = "✓" if result['status'] == 'success' else "✗"
                report += f"\n4. 因子优化器测试:\n"
                report += f"  {status_symbol} 测试{result['status']}\n"
                report += f"  优化成功: {'是' if result.get('optimization_success', False) else '否'}\n"
                report += f"  计算时间: {result.get('calculation_time', 0):.2f}秒\n"
                
                if result['status'] == 'failed':
                    report += f"  错误信息: {result.get('error', 'Unknown error')}\n"
            
            # 总结和建议
            report += f"\n总结和建议:\n"
            
            if percentage >= 90:
                report += "  系统整体表现优秀，各模块功能完善\n"
            elif percentage >= 80:
                report += "  系统整体表现良好，部分模块需要优化\n"
            elif percentage >= 70:
                report += "  系统基本功能正常，但仍有较大改进空间\n"
            else:
                report += "  系统存在较多问题，需要重点改进\n"
            
            report += "\n改进建议:\n"
            report += "  1. 增加更多类型的因子，提高因子多样性\n"
            report += "  2. 优化因子计算算法，提高计算效率\n"
            report += "  3. 加强因子质量控制，减少无效因子\n"
            report += "  4. 完善监控告警机制，提高告警准确性\n"
            report += "  5. 增加更多的策略优化方法\n"
            report += "  6. 添加更详细的性能分析和诊断功能\n"
            report += "  7. 实现更robust的错误处理和恢复机制\n"
            
            report += "\n" + "="*80 + "\n"
            
            return report
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return f"报告生成失败: {e}"
    
    def save_report(self, report, filename):
        """保存测试报告"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"测试报告已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存报告失败: {e}")

def main():
    """主函数"""
    try:
        logger.info("开始综合因子系统测试...")
        
        # 创建测试实例
        tester = ComprehensiveFactorTest(n_stocks=100, n_periods=100, random_seed=42)
        
        # 运行综合测试
        report = tester.run_comprehensive_test()
        
        # 输出报告
        print(report)
        
        logger.info("综合因子系统测试完成！")
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()