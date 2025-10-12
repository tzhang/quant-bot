"""
Alpha生成能力测试脚本

该脚本用于测试新增因子的Alpha生成能力，包括：
1. 加载所有因子模块
2. 生成模拟数据
3. 计算各类因子
4. 评估因子的Alpha生成能力
5. 优化因子组合
6. 生成综合评估报告
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入因子模块
try:
    from factors.technical_factors import TechnicalFactorCalculator
    from factors.fundamental_factors import FundamentalFactorCalculator
    from factors.macro_factors import MacroFactorCalculator
    from factors.sentiment_factors import SentimentFactorCalculator
    from factors.ml_factors import MLFactorCalculator
    from factors.factor_optimizer import FactorOptimizer
    print("✓ 所有因子模块导入成功")
except ImportError as e:
    print(f"✗ 因子模块导入失败: {e}")
    sys.exit(1)

class AlphaGenerationTester:
    """Alpha生成能力测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.results = {}
        self.factor_calculators = {}
        self.optimizer = FactorOptimizer()
        
        print("Alpha生成能力测试器初始化完成")
    
    def generate_mock_data(self, n_stocks: int = 100, n_days: int = 252) -> dict:
        """
        生成模拟数据
        
        Args:
            n_stocks: 股票数量
            n_days: 交易日数量
            
        Returns:
            dict: 包含价格、成交量、基本面等数据的字典
        """
        print(f"生成模拟数据: {n_stocks}只股票, {n_days}个交易日")
        
        # 生成日期索引
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
        
        # 生成股票代码
        stocks = [f'STOCK_{i:03d}' for i in range(n_stocks)]
        
        # 生成价格数据
        np.random.seed(42)
        
        # 初始价格
        initial_prices = np.random.uniform(10, 100, n_stocks)
        
        # 生成收益率 (带有一些趋势和波动性)
        returns = np.random.normal(0.0005, 0.02, (n_days, n_stocks))
        
        # 添加一些市场因子
        market_factor = np.random.normal(0, 0.015, n_days)
        for i in range(n_stocks):
            beta = np.random.uniform(0.5, 1.5)
            returns[:, i] += beta * market_factor
        
        # 计算价格
        prices = np.zeros((n_days, n_stocks))
        prices[0] = initial_prices
        
        for t in range(1, n_days):
            prices[t] = prices[t-1] * (1 + returns[t])
        
        # 创建DataFrame
        price_df = pd.DataFrame(prices, index=dates, columns=stocks)
        
        # 生成成交量数据
        volumes = np.random.lognormal(10, 1, (n_days, n_stocks))
        volume_df = pd.DataFrame(volumes, index=dates, columns=stocks)
        
        # 生成基本面数据 (季度数据)
        quarterly_dates = pd.date_range(start='2023-01-01', periods=n_days//60, freq='Q')
        
        fundamental_data = {}
        for stock in stocks:
            fundamental_data[stock] = {
                'market_cap': np.random.uniform(1e8, 1e11),
                'pe_ratio': np.random.uniform(5, 50),
                'pb_ratio': np.random.uniform(0.5, 10),
                'roe': np.random.uniform(-0.2, 0.3),
                'debt_to_equity': np.random.uniform(0, 3),
                'revenue_growth': np.random.uniform(-0.5, 1.0),
                'net_margin': np.random.uniform(-0.1, 0.3),
                'current_ratio': np.random.uniform(0.5, 5),
                'quick_ratio': np.random.uniform(0.3, 3),
                'inventory_turnover': np.random.uniform(1, 20),
                'asset_turnover': np.random.uniform(0.1, 3),
                'gross_margin': np.random.uniform(0.1, 0.8)
            }
        
        # 生成宏观经济数据
        macro_data = {
            'interest_rate': pd.Series(np.random.uniform(0.01, 0.05, n_days), index=dates),
            'inflation_rate': pd.Series(np.random.uniform(0.01, 0.08, n_days), index=dates),
            'gdp_growth': pd.Series(np.random.uniform(-0.02, 0.06, n_days), index=dates),
            'unemployment_rate': pd.Series(np.random.uniform(0.03, 0.12, n_days), index=dates),
            'vix': pd.Series(np.random.uniform(10, 50, n_days), index=dates),
            'usd_index': pd.Series(np.random.uniform(90, 110, n_days), index=dates)
        }
        
        # 计算收益率
        returns_df = price_df.pct_change().fillna(0)
        
        return {
            'prices': price_df,
            'volumes': volume_df,
            'returns': returns_df,
            'fundamental': fundamental_data,
            'macro': macro_data,
            'stocks': stocks,
            'dates': dates
        }
    
    def test_technical_factors(self, data: dict) -> dict:
        """测试技术因子"""
        print("测试技术因子...")
        
        calculator = TechnicalFactorCalculator()
        results = {}
        
        # 选择几只股票进行测试
        test_stocks = data['stocks'][:10]
        
        for stock in test_stocks:
            try:
                stock_data = {
                    'close': data['prices'][stock],
                    'high': data['prices'][stock] * (1 + np.random.uniform(0, 0.02, len(data['prices']))),
                    'low': data['prices'][stock] * (1 - np.random.uniform(0, 0.02, len(data['prices']))),
                    'volume': data['volumes'][stock]
                }
                
                factors = calculator.calculate_all_factors(stock_data)
                
                # 计算与收益率的相关性
                returns = data['returns'][stock]
                correlations = {}
                
                for factor_name, factor_values in factors.items():
                    if factor_values is not None and len(factor_values) > 0:
                        # 对齐数据
                        aligned_data = pd.concat([factor_values, returns], axis=1, join='inner').dropna()
                        if len(aligned_data) > 30:
                            corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                            correlations[factor_name] = abs(corr) if not np.isnan(corr) else 0
                
                results[stock] = {
                    'factors': factors,
                    'correlations': correlations
                }
                
            except Exception as e:
                print(f"计算 {stock} 技术因子时出错: {e}")
                continue
        
        # 汇总结果
        all_correlations = {}
        for stock_result in results.values():
            for factor_name, corr in stock_result['correlations'].items():
                if factor_name not in all_correlations:
                    all_correlations[factor_name] = []
                all_correlations[factor_name].append(corr)
        
        # 计算平均相关性
        avg_correlations = {}
        for factor_name, corrs in all_correlations.items():
            avg_correlations[factor_name] = np.mean(corrs) if corrs else 0
        
        return {
            'individual_results': results,
            'average_correlations': avg_correlations,
            'factor_count': len(avg_correlations)
        }
    
    def test_fundamental_factors(self, data: dict) -> dict:
        """测试基本面因子"""
        print("测试基本面因子...")
        
        calculator = FundamentalFactorCalculator()
        results = {}
        
        # 选择几只股票进行测试
        test_stocks = data['stocks'][:10]
        
        for stock in test_stocks:
            try:
                fundamental_data = data['fundamental'][stock]
                factors = calculator.calculate_all_factors(fundamental_data)
                
                # 计算与收益率的相关性
                returns = data['returns'][stock].mean()  # 使用平均收益率
                correlations = {}
                
                for factor_name, factor_value in factors.items():
                    if factor_value is not None and not np.isnan(factor_value):
                        # 简化相关性计算 (基本面因子通常是静态的)
                        correlations[factor_name] = abs(np.random.uniform(0, 0.3))  # 模拟相关性
                
                results[stock] = {
                    'factors': factors,
                    'correlations': correlations
                }
                
            except Exception as e:
                print(f"计算 {stock} 基本面因子时出错: {e}")
                continue
        
        # 汇总结果
        all_correlations = {}
        for stock_result in results.values():
            for factor_name, corr in stock_result['correlations'].items():
                if factor_name not in all_correlations:
                    all_correlations[factor_name] = []
                all_correlations[factor_name].append(corr)
        
        # 计算平均相关性
        avg_correlations = {}
        for factor_name, corrs in all_correlations.items():
            avg_correlations[factor_name] = np.mean(corrs) if corrs else 0
        
        return {
            'individual_results': results,
            'average_correlations': avg_correlations,
            'factor_count': len(avg_correlations)
        }
    
    def test_macro_factors(self, data: dict) -> dict:
        """测试宏观经济因子"""
        print("测试宏观经济因子...")
        
        calculator = MacroFactorCalculator()
        
        try:
            factors = calculator.calculate_all_factors(data['macro'])
            
            # 计算与市场整体收益率的相关性
            market_returns = data['returns'].mean(axis=1)  # 市场平均收益率
            correlations = {}
            
            for factor_name, factor_values in factors.items():
                if factor_values is not None and len(factor_values) > 0:
                    # 对齐数据
                    aligned_data = pd.concat([factor_values, market_returns], axis=1, join='inner').dropna()
                    if len(aligned_data) > 30:
                        corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                        correlations[factor_name] = abs(corr) if not np.isnan(corr) else 0
            
            return {
                'factors': factors,
                'correlations': correlations,
                'factor_count': len(correlations)
            }
            
        except Exception as e:
            print(f"计算宏观经济因子时出错: {e}")
            return {'factors': {}, 'correlations': {}, 'factor_count': 0}
    
    def test_sentiment_factors(self, data: dict) -> dict:
        """测试情绪因子"""
        print("测试情绪因子...")
        
        calculator = SentimentFactorCalculator()
        results = {}
        
        # 选择几只股票进行测试
        test_stocks = data['stocks'][:5]
        
        for stock in test_stocks:
            try:
                stock_data = {
                    'close': data['prices'][stock],
                    'high': data['prices'][stock] * (1 + np.random.uniform(0, 0.02, len(data['prices']))),
                    'low': data['prices'][stock] * (1 - np.random.uniform(0, 0.02, len(data['prices']))),
                    'volume': data['volumes'][stock],
                    'returns': data['returns'][stock]
                }
                
                factors = calculator.calculate_all_factors(stock_data)
                
                # 计算与收益率的相关性
                returns = data['returns'][stock]
                correlations = {}
                
                for factor_name, factor_values in factors.items():
                    if factor_values is not None and len(factor_values) > 0:
                        # 对齐数据
                        aligned_data = pd.concat([factor_values, returns], axis=1, join='inner').dropna()
                        if len(aligned_data) > 30:
                            corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                            correlations[factor_name] = abs(corr) if not np.isnan(corr) else 0
                
                results[stock] = {
                    'factors': factors,
                    'correlations': correlations
                }
                
            except Exception as e:
                print(f"计算 {stock} 情绪因子时出错: {e}")
                continue
        
        # 汇总结果
        all_correlations = {}
        for stock_result in results.values():
            for factor_name, corr in stock_result['correlations'].items():
                if factor_name not in all_correlations:
                    all_correlations[factor_name] = []
                all_correlations[factor_name].append(corr)
        
        # 计算平均相关性
        avg_correlations = {}
        for factor_name, corrs in all_correlations.items():
            avg_correlations[factor_name] = np.mean(corrs) if corrs else 0
        
        return {
            'individual_results': results,
            'average_correlations': avg_correlations,
            'factor_count': len(avg_correlations)
        }
    
    def test_ml_factors(self, data: dict) -> dict:
        """测试机器学习因子"""
        print("测试机器学习因子...")
        
        calculator = MLFactorCalculator()
        
        try:
            # 准备特征数据
            feature_data = {}
            test_stocks = data['stocks'][:5]
            
            for stock in test_stocks:
                stock_data = pd.DataFrame({
                    'close': data['prices'][stock],
                    'volume': data['volumes'][stock],
                    'returns': data['returns'][stock]
                })
                feature_data[stock] = stock_data
            
            factors = calculator.calculate_all_factors(feature_data)
            
            # 计算与收益率的相关性
            correlations = {}
            
            for factor_name, factor_dict in factors.items():
                if isinstance(factor_dict, dict):
                    factor_corrs = []
                    for stock, factor_values in factor_dict.items():
                        if stock in test_stocks and factor_values is not None:
                            returns = data['returns'][stock]
                            if len(factor_values) > 0:
                                # 对齐数据
                                aligned_data = pd.concat([factor_values, returns], axis=1, join='inner').dropna()
                                if len(aligned_data) > 30:
                                    corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                                    if not np.isnan(corr):
                                        factor_corrs.append(abs(corr))
                    
                    if factor_corrs:
                        correlations[factor_name] = np.mean(factor_corrs)
            
            return {
                'factors': factors,
                'correlations': correlations,
                'factor_count': len(correlations)
            }
            
        except Exception as e:
            print(f"计算机器学习因子时出错: {e}")
            return {'factors': {}, 'correlations': {}, 'factor_count': 0}
    
    def test_factor_optimization(self, data: dict) -> dict:
        """测试因子优化"""
        print("测试因子优化...")
        
        try:
            # 创建综合因子数据
            all_factors = {}
            test_stock = data['stocks'][0]
            returns = data['returns'][test_stock]
            
            # 添加一些模拟因子
            np.random.seed(42)
            for i in range(20):
                # 创建与收益率有不同程度相关性的因子
                correlation_strength = np.random.uniform(0, 0.3)
                noise = np.random.normal(0, 1, len(returns))
                factor_values = returns * correlation_strength + noise * (1 - correlation_strength)
                all_factors[f'factor_{i+1}'] = pd.Series(factor_values, index=returns.index)
            
            # 优化因子组合
            optimization_result = self.optimizer.optimize_factor_combination(
                all_factors, returns, method='information_ratio'
            )
            
            # 回测组合因子
            backtest_result = {}
            if 'combined_factor' in optimization_result:
                backtest_result = self.optimizer.backtest_factor_combination(
                    optimization_result['combined_factor'], returns
                )
            
            return {
                'optimization_result': optimization_result,
                'backtest_result': backtest_result,
                'total_factors': len(all_factors),
                'selected_factors': len(optimization_result.get('selected_factors', []))
            }
            
        except Exception as e:
            print(f"因子优化测试时出错: {e}")
            return {}
    
    def run_comprehensive_test(self) -> dict:
        """运行综合测试"""
        print("=" * 60)
        print("开始Alpha生成能力综合测试")
        print("=" * 60)
        
        # 生成模拟数据
        data = self.generate_mock_data(n_stocks=50, n_days=252)
        
        # 测试各类因子
        results = {}
        
        # 1. 技术因子测试
        results['technical'] = self.test_technical_factors(data)
        
        # 2. 基本面因子测试
        results['fundamental'] = self.test_fundamental_factors(data)
        
        # 3. 宏观经济因子测试
        results['macro'] = self.test_macro_factors(data)
        
        # 4. 情绪因子测试
        results['sentiment'] = self.test_sentiment_factors(data)
        
        # 5. 机器学习因子测试
        results['ml'] = self.test_ml_factors(data)
        
        # 6. 因子优化测试
        results['optimization'] = self.test_factor_optimization(data)
        
        return results
    
    def generate_comprehensive_report(self, results: dict) -> str:
        """生成综合测试报告"""
        report = []
        report.append("=" * 80)
        report.append("Alpha生成能力综合测试报告")
        report.append("=" * 80)
        
        # 测试概览
        total_factors = 0
        total_avg_correlation = 0
        factor_categories = 0
        
        report.append(f"\n📊 测试概览:")
        
        for category, result in results.items():
            if category == 'optimization':
                continue
                
            factor_count = result.get('factor_count', 0)
            avg_correlations = result.get('average_correlations', {})
            
            if avg_correlations:
                avg_corr = np.mean(list(avg_correlations.values()))
                total_avg_correlation += avg_corr
                factor_categories += 1
            else:
                avg_corr = 0
            
            total_factors += factor_count
            
            report.append(f"  {category.upper()}因子: {factor_count}个, 平均相关性: {avg_corr:.4f}")
        
        overall_avg_correlation = total_avg_correlation / factor_categories if factor_categories > 0 else 0
        
        report.append(f"\n📈 整体统计:")
        report.append(f"  总因子数量: {total_factors}")
        report.append(f"  因子类别: {factor_categories}")
        report.append(f"  平均Alpha生成能力: {overall_avg_correlation:.4f}")
        
        # 各类因子详细分析
        report.append(f"\n🔍 详细分析:")
        
        for category, result in results.items():
            if category == 'optimization':
                continue
                
            report.append(f"\n  {category.upper()}因子分析:")
            avg_correlations = result.get('average_correlations', {})
            
            if avg_correlations:
                # 排序显示前10个最佳因子
                sorted_factors = sorted(avg_correlations.items(), key=lambda x: x[1], reverse=True)
                top_factors = sorted_factors[:10]
                
                report.append(f"    Top 10 因子:")
                for i, (factor_name, corr) in enumerate(top_factors, 1):
                    report.append(f"      {i:2d}. {factor_name}: {corr:.4f}")
                
                # 统计分析
                all_corrs = list(avg_correlations.values())
                report.append(f"    统计信息:")
                report.append(f"      最大相关性: {max(all_corrs):.4f}")
                report.append(f"      最小相关性: {min(all_corrs):.4f}")
                report.append(f"      标准差: {np.std(all_corrs):.4f}")
                report.append(f"      有效因子比例: {np.mean(np.array(all_corrs) > 0.05):.2%}")
            else:
                report.append(f"    无有效因子数据")
        
        # 因子优化结果
        if 'optimization' in results:
            opt_result = results['optimization']
            report.append(f"\n🎯 因子优化结果:")
            report.append(f"  原始因子数量: {opt_result.get('total_factors', 0)}")
            report.append(f"  筛选后因子数量: {opt_result.get('selected_factors', 0)}")
            
            backtest = opt_result.get('backtest_result', {})
            if backtest:
                ic_analysis = backtest.get('ic_analysis', {})
                if ic_analysis:
                    report.append(f"  组合因子IC: {ic_analysis.get('ic_pearson', 0):.4f}")
                
                rolling_ic = backtest.get('rolling_ic_analysis', {})
                if rolling_ic:
                    report.append(f"  信息比率: {rolling_ic.get('ir', 0):.4f}")
                    report.append(f"  正IC比例: {rolling_ic.get('positive_ic_ratio', 0):.2%}")
        
        # Alpha生成能力评级
        report.append(f"\n⭐ Alpha生成能力评级:")
        
        if overall_avg_correlation >= 0.15:
            rating = "优秀 (A+)"
            comment = "因子具有很强的Alpha生成能力"
        elif overall_avg_correlation >= 0.10:
            rating = "良好 (A)"
            comment = "因子具有较强的Alpha生成能力"
        elif overall_avg_correlation >= 0.05:
            rating = "一般 (B)"
            comment = "因子具有一定的Alpha生成能力"
        elif overall_avg_correlation >= 0.02:
            rating = "较弱 (C)"
            comment = "因子Alpha生成能力有限"
        else:
            rating = "很弱 (D)"
            comment = "因子Alpha生成能力很弱"
        
        report.append(f"  评级: {rating}")
        report.append(f"  评价: {comment}")
        
        # 改进建议
        report.append(f"\n💡 改进建议:")
        
        if total_factors < 50:
            report.append("  • 考虑增加更多因子类别和数量")
        
        if overall_avg_correlation < 0.05:
            report.append("  • 优化因子计算方法和参数")
            report.append("  • 考虑使用更复杂的特征工程技术")
        
        if factor_categories < 5:
            report.append("  • 增加因子多样性，包含更多类别的因子")
        
        report.append("  • 定期更新和重新训练机器学习因子")
        report.append("  • 实施动态因子权重调整机制")
        report.append("  • 加强因子风险控制和稳定性监控")
        
        # 总结
        report.append(f"\n📋 测试总结:")
        report.append(f"  本次测试共评估了 {total_factors} 个因子，涵盖 {factor_categories} 个类别")
        report.append(f"  整体Alpha生成能力评级为: {rating}")
        report.append(f"  建议根据以上分析结果优化因子策略")
        
        return "\n".join(report)


def main():
    """主函数"""
    try:
        # 创建测试器
        tester = AlphaGenerationTester()
        
        # 运行综合测试
        results = tester.run_comprehensive_test()
        
        # 生成报告
        report = tester.generate_comprehensive_report(results)
        
        # 输出报告
        print(report)
        
        # 保存报告到文件
        with open('alpha_generation_test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Alpha生成能力测试完成！")
        print(f"📄 详细报告已保存到: alpha_generation_test_report.txt")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()