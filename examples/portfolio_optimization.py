#!/usr/bin/env python3
"""
投资组合优化分析工具

基于现代投资组合理论，帮助构建风险调整后的最优投资组合
注意：这仅是分析工具，不构成投资建议
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.indicators import TechnicalIndicators

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PortfolioOptimizer:
    """投资组合优化器"""
    
    def __init__(self):
        """初始化投资组合优化器"""
        self.data_cache_dir = Path("data_cache")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 股票池 - 不同板块的代表性股票
        self.stock_universe = {
            '科技股': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            '金融股': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            '医疗股': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            '消费股': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
            '工业股': ['BA', 'CAT', 'GE', 'MMM', 'HON']
        }
        
        print("🚀 投资组合优化器初始化完成")
    
    def _load_cached_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        加载缓存的股票数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票数据DataFrame，如果加载失败返回None
        """
        try:
            # 查找匹配的缓存文件
            cache_files = list(self.data_cache_dir.glob(f"ohlcv_{symbol}_*.csv"))
            if not cache_files:
                print(f"⚠️ 未找到 {symbol} 的缓存数据")
                return None
            
            # 使用最新的缓存文件
            cache_file = sorted(cache_files)[-1]
            
            # 读取CSV文件，跳过前两行元数据
            df = pd.read_csv(cache_file, skiprows=2)
            
            # 重新命名列（去掉Price列，重命名其他列）
            df = df.drop(df.columns[0], axis=1)  # 删除第一列Price
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # 设置日期索引
            df.index = pd.to_datetime(df.index)
            
            return df
            
        except Exception as e:
            print(f"❌ 加载 {symbol} 数据失败: {e}")
            return None
    
    def load_portfolio_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        加载投资组合数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            股票数据字典
        """
        print(f"📊 正在加载 {len(symbols)} 只股票的数据...")
        
        portfolio_data = {}
        for symbol in symbols:
            data = self._load_cached_stock_data(symbol)
            if data is not None:
                portfolio_data[symbol] = data
                print(f"✅ {symbol}: {len(data)} 条数据")
            else:
                print(f"❌ {symbol}: 数据加载失败")
        
        print(f"📈 成功加载 {len(portfolio_data)} 只股票数据")
        return portfolio_data
    
    def calculate_returns(self, portfolio_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算股票收益率
        
        Args:
            portfolio_data: 股票数据字典
            
        Returns:
            收益率DataFrame
        """
        returns_data = {}
        
        for symbol, data in portfolio_data.items():
            # 计算日收益率
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        # 合并所有收益率数据
        returns_df = pd.DataFrame(returns_data)
        
        # 只保留所有股票都有数据的日期
        returns_df = returns_df.dropna()
        
        print(f"📊 收益率数据: {len(returns_df)} 个交易日")
        return returns_df
    
    def calculate_portfolio_metrics(self, returns: pd.DataFrame) -> Dict:
        """
        计算投资组合指标
        
        Args:
            returns: 收益率DataFrame
            
        Returns:
            投资组合指标字典
        """
        # 年化收益率
        annual_returns = returns.mean() * 252
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility
        
        # 相关性矩阵
        correlation_matrix = returns.corr()
        
        # 协方差矩阵
        covariance_matrix = returns.cov() * 252  # 年化
        
        return {
            'annual_returns': annual_returns,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'correlation_matrix': correlation_matrix,
            'covariance_matrix': covariance_matrix
        }
    
    def optimize_portfolio(self, returns: pd.DataFrame, target_return: float = None) -> Dict:
        """
        优化投资组合
        
        Args:
            returns: 收益率DataFrame
            target_return: 目标收益率
            
        Returns:
            优化结果字典
        """
        print("🎯 正在进行投资组合优化...")
        
        metrics = self.calculate_portfolio_metrics(returns)
        
        n_assets = len(returns.columns)
        
        # 等权重组合
        equal_weights = np.array([1/n_assets] * n_assets)
        
        # 计算等权重组合的指标
        equal_portfolio_return = np.sum(equal_weights * metrics['annual_returns'])
        equal_portfolio_volatility = np.sqrt(
            np.dot(equal_weights.T, np.dot(metrics['covariance_matrix'], equal_weights))
        )
        equal_portfolio_sharpe = (equal_portfolio_return - 0.03) / equal_portfolio_volatility
        
        # 最小方差组合（简化版本）
        # 这里使用简化的方法，实际应用中可以使用scipy.optimize
        inv_volatility_weights = 1 / metrics['annual_volatility']
        min_var_weights = inv_volatility_weights / inv_volatility_weights.sum()
        
        min_var_portfolio_return = np.sum(min_var_weights * metrics['annual_returns'])
        min_var_portfolio_volatility = np.sqrt(
            np.dot(min_var_weights.T, np.dot(metrics['covariance_matrix'], min_var_weights))
        )
        min_var_portfolio_sharpe = (min_var_portfolio_return - 0.03) / min_var_portfolio_volatility
        
        # 最大夏普比率组合（简化版本）
        max_sharpe_weights = metrics['sharpe_ratio'] / metrics['sharpe_ratio'].sum()
        max_sharpe_weights = np.maximum(max_sharpe_weights, 0)  # 确保权重非负
        max_sharpe_weights = max_sharpe_weights / max_sharpe_weights.sum()  # 重新标准化
        
        max_sharpe_portfolio_return = np.sum(max_sharpe_weights * metrics['annual_returns'])
        max_sharpe_portfolio_volatility = np.sqrt(
            np.dot(max_sharpe_weights.T, np.dot(metrics['covariance_matrix'], max_sharpe_weights))
        )
        max_sharpe_portfolio_sharpe = (max_sharpe_portfolio_return - 0.03) / max_sharpe_portfolio_volatility
        
        return {
            'metrics': metrics,
            'portfolios': {
                'equal_weight': {
                    'weights': equal_weights,
                    'return': equal_portfolio_return,
                    'volatility': equal_portfolio_volatility,
                    'sharpe': equal_portfolio_sharpe
                },
                'min_variance': {
                    'weights': min_var_weights,
                    'return': min_var_portfolio_return,
                    'volatility': min_var_portfolio_volatility,
                    'sharpe': min_var_portfolio_sharpe
                },
                'max_sharpe': {
                    'weights': max_sharpe_weights,
                    'return': max_sharpe_portfolio_return,
                    'volatility': max_sharpe_portfolio_volatility,
                    'sharpe': max_sharpe_portfolio_sharpe
                }
            }
        }
    
    def create_portfolio_visualization(self, optimization_results: Dict, symbols: List[str]):
        """
        创建投资组合可视化图表
        
        Args:
            optimization_results: 优化结果
            symbols: 股票代码列表
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('投资组合优化分析', fontsize=16, fontweight='bold')
        
        # 1. 个股风险收益散点图
        ax1 = axes[0, 0]
        metrics = optimization_results['metrics']
        
        scatter = ax1.scatter(
            metrics['annual_volatility'] * 100,
            metrics['annual_returns'] * 100,
            c=metrics['sharpe_ratio'],
            cmap='RdYlGn',
            s=100,
            alpha=0.7
        )
        
        for i, symbol in enumerate(symbols):
            ax1.annotate(
                symbol,
                (metrics['annual_volatility'].iloc[i] * 100, metrics['annual_returns'].iloc[i] * 100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        ax1.set_xlabel('年化波动率 (%)')
        ax1.set_ylabel('年化收益率 (%)')
        ax1.set_title('个股风险收益分布')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='夏普比率')
        
        # 2. 投资组合权重对比
        ax2 = axes[0, 1]
        portfolios = optimization_results['portfolios']
        
        x = np.arange(len(symbols))
        width = 0.25
        
        ax2.bar(x - width, portfolios['equal_weight']['weights'], width, 
                label='等权重', alpha=0.8)
        ax2.bar(x, portfolios['min_variance']['weights'], width, 
                label='最小方差', alpha=0.8)
        ax2.bar(x + width, portfolios['max_sharpe']['weights'], width, 
                label='最大夏普', alpha=0.8)
        
        ax2.set_xlabel('股票')
        ax2.set_ylabel('权重')
        ax2.set_title('不同策略的投资组合权重')
        ax2.set_xticks(x)
        ax2.set_xticklabels(symbols, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 相关性热力图
        ax3 = axes[1, 0]
        correlation_matrix = metrics['correlation_matrix']
        
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            ax=ax3,
            cbar_kws={'label': '相关系数'}
        )
        ax3.set_title('股票相关性矩阵')
        
        # 4. 投资组合性能对比
        ax4 = axes[1, 1]
        
        portfolio_names = ['等权重', '最小方差', '最大夏普']
        returns = [p['return'] * 100 for p in portfolios.values()]
        volatilities = [p['volatility'] * 100 for p in portfolios.values()]
        sharpe_ratios = [p['sharpe'] for p in portfolios.values()]
        
        colors = ['skyblue', 'lightgreen', 'salmon']
        
        scatter = ax4.scatter(volatilities, returns, c=sharpe_ratios, 
                            s=200, cmap='RdYlGn', alpha=0.8)
        
        for i, name in enumerate(portfolio_names):
            ax4.annotate(name, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('年化波动率 (%)')
        ax4.set_ylabel('年化收益率 (%)')
        ax4.set_title('投资组合性能对比')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='夏普比率')
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.results_dir / 'portfolio_optimization.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 投资组合分析图表已保存: {chart_path}")
        
        plt.show()
    
    def generate_investment_report(self, optimization_results: Dict, symbols: List[str]):
        """
        生成投资分析报告
        
        Args:
            optimization_results: 优化结果
            symbols: 股票代码列表
        """
        print("\n" + "="*60)
        print("📊 投资组合优化分析报告")
        print("="*60)
        
        metrics = optimization_results['metrics']
        portfolios = optimization_results['portfolios']
        
        # 个股分析
        print("\n🏢 个股分析:")
        print("-" * 40)
        for symbol in symbols:
            annual_return = metrics['annual_returns'][symbol] * 100
            volatility = metrics['annual_volatility'][symbol] * 100
            sharpe = metrics['sharpe_ratio'][symbol]
            
            print(f"{symbol:>6}: 收益率 {annual_return:6.2f}% | "
                  f"波动率 {volatility:6.2f}% | 夏普比率 {sharpe:6.2f}")
        
        # 投资组合对比
        print("\n📈 投资组合策略对比:")
        print("-" * 40)
        
        for name, portfolio in portfolios.items():
            name_cn = {'equal_weight': '等权重组合', 
                      'min_variance': '最小方差组合', 
                      'max_sharpe': '最大夏普组合'}[name]
            
            print(f"\n{name_cn}:")
            print(f"  预期年化收益率: {portfolio['return']*100:.2f}%")
            print(f"  预期年化波动率: {portfolio['volatility']*100:.2f}%")
            print(f"  夏普比率: {portfolio['sharpe']:.2f}")
            
            print("  权重分配:")
            for i, symbol in enumerate(symbols):
                weight = portfolio['weights'][i] * 100
                if weight > 1:  # 只显示权重大于1%的股票
                    print(f"    {symbol}: {weight:.1f}%")
        
        # 风险分析
        print("\n⚠️ 风险分析:")
        print("-" * 40)
        
        # 计算投资组合间的相关性
        avg_correlation = metrics['correlation_matrix'].values[
            np.triu_indices_from(metrics['correlation_matrix'].values, k=1)
        ].mean()
        
        print(f"平均股票相关性: {avg_correlation:.3f}")
        
        if avg_correlation > 0.7:
            print("⚠️ 高相关性警告: 股票间相关性较高，分散化效果有限")
        elif avg_correlation > 0.5:
            print("⚠️ 中等相关性: 存在一定的集中风险")
        else:
            print("✅ 良好分散化: 股票间相关性较低")
        
        # 投资建议
        print("\n💡 分析总结:")
        print("-" * 40)
        
        best_sharpe_portfolio = max(portfolios.items(), key=lambda x: x[1]['sharpe'])
        best_return_portfolio = max(portfolios.items(), key=lambda x: x[1]['return'])
        min_risk_portfolio = min(portfolios.items(), key=lambda x: x[1]['volatility'])
        
        portfolio_names = {'equal_weight': '等权重组合', 
                          'min_variance': '最小方差组合', 
                          'max_sharpe': '最大夏普组合'}
        
        print(f"• 风险调整后最佳表现: {portfolio_names[best_sharpe_portfolio[0]]}")
        print(f"• 最高预期收益: {portfolio_names[best_return_portfolio[0]]}")
        print(f"• 最低风险: {portfolio_names[min_risk_portfolio[0]]}")
        
        print("\n📝 投资建议框架:")
        print("1. 根据个人风险承受能力选择合适的投资组合策略")
        print("2. 定期重新平衡投资组合权重")
        print("3. 关注宏观经济环境变化")
        print("4. 考虑加入不同资产类别以进一步分散风险")
        
        print("\n" + "="*60)
        print("⚠️ 重要提醒")
        print("="*60)
        print("1. 以上分析基于历史数据，不保证未来表现")
        print("2. 投资有风险，请根据自身情况谨慎决策")
        print("3. 建议咨询专业投资顾问")
        print("4. 请做好风险管理和资金配置")


def main():
    """主函数"""
    print("🚀 启动投资组合优化分析...")
    
    optimizer = PortfolioOptimizer()
    
    # 选择分析的股票（来自不同板块）
    selected_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'JNJ']
    
    print(f"📊 分析股票池: {', '.join(selected_stocks)}")
    
    # 加载数据
    portfolio_data = optimizer.load_portfolio_data(selected_stocks)
    
    if len(portfolio_data) < 3:
        print("❌ 可用数据不足，无法进行投资组合优化")
        return
    
    # 计算收益率
    returns = optimizer.calculate_returns(portfolio_data)
    
    # 投资组合优化
    optimization_results = optimizer.optimize_portfolio(returns)
    
    # 生成报告
    optimizer.generate_investment_report(optimization_results, list(portfolio_data.keys()))
    
    # 创建可视化
    optimizer.create_portfolio_visualization(optimization_results, list(portfolio_data.keys()))
    
    print("\n✅ 投资组合优化分析完成！")
    print("📝 请记住：这只是分析工具，不构成投资建议！")


if __name__ == "__main__":
    main()