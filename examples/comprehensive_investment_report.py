#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合投资报告生成器

整合投资分析框架、投资组合优化和市场情绪分析，
生成完整的投资分析报告。

作者: 量化交易系统
日期: 2024年
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_manager import DataManager
from src.factors.engine import FactorEngine
from src.utils.indicators import TechnicalIndicators
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveInvestmentReporter:
    """
    综合投资报告生成器
    
    整合多个分析工具，生成完整的投资分析报告
    """
    
    def __init__(self):
        """初始化报告生成器"""
        self.data_manager = DataManager()
        self.factor_engine = FactorEngine()
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 分析股票池
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'NFLX', 'JPM', 'JNJ', 'PG', 'KO']
        
        # 板块分类
        self.sectors = {
            '科技股': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            '新兴科技': ['TSLA', 'META', 'AMZN', 'NFLX'],
            '金融股': ['JPM'],
            '医疗消费': ['JNJ', 'PG', 'KO']
        }
        
        self.data = {}
        self.analysis_results = {}
        
    def load_data(self):
        """加载股票数据"""
        print("📊 正在加载股票数据...")
        
        try:
            # 从data_cache目录加载CSV缓存文件
            from pathlib import Path
            cache_dir = Path("data_cache")
            
            if not cache_dir.exists():
                print("⚠️ 未找到data_cache目录，请先运行数据获取脚本")
                return False
            
            loaded_count = 0
            for symbol in self.symbols:
                # 查找该股票的缓存文件
                cache_files = list(cache_dir.glob(f"ohlcv_{symbol}_*.csv"))
                
                if cache_files:
                    # 使用最新的缓存文件
                    cache_file = cache_files[0]
                    df = pd.read_csv(cache_file)
                    
                    # 处理多级列名格式
                    # 第一行是Price,Open,High,Low,Close,Volume
                    # 第二行是Ticker,AAPL,AAPL,AAPL,AAPL,AAPL
                    # 第三行是Date,,,,,
                    # 从第四行开始是实际数据
                    
                    # 跳过前三行，重新设置列名
                    df = df.iloc[3:].copy()
                    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # 转换日期和数值类型
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    # 转换数据类型为数值
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 删除包含NaN的行
                    df = df.dropna()
                    
                    # 确保数据有效
                    if len(df) > 0 and 'Close' in df.columns and pd.api.types.is_numeric_dtype(df['Close']):
                        self.data[symbol] = df
                        loaded_count += 1
                        print(f"   ✓ {symbol}: {len(df)} 条记录")
                    else:
                        print(f"   ⚠️ {symbol}: 数据格式不正确或为空")
                else:
                    print(f"   ❌ {symbol}: 未找到缓存文件")
            
            if loaded_count > 0:
                print(f"✅ 成功加载 {loaded_count} 只股票的缓存数据")
                return True
            else:
                print("⚠️ 未找到任何有效的缓存数据")
                return False
                
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def market_trend_analysis(self):
        """市场趋势分析"""
        print("\n📈 正在进行市场趋势分析...")
        
        trend_results = {}
        
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
                
            df = self.data[symbol].copy()
            
            # 计算技术指标
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = TechnicalIndicators.rsi(df['Close'])
            
            # 布林带
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
            df['BB_Upper'] = bb_upper
            df['BB_Lower'] = bb_lower
            
            # 最新数据
            latest = df.iloc[-1]
            
            # 趋势判断
            trend_score = 0
            trend_signals = []
            
            # SMA趋势
            if latest['Close'] > latest['SMA_20']:
                trend_score += 1
                trend_signals.append("价格在20日均线上方")
            if latest['SMA_20'] > latest['SMA_50']:
                trend_score += 1
                trend_signals.append("短期均线在长期均线上方")
            
            # RSI状态
            if 30 <= latest['RSI'] <= 70:
                trend_score += 1
                trend_signals.append("RSI处于正常区间")
            elif latest['RSI'] < 30:
                trend_signals.append("RSI超卖")
            elif latest['RSI'] > 70:
                trend_signals.append("RSI超买")
            
            # 布林带位置
            bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
            if 0.2 <= bb_position <= 0.8:
                trend_score += 1
                trend_signals.append("价格在布林带中轨附近")
            
            trend_results[symbol] = {
                'score': trend_score,
                'signals': trend_signals,
                'rsi': latest['RSI'],
                'bb_position': bb_position,
                'price_change': ((latest['Close'] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100) if len(df) > 21 else 0
            }
        
        self.analysis_results['market_trend'] = trend_results
        return trend_results
    
    def sector_rotation_analysis(self):
        """板块轮动分析"""
        print("🔄 正在进行板块轮动分析...")
        
        sector_performance = {}
        
        for sector, symbols in self.sectors.items():
            sector_returns = []
            
            for symbol in symbols:
                if symbol in self.data and len(self.data[symbol]) > 21:
                    df = self.data[symbol]
                    # 计算21日收益率
                    return_21d = (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100
                    sector_returns.append(return_21d)
            
            if sector_returns:
                avg_return = np.mean(sector_returns)
                sector_performance[sector] = {
                    'return': avg_return,
                    'count': len(sector_returns),
                    'symbols': symbols[:len(sector_returns)]
                }
        
        # 排序
        sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1]['return'], reverse=True)
        
        self.analysis_results['sector_rotation'] = dict(sorted_sectors)
        return dict(sorted_sectors)
    
    def multi_factor_analysis(self):
        """多因子分析"""
        print("🧮 正在进行多因子分析...")
        
        factor_scores = {}
        
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
                
            df = self.data[symbol].copy()
            
            # 计算各种因子
            df['RSI'] = TechnicalIndicators.rsi(df['Close'])
            df['MACD'], df['MACD_Signal'], _ = TechnicalIndicators.macd(df['Close'])
            
            # 布林带
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
            df['BB_Upper'] = bb_upper
            df['BB_Lower'] = bb_lower
            
            # 最新数据
            latest = df.iloc[-1]
            
            # 因子评分
            score = 0
            
            # 技术因子
            # RSI评分 (30-70为正常，偏离给负分)
            if 40 <= latest['RSI'] <= 60:
                score += 20
            elif 30 <= latest['RSI'] <= 70:
                score += 10
            elif latest['RSI'] < 30:
                score += 5  # 超卖可能反弹
            
            # MACD信号
            if latest['MACD'] > latest['MACD_Signal']:
                score += 15
            
            # 布林带位置
            bb_pos = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
            if 0.3 <= bb_pos <= 0.7:
                score += 15
            elif bb_pos < 0.2:
                score += 10  # 可能超卖
            
            # 价格动量
            if len(df) > 21:
                momentum_21 = (latest['Close'] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100
                if momentum_21 > 5:
                    score += 20
                elif momentum_21 > 0:
                    score += 10
                elif momentum_21 > -5:
                    score += 5
            
            # 波动率因子
            if len(df) > 21:
                volatility = df['Close'].pct_change().rolling(21).std().iloc[-1] * np.sqrt(252) * 100
                if volatility < 25:  # 低波动率
                    score += 10
                elif volatility < 35:
                    score += 5
            
            # 成交量因子
            if len(df) > 21:
                avg_volume = df['Volume'].rolling(21).mean().iloc[-1]
                recent_volume = df['Volume'].iloc[-5:].mean()
                volume_ratio = recent_volume / avg_volume
                if volume_ratio > 1.2:  # 成交量放大
                    score += 10
                elif volume_ratio > 1.0:
                    score += 5
            
            factor_scores[symbol] = {
                'total_score': score,
                'rsi': latest['RSI'],
                'macd_signal': 'MACD > Signal' if latest['MACD'] > latest['MACD_Signal'] else 'MACD < Signal',
                'bb_position': bb_pos,
                'momentum_21d': momentum_21 if len(df) > 21 else 0,
                'volatility': volatility if len(df) > 21 else 0
            }
        
        # 排序
        sorted_scores = sorted(factor_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        self.analysis_results['multi_factor'] = dict(sorted_scores)
        return dict(sorted_scores)
    
    def portfolio_optimization(self):
        """投资组合优化"""
        print("📊 正在进行投资组合优化...")
        
        # 计算收益率矩阵
        returns_data = {}
        
        for symbol in self.symbols:
            if symbol in self.data and len(self.data[symbol]) > 252:
                df = self.data[symbol]
                returns = df['Close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if not returns_data:
            print("❌ 没有足够的数据进行投资组合优化")
            return {}
        
        # 创建收益率DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # 计算年化收益率和协方差矩阵
        annual_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # 等权重组合
        n_assets = len(returns_df.columns)
        equal_weights = np.array([1/n_assets] * n_assets)
        
        equal_return = np.sum(annual_returns * equal_weights)
        equal_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
        equal_sharpe = equal_return / equal_vol if equal_vol > 0 else 0
        
        # 最小方差组合 (简化版)
        inv_cov = np.linalg.pinv(cov_matrix)
        ones = np.ones((n_assets, 1))
        min_var_weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
        min_var_weights = min_var_weights.flatten()
        
        min_var_return = np.sum(annual_returns * min_var_weights)
        min_var_vol = np.sqrt(np.dot(min_var_weights.T, np.dot(cov_matrix, min_var_weights)))
        min_var_sharpe = min_var_return / min_var_vol if min_var_vol > 0 else 0
        
        portfolio_results = {
            'equal_weight': {
                'weights': dict(zip(returns_df.columns, equal_weights)),
                'expected_return': equal_return,
                'volatility': equal_vol,
                'sharpe_ratio': equal_sharpe
            },
            'min_variance': {
                'weights': dict(zip(returns_df.columns, min_var_weights)),
                'expected_return': min_var_return,
                'volatility': min_var_vol,
                'sharpe_ratio': min_var_sharpe
            }
        }
        
        self.analysis_results['portfolio'] = portfolio_results
        return portfolio_results
    
    def market_sentiment_analysis(self):
        """市场情绪分析"""
        print("😊 正在进行市场情绪分析...")
        
        sentiment_scores = {}
        
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
                
            df = self.data[symbol].copy()
            
            # 计算技术指标
            df['RSI'] = TechnicalIndicators.rsi(df['Close'])
            df['MACD'], df['MACD_Signal'], _ = TechnicalIndicators.macd(df['Close'])
            
            # 布林带
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
            df['BB_Upper'] = bb_upper
            df['BB_Lower'] = bb_lower
            
            # 移动平均线
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            
            # 最新数据
            latest = df.iloc[-1]
            
            # 情绪评分 (-100 到 +100)
            sentiment = 0
            
            # RSI情绪
            if latest['RSI'] > 70:
                sentiment += 20  # 超买，乐观
            elif latest['RSI'] > 60:
                sentiment += 10
            elif latest['RSI'] < 30:
                sentiment -= 20  # 超卖，悲观
            elif latest['RSI'] < 40:
                sentiment -= 10
            
            # MACD情绪
            if latest['MACD'] > latest['MACD_Signal']:
                sentiment += 15
            else:
                sentiment -= 15
            
            # 价格相对布林带位置
            bb_pos = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
            if bb_pos > 0.8:
                sentiment += 15
            elif bb_pos > 0.6:
                sentiment += 5
            elif bb_pos < 0.2:
                sentiment -= 15
            elif bb_pos < 0.4:
                sentiment -= 5
            
            # 价格相对移动平均线
            if latest['Close'] > latest['SMA_20']:
                sentiment += 10
            else:
                sentiment -= 10
                
            if latest['SMA_20'] > latest['SMA_50']:
                sentiment += 10
            else:
                sentiment -= 10
            
            # 价格动量
            if len(df) > 21:
                momentum = (latest['Close'] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100
                if momentum > 10:
                    sentiment += 20
                elif momentum > 5:
                    sentiment += 10
                elif momentum > 0:
                    sentiment += 5
                elif momentum > -5:
                    sentiment -= 5
                elif momentum > -10:
                    sentiment -= 10
                else:
                    sentiment -= 20
            
            # 情绪标签
            if sentiment > 20:
                sentiment_label = "非常乐观"
            elif sentiment > 10:
                sentiment_label = "乐观"
            elif sentiment > -10:
                sentiment_label = "中性"
            elif sentiment > -20:
                sentiment_label = "悲观"
            else:
                sentiment_label = "非常悲观"
            
            sentiment_scores[symbol] = {
                'score': sentiment,
                'label': sentiment_label,
                'rsi': latest['RSI'],
                'bb_position': bb_pos,
                'momentum': momentum if len(df) > 21 else 0
            }
        
        # 计算整体市场情绪
        overall_sentiment = np.mean([s['score'] for s in sentiment_scores.values()])
        
        # 板块情绪
        sector_sentiment = {}
        for sector, symbols in self.sectors.items():
            sector_scores = [sentiment_scores[s]['score'] for s in symbols if s in sentiment_scores]
            if sector_scores:
                sector_sentiment[sector] = np.mean(sector_scores)
        
        sentiment_results = {
            'overall': overall_sentiment,
            'individual': sentiment_scores,
            'sectors': sector_sentiment
        }
        
        self.analysis_results['sentiment'] = sentiment_results
        return sentiment_results
    
    def generate_investment_recommendations(self):
        """生成投资建议"""
        print("💡 正在生成投资建议...")
        
        recommendations = {
            'market_outlook': '',
            'sector_recommendations': [],
            'stock_picks': {
                'strong_buy': [],
                'buy': [],
                'hold': [],
                'avoid': []
            },
            'portfolio_strategy': '',
            'risk_warnings': []
        }
        
        # 市场展望
        if 'sentiment' in self.analysis_results:
            overall_sentiment = self.analysis_results['sentiment']['overall']
            if overall_sentiment > 10:
                recommendations['market_outlook'] = "市场情绪整体乐观，但需注意估值风险"
            elif overall_sentiment > -10:
                recommendations['market_outlook'] = "市场情绪中性，建议保持谨慎乐观"
            else:
                recommendations['market_outlook'] = "市场情绪偏悲观，可关注超跌反弹机会"
        
        # 板块建议
        if 'sector_rotation' in self.analysis_results:
            sorted_sectors = list(self.analysis_results['sector_rotation'].items())
            if sorted_sectors:
                top_sector = sorted_sectors[0]
                recommendations['sector_recommendations'].append(f"重点关注{top_sector[0]}，近期表现强势")
                
                if len(sorted_sectors) > 1:
                    weak_sector = sorted_sectors[-1]
                    recommendations['sector_recommendations'].append(f"谨慎对待{weak_sector[0]}，表现相对疲弱")
        
        # 个股推荐
        if 'multi_factor' in self.analysis_results:
            sorted_stocks = list(self.analysis_results['multi_factor'].items())
            
            for i, (symbol, data) in enumerate(sorted_stocks):
                score = data['total_score']
                
                if score >= 80:
                    recommendations['stock_picks']['strong_buy'].append(symbol)
                elif score >= 60:
                    recommendations['stock_picks']['buy'].append(symbol)
                elif score >= 40:
                    recommendations['stock_picks']['hold'].append(symbol)
                else:
                    recommendations['stock_picks']['avoid'].append(symbol)
        
        # 投资组合策略
        if 'portfolio' in self.analysis_results:
            min_var_sharpe = self.analysis_results['portfolio']['min_variance']['sharpe_ratio']
            equal_sharpe = self.analysis_results['portfolio']['equal_weight']['sharpe_ratio']
            
            if min_var_sharpe > equal_sharpe:
                recommendations['portfolio_strategy'] = "建议采用最小方差组合策略，更好的风险调整收益"
            else:
                recommendations['portfolio_strategy'] = "建议采用等权重组合策略，简单有效的分散化"
        
        # 风险提示
        recommendations['risk_warnings'] = [
            "分析基于历史数据，不能保证未来表现",
            "市场环境变化可能影响分析结果",
            "建议结合基本面分析和宏观环境",
            "请根据个人风险承受能力调整仓位",
            "定期重新评估和调整投资组合"
        ]
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("📊 正在创建可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建综合分析图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('综合投资分析报告', fontsize=16, fontweight='bold')
        
        # 1. 个股多因子评分
        if 'multi_factor' in self.analysis_results:
            symbols = list(self.analysis_results['multi_factor'].keys())[:8]  # 取前8只
            scores = [self.analysis_results['multi_factor'][s]['total_score'] for s in symbols]
            
            bars = axes[0, 0].bar(symbols, scores, color=['green' if s >= 60 else 'orange' if s >= 40 else 'red' for s in scores])
            axes[0, 0].set_title('个股多因子评分排名')
            axes[0, 0].set_ylabel('综合评分')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, score in zip(bars, scores):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                               f'{score:.0f}', ha='center', va='bottom')
        
        # 2. 板块表现对比
        if 'sector_rotation' in self.analysis_results:
            sectors = list(self.analysis_results['sector_rotation'].keys())
            returns = [self.analysis_results['sector_rotation'][s]['return'] for s in sectors]
            
            colors = ['green' if r > 0 else 'red' for r in returns]
            bars = axes[0, 1].bar(sectors, returns, color=colors, alpha=0.7)
            axes[0, 1].set_title('板块表现对比 (21日收益率)')
            axes[0, 1].set_ylabel('收益率 (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for bar, ret in zip(bars, returns):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + (0.5 if ret > 0 else -1), 
                               f'{ret:.1f}%', ha='center', va='bottom' if ret > 0 else 'top')
        
        # 3. 市场情绪分布
        if 'sentiment' in self.analysis_results:
            sentiment_data = self.analysis_results['sentiment']['individual']
            symbols = list(sentiment_data.keys())
            sentiments = [sentiment_data[s]['score'] for s in symbols]
            
            colors = ['darkgreen' if s > 20 else 'green' if s > 0 else 'orange' if s > -20 else 'red' for s in sentiments]
            bars = axes[1, 0].bar(symbols, sentiments, color=colors, alpha=0.7)
            axes[1, 0].set_title('个股市场情绪评分')
            axes[1, 0].set_ylabel('情绪评分')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加情绪区间线
            axes[1, 0].axhline(y=20, color='green', linestyle='--', alpha=0.5, label='乐观')
            axes[1, 0].axhline(y=-20, color='red', linestyle='--', alpha=0.5, label='悲观')
            axes[1, 0].legend()
        
        # 4. 投资组合对比
        if 'portfolio' in self.analysis_results:
            portfolio_data = self.analysis_results['portfolio']
            strategies = ['等权重组合', '最小方差组合']
            returns = [portfolio_data['equal_weight']['expected_return'] * 100,
                      portfolio_data['min_variance']['expected_return'] * 100]
            volatilities = [portfolio_data['equal_weight']['volatility'] * 100,
                           portfolio_data['min_variance']['volatility'] * 100]
            
            x = np.arange(len(strategies))
            width = 0.35
            
            bars1 = axes[1, 1].bar(x - width/2, returns, width, label='预期收益率 (%)', alpha=0.8)
            bars2 = axes[1, 1].bar(x + width/2, volatilities, width, label='波动率 (%)', alpha=0.8)
            
            axes[1, 1].set_title('投资组合策略对比')
            axes[1, 1].set_ylabel('百分比 (%)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(strategies)
            axes[1, 1].legend()
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + 1,
                                   f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/comprehensive_investment_report.png', dpi=300, bbox_inches='tight')
        print("📊 综合分析图表已保存: results/comprehensive_investment_report.png")
        
        plt.show()
    
    def generate_report(self):
        """生成完整的投资分析报告"""
        print("=" * 60)
        print("📋 综合投资分析报告")
        print("=" * 60)
        print(f"📅 报告日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 分析股票: {', '.join(self.symbols)}")
        print("=" * 60)
        
        # 市场趋势分析结果
        if 'market_trend' in self.analysis_results:
            print("\n📈 市场趋势分析")
            print("-" * 40)
            trend_data = self.analysis_results['market_trend']
            
            # 按趋势评分排序
            sorted_trends = sorted(trend_data.items(), key=lambda x: x[1]['score'], reverse=True)
            
            for symbol, data in sorted_trends[:5]:  # 显示前5名
                trend_strength = "强势" if data['score'] >= 3 else "中性" if data['score'] >= 2 else "弱势"
                print(f"  {symbol:>6}: {data['score']}/4 | {trend_strength:>4} | RSI: {data['rsi']:.1f} | 21日涨跌: {data['price_change']:+.1f}%")
        
        # 板块轮动分析结果
        if 'sector_rotation' in self.analysis_results:
            print("\n🔄 板块轮动分析")
            print("-" * 40)
            for sector, data in self.analysis_results['sector_rotation'].items():
                emoji = "📈" if data['return'] > 5 else "📊" if data['return'] > 0 else "📉"
                print(f"  {sector:>8}: {data['return']:+6.1f}% {emoji} ({data['count']}只股票)")
        
        # 个股多因子分析结果
        if 'multi_factor' in self.analysis_results:
            print("\n🧮 个股多因子评分排名")
            print("-" * 40)
            factor_data = self.analysis_results['multi_factor']
            
            for i, (symbol, data) in enumerate(list(factor_data.items())[:8]):
                rating = "★★★★★" if data['total_score'] >= 80 else "★★★★☆" if data['total_score'] >= 60 else "★★★☆☆" if data['total_score'] >= 40 else "★★☆☆☆"
                print(f"  {i+1:>2}. {symbol:>6}: {data['total_score']:>3.0f}分 {rating} | RSI: {data['rsi']:>5.1f} | 动量: {data['momentum_21d']:+5.1f}%")
        
        # 投资组合优化结果
        if 'portfolio' in self.analysis_results:
            print("\n📊 投资组合优化结果")
            print("-" * 40)
            portfolio_data = self.analysis_results['portfolio']
            
            for strategy, data in portfolio_data.items():
                strategy_name = "等权重组合" if strategy == "equal_weight" else "最小方差组合"
                print(f"\n  {strategy_name}:")
                print(f"    预期年化收益率: {data['expected_return']*100:>6.1f}%")
                print(f"    预期年化波动率: {data['volatility']*100:>6.1f}%")
                print(f"    夏普比率:       {data['sharpe_ratio']:>6.2f}")
                
                # 显示权重最大的前5只股票
                sorted_weights = sorted(data['weights'].items(), key=lambda x: x[1], reverse=True)
                print("    主要持仓:")
                for symbol, weight in sorted_weights[:5]:
                    print(f"      {symbol}: {weight*100:>5.1f}%")
        
        # 市场情绪分析结果
        if 'sentiment' in self.analysis_results:
            print("\n😊 市场情绪分析")
            print("-" * 40)
            sentiment_data = self.analysis_results['sentiment']
            
            overall = sentiment_data['overall']
            overall_label = "非常乐观" if overall > 20 else "乐观" if overall > 10 else "中性" if overall > -10 else "悲观" if overall > -20 else "非常悲观"
            emoji = "😄" if overall > 20 else "😊" if overall > 10 else "😐" if overall > -10 else "😟" if overall > -20 else "😰"
            
            print(f"  整体市场情绪: {overall:+.1f} ({overall_label}) {emoji}")
            
            print("\n  板块情绪对比:")
            for sector, score in sentiment_data['sectors'].items():
                emoji = "📈" if score > 10 else "📊" if score > -10 else "📉"
                print(f"    {sector:>8}: {score:+6.1f} {emoji}")
            
            print("\n  个股情绪排名:")
            sorted_sentiment = sorted(sentiment_data['individual'].items(), key=lambda x: x[1]['score'], reverse=True)
            for i, (symbol, data) in enumerate(sorted_sentiment[:8]):
                emoji = "😄" if data['score'] > 20 else "😊" if data['score'] > 0 else "😐" if data['score'] > -20 else "😟"
                print(f"    {i+1:>2}. {symbol:>6}: {data['score']:+4.0f}分 | {data['label']:>6} {emoji}")
        
        # 投资建议
        if 'recommendations' in self.analysis_results:
            print("\n💡 投资建议")
            print("-" * 40)
            rec = self.analysis_results['recommendations']
            
            print(f"  市场展望: {rec['market_outlook']}")
            
            if rec['sector_recommendations']:
                print("\n  板块建议:")
                for suggestion in rec['sector_recommendations']:
                    print(f"    • {suggestion}")
            
            print("\n  个股推荐:")
            if rec['stock_picks']['strong_buy']:
                print(f"    强烈推荐: {', '.join(rec['stock_picks']['strong_buy'])}")
            if rec['stock_picks']['buy']:
                print(f"    推荐买入: {', '.join(rec['stock_picks']['buy'])}")
            if rec['stock_picks']['hold']:
                print(f"    建议持有: {', '.join(rec['stock_picks']['hold'])}")
            if rec['stock_picks']['avoid']:
                print(f"    建议回避: {', '.join(rec['stock_picks']['avoid'])}")
            
            print(f"\n  组合策略: {rec['portfolio_strategy']}")
            
            print("\n  风险提示:")
            for warning in rec['risk_warnings']:
                print(f"    ⚠️ {warning}")
        
        print("\n" + "=" * 60)
        print("📊 分析完成！图表已保存至 results/ 目录")
        print("⚠️ 重要提醒: 本分析仅供参考，不构成投资建议！")
        print("=" * 60)
    
    def run_comprehensive_analysis(self):
        """运行完整的综合分析"""
        print("🚀 启动综合投资分析...")
        
        # 1. 加载数据
        if not self.load_data():
            return
        
        # 2. 执行各项分析
        self.market_trend_analysis()
        self.sector_rotation_analysis()
        self.multi_factor_analysis()
        self.portfolio_optimization()
        self.market_sentiment_analysis()
        
        # 3. 生成投资建议
        self.generate_investment_recommendations()
        
        # 4. 创建可视化
        self.create_visualizations()
        
        # 5. 生成报告
        self.generate_report()

def main():
    """主函数"""
    print("🎯 综合投资分析报告生成器")
    print("=" * 50)
    
    # 创建分析器
    reporter = ComprehensiveInvestmentReporter()
    
    # 运行综合分析
    reporter.run_comprehensive_analysis()
    
    print("\n✅ 综合投资分析完成！")
    print("\n📝 请记住：")
    print("   • 这只是分析工具，不构成投资建议")
    print("   • 投资有风险，请谨慎决策")
    print("   • 建议结合基本面分析和宏观环境")
    print("   • 请根据个人风险承受能力调整策略")

if __name__ == "__main__":
    main()