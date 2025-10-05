#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资策略推荐系统
基于量化分析结果为用户提供未来3个月的投资策略建议

作者: 量化交易系统
创建时间: 2025-01-04
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class InvestmentStrategyRecommendation:
    """投资策略推荐系统"""
    
    def __init__(self):
        """初始化推荐系统"""
        self.data = {}
        self.analysis_results = {}
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 股票池 - 主要关注的优质股票
        self.stock_universe = {
            # 科技股
            'AAPL': {'sector': '科技', 'name': '苹果', 'risk_level': '中等'},
            'MSFT': {'sector': '科技', 'name': '微软', 'risk_level': '中等'},
            'GOOGL': {'sector': '科技', 'name': '谷歌', 'risk_level': '中等'},
            'NVDA': {'sector': '科技', 'name': '英伟达', 'risk_level': '高'},
            'TSLA': {'sector': '新能源', 'name': '特斯拉', 'risk_level': '高'},
            'META': {'sector': '科技', 'name': 'Meta', 'risk_level': '中等'},
            
            # 金融股
            'JPM': {'sector': '金融', 'name': '摩根大通', 'risk_level': '中等'},
            
            # 消费股
            'JNJ': {'sector': '医疗', 'name': '强生', 'risk_level': '低'},
            'KO': {'sector': '消费', 'name': '可口可乐', 'risk_level': '低'},
            'PG': {'sector': '消费', 'name': '宝洁', 'risk_level': '低'},
            
            # 流媒体
            'NFLX': {'sector': '媒体', 'name': '奈飞', 'risk_level': '中高'},
            'AMZN': {'sector': '科技', 'name': '亚马逊', 'risk_level': '中等'},
        }
        
        # 投资策略模板
        self.strategy_templates = {
            'conservative': {
                'name': '稳健型策略',
                'risk_tolerance': '低',
                'target_return': '8-12%',
                'max_drawdown': '10%',
                'sectors': ['医疗', '消费', '金融'],
                'allocation': {'股票': 60, '债券': 30, '现金': 10}
            },
            'balanced': {
                'name': '平衡型策略',
                'risk_tolerance': '中等',
                'target_return': '12-18%',
                'max_drawdown': '15%',
                'sectors': ['科技', '医疗', '消费', '金融'],
                'allocation': {'股票': 70, '债券': 20, '现金': 10}
            },
            'aggressive': {
                'name': '积极型策略',
                'risk_tolerance': '高',
                'target_return': '18-25%',
                'max_drawdown': '25%',
                'sectors': ['科技', '新能源', '成长股'],
                'allocation': {'股票': 85, '债券': 10, '现金': 5}
            }
        }

    def load_cached_data(self):
        """加载缓存的股票数据"""
        print("📊 正在加载股票数据...")
        
        try:
            cache_dir = Path("data_cache")
            if not cache_dir.exists():
                print("⚠️ 未找到data_cache目录")
                return False
            
            loaded_count = 0
            for symbol in self.stock_universe.keys():
                cache_files = list(cache_dir.glob(f"ohlcv_{symbol}_*.csv"))
                
                if cache_files:
                    cache_file = cache_files[0]
                    df = pd.read_csv(cache_file)
                    
                    # 处理CSV格式
                    df = df.iloc[3:].copy()
                    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    
                    if len(df) > 0:
                        self.data[symbol] = df
                        loaded_count += 1
                        print(f"   ✓ {symbol}: {len(df)} 条记录")
            
            print(f"✅ 成功加载 {loaded_count} 只股票数据")
            return loaded_count > 0
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False

    def analyze_market_conditions(self):
        """分析当前市场环境"""
        print("\n🔍 分析市场环境...")
        
        market_analysis = {
            'trend': 'neutral',
            'volatility': 'medium',
            'sector_rotation': {},
            'risk_sentiment': 'neutral'
        }
        
        if not self.data:
            return market_analysis
        
        # 计算市场整体趋势
        recent_returns = []
        volatilities = []
        
        for symbol, df in self.data.items():
            if len(df) >= 20:
                # 计算近期收益率
                recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
                recent_returns.append(recent_return)
                
                # 计算波动率
                daily_returns = df['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252) * 100
                volatilities.append(volatility)
        
        if recent_returns:
            avg_return = np.mean(recent_returns)
            avg_volatility = np.mean(volatilities)
            
            # 判断市场趋势
            if avg_return > 5:
                market_analysis['trend'] = 'bullish'
            elif avg_return < -5:
                market_analysis['trend'] = 'bearish'
            else:
                market_analysis['trend'] = 'neutral'
            
            # 判断波动率水平
            if avg_volatility > 30:
                market_analysis['volatility'] = 'high'
            elif avg_volatility < 20:
                market_analysis['volatility'] = 'low'
            else:
                market_analysis['volatility'] = 'medium'
        
        # 分析板块表现
        sector_performance = {}
        for symbol, info in self.stock_universe.items():
            if symbol in self.data and len(self.data[symbol]) >= 20:
                sector = info['sector']
                recent_return = (self.data[symbol]['Close'].iloc[-1] / 
                               self.data[symbol]['Close'].iloc[-20] - 1) * 100
                
                if sector not in sector_performance:
                    sector_performance[sector] = []
                sector_performance[sector].append(recent_return)
        
        for sector, returns in sector_performance.items():
            market_analysis['sector_rotation'][sector] = np.mean(returns)
        
        self.analysis_results['market_conditions'] = market_analysis
        return market_analysis

    def calculate_stock_scores(self):
        """计算个股评分"""
        print("📈 计算个股评分...")
        
        stock_scores = {}
        
        for symbol, df in self.data.items():
            if len(df) < 50:
                continue
                
            score = 0
            details = {}
            
            # 1. 趋势评分 (30分)
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            trend_score = 0
            if current_price > sma_20 > sma_50:
                trend_score = 30
            elif current_price > sma_20:
                trend_score = 20
            elif current_price > sma_50:
                trend_score = 10
            
            details['trend_score'] = trend_score
            score += trend_score
            
            # 2. 动量评分 (25分)
            returns_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
            returns_3m = (df['Close'].iloc[-1] / df['Close'].iloc[-60] - 1) * 100 if len(df) >= 60 else 0
            
            momentum_score = 0
            if returns_1m > 10:
                momentum_score += 15
            elif returns_1m > 5:
                momentum_score += 10
            elif returns_1m > 0:
                momentum_score += 5
            
            if returns_3m > 15:
                momentum_score += 10
            elif returns_3m > 5:
                momentum_score += 5
            
            details['momentum_score'] = momentum_score
            details['returns_1m'] = returns_1m
            details['returns_3m'] = returns_3m
            score += momentum_score
            
            # 3. 技术指标评分 (25分)
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            rsi_score = 0
            if 30 <= current_rsi <= 70:
                rsi_score = 15
            elif 20 <= current_rsi <= 80:
                rsi_score = 10
            else:
                rsi_score = 5
            
            details['rsi'] = current_rsi
            details['rsi_score'] = rsi_score
            score += rsi_score
            
            # 成交量确认
            volume_ma = df['Volume'].rolling(20).mean()
            recent_volume = df['Volume'].iloc[-5:].mean()
            volume_score = 10 if recent_volume > volume_ma.iloc[-1] else 5
            
            details['volume_score'] = volume_score
            score += volume_score
            
            # 4. 风险评分 (20分)
            daily_returns = df['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            
            risk_score = 0
            if volatility < 20:
                risk_score = 20
            elif volatility < 30:
                risk_score = 15
            elif volatility < 40:
                risk_score = 10
            else:
                risk_score = 5
            
            details['volatility'] = volatility
            details['risk_score'] = risk_score
            score += risk_score
            
            stock_scores[symbol] = {
                'total_score': score,
                'details': details,
                'sector': self.stock_universe[symbol]['sector'],
                'risk_level': self.stock_universe[symbol]['risk_level']
            }
        
        # 排序
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        self.analysis_results['stock_scores'] = dict(sorted_stocks)
        
        return dict(sorted_stocks)

    def generate_strategy_recommendations(self):
        """生成投资策略推荐"""
        print("💡 生成投资策略推荐...")
        
        market_conditions = self.analysis_results.get('market_conditions', {})
        stock_scores = self.analysis_results.get('stock_scores', {})
        
        recommendations = {}
        
        # 根据市场环境选择基础策略
        market_trend = market_conditions.get('trend', 'neutral')
        market_volatility = market_conditions.get('volatility', 'medium')
        
        if market_trend == 'bullish' and market_volatility == 'low':
            base_strategy = 'aggressive'
        elif market_trend == 'bearish' or market_volatility == 'high':
            base_strategy = 'conservative'
        else:
            base_strategy = 'balanced'
        
        # 为每种风险偏好生成推荐
        for strategy_type, template in self.strategy_templates.items():
            strategy_rec = template.copy()
            strategy_rec['recommended_stocks'] = []
            strategy_rec['portfolio_allocation'] = {}
            
            # 根据策略类型筛选股票
            suitable_stocks = []
            for symbol, score_data in stock_scores.items():
                stock_risk = self.stock_universe[symbol]['risk_level']
                stock_sector = score_data['sector']
                
                # 风险匹配
                risk_match = False
                if strategy_type == 'conservative' and stock_risk in ['低', '中等']:
                    risk_match = True
                elif strategy_type == 'balanced' and stock_risk in ['低', '中等', '中高']:
                    risk_match = True
                elif strategy_type == 'aggressive':
                    risk_match = True
                
                # 板块匹配
                sector_match = stock_sector in template['sectors']
                
                if risk_match and (sector_match or strategy_type == 'aggressive'):
                    suitable_stocks.append((symbol, score_data))
            
            # 选择前5-8只股票
            max_stocks = 5 if strategy_type == 'conservative' else 8
            selected_stocks = suitable_stocks[:max_stocks]
            
            total_weight = 0
            for symbol, score_data in selected_stocks:
                # 根据评分分配权重
                base_weight = score_data['total_score'] / 100 * 15  # 基础权重
                
                # 调整权重
                if strategy_type == 'conservative':
                    if self.stock_universe[symbol]['risk_level'] == '低':
                        weight = min(base_weight * 1.2, 25)
                    else:
                        weight = min(base_weight * 0.8, 15)
                elif strategy_type == 'aggressive':
                    if self.stock_universe[symbol]['risk_level'] == '高':
                        weight = min(base_weight * 1.3, 20)
                    else:
                        weight = base_weight
                else:
                    weight = base_weight
                
                strategy_rec['recommended_stocks'].append({
                    'symbol': symbol,
                    'name': self.stock_universe[symbol]['name'],
                    'sector': score_data['sector'],
                    'score': score_data['total_score'],
                    'weight': round(weight, 1),
                    'risk_level': self.stock_universe[symbol]['risk_level']
                })
                
                total_weight += weight
            
            # 标准化权重
            if total_weight > 0:
                for stock in strategy_rec['recommended_stocks']:
                    stock['weight'] = round(stock['weight'] / total_weight * template['allocation']['股票'], 1)
            
            recommendations[strategy_type] = strategy_rec
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations

    def create_recommendation_report(self):
        """生成推荐报告"""
        print("📋 生成投资策略报告...")
        
        market_conditions = self.analysis_results.get('market_conditions', {})
        recommendations = self.analysis_results.get('recommendations', {})
        
        print("\n" + "="*60)
        print("🎯 未来3个月投资策略推荐报告")
        print("="*60)
        
        # 市场环境分析
        print(f"\n📊 当前市场环境分析")
        print("-" * 40)
        trend_desc = {
            'bullish': '上涨趋势 📈',
            'bearish': '下跌趋势 📉',
            'neutral': '震荡整理 📊'
        }
        vol_desc = {
            'high': '高波动 ⚡',
            'medium': '中等波动 📊',
            'low': '低波动 😌'
        }
        
        print(f"市场趋势: {trend_desc.get(market_conditions.get('trend', 'neutral'), '未知')}")
        print(f"波动水平: {vol_desc.get(market_conditions.get('volatility', 'medium'), '未知')}")
        
        # 板块表现
        if market_conditions.get('sector_rotation'):
            print(f"\n板块表现排名:")
            sorted_sectors = sorted(market_conditions['sector_rotation'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for i, (sector, performance) in enumerate(sorted_sectors, 1):
                emoji = "🔥" if performance > 5 else "📈" if performance > 0 else "📉"
                print(f"  {i}. {sector}: {performance:+.1f}% {emoji}")
        
        # 投资策略推荐
        print(f"\n💡 投资策略推荐")
        print("-" * 40)
        
        strategy_names = {
            'conservative': '稳健型策略 🛡️',
            'balanced': '平衡型策略 ⚖️',
            'aggressive': '积极型策略 🚀'
        }
        
        for strategy_type, strategy_data in recommendations.items():
            print(f"\n{strategy_names.get(strategy_type, strategy_type)}")
            print(f"目标收益: {strategy_data['target_return']}")
            print(f"最大回撤: {strategy_data['max_drawdown']}")
            
            print(f"\n推荐股票组合:")
            for stock in strategy_data['recommended_stocks']:
                risk_emoji = {
                    '低': '🟢',
                    '中等': '🟡', 
                    '中高': '🟠',
                    '高': '🔴'
                }
                print(f"  • {stock['symbol']} ({stock['name']}) - {stock['weight']}% "
                      f"{risk_emoji.get(stock['risk_level'], '⚪')} "
                      f"评分: {stock['score']}/100")
            
            print(f"\n资产配置:")
            for asset, allocation in strategy_data['allocation'].items():
                print(f"  • {asset}: {allocation}%")
        
        # 风险提示
        print(f"\n⚠️ 重要风险提示")
        print("-" * 40)
        print("• 本分析基于历史数据和技术指标，不能保证未来表现")
        print("• 投资有风险，入市需谨慎，请根据个人情况调整")
        print("• 建议定期重新评估和调整投资组合")
        print("• 请关注宏观经济环境和公司基本面变化")
        print("• 建议分批建仓，控制单次投资金额")
        
        # 操作建议
        print(f"\n📅 未来3个月操作建议")
        print("-" * 40)
        
        current_month = datetime.now().month
        seasons = {
            (12, 1, 2): "冬季",
            (3, 4, 5): "春季", 
            (6, 7, 8): "夏季",
            (9, 10, 11): "秋季"
        }
        
        current_season = "未知"
        for months, season in seasons.items():
            if current_month in months:
                current_season = season
                break
        
        print(f"当前季节: {current_season}")
        
        if market_conditions.get('trend') == 'bullish':
            print("• 第1个月: 积极建仓，重点关注科技和成长股")
            print("• 第2个月: 持续加仓，关注业绩预期较好的个股")
            print("• 第3个月: 适当获利了结，为下一轮布局做准备")
        elif market_conditions.get('trend') == 'bearish':
            print("• 第1个月: 谨慎观望，重点关注防御性股票")
            print("• 第2个月: 逢低分批建仓优质股票")
            print("• 第3个月: 等待市场企稳信号，准备加仓")
        else:
            print("• 第1个月: 均衡配置，关注业绩确定性较高的股票")
            print("• 第2个月: 根据市场变化调整仓位结构")
            print("• 第3个月: 为下一季度投资做好准备")

    def create_visualization(self):
        """创建可视化图表"""
        print("📊 生成可视化图表...")
        
        try:
            stock_scores = self.analysis_results.get('stock_scores', {})
            recommendations = self.analysis_results.get('recommendations', {})
            
            if not stock_scores or not recommendations:
                print("⚠️ 缺少分析数据，跳过图表生成")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('投资策略推荐分析报告', fontsize=16, fontweight='bold')
            
            # 1. 个股评分排名
            symbols = list(stock_scores.keys())[:10]
            scores = [stock_scores[s]['total_score'] for s in symbols]
            colors = ['#2E8B57' if s >= 70 else '#FFD700' if s >= 50 else '#FF6347' for s in scores]
            
            bars = ax1.barh(symbols, scores, color=colors)
            ax1.set_xlabel('综合评分')
            ax1.set_title('个股综合评分排名 (Top 10)')
            ax1.set_xlim(0, 100)
            
            # 添加数值标签
            for bar, score in zip(bars, scores):
                ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        f'{score:.0f}', va='center', fontsize=9)
            
            # 2. 板块表现对比
            market_conditions = self.analysis_results.get('market_conditions', {})
            sector_data = market_conditions.get('sector_rotation', {})
            
            if sector_data:
                sectors = list(sector_data.keys())
                performance = list(sector_data.values())
                colors = ['#2E8B57' if p > 0 else '#FF6347' for p in performance]
                
                bars = ax2.bar(sectors, performance, color=colors)
                ax2.set_ylabel('收益率 (%)')
                ax2.set_title('板块表现对比 (近20日)')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # 添加数值标签
                for bar, perf in zip(bars, performance):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                            f'{perf:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
            
            # 3. 策略风险收益对比
            strategies = []
            returns = []
            risks = []
            
            for strategy_type, strategy_data in recommendations.items():
                strategies.append(strategy_data['name'])
                # 提取目标收益率的中位数
                target_return = strategy_data['target_return']
                if '-' in target_return:
                    low, high = target_return.replace('%', '').split('-')
                    avg_return = (float(low) + float(high)) / 2
                else:
                    avg_return = float(target_return.replace('%', ''))
                returns.append(avg_return)
                
                # 提取最大回撤
                max_dd = float(strategy_data['max_drawdown'].replace('%', ''))
                risks.append(max_dd)
            
            scatter = ax3.scatter(risks, returns, s=200, alpha=0.7, 
                                c=['#2E8B57', '#FFD700', '#FF6347'])
            ax3.set_xlabel('最大回撤 (%)')
            ax3.set_ylabel('目标收益率 (%)')
            ax3.set_title('策略风险收益特征')
            
            # 添加策略标签
            for i, strategy in enumerate(strategies):
                ax3.annotate(strategy.replace('策略', ''), (risks[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # 4. 推荐股票权重分布 (平衡型策略)
            if 'balanced' in recommendations:
                balanced_stocks = recommendations['balanced']['recommended_stocks']
                if balanced_stocks:
                    symbols = [s['symbol'] for s in balanced_stocks]
                    weights = [s['weight'] for s in balanced_stocks]
                    
                    wedges, texts, autotexts = ax4.pie(weights, labels=symbols, autopct='%1.1f%%',
                                                      startangle=90, colors=plt.cm.Set3.colors)
                    ax4.set_title('平衡型策略 - 股票权重分配')
            
            plt.tight_layout()
            
            # 保存图表
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            chart_path = results_dir / "investment_strategy_recommendation.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f"✅ 图表已保存: {chart_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ 图表生成失败: {e}")
            import traceback
            traceback.print_exc()

    def run_analysis(self):
        """运行完整的投资策略分析"""
        print("🎯 投资策略推荐系统")
        print("="*50)
        
        # 1. 加载数据
        if not self.load_cached_data():
            print("❌ 数据加载失败，无法进行分析")
            return
        
        # 2. 分析市场环境
        self.analyze_market_conditions()
        
        # 3. 计算个股评分
        self.calculate_stock_scores()
        
        # 4. 生成策略推荐
        self.generate_strategy_recommendations()
        
        # 5. 生成报告
        self.create_recommendation_report()
        
        # 6. 创建可视化
        self.create_visualization()
        
        print("\n" + "="*60)
        print("✅ 投资策略推荐分析完成！")
        print("="*60)
        
        print("\n📝 重要提醒:")
        print("• 本分析仅供参考，不构成投资建议")
        print("• 投资有风险，请根据个人情况谨慎决策")
        print("• 建议结合基本面分析和宏观环境")
        print("• 请定期重新评估和调整投资组合")

def main():
    """主函数"""
    try:
        recommender = InvestmentStrategyRecommendation()
        recommender.run_analysis()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断分析")
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()