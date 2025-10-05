#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场情绪分析工具
分析A股市场的情绪指标，包括技术指标、板块轮动、风险指标等
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


class MarketSentimentAnalyzer:
    """市场情绪分析器"""
    
    def __init__(self):
        # A股代表性股票池
        self.stock_pool = {
            '000001.SZ': '平安银行',
            '000002.SZ': '万科A',
            '600036.SH': '招商银行',
            '600519.SH': '贵州茅台',
            '000858.SZ': '五粮液',
            '600276.SH': '恒瑞医药',
            '000725.SZ': '京东方A',
            '002415.SZ': '海康威视',
            '300015.SZ': '爱尔眼科',
            '600887.SH': '伊利股份'
        }
        
        # 板块映射
        self.sector_mapping = {
            '000001.SZ': '金融',
            '000002.SZ': '房地产',
            '600036.SH': '金融',
            '600519.SH': '消费',
            '000858.SZ': '消费',
            '600276.SH': '医药',
            '000725.SZ': '科技',
            '002415.SZ': '科技',
            '300015.SZ': '医药',
            '600887.SH': '消费'
        }
    
    def _load_cached_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载缓存的股票数据（模拟数据）"""
        try:
            # 模拟股票数据
            np.random.seed(hash(symbol) % 2**32)
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            
            # 生成模拟价格数据
            base_price = 10 + (hash(symbol) % 100)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # 生成成交量数据
            volumes = np.random.lognormal(15, 0.5, len(dates))
            
            data = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            data.set_index('date', inplace=True)
            return data
            
        except Exception as e:
            print(f"加载 {symbol} 数据失败: {e}")
            return None
    
    def calculate_sentiment_indicators(self, data: pd.DataFrame) -> Dict:
        """计算情绪指标"""
        if data is None or data.empty:
            return {}
        
        try:
            # RSI
            rsi = self.calculate_rsi(data['close'])
            
            # MACD
            macd_line, signal_line, histogram = self.calculate_macd(data['close'])
            
            # 布林带
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(data['close'])
            
            # 动量指标
            momentum = self.calculate_momentum(data['close'])
            
            # 成交量比率
            volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
            
            # 布林带位置
            bb_position = (data['close'].iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            
            return {
                'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                'macd': macd_line.iloc[-1] if not macd_line.empty else 0,
                'macd_signal': signal_line.iloc[-1] if not signal_line.empty else 0,
                'macd_histogram': histogram.iloc[-1] if not histogram.empty else 0,
                'bb_position': bb_position if not np.isnan(bb_position) else 0.5,
                'momentum': momentum.iloc[-1] if not momentum.empty else 0,
                'volume_ratio': volume_ratio if not np.isnan(volume_ratio) else 1.0
            }
            
        except Exception as e:
            print(f"计算指标失败: {e}")
            return {}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """计算布林带"""
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    def calculate_momentum(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """计算动量指标"""
        return prices.pct_change(period)
    
    def analyze_market_sentiment(self, symbol: str) -> Dict:
        """分析市场情绪"""
        data = self._load_cached_stock_data(symbol)
        if data is None or data.empty:
            return {}
        
        # 计算技术指标
        indicators = self.calculate_sentiment_indicators(data)
        if not indicators:
            return {}
        
        # 计算价格变化
        price_change = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
        
        # 计算情绪得分
        sentiment_score = self._calculate_sentiment_score(indicators)
        
        return {
            'symbol': symbol,
            'name': self.stock_pool.get(symbol, symbol),
            'price_change': price_change,
            'indicators': indicators,
            'sentiment_score': sentiment_score
        }
    
    def _calculate_sentiment_score(self, indicators: Dict) -> Dict:
        """计算情绪得分"""
        scores = {}
        total_score = 0
        total_weight = 0
        
        # RSI得分 (权重: 20%)
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            rsi_score = 20  # 超买，负面
            rsi_signal = '超买'
        elif rsi < 30:
            rsi_score = 80  # 超卖，可能反弹
            rsi_signal = '超卖'
        else:
            rsi_score = 50 + (rsi - 50) * 0.5  # 中性区间
            rsi_signal = '中性'
        
        scores['rsi'] = {'score': rsi_score, 'signal': rsi_signal}
        total_score += rsi_score * 0.2
        total_weight += 0.2
        
        # MACD得分 (权重: 25%)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_histogram = indicators.get('macd_histogram', 0)
        
        if macd > macd_signal and macd_histogram > 0:
            macd_score = 75
            macd_signal_text = '金叉上涨'
        elif macd < macd_signal and macd_histogram < 0:
            macd_score = 25
            macd_signal_text = '死叉下跌'
        else:
            macd_score = 50
            macd_signal_text = '震荡'
        
        scores['macd'] = {'score': macd_score, 'signal': macd_signal_text}
        total_score += macd_score * 0.25
        total_weight += 0.25
        
        # 布林带得分 (权重: 20%)
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position > 0.8:
            bb_score = 20  # 接近上轨，可能回调
            bb_signal = '接近上轨'
        elif bb_position < 0.2:
            bb_score = 80  # 接近下轨，可能反弹
            bb_signal = '接近下轨'
        else:
            bb_score = 50 + (bb_position - 0.5) * 40
            bb_signal = '中轨附近'
        
        scores['bollinger'] = {'score': bb_score, 'signal': bb_signal}
        total_score += bb_score * 0.2
        total_weight += 0.2
        
        # 动量得分 (权重: 15%)
        momentum = indicators.get('momentum', 0)
        momentum_score = 50 + momentum * 1000  # 放大动量影响
        momentum_score = max(0, min(100, momentum_score))  # 限制在0-100
        
        scores['momentum'] = {'score': momentum_score, 'signal': '正动量' if momentum > 0 else '负动量'}
        total_score += momentum_score * 0.15
        total_weight += 0.15
        
        # 成交量得分 (权重: 20%)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            volume_score = 70  # 放量
            volume_signal = '放量'
        elif volume_ratio < 0.7:
            volume_score = 30  # 缩量
            volume_signal = '缩量'
        else:
            volume_score = 50
            volume_signal = '正常'
        
        scores['volume'] = {'score': volume_score, 'signal': volume_signal}
        total_score += volume_score * 0.2
        total_weight += 0.2
        
        # 计算总得分
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        return {
            'overall_score': overall_score,
            'level': self._get_sentiment_level(overall_score),
            'details': scores
        }
    
    def _get_sentiment_level(self, score: float) -> str:
        """获取情绪等级"""
        if score >= 70:
            return '积极'
        elif score >= 60:
            return '中性偏积极'
        elif score >= 40:
            return '中性偏消极'
        else:
            return '消极'
    
    def analyze_sector_rotation(self) -> Dict:
        """分析板块轮动"""
        sector_performance = {}
        
        for symbol, sector in self.sector_mapping.items():
            data = self._load_cached_stock_data(symbol)
            if data is not None and not data.empty:
                # 计算近期收益率
                returns = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
                volatility = data['close'].pct_change().rolling(20).std().iloc[-1] * 100
                
                if sector not in sector_performance:
                    sector_performance[sector] = []
                
                sector_performance[sector].append({
                    'symbol': symbol,
                    'returns': returns,
                    'volatility': volatility
                })
        
        # 计算板块平均表现
        sector_summary = {}
        for sector, stocks in sector_performance.items():
            avg_returns = np.mean([s['returns'] for s in stocks])
            avg_volatility = np.mean([s['volatility'] for s in stocks])
            sharpe_ratio = avg_returns / avg_volatility if avg_volatility > 0 else 0
            
            sector_summary[sector] = {
                'sector': sector,
                'returns': avg_returns,
                'volatility': avg_volatility,
                'sharpe_ratio': sharpe_ratio,
                'stock_count': len(stocks)
            }
        
        # 排序
        leading_sectors = sorted(sector_summary.values(), 
                               key=lambda x: x['sharpe_ratio'], reverse=True)
        lagging_sectors = sorted(sector_summary.values(), 
                               key=lambda x: x['sharpe_ratio'])
        
        return {
            'leading_sectors': leading_sectors[:3],
            'lagging_sectors': lagging_sectors[:3],
            'sector_details': sector_summary
        }
    
    def calculate_risk_metrics(self, symbol: str) -> Dict:
        """计算风险指标"""
        data = self._load_cached_stock_data(symbol)
        if data is None or data.empty:
            return {}
        
        returns = data['close'].pct_change().dropna()
        
        # 基础统计
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率 (假设无风险利率为3%)
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # VaR (95%置信度)
        var_95 = np.percentile(returns, 5)
        
        # CVaR (条件VaR)
        cvar_95 = returns[returns <= var_95].mean()
        
        # 卡尔马比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 偏度和峰度
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # 风险等级评估
        risk_level = self._assess_risk_level(annual_volatility, max_drawdown, sharpe_ratio)
        
        return {
            'symbol': symbol,
            'name': self.stock_pool.get(symbol, symbol),
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'risk_level': risk_level
        }
    
    def _assess_risk_level(self, volatility: float, max_drawdown: float, sharpe_ratio: float) -> str:
        """评估风险等级"""
        risk_score = 0
        
        # 波动率评分
        if volatility > 0.3:
            risk_score += 3
        elif volatility > 0.2:
            risk_score += 2
        else:
            risk_score += 1
        
        # 最大回撤评分
        if abs(max_drawdown) > 0.3:
            risk_score += 3
        elif abs(max_drawdown) > 0.2:
            risk_score += 2
        else:
            risk_score += 1
        
        # 夏普比率评分（反向）
        if sharpe_ratio < 0:
            risk_score += 3
        elif sharpe_ratio < 0.5:
            risk_score += 2
        else:
            risk_score += 1
        
        # 风险等级
        if risk_score >= 8:
            return 'C'  # 高风险
        elif risk_score >= 6:
            return 'B'  # 中等风险
        elif risk_score >= 4:
            return 'B+'  # 中低风险
        else:
            return 'A'  # 低风险
    
    def create_sentiment_visualization(self, sentiment_data: List[Dict]):
        """创建情绪分析可视化图表"""
        if not sentiment_data:
            print("没有情绪数据可供可视化")
            return
        
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表
            fig = plt.figure(figsize=(20, 15))
            
            # 1. 整体情绪得分柱状图
            ax1 = plt.subplot(3, 3, 1)
            symbols = [item['symbol'] for item in sentiment_data]
            scores = [item['sentiment_score']['overall_score'] for item in sentiment_data]
            colors = ['green' if score >= 60 else 'orange' if score >= 40 else 'red' for score in scores]
            
            bars = ax1.bar(range(len(symbols)), scores, color=colors, alpha=0.7)
            ax1.set_title('整体情绪得分', fontsize=14, fontweight='bold')
            ax1.set_xlabel('股票代码')
            ax1.set_ylabel('情绪得分')
            ax1.set_xticks(range(len(symbols)))
            ax1.set_xticklabels([s.split('.')[0] for s in symbols], rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{score:.1f}', ha='center', va='bottom')
            
            # 2. RSI指标分布
            ax2 = plt.subplot(3, 3, 2)
            rsi_values = [item['indicators']['rsi'] for item in sentiment_data if 'rsi' in item['indicators']]
            if rsi_values:
                ax2.hist(rsi_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
                ax2.axvline(x=70, color='red', linestyle='--', label='超买线(70)')
                ax2.axvline(x=30, color='green', linestyle='--', label='超卖线(30)')
                ax2.set_title('RSI指标分布')
                ax2.set_xlabel('RSI值')
                ax2.set_ylabel('频次')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. 动量指标散点图
            ax3 = plt.subplot(3, 3, 3)
            momentum_values = [item['indicators']['momentum'] * 100 for item in sentiment_data if 'momentum' in item['indicators']]
            price_changes = [item['price_change'] for item in sentiment_data]
            if momentum_values and price_changes:
                scatter = ax3.scatter(momentum_values, price_changes, 
                                    c=scores, cmap='RdYlGn', alpha=0.7, s=100)
                ax3.set_title('动量 vs 价格变化')
                ax3.set_xlabel('动量指标 (%)')
                ax3.set_ylabel('价格变化 (%)')
                ax3.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax3, label='情绪得分')
            
            # 4. 情绪等级饼图
            ax4 = plt.subplot(3, 3, 4)
            sentiment_levels = [item['sentiment_score']['level'] for item in sentiment_data]
            level_counts = pd.Series(sentiment_levels).value_counts()
            colors_pie = ['green', 'lightgreen', 'orange', 'red']
            ax4.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
                   colors=colors_pie[:len(level_counts)], startangle=90)
            ax4.set_title('情绪等级分布')
            
            # 5. 成交量比率柱状图
            ax5 = plt.subplot(3, 3, 5)
            volume_ratios = [item['indicators']['volume_ratio'] for item in sentiment_data if 'volume_ratio' in item['indicators']]
            if volume_ratios:
                bars5 = ax5.bar(range(len(symbols)), volume_ratios, 
                              color=['red' if vr > 1.5 else 'green' if vr < 0.7 else 'blue' for vr in volume_ratios],
                              alpha=0.7)
                ax5.set_title('成交量比率')
                ax5.set_xlabel('股票代码')
                ax5.set_ylabel('成交量比率')
                ax5.set_xticks(range(len(symbols)))
                ax5.set_xticklabels([s.split('.')[0] for s in symbols], rotation=45)
                ax5.axhline(y=1, color='black', linestyle='-', alpha=0.5)
                ax5.grid(True, alpha=0.3)
            
            # 6. MACD信号分布
            ax6 = plt.subplot(3, 3, 6)
            macd_signals = [item['sentiment_score']['details']['macd']['signal'] for item in sentiment_data 
                          if 'macd' in item['sentiment_score']['details']]
            if macd_signals:
                signal_counts = pd.Series(macd_signals).value_counts()
                ax6.bar(signal_counts.index, signal_counts.values, 
                       color=['green', 'red', 'gray'], alpha=0.7)
                ax6.set_title('MACD信号分布')
                ax6.set_xlabel('信号类型')
                ax6.set_ylabel('数量')
                ax6.grid(True, alpha=0.3)
            
            # 7. 布林带位置分布
            ax7 = plt.subplot(3, 3, 7)
            bb_positions = [item['indicators']['bb_position'] for item in sentiment_data if 'bb_position' in item['indicators']]
            if bb_positions:
                ax7.hist(bb_positions, bins=10, alpha=0.7, color='purple', edgecolor='black')
                ax7.axvline(x=0.8, color='red', linestyle='--', label='上轨(0.8)')
                ax7.axvline(x=0.2, color='green', linestyle='--', label='下轨(0.2)')
                ax7.set_title('布林带位置分布')
                ax7.set_xlabel('布林带位置')
                ax7.set_ylabel('频次')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
            
            # 8. 价格变化vs情绪得分
            ax8 = plt.subplot(3, 3, 8)
            ax8.scatter(price_changes, scores, alpha=0.7, s=100, c='blue')
            ax8.set_title('价格变化 vs 情绪得分')
            ax8.set_xlabel('价格变化 (%)')
            ax8.set_ylabel('情绪得分')
            ax8.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(price_changes) > 1:
                z = np.polyfit(price_changes, scores, 1)
                p = np.poly1d(z)
                ax8.plot(sorted(price_changes), p(sorted(price_changes)), "r--", alpha=0.8)
            
            # 9. 综合指标雷达图
            ax9 = plt.subplot(3, 3, 9, projection='polar')
            if sentiment_data:
                # 选择第一个股票作为示例
                sample_data = sentiment_data[0]
                indicators = ['RSI', 'MACD', '布林带', '动量', '成交量']
                values = [
                    sample_data['sentiment_score']['details'].get('rsi', {}).get('score', 50),
                    sample_data['sentiment_score']['details'].get('macd', {}).get('score', 50),
                    sample_data['sentiment_score']['details'].get('bollinger', {}).get('score', 50),
                    sample_data['sentiment_score']['details'].get('momentum', {}).get('score', 50),
                    sample_data['sentiment_score']['details'].get('volume', {}).get('score', 50)
                ]
                
                angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False)
                values += values[:1]  # 闭合图形
                angles = np.concatenate((angles, [angles[0]]))
                
                ax9.plot(angles, values, 'o-', linewidth=2, label=sample_data['symbol'])
                ax9.fill(angles, values, alpha=0.25)
                ax9.set_xticks(angles[:-1])
                ax9.set_xticklabels(indicators)
                ax9.set_ylim(0, 100)
                ax9.set_title(f'指标雷达图 - {sample_data["symbol"]}')
                ax9.grid(True)
            
            plt.tight_layout()
            plt.savefig('market_sentiment_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("情绪分析可视化图表已保存为 'market_sentiment_analysis.png'")
            
        except Exception as e:
            print(f"创建可视化图表时出错: {e}")
    
    def create_risk_visualization(self, risk_data: List[Dict]):
        """创建风险指标可视化图表"""
        if not risk_data:
            print("没有风险数据可供可视化")
            return
        
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表
            fig = plt.figure(figsize=(20, 15))
            
            # 1. 风险等级饼图
            ax1 = plt.subplot(3, 3, 1)
            risk_levels = [item['risk_level'] for item in risk_data]
            level_counts = pd.Series(risk_levels).value_counts()
            colors = ['green', 'yellow', 'orange', 'red']
            ax1.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
                   colors=colors[:len(level_counts)], startangle=90)
            ax1.set_title('风险等级分布')
            
            # 2. VaR分布直方图
            ax2 = plt.subplot(3, 3, 2)
            var_values = [item['var_95'] * 100 for item in risk_data]
            ax2.hist(var_values, bins=10, alpha=0.7, color='red', edgecolor='black')
            ax2.set_title('VaR(95%)分布')
            ax2.set_xlabel('VaR (%)')
            ax2.set_ylabel('频次')
            ax2.grid(True, alpha=0.3)
            
            # 3. 收益率vs波动率散点图
            ax3 = plt.subplot(3, 3, 3)
            returns = [item['annual_return'] * 100 for item in risk_data]
            volatilities = [item['annual_volatility'] * 100 for item in risk_data]
            sharpe_ratios = [item['sharpe_ratio'] for item in risk_data]
            
            scatter = ax3.scatter(volatilities, returns, c=sharpe_ratios, 
                                cmap='RdYlGn', alpha=0.7, s=100)
            ax3.set_title('收益率 vs 波动率')
            ax3.set_xlabel('年化波动率 (%)')
            ax3.set_ylabel('年化收益率 (%)')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='夏普比率')
            
            # 4. 夏普比率柱状图
            ax4 = plt.subplot(3, 3, 4)
            symbols = [item['symbol'].split('.')[0] for item in risk_data]
            colors_sharpe = ['green' if sr > 1 else 'orange' if sr > 0 else 'red' for sr in sharpe_ratios]
            bars4 = ax4.bar(range(len(symbols)), sharpe_ratios, color=colors_sharpe, alpha=0.7)
            ax4.set_title('夏普比率对比')
            ax4.set_xlabel('股票代码')
            ax4.set_ylabel('夏普比率')
            ax4.set_xticks(range(len(symbols)))
            ax4.set_xticklabels(symbols, rotation=45)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3)
            
            # 5. 最大回撤对比图
            ax5 = plt.subplot(3, 3, 5)
            max_drawdowns = [abs(item['max_drawdown']) * 100 for item in risk_data]
            bars5 = ax5.bar(range(len(symbols)), max_drawdowns, 
                          color=['red' if md > 20 else 'orange' if md > 10 else 'green' for md in max_drawdowns],
                          alpha=0.7)
            ax5.set_title('最大回撤对比')
            ax5.set_xlabel('股票代码')
            ax5.set_ylabel('最大回撤 (%)')
            ax5.set_xticks(range(len(symbols)))
            ax5.set_xticklabels(symbols, rotation=45)
            ax5.grid(True, alpha=0.3)
            
            # 6. CVaR vs VaR散点图
            ax6 = plt.subplot(3, 3, 6)
            cvar_values = [item['cvar_95'] * 100 for item in risk_data]
            ax6.scatter(var_values, cvar_values, alpha=0.7, s=100, c='purple')
            ax6.set_title('CVaR vs VaR')
            ax6.set_xlabel('VaR(95%) (%)')
            ax6.set_ylabel('CVaR(95%) (%)')
            ax6.grid(True, alpha=0.3)
            
            # 添加对角线
            min_val = min(min(var_values), min(cvar_values))
            max_val = max(max(var_values), max(cvar_values))
            ax6.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
            
            # 7. 索提诺比率vs夏普比率
            ax7 = plt.subplot(3, 3, 7)
            sortino_ratios = [item['sortino_ratio'] for item in risk_data]
            ax7.scatter(sharpe_ratios, sortino_ratios, alpha=0.7, s=100, c='blue')
            ax7.set_title('索提诺比率 vs 夏普比率')
            ax7.set_xlabel('夏普比率')
            ax7.set_ylabel('索提诺比率')
            ax7.grid(True, alpha=0.3)
            
            # 8. 偏度vs峰度分布
            ax8 = plt.subplot(3, 3, 8)
            skewness_values = [item['skewness'] for item in risk_data]
            kurtosis_values = [item['kurtosis'] for item in risk_data]
            ax8.scatter(skewness_values, kurtosis_values, alpha=0.7, s=100, c='orange')
            ax8.set_title('偏度 vs 峰度分布')
            ax8.set_xlabel('偏度')
            ax8.set_ylabel('峰度')
            ax8.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax8.grid(True, alpha=0.3)
            
            # 9. 卡尔马比率排名
            ax9 = plt.subplot(3, 3, 9)
            calmar_ratios = [item['calmar_ratio'] for item in risk_data]
            sorted_indices = sorted(range(len(calmar_ratios)), key=lambda i: calmar_ratios[i], reverse=True)
            sorted_symbols = [symbols[i] for i in sorted_indices]
            sorted_calmar = [calmar_ratios[i] for i in sorted_indices]
            
            bars9 = ax9.bar(range(len(sorted_symbols)), sorted_calmar, 
                          color=['green' if cr > 0.5 else 'orange' if cr > 0 else 'red' for cr in sorted_calmar],
                          alpha=0.7)
            ax9.set_title('卡尔马比率排名')
            ax9.set_xlabel('股票代码')
            ax9.set_ylabel('卡尔马比率')
            ax9.set_xticks(range(len(sorted_symbols)))
            ax9.set_xticklabels(sorted_symbols, rotation=45)
            ax9.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('risk_metrics_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("风险指标可视化图表已保存为 'risk_metrics_analysis.png'")
            
        except Exception as e:
            print(f"创建风险可视化图表时出错: {e}")
    
    def create_sector_rotation_visualization(self, sector_rotation):
        """创建板块轮动可视化图表"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表
            fig = plt.figure(figsize=(20, 16))
            
            # 准备数据
            leading_sectors = sector_rotation['leading_sectors']
            lagging_sectors = sector_rotation['lagging_sectors']
            
            # 合并所有板块数据
            all_sectors = leading_sectors + lagging_sectors
            sector_names = [sector['sector'] for sector in all_sectors]
            returns = [sector['returns'] for sector in all_sectors]
            volatilities = [sector['volatility'] for sector in all_sectors]
            sharpe_ratios = [sector['sharpe_ratio'] for sector in all_sectors]
            
            # 1. 板块收益率对比柱状图
            plt.subplot(3, 3, 1)
            colors = ['green' if r > 0 else 'red' for r in returns]
            bars = plt.bar(range(len(sector_names)), returns, color=colors, alpha=0.7)
            plt.title('板块收益率对比', fontsize=14, fontweight='bold')
            plt.xlabel('板块')
            plt.ylabel('收益率 (%)')
            plt.xticks(range(len(sector_names)), sector_names, rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar, ret) in enumerate(zip(bars, returns)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1 if ret > 0 else bar.get_height() - 0.3,
                        f'{ret:.1f}%', ha='center', va='bottom' if ret > 0 else 'top', fontsize=10)
            
            # 2. 板块波动率对比
            plt.subplot(3, 3, 2)
            plt.bar(range(len(sector_names)), volatilities, color='orange', alpha=0.7)
            plt.title('板块波动率对比', fontsize=14, fontweight='bold')
            plt.xlabel('板块')
            plt.ylabel('波动率 (%)')
            plt.xticks(range(len(sector_names)), sector_names, rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 3. 夏普比率对比
            plt.subplot(3, 3, 3)
            colors = ['darkgreen' if s > 1 else 'orange' if s > 0 else 'red' for s in sharpe_ratios]
            plt.bar(range(len(sector_names)), sharpe_ratios, color=colors, alpha=0.7)
            plt.title('板块夏普比率对比', fontsize=14, fontweight='bold')
            plt.xlabel('板块')
            plt.ylabel('夏普比率')
            plt.xticks(range(len(sector_names)), sector_names, rotation=45)
            plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='优秀线(1.0)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 4. 收益率vs波动率散点图
            plt.subplot(3, 3, 4)
            scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='RdYlGn', 
                                s=100, alpha=0.7, edgecolors='black')
            plt.title('收益率 vs 波动率', fontsize=14, fontweight='bold')
            plt.xlabel('波动率 (%)')
            plt.ylabel('收益率 (%)')
            plt.grid(True, alpha=0.3)
            
            # 添加板块标签
            for i, name in enumerate(sector_names):
                plt.annotate(name, (volatilities[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('夏普比率', rotation=270, labelpad=15)
            
            # 5. 领涨板块饼图
            plt.subplot(3, 3, 5)
            leading_names = [sector['sector'] for sector in leading_sectors]
            leading_returns = [max(0.1, sector['returns']) for sector in leading_sectors]  # 避免负值
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(leading_names)))
            
            wedges, texts, autotexts = plt.pie(leading_returns, labels=leading_names, autopct='%1.1f%%',
                                              colors=colors_pie, startangle=90)
            plt.title('领涨板块分布', fontsize=14, fontweight='bold')
            
            # 6. 板块轮动强度雷达图
            plt.subplot(3, 3, 6, projection='polar')
            
            # 计算轮动强度（基于收益率和夏普比率）
            rotation_strength = []
            for sector in all_sectors:
                strength = (sector['returns'] * 0.6 + sector['sharpe_ratio'] * 10 * 0.4)
                rotation_strength.append(max(0, strength))  # 确保非负
            
            angles = np.linspace(0, 2 * np.pi, len(sector_names), endpoint=False)
            rotation_strength += rotation_strength[:1]  # 闭合图形
            angles = np.concatenate((angles, [angles[0]]))
            
            plt.plot(angles, rotation_strength, 'o-', linewidth=2, color='blue', alpha=0.7)
            plt.fill(angles, rotation_strength, alpha=0.25, color='blue')
            plt.xticks(angles[:-1], sector_names)
            plt.title('板块轮动强度雷达图', fontsize=14, fontweight='bold', pad=20)
            plt.grid(True)
            
            # 7. 板块表现排名
            plt.subplot(3, 3, 7)
            # 按收益率排序
            sorted_data = sorted(zip(sector_names, returns), key=lambda x: x[1], reverse=True)
            sorted_names, sorted_returns = zip(*sorted_data)
            
            colors = ['darkgreen' if r > 5 else 'green' if r > 0 else 'red' for r in sorted_returns]
            plt.barh(range(len(sorted_names)), sorted_returns, color=colors, alpha=0.7)
            plt.title('板块表现排名', fontsize=14, fontweight='bold')
            plt.xlabel('收益率 (%)')
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.grid(True, alpha=0.3)
            
            # 8. 风险调整后收益对比
            plt.subplot(3, 3, 8)
            risk_adj_returns = [ret / vol if vol > 0 else 0 for ret, vol in zip(returns, volatilities)]
            colors = ['darkgreen' if r > 0.5 else 'orange' if r > 0 else 'red' for r in risk_adj_returns]
            plt.bar(range(len(sector_names)), risk_adj_returns, color=colors, alpha=0.7)
            plt.title('风险调整后收益对比', fontsize=14, fontweight='bold')
            plt.xlabel('板块')
            plt.ylabel('收益率/波动率')
            plt.xticks(range(len(sector_names)), sector_names, rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 9. 板块轮动趋势图（模拟时间序列）
            plt.subplot(3, 3, 9)
            # 模拟30天的轮动趋势
            days = np.arange(1, 31)
            
            for i, (name, ret) in enumerate(zip(sector_names[:3], returns[:3])):  # 只显示前3个板块
                # 基于当前收益率生成趋势
                base_trend = ret + np.random.normal(0, 2, 30).cumsum() * 0.1
                plt.plot(days, base_trend, marker='o', markersize=3, 
                        label=f'{name} ({ret:.1f}%)', alpha=0.8)
            
            plt.title('板块轮动趋势（30天）', fontsize=14, fontweight='bold')
            plt.xlabel('天数')
            plt.ylabel('累计收益率 (%)')
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            filename = 'sector_rotation_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"板块轮动可视化图表已保存为 '{filename}'")
            
        except Exception as e:
            print(f"创建板块轮动可视化图表时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_report(self, sentiment_data: List[Dict], sector_rotation: Dict, risk_analysis: List[Dict]):
        """生成综合分析报告"""
        print("\n" + "="*80)
        print("市场情绪分析报告")
        print("="*80)
        
        # 情绪分析摘要
        if sentiment_data:
            avg_sentiment = np.mean([item['sentiment_score']['overall_score'] for item in sentiment_data])
            sentiment_levels = [item['sentiment_score']['level'] for item in sentiment_data]
            level_counts = pd.Series(sentiment_levels).value_counts()
            
            print(f"\n📊 情绪分析摘要:")
            print(f"平均情绪得分: {avg_sentiment:.1f}")
            print("情绪等级分布:")
            for level, count in level_counts.items():
                print(f"  {level}: {count}只股票")
        
        # 板块轮动摘要
        if sector_rotation:
            print(f"\n🔄 板块轮动分析:")
            print("领涨板块:")
            for sector_info in sector_rotation['leading_sectors']:
                sector = sector_info['sector']
                data = sector_info
                print(f"  {sector}: 收益率 {data['returns']:.2f}%, 夏普比率 {data['sharpe_ratio']:.2f}")
            
            print("落后板块:")
            for sector_info in sector_rotation['lagging_sectors']:
                sector = sector_info['sector']
                data = sector_info
                print(f"  {sector}: 收益率 {data['returns']:.2f}%, 夏普比率 {data['sharpe_ratio']:.2f}")
        
        # 风险分析摘要
        if risk_analysis:
            print(f"\n⚠️ 风险分析摘要:")
            avg_volatility = np.mean([item['annual_volatility'] for item in risk_analysis]) * 100
            avg_sharpe = np.mean([item['sharpe_ratio'] for item in risk_analysis])
            print(f"平均年化波动率: {avg_volatility:.2f}%")
            print(f"平均夏普比率: {avg_sharpe:.2f}")
            
            # 风险等级分布
            risk_levels = [item['risk_level'] for item in risk_analysis]
            risk_counts = pd.Series(risk_levels).value_counts()
            print("风险等级分布:")
            for level, count in risk_counts.items():
                print(f"  {level}级: {count}只股票")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    print("开始市场情绪分析...")
    
    analyzer = MarketSentimentAnalyzer()
    
    # 情绪分析
    sentiment_results = []
    print("\n正在分析各股票情绪指标...")
    
    for symbol in analyzer.stock_pool.keys():
        print(f"分析 {symbol} ({analyzer.stock_pool[symbol]})...")
        sentiment = analyzer.analyze_market_sentiment(symbol)
        if sentiment:
            sentiment_results.append(sentiment)
    
    # 显示情绪分析结果
    if sentiment_results:
        print(f"\n📈 情绪分析结果:")
        print("-" * 80)
        for result in sentiment_results:
            score = result['sentiment_score']['overall_score']
            level = result['sentiment_score']['level']
            price_change = result['price_change']
            print(f"{result['symbol']} ({result['name']}): "
                  f"情绪得分 {score:.1f} ({level}), "
                  f"价格变化 {price_change:+.2f}%")
    
    # 板块轮动分析
    print(f"\n🔄 板块轮动分析:")
    print("-" * 50)
    sector_rotation = analyzer.analyze_sector_rotation()
    
    if sector_rotation:
        print("领涨板块:")
        for sector_info in sector_rotation['leading_sectors']:
            sector = sector_info['sector']
            data = sector_info
            print(f"  {sector}: 收益率 {data['returns']:.2f}%, 夏普比率 {data['sharpe_ratio']:.2f}")
        
        print("落后板块:")
        for sector_info in sector_rotation['lagging_sectors']:
            sector = sector_info['sector']
            data = sector_info
            print(f"  {sector}: 收益率 {data['returns']:.2f}%, 夏普比率 {data['sharpe_ratio']:.2f}")
    
    # 风险指标分析
    print(f"\n⚠️ 风险指标分析:")
    print("-" * 50)
    risk_analysis = []
    
    for symbol in analyzer.stock_pool.keys():
        risk_metrics = analyzer.calculate_risk_metrics(symbol)
        if risk_metrics:
            risk_analysis.append(risk_metrics)
            print(f"{risk_metrics['symbol']} ({risk_metrics['name']}):")
            print(f"  年化收益率: {risk_metrics['annual_return']*100:.2f}%")
            print(f"  年化波动率: {risk_metrics['annual_volatility']*100:.2f}%")
            print(f"  夏普比率: {risk_metrics['sharpe_ratio']:.2f}")
            print(f"  索提诺比率: {risk_metrics['sortino_ratio']:.2f}")
            print(f"  最大回撤: {risk_metrics['max_drawdown']*100:.2f}%")
            print(f"  VaR(95%): {risk_metrics['var_95']*100:.2f}%")
            print(f"  风险等级: {risk_metrics['risk_level']}")
            print()
    
    # 风险等级统计
    if risk_analysis:
        risk_levels = [item['risk_level'] for item in risk_analysis]
        risk_counts = pd.Series(risk_levels).value_counts()
        print("风险等级分布:")
        for level, count in risk_counts.items():
            print(f"  {level}级: {count}只股票")
        
        # 风险提醒
        high_risk_stocks = [item for item in risk_analysis if item['risk_level'] == 'C']
        if high_risk_stocks:
            print(f"\n⚠️ 高风险股票提醒:")
            for stock in high_risk_stocks:
                print(f"  {stock['symbol']} ({stock['name']}): 波动率 {stock['annual_volatility']*100:.1f}%, "
                      f"最大回撤 {abs(stock['max_drawdown'])*100:.1f}%")
    
    # 创建可视化图表
    print(f"\n📊 生成可视化图表...")
    print("-" * 30)
    if sentiment_results:
        analyzer.create_sentiment_visualization(sentiment_results)
    
    # 创建风险指标可视化图表
    print(f"\n📊 生成风险指标可视化图表...")
    print("-" * 30)
    if risk_analysis:
        analyzer.create_risk_visualization(risk_analysis)
    
    # 创建板块轮动可视化图表
    print(f"\n📊 生成板块轮动可视化图表...")
    print("-" * 30)
    if sector_rotation:
        analyzer.create_sector_rotation_visualization(sector_rotation)
    
    # 生成综合报告
    analyzer.generate_report(sentiment_results, sector_rotation, risk_analysis)
    
    print("\n✅ 市场情绪分析完成!")


if __name__ == "__main__":
    main()