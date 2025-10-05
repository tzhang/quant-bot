#!/usr/bin/env python3
"""
市场情绪分析工具

基于技术指标和价格行为分析市场情绪，帮助判断市场趋势
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


class MarketSentimentAnalyzer:
    """市场情绪分析器"""
    
    def __init__(self):
        """初始化市场情绪分析器"""
        self.data_cache_dir = Path("data_cache")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 市场代表性指数和股票
        self.market_indicators = {
            '大盘指数': ['SPY', 'QQQ', 'IWM'],  # S&P500, 纳斯达克, 小盘股
            '科技龙头': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            '传统价值': ['JPM', 'JNJ', 'PG', 'KO'],
            '成长股': ['TSLA', 'AMZN', 'META', 'NFLX']
        }
        
        print("🎯 市场情绪分析器初始化完成")
    
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
            return None
    
    def calculate_sentiment_indicators(self, data: pd.DataFrame) -> Dict:
        """
        计算情绪指标
        
        Args:
            data: 股票数据
            
        Returns:
            情绪指标字典
        """
        indicators = {}
        
        # RSI - 相对强弱指标
        indicators['RSI'] = TechnicalIndicators.rsi(data['Close'])
        
        # 布林带
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(data['Close'])
        indicators['BB_Upper'] = bb_upper
        indicators['BB_Middle'] = bb_middle
        indicators['BB_Lower'] = bb_lower
        indicators['BB_Position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        macd_line, macd_signal, macd_histogram = TechnicalIndicators.macd(data['Close'])
        indicators['MACD'] = macd_line
        indicators['MACD_Signal'] = macd_signal
        indicators['MACD_Histogram'] = macd_histogram
        
        # 移动平均线
        indicators['SMA_20'] = TechnicalIndicators.sma(data['Close'], 20)
        indicators['SMA_50'] = TechnicalIndicators.sma(data['Close'], 50)
        indicators['EMA_12'] = TechnicalIndicators.ema(data['Close'], 12)
        
        # 价格动量
        indicators['Price_Change_1d'] = data['Close'].pct_change(1)
        indicators['Price_Change_5d'] = data['Close'].pct_change(5)
        indicators['Price_Change_20d'] = data['Close'].pct_change(20)
        
        # 成交量指标
        indicators['Volume_SMA'] = TechnicalIndicators.sma(data['Volume'], 20)
        indicators['Volume_Ratio'] = data['Volume'] / indicators['Volume_SMA']
        
        return indicators
    
    def analyze_market_sentiment(self, symbol: str) -> Dict:
        """
        分析单个股票的市场情绪
        
        Args:
            symbol: 股票代码
            
        Returns:
            情绪分析结果
        """
        data = self._load_cached_stock_data(symbol)
        if data is None:
            return None
        
        indicators = self.calculate_sentiment_indicators(data)
        
        # 获取最新值
        latest_data = {}
        for key, value in indicators.items():
            if isinstance(value, pd.Series) and len(value) > 0:
                latest_data[key] = value.iloc[-1]
        
        # 情绪评分
        sentiment_score = self._calculate_sentiment_score(latest_data)
        
        # 趋势分析
        trend_analysis = self._analyze_trend(data, indicators)
        
        return {
            'symbol': symbol,
            'latest_data': latest_data,
            'sentiment_score': sentiment_score,
            'trend_analysis': trend_analysis,
            'raw_indicators': indicators
        }
    
    def _calculate_sentiment_score(self, latest_data: Dict) -> Dict:
        """
        计算情绪评分
        
        Args:
            latest_data: 最新指标数据
            
        Returns:
            情绪评分字典
        """
        scores = {}
        
        # RSI评分 (0-100)
        rsi = latest_data.get('RSI', 50)
        if rsi > 70:
            scores['RSI'] = {'value': rsi, 'signal': '超买', 'score': -20}
        elif rsi < 30:
            scores['RSI'] = {'value': rsi, 'signal': '超卖', 'score': 20}
        else:
            scores['RSI'] = {'value': rsi, 'signal': '中性', 'score': 0}
        
        # 布林带位置评分
        bb_pos = latest_data.get('BB_Position', 0.5)
        if bb_pos > 0.8:
            scores['Bollinger'] = {'value': bb_pos, 'signal': '接近上轨', 'score': -10}
        elif bb_pos < 0.2:
            scores['Bollinger'] = {'value': bb_pos, 'signal': '接近下轨', 'score': 10}
        else:
            scores['Bollinger'] = {'value': bb_pos, 'signal': '中性', 'score': 0}
        
        # MACD评分
        macd = latest_data.get('MACD', 0)
        macd_signal = latest_data.get('MACD_Signal', 0)
        if macd > macd_signal:
            scores['MACD'] = {'signal': '金叉', 'score': 15}
        else:
            scores['MACD'] = {'signal': '死叉', 'score': -15}
        
        # 价格动量评分
        price_change_20d = latest_data.get('Price_Change_20d', 0) * 100
        if price_change_20d > 10:
            scores['Momentum'] = {'value': price_change_20d, 'signal': '强势上涨', 'score': 20}
        elif price_change_20d > 5:
            scores['Momentum'] = {'value': price_change_20d, 'signal': '温和上涨', 'score': 10}
        elif price_change_20d < -10:
            scores['Momentum'] = {'value': price_change_20d, 'signal': '大幅下跌', 'score': -20}
        elif price_change_20d < -5:
            scores['Momentum'] = {'value': price_change_20d, 'signal': '温和下跌', 'score': -10}
        else:
            scores['Momentum'] = {'value': price_change_20d, 'signal': '横盘整理', 'score': 0}
        
        # 成交量评分
        volume_ratio = latest_data.get('Volume_Ratio', 1)
        if volume_ratio > 1.5:
            scores['Volume'] = {'value': volume_ratio, 'signal': '放量', 'score': 10}
        elif volume_ratio < 0.7:
            scores['Volume'] = {'value': volume_ratio, 'signal': '缩量', 'score': -5}
        else:
            scores['Volume'] = {'value': volume_ratio, 'signal': '正常', 'score': 0}
        
        # 综合评分
        total_score = sum([s['score'] for s in scores.values()])
        
        if total_score > 20:
            overall_sentiment = '非常乐观'
        elif total_score > 10:
            overall_sentiment = '乐观'
        elif total_score > -10:
            overall_sentiment = '中性'
        elif total_score > -20:
            overall_sentiment = '悲观'
        else:
            overall_sentiment = '非常悲观'
        
        return {
            'individual_scores': scores,
            'total_score': total_score,
            'overall_sentiment': overall_sentiment
        }
    
    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """
        分析趋势
        
        Args:
            data: 股票数据
            indicators: 技术指标
            
        Returns:
            趋势分析结果
        """
        current_price = data['Close'].iloc[-1]
        sma_20 = indicators['SMA_20'].iloc[-1]
        sma_50 = indicators['SMA_50'].iloc[-1]
        
        # 趋势判断
        if current_price > sma_20 > sma_50:
            trend = '上升趋势'
            trend_strength = '强'
        elif current_price > sma_20:
            trend = '短期上升'
            trend_strength = '中'
        elif current_price < sma_20 < sma_50:
            trend = '下降趋势'
            trend_strength = '强'
        elif current_price < sma_20:
            trend = '短期下降'
            trend_strength = '中'
        else:
            trend = '横盘整理'
            trend_strength = '弱'
        
        # 支撑阻力位
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        
        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'resistance': recent_high,
            'support': recent_low
        }
    
    def analyze_market_overview(self) -> Dict:
        """
        分析整体市场情绪
        
        Returns:
            市场概览分析结果
        """
        print("🔍 正在分析整体市场情绪...")
        
        all_results = {}
        category_sentiments = {}
        
        for category, symbols in self.market_indicators.items():
            print(f"📊 分析 {category}...")
            category_results = []
            
            for symbol in symbols:
                result = self.analyze_market_sentiment(symbol)
                if result:
                    all_results[symbol] = result
                    category_results.append(result)
                    print(f"  ✅ {symbol}: {result['sentiment_score']['overall_sentiment']}")
                else:
                    print(f"  ❌ {symbol}: 数据不可用")
            
            if category_results:
                # 计算类别平均情绪
                avg_score = np.mean([r['sentiment_score']['total_score'] for r in category_results])
                category_sentiments[category] = {
                    'average_score': avg_score,
                    'count': len(category_results),
                    'results': category_results
                }
        
        return {
            'individual_results': all_results,
            'category_sentiments': category_sentiments
        }
    
    def create_sentiment_visualization(self, market_analysis: Dict):
        """
        创建情绪分析可视化图表
        
        Args:
            market_analysis: 市场分析结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('市场情绪分析仪表板', fontsize=16, fontweight='bold')
        
        # 1. 整体市场情绪分布
        ax1 = axes[0, 0]
        
        sentiments = []
        symbols = []
        scores = []
        
        for symbol, result in market_analysis['individual_results'].items():
            sentiments.append(result['sentiment_score']['overall_sentiment'])
            symbols.append(symbol)
            scores.append(result['sentiment_score']['total_score'])
        
        # 情绪分布饼图
        sentiment_counts = pd.Series(sentiments).value_counts()
        colors = ['red', 'orange', 'gray', 'lightgreen', 'green']
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors[:len(sentiment_counts)])
        ax1.set_title('整体市场情绪分布')
        
        # 2. 各板块情绪对比
        ax2 = axes[0, 1]
        
        categories = list(market_analysis['category_sentiments'].keys())
        category_scores = [market_analysis['category_sentiments'][cat]['average_score'] 
                          for cat in categories]
        
        bars = ax2.bar(categories, category_scores, 
                       color=['green' if score > 0 else 'red' for score in category_scores])
        ax2.set_title('各板块情绪评分')
        ax2.set_ylabel('平均情绪评分')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, category_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{score:.1f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. 个股情绪评分排名
        ax3 = axes[1, 0]
        
        # 按评分排序
        sorted_data = sorted(zip(symbols, scores), key=lambda x: x[1], reverse=True)
        sorted_symbols, sorted_scores = zip(*sorted_data)
        
        colors = ['green' if score > 10 else 'lightgreen' if score > 0 else 
                 'orange' if score > -10 else 'red' for score in sorted_scores]
        
        bars = ax3.barh(range(len(sorted_symbols)), sorted_scores, color=colors)
        ax3.set_yticks(range(len(sorted_symbols)))
        ax3.set_yticklabels(sorted_symbols)
        ax3.set_xlabel('情绪评分')
        ax3.set_title('个股情绪评分排名')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # 4. RSI分布热力图
        ax4 = axes[1, 1]
        
        # 创建RSI数据矩阵
        rsi_data = []
        rsi_symbols = []
        
        for symbol, result in market_analysis['individual_results'].items():
            rsi_value = result['sentiment_score']['individual_scores'].get('RSI', {}).get('value', 50)
            rsi_data.append(rsi_value)
            rsi_symbols.append(symbol)
        
        # 将RSI值重新整理为矩阵形式
        n_cols = 4
        n_rows = (len(rsi_data) + n_cols - 1) // n_cols
        
        rsi_matrix = np.full((n_rows, n_cols), np.nan)
        symbol_matrix = [[''] * n_cols for _ in range(n_rows)]
        
        for i, (rsi, symbol) in enumerate(zip(rsi_data, rsi_symbols)):
            row, col = divmod(i, n_cols)
            rsi_matrix[row, col] = rsi
            symbol_matrix[row][col] = symbol
        
        # 创建热力图
        im = ax4.imshow(rsi_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
        
        # 添加文本标签
        for i in range(n_rows):
            for j in range(n_cols):
                if not np.isnan(rsi_matrix[i, j]):
                    text = f"{symbol_matrix[i][j]}\n{rsi_matrix[i, j]:.1f}"
                    ax4.text(j, i, text, ha="center", va="center", fontsize=8)
        
        ax4.set_title('RSI热力图')
        ax4.set_xticks([])
        ax4.set_yticks([])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('RSI值')
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.results_dir / 'market_sentiment_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 市场情绪分析图表已保存: {chart_path}")
        
        plt.show()
    
    def generate_sentiment_report(self, market_analysis: Dict):
        """
        生成市场情绪分析报告
        
        Args:
            market_analysis: 市场分析结果
        """
        print("\n" + "="*60)
        print("🎯 市场情绪分析报告")
        print("="*60)
        
        # 整体市场情绪
        all_scores = [result['sentiment_score']['total_score'] 
                     for result in market_analysis['individual_results'].values()]
        avg_market_score = np.mean(all_scores)
        
        print(f"\n📊 整体市场情绪评分: {avg_market_score:.1f}")
        
        if avg_market_score > 15:
            market_mood = "非常乐观 🚀"
        elif avg_market_score > 5:
            market_mood = "乐观 📈"
        elif avg_market_score > -5:
            market_mood = "中性 ➡️"
        elif avg_market_score > -15:
            market_mood = "悲观 📉"
        else:
            market_mood = "非常悲观 ⚠️"
        
        print(f"市场整体情绪: {market_mood}")
        
        # 板块分析
        print("\n🏭 板块情绪分析:")
        print("-" * 40)
        
        for category, data in market_analysis['category_sentiments'].items():
            score = data['average_score']
            count = data['count']
            
            if score > 10:
                mood_icon = "🚀"
            elif score > 0:
                mood_icon = "📈"
            elif score > -10:
                mood_icon = "📉"
            else:
                mood_icon = "⚠️"
            
            print(f"{category:>8}: {score:6.1f} {mood_icon} ({count}只股票)")
        
        # 个股详细分析
        print("\n📈 个股情绪详情:")
        print("-" * 60)
        
        # 按评分排序
        sorted_results = sorted(
            market_analysis['individual_results'].items(),
            key=lambda x: x[1]['sentiment_score']['total_score'],
            reverse=True
        )
        
        for symbol, result in sorted_results:
            score = result['sentiment_score']['total_score']
            sentiment = result['sentiment_score']['overall_sentiment']
            trend = result['trend_analysis']['trend']
            
            # 获取关键指标
            rsi = result['sentiment_score']['individual_scores'].get('RSI', {}).get('value', 0)
            momentum = result['sentiment_score']['individual_scores'].get('Momentum', {}).get('value', 0)
            
            print(f"{symbol:>6}: {score:4.0f}分 | {sentiment:>6} | {trend:>8} | "
                  f"RSI:{rsi:5.1f} | 动量:{momentum:5.1f}%")
        
        # 市场信号分析
        print("\n🔍 市场信号分析:")
        print("-" * 40)
        
        # 统计各种信号
        rsi_overbought = sum(1 for result in market_analysis['individual_results'].values()
                            if result['sentiment_score']['individual_scores'].get('RSI', {}).get('signal') == '超买')
        rsi_oversold = sum(1 for result in market_analysis['individual_results'].values()
                          if result['sentiment_score']['individual_scores'].get('RSI', {}).get('signal') == '超卖')
        
        macd_bullish = sum(1 for result in market_analysis['individual_results'].values()
                          if result['sentiment_score']['individual_scores'].get('MACD', {}).get('signal') == '金叉')
        
        strong_momentum = sum(1 for result in market_analysis['individual_results'].values()
                             if result['sentiment_score']['individual_scores'].get('Momentum', {}).get('signal') in ['强势上涨', '温和上涨'])
        
        total_stocks = len(market_analysis['individual_results'])
        
        print(f"RSI超买股票: {rsi_overbought}/{total_stocks} ({rsi_overbought/total_stocks*100:.1f}%)")
        print(f"RSI超卖股票: {rsi_oversold}/{total_stocks} ({rsi_oversold/total_stocks*100:.1f}%)")
        print(f"MACD金叉股票: {macd_bullish}/{total_stocks} ({macd_bullish/total_stocks*100:.1f}%)")
        print(f"动量向上股票: {strong_momentum}/{total_stocks} ({strong_momentum/total_stocks*100:.1f}%)")
        
        # 投资建议框架
        print("\n💡 市场观察总结:")
        print("-" * 40)
        
        if avg_market_score > 10:
            print("• 市场情绪偏向乐观，但需注意是否过热")
            print("• 关注高估值股票的回调风险")
            print("• 可考虑适当获利了结")
        elif avg_market_score > 0:
            print("• 市场情绪温和积极")
            print("• 可关注基本面良好的成长股")
            print("• 保持适度乐观，控制仓位")
        elif avg_market_score > -10:
            print("• 市场情绪相对中性")
            print("• 可关注被低估的价值股")
            print("• 保持观望，等待更明确信号")
        else:
            print("• 市场情绪偏向悲观")
            print("• 可关注防御性股票")
            print("• 控制风险，保持现金比例")
        
        print("\n" + "="*60)
        print("⚠️ 重要提醒")
        print("="*60)
        print("1. 市场情绪分析基于技术指标，仅供参考")
        print("2. 投资决策应结合基本面分析")
        print("3. 市场情绪变化快速，需持续关注")
        print("4. 请根据个人风险承受能力做出投资决策")


def main():
    """主函数"""
    print("🎯 启动市场情绪分析...")
    
    analyzer = MarketSentimentAnalyzer()
    
    # 分析整体市场情绪
    market_analysis = analyzer.analyze_market_overview()
    
    if not market_analysis['individual_results']:
        print("❌ 没有可用的市场数据")
        return
    
    # 生成报告
    analyzer.generate_sentiment_report(market_analysis)
    
    # 创建可视化
    analyzer.create_sentiment_visualization(market_analysis)
    
    print("\n✅ 市场情绪分析完成！")
    print("📝 请记住：这只是分析工具，不构成投资建议！")


if __name__ == "__main__":
    main()