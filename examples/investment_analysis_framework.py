#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资分析框架
===========

这个脚本提供了一个系统性的投资分析框架，帮助您：
1. 分析市场趋势和板块轮动
2. 评估个股的技术面和基本面
3. 构建多因子评分模型
4. 进行风险评估和组合优化

注意：这只是分析工具，不构成投资建议！
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from src.data.data_manager import DataManager
from src.factors.engine import FactorEngine
from src.utils.indicators import TechnicalIndicators

class InvestmentAnalysisFramework:
    """投资分析框架类"""
    
    def __init__(self):
        """初始化分析框架"""
        self.data_manager = DataManager()
        self.factor_engine = FactorEngine()
        self.tech_indicators = TechnicalIndicators()
        
        # 定义不同板块的代表性股票
        self.sectors = {
            '科技股': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            '金融股': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            '消费股': ['KO', 'PEP', 'WMT', 'HD', 'DIS'],
            '医疗股': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
            '能源股': ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
        }
        
        print("🔍 投资分析框架初始化完成")
        print("📊 支持的分析功能：")
        print("   1. 市场趋势分析")
        print("   2. 板块轮动分析") 
        print("   3. 个股多因子评分")
        print("   4. 风险收益分析")
        print("   5. 投资组合优化建议")
    
    def analyze_market_trend(self, lookback_days=90):
        """分析市场整体趋势"""
        print(f"\n📈 正在分析最近{lookback_days}天的市场趋势...")
        
        # 获取市场指数数据（使用SPY作为市场代理）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')
        
        try:
            market_data = self.data_manager.get_data('SPY', start_date, end_date)
            if market_data is None or market_data.empty:
                print("❌ 无法获取市场数据，使用缓存数据分析")
                return self._analyze_cached_market_data()
            
            # 计算技术指标
            market_data['SMA_20'] = market_data['Close'].rolling(20).mean()
            market_data['SMA_50'] = market_data['Close'].rolling(50).mean()
            market_data['RSI'] = TechnicalIndicators.rsi(market_data['Close'])
            
            # 分析趋势
            current_price = market_data['Close'].iloc[-1]
            sma_20 = market_data['SMA_20'].iloc[-1]
            sma_50 = market_data['SMA_50'].iloc[-1]
            rsi = market_data['RSI'].iloc[-1]
            
            # 趋势判断
            trend_signals = []
            if current_price > sma_20 > sma_50:
                trend_signals.append("📈 短期上升趋势")
            elif current_price < sma_20 < sma_50:
                trend_signals.append("📉 短期下降趋势")
            else:
                trend_signals.append("📊 趋势不明确")
            
            if rsi > 70:
                trend_signals.append("⚠️ 市场可能超买")
            elif rsi < 30:
                trend_signals.append("💡 市场可能超卖")
            else:
                trend_signals.append("✅ RSI处于正常区间")
            
            print("🎯 市场趋势分析结果：")
            for signal in trend_signals:
                print(f"   {signal}")
            
            return {
                'trend_signals': trend_signals,
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi
            }
            
        except Exception as e:
            print(f"❌ 市场趋势分析出错: {e}")
            return None
    
    def analyze_sector_rotation(self):
        """分析板块轮动情况"""
        print("\n🔄 正在分析板块轮动情况...")
        
        sector_performance = {}
        
        for sector_name, stocks in self.sectors.items():
            print(f"   分析 {sector_name}...")
            
            sector_returns = []
            for stock in stocks[:3]:  # 只分析前3只股票以节省时间
                try:
                    # 尝试从缓存加载数据
                    data = self._load_cached_stock_data(stock)
                    if data is not None and len(data) > 20:
                        # 计算最近20天收益率
                        recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
                        sector_returns.append(recent_return)
                except Exception as e:
                    print(f"     ⚠️ {stock} 数据获取失败: {e}")
                    continue
            
            if sector_returns:
                avg_return = np.mean(sector_returns)
                sector_performance[sector_name] = avg_return
                print(f"     {sector_name}: {avg_return:.2f}%")
        
        # 排序板块表现
        if sector_performance:
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            
            print("\n🏆 板块表现排名（最近20天）：")
            for i, (sector, performance) in enumerate(sorted_sectors, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                print(f"   {emoji} {i}. {sector}: {performance:.2f}%")
            
            return sorted_sectors
        else:
            print("❌ 无法获取板块数据")
            return None
    
    def analyze_stock_multifactor(self, symbol):
        """对个股进行多因子分析"""
        print(f"\n🔍 正在对 {symbol} 进行多因子分析...")
        
        try:
            # 尝试从缓存加载数据
            data = self._load_cached_stock_data(symbol)
            if data is None or len(data) < 50:
                print(f"❌ {symbol} 数据不足，跳过分析")
                return None
            
            # 计算各类因子
            factors = {}
            
            # 技术因子
            factors['RSI'] = TechnicalIndicators.rsi(data['Close']).iloc[-1]
            factors['MACD_Signal'] = self._calculate_macd_signal(data)
            factors['Bollinger_Position'] = self._calculate_bollinger_position(data)
            
            # 动量因子
            factors['Momentum_20d'] = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
            factors['Momentum_60d'] = (data['Close'].iloc[-1] / data['Close'].iloc[-60] - 1) * 100 if len(data) > 60 else 0
            
            # 波动率因子
            factors['Volatility_20d'] = data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            # 成交量因子
            factors['Volume_Ratio'] = data['Volume'].iloc[-5:].mean() / data['Volume'].iloc[-20:-5].mean() if len(data) > 20 else 1
            
            # 计算综合评分
            score = self._calculate_composite_score(factors)
            
            print(f"📊 {symbol} 多因子分析结果：")
            print(f"   RSI: {factors['RSI']:.1f}")
            print(f"   20日动量: {factors['Momentum_20d']:.2f}%")
            print(f"   60日动量: {factors['Momentum_60d']:.2f}%")
            print(f"   波动率: {factors['Volatility_20d']:.2f}%")
            print(f"   成交量比率: {factors['Volume_Ratio']:.2f}")
            print(f"   🎯 综合评分: {score:.1f}/100")
            
            return {
                'symbol': symbol,
                'factors': factors,
                'composite_score': score
            }
            
        except Exception as e:
            print(f"❌ {symbol} 多因子分析出错: {e}")
            return None
    
    def generate_investment_framework(self):
        """生成投资分析框架报告"""
        print("\n" + "="*60)
        print("📋 投资分析框架报告")
        print("="*60)
        
        # 1. 市场趋势分析
        market_analysis = self.analyze_market_trend()
        
        # 2. 板块轮动分析
        sector_analysis = self.analyze_sector_rotation()
        
        # 3. 重点股票分析
        print("\n🎯 重点股票多因子分析：")
        focus_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
        stock_scores = []
        
        for stock in focus_stocks:
            result = self.analyze_stock_multifactor(stock)
            if result:
                stock_scores.append(result)
        
        # 4. 生成投资建议框架
        self._generate_investment_suggestions(market_analysis, sector_analysis, stock_scores)
    
    def _load_cached_stock_data(self, symbol):
        """从缓存加载股票数据"""
        try:
            cache_dir = "/Users/tony/codebase/my-quant/data_cache"
            
            # 查找缓存文件
            import glob
            pattern = f"{cache_dir}/ohlcv_{symbol}_*.csv"
            files = glob.glob(pattern)
            
            if files:
                # 使用最新的文件
                latest_file = max(files, key=os.path.getctime)
                
                # 读取CSV文件，跳过前两行
                df = pd.read_csv(latest_file, skiprows=2)
                
                # 重新命名列
                if len(df.columns) >= 6:
                    df = df.iloc[:, 1:6]  # 去掉第一列Price
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # 设置日期索引
                    df.index = pd.to_datetime(df.index)
                    
                    return df
            
            return None
            
        except Exception as e:
            print(f"     ⚠️ 加载 {symbol} 缓存数据失败: {e}")
            return None
    
    def _calculate_macd_signal(self, data):
        """计算MACD信号"""
        try:
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            if macd.iloc[-1] > signal.iloc[-1]:
                return 1  # 买入信号
            else:
                return -1  # 卖出信号
        except:
            return 0
    
    def _calculate_bollinger_position(self, data):
        """计算布林带位置"""
        try:
            sma = data['Close'].rolling(20).mean()
            std = data['Close'].rolling(20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            
            current_price = data['Close'].iloc[-1]
            position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            return position
        except:
            return 0.5
    
    def _calculate_composite_score(self, factors):
        """计算综合评分"""
        score = 50  # 基础分
        
        # RSI评分
        rsi = factors.get('RSI', 50)
        if 30 <= rsi <= 70:
            score += 10
        elif rsi < 30:
            score += 5  # 超卖可能是机会
        else:
            score -= 10  # 超买风险
        
        # 动量评分
        momentum_20 = factors.get('Momentum_20d', 0)
        if momentum_20 > 5:
            score += 15
        elif momentum_20 > 0:
            score += 5
        else:
            score -= 10
        
        # MACD评分
        macd_signal = factors.get('MACD_Signal', 0)
        score += macd_signal * 10
        
        # 布林带位置评分
        bb_pos = factors.get('Bollinger_Position', 0.5)
        if 0.2 <= bb_pos <= 0.8:
            score += 10
        
        # 成交量评分
        vol_ratio = factors.get('Volume_Ratio', 1)
        if vol_ratio > 1.2:
            score += 5
        
        return max(0, min(100, score))
    
    def _analyze_cached_market_data(self):
        """使用缓存数据分析市场"""
        print("   使用AAPL作为市场代理进行分析...")
        
        data = self._load_cached_stock_data('AAPL')
        if data is not None and len(data) > 20:
            current_price = data['Close'].iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            rsi = TechnicalIndicators.rsi(data['Close']).iloc[-1]
            
            trend_signals = []
            if current_price > sma_20:
                trend_signals.append("📈 短期趋势向上")
            else:
                trend_signals.append("📉 短期趋势向下")
            
            if rsi > 70:
                trend_signals.append("⚠️ 可能超买")
            elif rsi < 30:
                trend_signals.append("💡 可能超卖")
            else:
                trend_signals.append("✅ RSI正常")
            
            print("🎯 市场趋势分析结果（基于AAPL）：")
            for signal in trend_signals:
                print(f"   {signal}")
            
            return {'trend_signals': trend_signals}
        
        return None
    
    def _generate_investment_suggestions(self, market_analysis, sector_analysis, stock_scores):
        """生成投资建议框架"""
        print("\n" + "="*60)
        print("💡 投资分析框架总结")
        print("="*60)
        
        print("\n📊 基于数据的观察（仅供参考）：")
        
        # 市场环境评估
        if market_analysis and market_analysis.get('trend_signals'):
            print("\n🌍 市场环境：")
            for signal in market_analysis['trend_signals']:
                print(f"   • {signal}")
        
        # 板块建议
        if sector_analysis:
            print(f"\n🏭 表现较好的板块：")
            for i, (sector, performance) in enumerate(sector_analysis[:3]):
                print(f"   • {sector}: {performance:.2f}%")
        
        # 个股评分
        if stock_scores:
            sorted_stocks = sorted(stock_scores, key=lambda x: x['composite_score'], reverse=True)
            print(f"\n📈 个股综合评分排名：")
            for stock in sorted_stocks:
                print(f"   • {stock['symbol']}: {stock['composite_score']:.1f}/100")
        
        print("\n" + "="*60)
        print("⚠️  重要提醒")
        print("="*60)
        print("1. 以上分析仅基于历史数据和技术指标")
        print("2. 不构成投资建议，请结合自身情况判断")
        print("3. 投资有风险，入市需谨慎")
        print("4. 建议咨询专业投资顾问")
        print("5. 做好风险管理和资金配置")

def main():
    """主函数"""
    print("🚀 启动投资分析框架...")
    
    # 创建分析框架
    framework = InvestmentAnalysisFramework()
    
    # 生成完整的投资分析报告
    framework.generate_investment_framework()
    
    print(f"\n✅ 投资分析框架运行完成！")
    print(f"📝 请记住：这只是分析工具，不是投资建议！")

if __name__ == "__main__":
    main()