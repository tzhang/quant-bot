#!/usr/bin/env python3
"""
创建更真实的模拟数据用于市场情绪分析测试
包含更复杂的价格模式、趋势、波动率变化等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import math

class RealisticStockDataGenerator:
    """真实股票数据生成器"""
    
    def __init__(self):
        # 股票基础信息
        self.stock_info = {
            'AAPL': {'base_price': 150, 'volatility': 0.25, 'trend': 0.08, 'sector': 'tech'},
            'MSFT': {'base_price': 300, 'volatility': 0.22, 'trend': 0.12, 'sector': 'tech'},
            'GOOGL': {'base_price': 2500, 'volatility': 0.28, 'trend': 0.06, 'sector': 'tech'},
            'SPY': {'base_price': 400, 'volatility': 0.16, 'trend': 0.10, 'sector': 'index'},
            'QQQ': {'base_price': 350, 'volatility': 0.20, 'trend': 0.15, 'sector': 'tech_index'},
            'NVDA': {'base_price': 800, 'volatility': 0.45, 'trend': 0.25, 'sector': 'tech'},
            'TSLA': {'base_price': 200, 'volatility': 0.55, 'trend': 0.05, 'sector': 'auto'},
            'AMZN': {'base_price': 3000, 'volatility': 0.30, 'trend': 0.08, 'sector': 'tech'},
            'META': {'base_price': 250, 'volatility': 0.35, 'trend': 0.02, 'sector': 'tech'},
            'NFLX': {'base_price': 400, 'volatility': 0.40, 'trend': -0.05, 'sector': 'media'},
            'JPM': {'base_price': 150, 'volatility': 0.25, 'trend': 0.06, 'sector': 'finance'},
            'JNJ': {'base_price': 160, 'volatility': 0.15, 'trend': 0.04, 'sector': 'healthcare'},
            'PG': {'base_price': 140, 'volatility': 0.12, 'trend': 0.03, 'sector': 'consumer'},
            'KO': {'base_price': 60, 'volatility': 0.14, 'trend': 0.02, 'sector': 'consumer'},
            'IWM': {'base_price': 180, 'volatility': 0.22, 'trend': 0.08, 'sector': 'small_cap'}
        }
        
        # 市场状态参数
        self.market_regimes = {
            'bull': {'trend_multiplier': 1.5, 'volatility_multiplier': 0.8},
            'bear': {'trend_multiplier': -1.2, 'volatility_multiplier': 1.5},
            'sideways': {'trend_multiplier': 0.1, 'volatility_multiplier': 1.0}
        }
    
    def generate_market_regime_schedule(self, days: int) -> list:
        """生成市场状态时间表"""
        regimes = []
        current_day = 0
        
        while current_day < days:
            # 随机选择市场状态和持续时间
            regime = np.random.choice(['bull', 'bear', 'sideways'], p=[0.4, 0.2, 0.4])
            duration = np.random.randint(5, 20)  # 5-20天的市场状态
            
            for _ in range(min(duration, days - current_day)):
                regimes.append(regime)
                current_day += 1
        
        return regimes[:days]
    
    def add_intraday_patterns(self, open_price: float, close_price: float) -> tuple:
        """添加日内价格模式"""
        # 生成更真实的高低价
        price_range = abs(close_price - open_price)
        base_volatility = max(price_range, open_price * 0.005)  # 最小0.5%的日内波动
        
        # 随机决定是否有突破或回调
        breakthrough = np.random.random() < 0.15  # 15%概率出现突破
        
        if breakthrough:
            # 突破模式：高低价范围更大
            high_extension = np.random.uniform(0.5, 2.0) * base_volatility
            low_extension = np.random.uniform(0.5, 2.0) * base_volatility
        else:
            # 正常模式
            high_extension = np.random.uniform(0.2, 0.8) * base_volatility
            low_extension = np.random.uniform(0.2, 0.8) * base_volatility
        
        high = max(open_price, close_price) + high_extension
        low = min(open_price, close_price) - low_extension
        
        # 确保价格合理性
        low = max(low, min(open_price, close_price) * 0.95)
        
        return high, low
    
    def generate_volume_pattern(self, symbol: str, price_change_pct: float, 
                              base_volume: int, day_index: int) -> int:
        """生成更真实的成交量模式"""
        info = self.stock_info[symbol]
        
        # 基础成交量根据股票类型调整
        if info['sector'] == 'tech':
            volume_multiplier = 1.5
        elif info['sector'] == 'index':
            volume_multiplier = 3.0
        else:
            volume_multiplier = 1.0
        
        # 价格变化与成交量的关系
        price_volume_correlation = 1 + abs(price_change_pct) * 2
        
        # 周期性模式（周一和周五成交量通常更高）
        weekday = day_index % 5
        if weekday in [0, 4]:  # 周一和周五
            weekly_multiplier = 1.2
        else:
            weekly_multiplier = 1.0
        
        # 随机波动
        random_factor = np.random.lognormal(0, 0.3)
        
        volume = int(base_volume * volume_multiplier * price_volume_correlation * 
                    weekly_multiplier * random_factor)
        
        return max(volume, 1000)  # 最小成交量
    
    def add_earnings_events(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """添加财报事件影响"""
        # 随机选择1-2个财报日期
        earnings_days = np.random.choice(len(data), size=np.random.randint(1, 3), replace=False)
        
        for day in earnings_days:
            if day < len(data) - 1:
                # 财报前的不确定性（降低成交量，增加波动）
                if day > 0:
                    data.iloc[day-1, data.columns.get_loc('Volume')] *= 0.7
                
                # 财报日（大幅价格变动和成交量）
                surprise = np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.08)  # 3-8%的意外变动
                current_close = data.iloc[day]['Close']
                new_close = current_close * (1 + surprise)
                
                data.iloc[day, data.columns.get_loc('Close')] = new_close
                data.iloc[day, data.columns.get_loc('Volume')] *= np.random.uniform(2.0, 4.0)
                
                # 调整后续几天的开盘价
                if day < len(data) - 1:
                    data.iloc[day+1, data.columns.get_loc('Open')] = new_close * (1 + np.random.normal(0, 0.01))
        
        return data
    
    def generate_realistic_stock_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        生成更真实的股票数据
        
        Args:
            symbol: 股票代码
            days: 天数
            
        Returns:
            真实模拟股票数据DataFrame
        """
        # 设置随机种子
        np.random.seed(hash(symbol) % 2**32)
        
        # 生成交易日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.4))  # 多生成一些日期以过滤周末
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_dates = all_dates[all_dates.weekday < 5][:days]  # 只保留工作日
        
        # 获取股票信息
        info = self.stock_info[symbol]
        base_price = info['base_price']
        annual_volatility = info['volatility']
        annual_trend = info['trend']
        
        # 生成市场状态时间表
        market_regimes = self.generate_market_regime_schedule(len(trading_dates))
        
        # 生成价格序列
        prices = [base_price]
        volumes = []
        
        for i, (date, regime) in enumerate(zip(trading_dates[1:], market_regimes[1:]), 1):
            # 获取市场状态参数
            regime_params = self.market_regimes[regime]
            
            # 计算日收益率
            daily_trend = (annual_trend * regime_params['trend_multiplier']) / 252
            daily_volatility = (annual_volatility * regime_params['volatility_multiplier']) / math.sqrt(252)
            
            # 添加均值回归效应
            price_deviation = (prices[-1] - base_price) / base_price
            mean_reversion = -0.1 * price_deviation  # 轻微的均值回归
            
            # 生成随机收益率
            random_return = np.random.normal(daily_trend + mean_reversion, daily_volatility)
            
            # 计算新价格
            new_price = prices[-1] * (1 + random_return)
            new_price = max(new_price, base_price * 0.3)  # 防止价格过低
            prices.append(new_price)
        
        # 生成OHLCV数据
        data = []
        base_volume = int(np.random.lognormal(15, 0.5))
        
        for i, (date, close_price) in enumerate(zip(trading_dates, prices)):
            # 生成开盘价
            if i == 0:
                open_price = close_price * (1 + np.random.normal(0, 0.002))
            else:
                # 考虑隔夜跳空
                gap = np.random.normal(0, 0.005)
                if abs(gap) > 0.015:  # 大跳空概率较低
                    gap = gap * 0.3
                open_price = prices[i-1] * (1 + gap)
            
            # 生成高低价
            high, low = self.add_intraday_patterns(open_price, close_price)
            
            # 生成成交量
            price_change_pct = (close_price - open_price) / open_price if open_price > 0 else 0
            volume = self.generate_volume_pattern(symbol, price_change_pct, base_volume, i)
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=trading_dates)
        
        # 添加特殊事件影响
        df = self.add_earnings_events(df, symbol)
        
        # 最终数据清理
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return df

def create_realistic_cache_files():
    """创建更真实的缓存文件"""
    print("🚀 创建更真实的模拟数据缓存文件...")
    
    # 创建缓存目录
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # 清理旧的缓存文件
    try:
        for old_file in cache_dir.glob("*.csv"):
            if old_file.exists():
                old_file.unlink()
        print("  🧹 清理旧缓存文件")
    except Exception as e:
        print(f"  ⚠️ 清理缓存文件时出现问题: {e}")
        print("  继续生成新数据...")
    
    # 初始化生成器
    generator = RealisticStockDataGenerator()
    
    # 股票列表
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'NVDA', 'TSLA', 
               'AMZN', 'META', 'NFLX', 'JPM', 'JNJ', 'PG', 'KO', 'IWM']
    
    success_count = 0
    
    for symbol in symbols:
        try:
            # 生成更真实的数据
            data = generator.generate_realistic_stock_data(symbol, days=60)
            
            # 创建缓存文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ohlcv_{symbol}_{timestamp}.csv"
            filepath = cache_dir / filename
            
            # 写入文件（包含元数据）
            with open(filepath, 'w') as f:
                f.write(f"# Symbol: {symbol}\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Type: Realistic Simulation\n")
                f.write(f"# Base Price: {generator.stock_info[symbol]['base_price']}\n")
                f.write(f"# Volatility: {generator.stock_info[symbol]['volatility']:.2%}\n")
                f.write(f"# Trend: {generator.stock_info[symbol]['trend']:.2%}\n")
                
                # 添加Price列（索引）然后是OHLCV数据
                data_with_price = data.copy()
                data_with_price.insert(0, 'Price', data_with_price.index.strftime('%Y-%m-%d'))
                data_with_price.to_csv(f, index=False)
            
            # 计算一些统计信息
            returns = data['Close'].pct_change().dropna()
            realized_vol = returns.std() * math.sqrt(252)
            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            
            print(f"  ✅ {symbol}: {len(data)} 条记录 | 收益率: {total_return:+.1f}% | 波动率: {realized_vol:.1%}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ {symbol}: 生成失败 - {e}")
    
    print(f"\n🎯 成功创建 {success_count}/{len(symbols)} 个股票的真实模拟缓存文件")
    print("✅ 更真实的模拟数据创建完成！")
    print("📊 数据特点：")
    print("   • 包含市场状态变化（牛市/熊市/横盘）")
    print("   • 真实的价量关系")
    print("   • 财报事件影响")
    print("   • 行业特征差异")
    print("   • 日内价格模式")

if __name__ == "__main__":
    create_realistic_cache_files()