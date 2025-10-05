#!/usr/bin/env python3
"""
投资组合策略分析工具
基于用户实际持仓制定未来3个月的交易策略

作者: Quant Bot
日期: 2025-01-05
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PortfolioStrategyAnalyzer:
    def __init__(self):
        """初始化投资组合策略分析器"""
        # 用户当前持仓（基于提供的截图数据）
        self.current_holdings = {
            'HUBS': {'shares': 10, 'current_price': 452.60, 'change': 7.44, 'change_pct': 1.67},
            'MDB': {'shares': 1, 'current_price': 320.00, 'change': -6.29, 'change_pct': -1.93},
            'NIO': {'shares': 78, 'current_price': 7.68, 'change': -0.21, 'change_pct': -2.66},
            'OKTA': {'shares': 5, 'current_price': 93.45, 'change': -1.47, 'change_pct': -1.55},
            'TSLA': {'shares': 6, 'current_price': 429.90, 'change': -6.10, 'change_pct': -1.40}
        }
        
        # 股票基本信息
        self.stock_info = {
            'HUBS': {'name': 'HubSpot', 'sector': 'Technology', 'industry': 'Software'},
            'MDB': {'name': 'MongoDB', 'sector': 'Technology', 'industry': 'Software'},
            'NIO': {'name': 'NIO Inc', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers'},
            'OKTA': {'name': 'Okta', 'sector': 'Technology', 'industry': 'Software'},
            'TSLA': {'name': 'Tesla', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers'}
        }
        
        self.symbols = list(self.current_holdings.keys())
        
    def calculate_portfolio_value(self):
        """计算投资组合总价值"""
        total_value = 0
        portfolio_details = {}
        
        for symbol, holding in self.current_holdings.items():
            market_value = holding['shares'] * holding['current_price']
            total_value += market_value
            
            portfolio_details[symbol] = {
                'shares': holding['shares'],
                'price': holding['current_price'],
                'market_value': market_value,
                'daily_pnl': holding['shares'] * holding['change'],
                'weight': 0  # 将在后面计算
            }
        
        # 计算权重
        for symbol in portfolio_details:
            portfolio_details[symbol]['weight'] = portfolio_details[symbol]['market_value'] / total_value
            
        return total_value, portfolio_details
    
    def fetch_historical_data(self, period='1y'):
        """获取历史数据"""
        print("📊 获取历史市场数据...")
        
        data = {}
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                data[symbol] = hist
                print(f"✅ {symbol}: 获取 {len(hist)} 天数据")
            except Exception as e:
                print(f"❌ {symbol}: 数据获取失败 - {e}")
                
        return data
    
    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        print("📈 计算技术指标...")
        
        indicators = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # 移动平均线
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # 布林带
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # 波动率
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            indicators[symbol] = df
            
        return indicators
    
    def analyze_risk_metrics(self, data):
        """分析风险指标"""
        print("⚠️ 分析投资组合风险...")
        
        risk_metrics = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            returns = df['Close'].pct_change().dropna()
            
            # 基本风险指标
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # VaR (95%)
            var_95 = np.percentile(returns, 5)
            
            # 最大回撤
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Beta (相对于SPY)
            try:
                spy = yf.Ticker('SPY').history(period='1y')['Close'].pct_change().dropna()
                if len(spy) > 0 and len(returns) > 0:
                    # 对齐数据
                    common_dates = returns.index.intersection(spy.index)
                    if len(common_dates) > 20:
                        stock_returns = returns.loc[common_dates]
                        market_returns = spy.loc[common_dates]
                        beta = np.cov(stock_returns, market_returns)[0, 1] / np.var(market_returns)
                    else:
                        beta = 1.0
                else:
                    beta = 1.0
            except:
                beta = 1.0
            
            risk_metrics[symbol] = {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'beta': beta,
                'current_rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
            }
            
        return risk_metrics
    
    def generate_sector_analysis(self):
        """生成行业分析"""
        print("🏭 分析行业分布...")
        
        total_value, portfolio_details = self.calculate_portfolio_value()
        
        sector_allocation = {}
        for symbol, details in portfolio_details.items():
            sector = self.stock_info[symbol]['sector']
            if sector not in sector_allocation:
                sector_allocation[sector] = {'weight': 0, 'value': 0, 'stocks': []}
            
            sector_allocation[sector]['weight'] += details['weight']
            sector_allocation[sector]['value'] += details['market_value']
            sector_allocation[sector]['stocks'].append(symbol)
        
        return sector_allocation, total_value, portfolio_details
    
    def generate_trading_signals(self, indicators, risk_metrics):
        """生成交易信号"""
        print("🎯 生成交易信号...")
        
        signals = {}
        
        for symbol in self.symbols:
            if symbol not in indicators or symbol not in risk_metrics:
                continue
                
            df = indicators[symbol]
            risk = risk_metrics[symbol]
            
            # 获取最新数据
            current_price = df['Close'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            ma200 = df['MA200'].iloc[-1]
            rsi = risk['current_rsi']
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            
            # 技术信号评分
            technical_score = 0
            
            # 移动平均线信号
            if current_price > ma20 > ma50:
                technical_score += 2
            elif current_price > ma20:
                technical_score += 1
            elif current_price < ma20:
                technical_score -= 1
                
            # RSI信号
            if rsi < 30:
                technical_score += 2  # 超卖
            elif rsi < 40:
                technical_score += 1
            elif rsi > 70:
                technical_score -= 2  # 超买
            elif rsi > 60:
                technical_score -= 1
                
            # MACD信号
            if macd > macd_signal:
                technical_score += 1
            else:
                technical_score -= 1
                
            # 风险调整
            if risk['sharpe_ratio'] < 0:
                technical_score -= 2
            elif risk['max_drawdown'] < -0.3:
                technical_score -= 1
                
            # 生成建议
            if technical_score >= 3:
                action = "强烈买入"
                confidence = "高"
            elif technical_score >= 1:
                action = "买入"
                confidence = "中"
            elif technical_score >= -1:
                action = "持有"
                confidence = "中"
            elif technical_score >= -3:
                action = "卖出"
                confidence = "中"
            else:
                action = "强烈卖出"
                confidence = "高"
                
            signals[symbol] = {
                'action': action,
                'confidence': confidence,
                'technical_score': technical_score,
                'current_price': current_price,
                'target_price': current_price * (1 + 0.1 * technical_score / 5),
                'stop_loss': current_price * 0.9,
                'reasons': []
            }
            
            # 添加具体原因
            if rsi < 30:
                signals[symbol]['reasons'].append(f"RSI超卖({rsi:.1f})")
            elif rsi > 70:
                signals[symbol]['reasons'].append(f"RSI超买({rsi:.1f})")
                
            if current_price > ma20 > ma50:
                signals[symbol]['reasons'].append("均线多头排列")
            elif current_price < ma20 < ma50:
                signals[symbol]['reasons'].append("均线空头排列")
                
            if risk['sharpe_ratio'] < 0:
                signals[symbol]['reasons'].append("夏普比率为负")
                
        return signals
    
    def create_strategy_visualization(self, sector_allocation, portfolio_details, risk_metrics, signals):
        """创建策略可视化图表"""
        print("📊 生成策略分析图表...")
        
        # 设置图表样式
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 投资组合分布饼图
        ax1 = plt.subplot(3, 3, 1)
        weights = [details['weight'] for details in portfolio_details.values()]
        labels = [f"{symbol}\n({weight:.1%})" for symbol, weight in 
                 zip(portfolio_details.keys(), weights)]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax1.pie(weights, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('投资组合分布', fontsize=14, fontweight='bold')
        
        # 2. 行业分布
        ax2 = plt.subplot(3, 3, 2)
        sector_weights = [allocation['weight'] for allocation in sector_allocation.values()]
        sector_labels = list(sector_allocation.keys())
        
        ax2.pie(sector_weights, labels=sector_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('行业分布', fontsize=14, fontweight='bold')
        
        # 3. 风险收益散点图
        ax3 = plt.subplot(3, 3, 3)
        returns = [risk_metrics[symbol]['annual_return'] * 100 for symbol in self.symbols 
                  if symbol in risk_metrics]
        volatilities = [risk_metrics[symbol]['annual_volatility'] * 100 for symbol in self.symbols 
                       if symbol in risk_metrics]
        
        scatter = ax3.scatter(volatilities, returns, s=100, alpha=0.7, c=range(len(returns)), cmap='viridis')
        
        for i, symbol in enumerate([s for s in self.symbols if s in risk_metrics]):
            ax3.annotate(symbol, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax3.set_xlabel('年化波动率 (%)')
        ax3.set_ylabel('年化收益率 (%)')
        ax3.set_title('风险收益分布', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 夏普比率对比
        ax4 = plt.subplot(3, 3, 4)
        sharpe_ratios = [risk_metrics[symbol]['sharpe_ratio'] for symbol in self.symbols 
                        if symbol in risk_metrics]
        symbols_with_data = [symbol for symbol in self.symbols if symbol in risk_metrics]
        
        bars = ax4.bar(symbols_with_data, sharpe_ratios, 
                      color=['green' if sr > 0 else 'red' for sr in sharpe_ratios])
        ax4.set_title('夏普比率对比', fontsize=14, fontweight='bold')
        ax4.set_ylabel('夏普比率')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 最大回撤对比
        ax5 = plt.subplot(3, 3, 5)
        max_drawdowns = [risk_metrics[symbol]['max_drawdown'] * 100 for symbol in self.symbols 
                        if symbol in risk_metrics]
        
        bars = ax5.bar(symbols_with_data, max_drawdowns, color='red', alpha=0.7)
        ax5.set_title('最大回撤对比', fontsize=14, fontweight='bold')
        ax5.set_ylabel('最大回撤 (%)')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. 交易信号热力图
        ax6 = plt.subplot(3, 3, 6)
        signal_scores = [signals[symbol]['technical_score'] for symbol in self.symbols 
                        if symbol in signals]
        signal_matrix = np.array(signal_scores).reshape(1, -1)
        
        im = ax6.imshow(signal_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
        ax6.set_xticks(range(len(symbols_with_data)))
        ax6.set_xticklabels(symbols_with_data, rotation=45)
        ax6.set_yticks([])
        ax6.set_title('技术信号强度', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax6, orientation='horizontal', pad=0.1)
        cbar.set_label('信号强度 (负值=卖出, 正值=买入)')
        
        # 7. RSI分布
        ax7 = plt.subplot(3, 3, 7)
        rsi_values = [risk_metrics[symbol]['current_rsi'] for symbol in self.symbols 
                     if symbol in risk_metrics]
        
        bars = ax7.bar(symbols_with_data, rsi_values)
        ax7.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='超买线(70)')
        ax7.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='超卖线(30)')
        ax7.set_title('RSI指标分布', fontsize=14, fontweight='bold')
        ax7.set_ylabel('RSI')
        ax7.legend()
        ax7.tick_params(axis='x', rotation=45)
        
        # 8. Beta系数对比
        ax8 = plt.subplot(3, 3, 8)
        betas = [risk_metrics[symbol]['beta'] for symbol in self.symbols 
                if symbol in risk_metrics]
        
        bars = ax8.bar(symbols_with_data, betas, 
                      color=['blue' if beta < 1 else 'orange' for beta in betas])
        ax8.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='市场Beta=1')
        ax8.set_title('Beta系数对比', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Beta')
        ax8.legend()
        ax8.tick_params(axis='x', rotation=45)
        
        # 9. 投资组合价值分布
        ax9 = plt.subplot(3, 3, 9)
        values = [details['market_value'] for details in portfolio_details.values()]
        
        bars = ax9.bar(portfolio_details.keys(), values, color=colors[:len(values)])
        ax9.set_title('持仓市值分布', fontsize=14, fontweight='bold')
        ax9.set_ylabel('市值 ($)')
        ax9.tick_params(axis='x', rotation=45)
        
        # 格式化y轴显示
        ax9.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig('portfolio_strategy_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ 策略分析图表已保存为 'portfolio_strategy_analysis.png'")
        
    def generate_strategy_report(self, sector_allocation, portfolio_details, risk_metrics, signals, total_value):
        """生成策略报告"""
        print("\n" + "="*80)
        print("📊 投资组合策略分析报告")
        print("="*80)
        
        # 投资组合概览
        print(f"\n💰 投资组合概览:")
        print(f"总市值: ${total_value:,.2f}")
        print(f"持仓股票数: {len(self.current_holdings)}")
        
        daily_pnl = sum([details['daily_pnl'] for details in portfolio_details.values()])
        print(f"今日盈亏: ${daily_pnl:+,.2f}")
        
        # 行业分布
        print(f"\n🏭 行业分布:")
        for sector, allocation in sector_allocation.items():
            print(f"  {sector}: {allocation['weight']:.1%} (${allocation['value']:,.2f})")
            print(f"    包含股票: {', '.join(allocation['stocks'])}")
        
        # 个股分析
        print(f"\n📈 个股详细分析:")
        for symbol in self.symbols:
            if symbol in risk_metrics and symbol in signals:
                risk = risk_metrics[symbol]
                signal = signals[symbol]
                holding = self.current_holdings[symbol]
                
                print(f"\n  {symbol} ({self.stock_info[symbol]['name']}):")
                print(f"    当前价格: ${holding['current_price']:.2f} ({holding['change']:+.2f}, {holding['change_pct']:+.2f}%)")
                print(f"    持仓数量: {holding['shares']} 股")
                print(f"    市值: ${holding['shares'] * holding['current_price']:,.2f}")
                print(f"    年化收益率: {risk['annual_return']*100:+.1f}%")
                print(f"    年化波动率: {risk['annual_volatility']*100:.1f}%")
                print(f"    夏普比率: {risk['sharpe_ratio']:.2f}")
                print(f"    最大回撤: {risk['max_drawdown']*100:.1f}%")
                print(f"    Beta系数: {risk['beta']:.2f}")
                print(f"    RSI: {risk['current_rsi']:.1f}")
                print(f"    交易建议: {signal['action']} (信心度: {signal['confidence']})")
                print(f"    目标价格: ${signal['target_price']:.2f}")
                print(f"    止损价格: ${signal['stop_loss']:.2f}")
                if signal['reasons']:
                    print(f"    理由: {', '.join(signal['reasons'])}")
        
        # 风险评估
        print(f"\n⚠️ 投资组合风险评估:")
        avg_volatility = np.mean([risk_metrics[symbol]['annual_volatility'] 
                                 for symbol in self.symbols if symbol in risk_metrics])
        avg_sharpe = np.mean([risk_metrics[symbol]['sharpe_ratio'] 
                             for symbol in self.symbols if symbol in risk_metrics])
        
        print(f"  平均年化波动率: {avg_volatility*100:.1f}%")
        print(f"  平均夏普比率: {avg_sharpe:.2f}")
        
        # 行业集中度风险
        max_sector_weight = max([allocation['weight'] for allocation in sector_allocation.values()])
        if max_sector_weight > 0.6:
            print(f"  ⚠️ 行业集中度风险: 单一行业占比{max_sector_weight:.1%}，建议分散投资")
        
        # 个股集中度风险
        max_stock_weight = max([details['weight'] for details in portfolio_details.values()])
        if max_stock_weight > 0.4:
            print(f"  ⚠️ 个股集中度风险: 单一股票占比{max_stock_weight:.1%}，建议降低仓位")
        
        # 3个月交易策略
        print(f"\n🎯 未来3个月交易策略建议:")
        
        # 按信号强度分类
        strong_buy = [symbol for symbol, signal in signals.items() if signal['technical_score'] >= 3]
        buy = [symbol for symbol, signal in signals.items() if 1 <= signal['technical_score'] < 3]
        hold = [symbol for symbol, signal in signals.items() if -1 <= signal['technical_score'] < 1]
        sell = [symbol for symbol, signal in signals.items() if -3 <= signal['technical_score'] < -1]
        strong_sell = [symbol for symbol, signal in signals.items() if signal['technical_score'] < -3]
        
        if strong_buy:
            print(f"  🟢 强烈买入: {', '.join(strong_buy)}")
            print(f"     建议: 增加仓位，分批买入")
        
        if buy:
            print(f"  🟡 买入: {', '.join(buy)}")
            print(f"     建议: 适度增仓")
        
        if hold:
            print(f"  🔵 持有: {', '.join(hold)}")
            print(f"     建议: 维持现有仓位，观察市场变化")
        
        if sell:
            print(f"  🟠 卖出: {', '.join(sell)}")
            print(f"     建议: 减少仓位，分批卖出")
        
        if strong_sell:
            print(f"  🔴 强烈卖出: {', '.join(strong_sell)}")
            print(f"     建议: 大幅减仓或清仓")
        
        # 具体操作建议
        print(f"\n📋 具体操作建议:")
        print(f"  1. 短期(1个月内):")
        for symbol, signal in signals.items():
            if signal['technical_score'] >= 2:
                print(f"     • {symbol}: 考虑增仓10-20%")
            elif signal['technical_score'] <= -2:
                print(f"     • {symbol}: 考虑减仓20-30%")
        
        print(f"  2. 中期(1-3个月):")
        print(f"     • 关注行业轮动，科技股可能面临调整")
        print(f"     • 新能源汽车板块波动较大，注意风险控制")
        print(f"     • 建议设置止损位，控制单笔损失在10%以内")
        
        print(f"  3. 风险管理:")
        print(f"     • 建议保持20-30%的现金仓位")
        print(f"     • 单一股票仓位不超过总资产的25%")
        print(f"     • 定期重新平衡投资组合")
        
        print("\n" + "="*80)
        print("✅ 策略分析完成!")
        print("="*80)

def main():
    """主函数"""
    print("🚀 启动投资组合策略分析...")
    
    # 创建分析器
    analyzer = PortfolioStrategyAnalyzer()
    
    # 计算投资组合价值
    total_value, portfolio_details = analyzer.calculate_portfolio_value()
    
    # 获取历史数据
    historical_data = analyzer.fetch_historical_data()
    
    # 计算技术指标
    indicators = analyzer.calculate_technical_indicators(historical_data)
    
    # 分析风险指标
    risk_metrics = analyzer.analyze_risk_metrics(indicators)
    
    # 生成行业分析
    sector_allocation, total_value, portfolio_details = analyzer.generate_sector_analysis()
    
    # 生成交易信号
    signals = analyzer.generate_trading_signals(indicators, risk_metrics)
    
    # 创建可视化图表
    analyzer.create_strategy_visualization(sector_allocation, portfolio_details, risk_metrics, signals)
    
    # 生成策略报告
    analyzer.generate_strategy_report(sector_allocation, portfolio_details, risk_metrics, signals, total_value)

if __name__ == "__main__":
    main()