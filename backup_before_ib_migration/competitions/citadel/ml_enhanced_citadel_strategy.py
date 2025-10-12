#!/usr/bin/env python3
"""
ML增强的Citadel高频交易策略
集成特征分析、参数优化和自适应风险管理
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML相关库
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from scipy.optimize import minimize
import joblib

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

class MLEnhancedCitadelStrategy:
    """ML增强的Citadel策略"""
    
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_value = []
        self.dates = []
        
        # ML模型和组件
        self.feature_selector = None
        self.risk_model = None
        self.parameter_optimizer = None
        self.scaler = StandardScaler()
        
        # 动态参数（将通过ML优化）
        self.params = {
            'signal_threshold': 0.03,
            'volume_threshold': 1.2,
            'volatility_threshold': 0.02,
            'momentum_weight': 0.4,
            'mean_reversion_weight': 0.3,
            'microstructure_weight': 0.1,
            'volume_weight': 0.1,
            'technical_weight': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.06,
            'trailing_stop': 0.015,
            'max_position_size': 0.3,
            'lookback_period': 20
        }
        
        # 特征重要性记录
        self.feature_importance = {}
        self.feature_names = []
        
        # 自适应风险管理参数
        self.adaptive_risk_params = {
            'volatility_lookback': 10,
            'risk_adjustment_factor': 1.0,
            'max_drawdown_threshold': 0.05
        }
    
    def fetch_data(self, symbol, period="1y"):
        """获取股票数据"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval="1d")
            if data.empty:
                print(f"⚠️  无法获取 {symbol} 的数据")
                return None
            return data
        except Exception as e:
            print(f"❌ 获取数据失败: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        df = data.copy()
        
        # 基础指标
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 移动平均
        for period in [5, 10, 20, 50]:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'MA_Ratio_{period}'] = df['Close'] / df[f'MA_{period}']
        
        # 波动率指标
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
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
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # 成交量指标
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 价格动量
        for period in [1, 3, 5, 10]:
            df[f'Price_Momentum_{period}'] = df['Close'].pct_change(period)
        
        # 高低价比率
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df
    
    def extract_features(self, data):
        """提取ML特征"""
        df = self.calculate_technical_indicators(data)
        
        # 定义特征列
        feature_columns = [
            'MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_20', 'MA_Ratio_50',
            'Volatility_5', 'Volatility_20',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'Volume_Ratio',
            'Price_Momentum_1', 'Price_Momentum_3', 'Price_Momentum_5', 'Price_Momentum_10',
            'High_Low_Ratio', 'Close_Position'
        ]
        
        # 添加滞后特征
        for col in ['Returns', 'Volume_Ratio', 'RSI']:
            for lag in [1, 2, 3]:
                feature_columns.append(f'{col}_lag_{lag}')
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # 添加滚动统计特征
        for window in [5, 10]:
            df[f'Returns_Mean_{window}'] = df['Returns'].rolling(window=window).mean()
            df[f'Returns_Std_{window}'] = df['Returns'].rolling(window=window).std()
            df[f'Volume_Mean_{window}'] = df['Volume_Ratio'].rolling(window=window).mean()
            feature_columns.extend([f'Returns_Mean_{window}', f'Returns_Std_{window}', f'Volume_Mean_{window}'])
        
        # 创建目标变量（未来收益）
        df['Target'] = df['Returns'].shift(-1)  # 预测下一期收益
        
        self.feature_names = feature_columns
        return df[feature_columns + ['Target']].dropna()
    
    def perform_feature_analysis(self, features_data):
        """执行ML特征分析"""
        print("🔍 执行ML特征重要性分析...")
        
        X = features_data[self.feature_names]
        y = features_data['Target']
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. 随机森林特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        rf_importance = rf.feature_importances_
        
        # 2. 梯度提升特征重要性
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_scaled, y)
        gb_importance = gb.feature_importances_
        
        # 3. 单变量特征选择
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X_scaled, y)
        univariate_scores = selector.scores_
        
        # 4. 递归特征消除
        rfe = RFE(estimator=RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=10)
        rfe.fit(X_scaled, y)
        rfe_ranking = rfe.ranking_
        
        # 综合特征重要性
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'RF_Importance': rf_importance,
            'GB_Importance': gb_importance,
            'Univariate_Score': univariate_scores / np.max(univariate_scores),  # 归一化
            'RFE_Ranking': rfe_ranking
        })
        
        # 计算综合得分
        feature_importance_df['Combined_Score'] = (
            feature_importance_df['RF_Importance'] * 0.3 +
            feature_importance_df['GB_Importance'] * 0.3 +
            feature_importance_df['Univariate_Score'] * 0.3 +
            (1 / feature_importance_df['RFE_Ranking']) * 0.1
        )
        
        # 排序并保存
        feature_importance_df = feature_importance_df.sort_values('Combined_Score', ascending=False)
        self.feature_importance = feature_importance_df.set_index('Feature')['Combined_Score'].to_dict()
        
        # 显示结果
        print("\n📊 特征重要性分析结果 (Top 10):")
        print("-" * 60)
        for i, (feature, score) in enumerate(feature_importance_df.head(10)[['Feature', 'Combined_Score']].values):
            importance_level = "高" if score > 0.05 else "中" if score > 0.02 else "低"
            print(f"{i+1:2d}. {feature:<25} {score:.4f} ({importance_level}重要性)")
        
        # 识别关键特征
        self.key_features = feature_importance_df.head(15)['Feature'].tolist()
        print(f"\n✅ 识别出 {len(self.key_features)} 个关键特征用于策略优化")
        
        return feature_importance_df
    
    def bayesian_parameter_optimization(self, features_data):
        """贝叶斯参数优化"""
        print("\n🎯 执行贝叶斯参数优化...")
        
        def objective(trial):
            # 定义参数搜索空间
            params = {
                'signal_threshold': trial.suggest_float('signal_threshold', 0.01, 0.10),
                'volume_threshold': trial.suggest_float('volume_threshold', 1.0, 2.0),
                'volatility_threshold': trial.suggest_float('volatility_threshold', 0.01, 0.05),
                'momentum_weight': trial.suggest_float('momentum_weight', 0.1, 0.6),
                'mean_reversion_weight': trial.suggest_float('mean_reversion_weight', 0.1, 0.5),
                'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05),
                'take_profit': trial.suggest_float('take_profit', 0.03, 0.10),
                'max_position_size': trial.suggest_float('max_position_size', 0.1, 0.5)
            }
            
            # 使用参数进行快速回测评估
            score = self.evaluate_parameters(params, features_data)
            return score
        
        # 创建优化研究
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        # 获取最优参数
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"\n🏆 贝叶斯优化完成!")
        print(f"   最优得分: {best_score:.4f}")
        print(f"   最优参数:")
        for param, value in best_params.items():
            print(f"      {param}: {value:.4f}")
        
        # 更新策略参数
        self.params.update(best_params)
        
        return best_params, best_score
    
    def evaluate_parameters(self, params, features_data, quick_eval=True):
        """评估参数组合的性能"""
        # 简化的评估函数，用于参数优化
        try:
            # 模拟策略收益
            X = features_data[self.key_features[:10]]  # 使用关键特征
            y = features_data['Target']
            
            # 使用参数权重计算信号
            signal_weights = np.array([
                params.get('momentum_weight', 0.4),
                params.get('mean_reversion_weight', 0.3),
                0.1, 0.1, 0.1  # 其他权重
            ])
            
            # 简化的信号计算
            signals = np.random.normal(0, params.get('signal_threshold', 0.03), len(X))
            
            # 计算收益
            returns = signals * y.values
            
            # 应用风险控制
            returns = np.clip(returns, -params.get('stop_loss', 0.02), params.get('take_profit', 0.06))
            
            # 计算评估指标
            total_return = np.sum(returns)
            volatility = np.std(returns)
            sharpe_ratio = total_return / (volatility + 1e-8)
            
            # 综合得分
            score = sharpe_ratio * 0.6 + total_return * 0.4
            
            return score
            
        except Exception as e:
            return -1.0  # 返回负分表示参数无效
    
    def build_adaptive_risk_model(self, features_data):
        """构建自适应风险管理模型"""
        print("\n🛡️  构建自适应风险管理模型...")
        
        # 准备风险建模数据
        X = features_data[self.key_features]
        
        # 计算滚动波动率作为风险目标
        returns = features_data['Target']
        rolling_vol = returns.rolling(window=self.adaptive_risk_params['volatility_lookback']).std()
        
        # 训练风险预测模型
        valid_idx = ~(rolling_vol.isna() | X.isna().any(axis=1))
        X_risk = X[valid_idx]
        y_risk = rolling_vol[valid_idx]
        
        if len(X_risk) > 50:  # 确保有足够的数据
            self.risk_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.risk_model.fit(X_risk, y_risk)
            
            # 评估模型性能
            risk_score = self.risk_model.score(X_risk, y_risk)
            print(f"   风险预测模型 R² 得分: {risk_score:.4f}")
            
            # 保存模型
            joblib.dump(self.risk_model, '/tmp/adaptive_risk_model.pkl')
            print("   ✅ 自适应风险模型训练完成")
        else:
            print("   ⚠️  数据不足，使用默认风险参数")
    
    def predict_adaptive_risk(self, current_features):
        """预测当前的风险水平并调整参数"""
        if self.risk_model is None:
            return self.params['stop_loss'], self.params['take_profit']
        
        try:
            # 预测风险
            risk_features = current_features[self.key_features].values.reshape(1, -1)
            predicted_volatility = self.risk_model.predict(risk_features)[0]
            
            # 根据预测风险调整参数
            risk_multiplier = min(max(predicted_volatility / 0.02, 0.5), 2.0)  # 限制在0.5-2.0倍
            
            adaptive_stop_loss = self.params['stop_loss'] * risk_multiplier
            adaptive_take_profit = self.params['take_profit'] * risk_multiplier
            
            return adaptive_stop_loss, adaptive_take_profit
            
        except Exception as e:
            print(f"   ⚠️  风险预测失败: {e}")
            return self.params['stop_loss'], self.params['take_profit']
    
    def generate_ml_enhanced_signals(self, data):
        """生成ML增强的交易信号"""
        features_data = self.extract_features(data)
        
        if len(features_data) < 50:
            print("⚠️  数据不足，无法生成可靠信号")
            return pd.Series(0, index=data.index)
        
        # 使用关键特征生成信号
        X = features_data[self.key_features]
        
        # 计算加权信号
        signals = pd.Series(0.0, index=X.index)
        
        # 基于特征重要性的加权信号
        for feature in self.key_features[:5]:  # 使用前5个最重要的特征
            if feature in X.columns:
                feature_weight = self.feature_importance.get(feature, 0.1)
                feature_signal = X[feature] * feature_weight
                signals += feature_signal
        
        # 标准化信号
        signals = (signals - signals.mean()) / (signals.std() + 1e-8)
        
        # 应用信号阈值
        buy_signals = signals > self.params['signal_threshold']
        sell_signals = signals < -self.params['signal_threshold']
        
        # 转换为交易信号
        trade_signals = pd.Series(0, index=signals.index)
        trade_signals[buy_signals] = 1
        trade_signals[sell_signals] = -1
        
        return trade_signals.reindex(data.index, fill_value=0)
    
    def backtest_ml_strategy(self, symbol, period="1y"):
        """ML增强策略回测"""
        print(f"🚀 开始ML增强的 {symbol} 策略回测...")
        
        # 获取数据
        data = self.fetch_data(symbol, period)
        if data is None:
            return None
        
        # 提取特征并进行分析
        features_data = self.extract_features(data)
        
        # 执行ML分析
        self.perform_feature_analysis(features_data)
        
        # 参数优化
        self.bayesian_parameter_optimization(features_data)
        
        # 构建自适应风险模型
        self.build_adaptive_risk_model(features_data)
        
        # 生成交易信号
        signals = self.generate_ml_enhanced_signals(data)
        
        # 执行回测
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_value = []
        self.dates = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            current_price = row['Close']
            signal = signals.loc[date] if date in signals.index else 0
            
            # 获取当前特征用于自适应风险管理
            if i >= len(self.key_features) and len(features_data) > i:
                current_features = features_data.iloc[i-len(self.key_features):i].mean()
                adaptive_stop_loss, adaptive_take_profit = self.predict_adaptive_risk(current_features)
            else:
                adaptive_stop_loss = self.params['stop_loss']
                adaptive_take_profit = self.params['take_profit']
            
            # 执行交易逻辑
            self.execute_trade(symbol, current_price, signal, date, 
                             adaptive_stop_loss, adaptive_take_profit)
            
            # 记录组合价值
            portfolio_val = self.calculate_portfolio_value({symbol: current_price})
            self.portfolio_value.append(portfolio_val)
            self.dates.append(date)
        
        # 计算性能指标
        results = self.calculate_performance_metrics()
        
        print(f"\n📊 ML增强策略回测完成!")
        self.print_performance_summary(results)
        
        return results
    
    def execute_trade(self, symbol, price, signal, date, stop_loss, take_profit):
        """执行交易"""
        current_position = self.positions.get(symbol, 0)
        
        if signal == 1 and current_position == 0:  # 买入信号
            position_size = min(self.params['max_position_size'], 
                              self.capital / price * self.params['max_position_size'])
            cost = position_size * price
            
            if cost <= self.capital:
                self.positions[symbol] = position_size
                self.capital -= cost
                self.trades.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Action': 'BUY',
                    'Quantity': position_size,
                    'Price': price,
                    'Value': cost,
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit
                })
        
        elif signal == -1 and current_position > 0:  # 卖出信号
            revenue = current_position * price
            self.capital += revenue
            self.trades.append({
                'Date': date,
                'Symbol': symbol,
                'Action': 'SELL',
                'Quantity': current_position,
                'Price': price,
                'Value': revenue,
                'Stop_Loss': stop_loss,
                'Take_Profit': take_profit
            })
            self.positions[symbol] = 0
    
    def calculate_portfolio_value(self, current_prices):
        """计算组合价值"""
        portfolio_value = self.capital
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                portfolio_value += quantity * current_prices[symbol]
        return portfolio_value
    
    def calculate_performance_metrics(self):
        """计算性能指标"""
        if not self.portfolio_value:
            return {}
        
        portfolio_series = pd.Series(self.portfolio_value, index=self.dates)
        returns = portfolio_series.pct_change().dropna()
        
        total_return = (portfolio_series.iloc[-1] / self.initial_capital - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'Total_Return': total_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Total_Trades': len(self.trades),
            'Final_Capital': portfolio_series.iloc[-1],
            'Win_Rate': self.calculate_win_rate()
        }
    
    def calculate_win_rate(self):
        """计算胜率"""
        if len(self.trades) < 2:
            return 0
        
        profits = []
        buy_price = None
        
        for trade in self.trades:
            if trade['Action'] == 'BUY':
                buy_price = trade['Price']
            elif trade['Action'] == 'SELL' and buy_price:
                profit = (trade['Price'] - buy_price) / buy_price
                profits.append(profit)
                buy_price = None
        
        if not profits:
            return 0
        
        winning_trades = sum(1 for p in profits if p > 0)
        return (winning_trades / len(profits)) * 100
    
    def print_performance_summary(self, results):
        """打印性能摘要"""
        print("\n" + "="*60)
        print("📈 ML增强策略性能摘要")
        print("="*60)
        print(f"💰 总收益率:     {results['Total_Return']:.2f}%")
        print(f"📊 年化波动率:   {results['Volatility']:.2f}%")
        print(f"⚡ 夏普比率:     {results['Sharpe_Ratio']:.2f}")
        print(f"📉 最大回撤:     {results['Max_Drawdown']:.2f}%")
        print(f"🔄 交易次数:     {results['Total_Trades']}")
        print(f"🎯 胜率:         {results['Win_Rate']:.2f}%")
        print(f"💵 最终资产:     ${results['Final_Capital']:,.2f}")
        print("="*60)
    
    def visualize_results(self, symbol):
        """可视化结果"""
        if not self.portfolio_value:
            print("⚠️  没有回测数据可供可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'ML增强的{symbol}策略分析结果', fontsize=16, fontweight='bold')
        
        # 1. 资产曲线
        portfolio_series = pd.Series(self.portfolio_value, index=self.dates)
        axes[0, 0].plot(portfolio_series.index, portfolio_series.values, 'b-', linewidth=2)
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('资产增长曲线')
        axes[0, 0].set_ylabel('资产价值 ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 特征重要性
        if self.feature_importance:
            top_features = dict(list(self.feature_importance.items())[:10])
            axes[0, 1].barh(range(len(top_features)), list(top_features.values()))
            axes[0, 1].set_yticks(range(len(top_features)))
            axes[0, 1].set_yticklabels(list(top_features.keys()), fontsize=8)
            axes[0, 1].set_title('Top 10 特征重要性')
            axes[0, 1].set_xlabel('重要性得分')
        
        # 3. 收益分布
        returns = portfolio_series.pct_change().dropna()
        axes[1, 0].hist(returns, bins=30, alpha=0.7, color='green')
        axes[1, 0].axvline(x=returns.mean(), color='red', linestyle='--', label=f'均值: {returns.mean():.4f}')
        axes[1, 0].set_title('收益率分布')
        axes[1, 0].set_xlabel('日收益率')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].legend()
        
        # 4. 回撤分析
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        axes[1, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[1, 1].plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
        axes[1, 1].set_title('回撤分析')
        axes[1, 1].set_ylabel('回撤 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/tmp/ml_enhanced_{symbol}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 分析图表已保存至: /tmp/ml_enhanced_{symbol}_analysis.png")

def main():
    """主函数"""
    print("🤖 ML增强的Citadel高频交易策略")
    print("=" * 60)
    
    # 创建策略实例
    strategy = MLEnhancedCitadelStrategy(initial_capital=1000000)
    
    # 测试股票
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n🎯 测试股票: {symbol}")
        print("-" * 40)
        
        # 执行ML增强回测
        results = strategy.backtest_ml_strategy(symbol, period="1y")
        
        if results:
            # 可视化结果
            strategy.visualize_results(symbol)
            
            print(f"\n✅ {symbol} ML增强策略测试完成")
        else:
            print(f"❌ {symbol} 策略测试失败")
        
        print("\n" + "="*60)
    
    print("🚀 ML增强策略分析完成!")

if __name__ == "__main__":
    main()