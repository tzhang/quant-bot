#!/usr/bin/env python3
"""
自适应风险管理系统
基于市场状态和策略表现动态调整风险参数
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML相关库
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# 技术分析
import talib

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

class AdaptiveRiskManager:
    """自适应风险管理系统"""
    
    def __init__(self, initial_params=None):
        # 基础风险参数
        self.base_params = initial_params or {
            'stop_loss': 0.02,
            'take_profit': 0.06,
            'trailing_stop': 0.015,
            'max_position_size': 0.3,
            'risk_per_trade': 0.01
        }
        
        # 当前自适应参数
        self.current_params = self.base_params.copy()
        
        # ML模型
        self.volatility_predictor = None
        self.regime_classifier = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        
        # 市场状态
        self.market_regime = 'normal'  # normal, high_vol, trending, sideways
        self.volatility_forecast = 0.02
        self.risk_score = 0.5  # 0-1, 越高风险越大
        
        # 历史数据
        self.performance_history = []
        self.market_data_history = []
        self.risk_adjustments_history = []
        
        # 风险调整因子
        self.adjustment_factors = {
            'volatility_factor': 1.0,
            'regime_factor': 1.0,
            'performance_factor': 1.0,
            'drawdown_factor': 1.0
        }
    
    def calculate_market_features(self, data, lookback=20):
        """计算市场特征"""
        df = data.copy()
        
        # 基础特征
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 波动率特征
        df['realized_vol'] = df['returns'].rolling(window=lookback).std() * np.sqrt(252)
        df['vol_5d'] = df['returns'].rolling(window=5).std() * np.sqrt(252)
        df['vol_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['vol_ratio'] = df['vol_5d'] / df['vol_20d']
        
        # 趋势特征
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['trend_strength'] = (df['Close'] - df['sma_20']) / df['sma_20']
        df['trend_direction'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        # 动量特征
        df['momentum_5d'] = df['Close'].pct_change(5)
        df['momentum_20d'] = df['Close'].pct_change(20)
        df['rsi'] = talib.RSI(df['Close'].values, timeperiod=14)
        
        # 成交量特征
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # VIX代理指标（基于收益率分布）
        df['vix_proxy'] = df['returns'].rolling(window=20).apply(
            lambda x: np.percentile(np.abs(x), 95) * np.sqrt(252) * 100
        )
        
        # 市场压力指标
        df['stress_indicator'] = (
            df['vol_ratio'] * 0.3 +
            np.abs(df['trend_strength']) * 0.3 +
            (df['vix_proxy'] / 50) * 0.4
        )
        
        return df
    
    def train_volatility_predictor(self, market_data):
        """训练波动率预测模型"""
        print("📊 训练波动率预测模型...")
        
        # 计算特征
        features_df = self.calculate_market_features(market_data)
        
        # 准备训练数据
        feature_cols = [
            'vol_5d', 'vol_20d', 'vol_ratio', 'trend_strength', 
            'momentum_5d', 'momentum_20d', 'rsi', 'volume_ratio', 'stress_indicator'
        ]
        
        # 目标变量：未来5日波动率
        features_df['target_vol'] = features_df['realized_vol'].shift(-5)
        
        # 清理数据
        clean_data = features_df[feature_cols + ['target_vol']].dropna()
        
        if len(clean_data) < 50:
            print("⚠️  数据不足，使用默认波动率预测")
            return
        
        X = clean_data[feature_cols]
        y = clean_data['target_vol']
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.volatility_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.volatility_predictor.fit(X_scaled, y)
        
        # 评估模型
        predictions = self.volatility_predictor.predict(X_scaled)
        mse = mean_squared_error(y, predictions)
        r2 = self.volatility_predictor.score(X_scaled, y)
        
        print(f"   波动率预测模型 R² 得分: {r2:.4f}")
        print(f"   均方误差: {mse:.6f}")
        
        # 保存模型
        joblib.dump(self.volatility_predictor, '/tmp/volatility_predictor.pkl')
        joblib.dump(self.scaler, '/tmp/volatility_scaler.pkl')
    
    def train_regime_classifier(self, market_data):
        """训练市场状态分类器"""
        print("🎯 训练市场状态分类器...")
        
        features_df = self.calculate_market_features(market_data)
        
        # 定义市场状态
        def classify_regime(row):
            if row['realized_vol'] > 0.25:  # 高波动
                return 'high_vol'
            elif abs(row['trend_strength']) > 0.05:  # 趋势市场
                return 'trending'
            elif row['vol_ratio'] < 0.8:  # 低波动
                return 'low_vol'
            else:
                return 'normal'
        
        features_df['regime'] = features_df.apply(classify_regime, axis=1)
        
        # 准备训练数据
        feature_cols = [
            'realized_vol', 'vol_ratio', 'trend_strength', 'momentum_20d', 
            'rsi', 'volume_ratio', 'stress_indicator'
        ]
        
        clean_data = features_df[feature_cols + ['regime']].dropna()
        
        if len(clean_data) < 50:
            print("⚠️  数据不足，使用默认市场状态分类")
            return
        
        X = clean_data[feature_cols]
        y = clean_data['regime']
        
        # 使用聚类作为无监督分类器
        self.regime_classifier = KMeans(n_clusters=4, random_state=42)
        X_scaled = StandardScaler().fit_transform(X)
        cluster_labels = self.regime_classifier.fit_predict(X_scaled)
        
        # 映射聚类结果到市场状态
        regime_mapping = {}
        for i in range(4):
            cluster_mask = cluster_labels == i
            if cluster_mask.sum() > 0:
                most_common_regime = y[cluster_mask].mode().iloc[0] if len(y[cluster_mask].mode()) > 0 else 'normal'
                regime_mapping[i] = most_common_regime
        
        self.regime_mapping = regime_mapping
        
        print(f"   市场状态分类器训练完成，识别出 {len(regime_mapping)} 种状态")
        
        # 保存模型
        joblib.dump(self.regime_classifier, '/tmp/regime_classifier.pkl')
    
    def train_anomaly_detector(self, market_data):
        """训练异常检测模型"""
        print("🚨 训练异常检测模型...")
        
        features_df = self.calculate_market_features(market_data)
        
        feature_cols = [
            'realized_vol', 'vol_ratio', 'trend_strength', 'momentum_5d',
            'volume_ratio', 'stress_indicator'
        ]
        
        clean_data = features_df[feature_cols].dropna()
        
        if len(clean_data) < 30:
            print("⚠️  数据不足，跳过异常检测训练")
            return
        
        # 训练孤立森林
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # 假设10%的数据是异常
            random_state=42
        )
        
        X_scaled = StandardScaler().fit_transform(clean_data)
        self.anomaly_detector.fit(X_scaled)
        
        # 测试异常检测
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        anomaly_labels = self.anomaly_detector.predict(X_scaled)
        
        anomaly_rate = (anomaly_labels == -1).mean()
        print(f"   异常检测模型训练完成，检测到 {anomaly_rate:.1%} 的异常数据")
        
        # 保存模型
        joblib.dump(self.anomaly_detector, '/tmp/anomaly_detector.pkl')
    
    def predict_market_state(self, current_data):
        """预测当前市场状态"""
        try:
            # 计算当前特征
            features_df = self.calculate_market_features(current_data)
            latest_features = features_df.iloc[-1]
            
            # 预测波动率
            if self.volatility_predictor is not None:
                vol_features = [
                    'vol_5d', 'vol_20d', 'vol_ratio', 'trend_strength',
                    'momentum_5d', 'momentum_20d', 'rsi', 'volume_ratio', 'stress_indicator'
                ]
                
                vol_input = latest_features[vol_features].values.reshape(1, -1)
                vol_input_scaled = self.scaler.transform(vol_input)
                self.volatility_forecast = self.volatility_predictor.predict(vol_input_scaled)[0]
            
            # 预测市场状态
            if self.regime_classifier is not None:
                regime_features = [
                    'realized_vol', 'vol_ratio', 'trend_strength', 'momentum_20d',
                    'rsi', 'volume_ratio', 'stress_indicator'
                ]
                
                regime_input = latest_features[regime_features].values.reshape(1, -1)
                regime_input_scaled = StandardScaler().fit_transform(regime_input)
                cluster_pred = self.regime_classifier.predict(regime_input_scaled)[0]
                self.market_regime = self.regime_mapping.get(cluster_pred, 'normal')
            
            # 异常检测
            anomaly_score = 0.5  # 默认正常
            if self.anomaly_detector is not None:
                anomaly_features = [
                    'realized_vol', 'vol_ratio', 'trend_strength', 'momentum_5d',
                    'volume_ratio', 'stress_indicator'
                ]
                
                anomaly_input = latest_features[anomaly_features].values.reshape(1, -1)
                anomaly_input_scaled = StandardScaler().fit_transform(anomaly_input)
                anomaly_score = self.anomaly_detector.decision_function(anomaly_input_scaled)[0]
                
                # 转换为0-1风险分数
                self.risk_score = max(0, min(1, (1 - anomaly_score) / 2))
            
            return {
                'volatility_forecast': self.volatility_forecast,
                'market_regime': self.market_regime,
                'risk_score': self.risk_score,
                'anomaly_score': anomaly_score
            }
            
        except Exception as e:
            print(f"⚠️  市场状态预测失败: {e}")
            return {
                'volatility_forecast': 0.02,
                'market_regime': 'normal',
                'risk_score': 0.5,
                'anomaly_score': 0
            }
    
    def calculate_adjustment_factors(self, market_state, performance_metrics=None):
        """计算风险调整因子"""
        
        # 1. 波动率调整因子
        expected_vol = 0.15  # 基准年化波动率
        vol_ratio = market_state['volatility_forecast'] / expected_vol
        self.adjustment_factors['volatility_factor'] = np.clip(vol_ratio, 0.5, 2.0)
        
        # 2. 市场状态调整因子
        regime_adjustments = {
            'normal': 1.0,
            'high_vol': 1.5,    # 高波动时增加风险控制
            'trending': 0.8,    # 趋势市场可以适当放松
            'low_vol': 0.9,     # 低波动时略微放松
            'sideways': 1.2     # 震荡市场增加控制
        }
        self.adjustment_factors['regime_factor'] = regime_adjustments.get(
            market_state['market_regime'], 1.0
        )
        
        # 3. 风险分数调整因子
        risk_factor = 1 + market_state['risk_score']  # 1.0 - 2.0
        self.adjustment_factors['risk_factor'] = risk_factor
        
        # 4. 表现调整因子
        if performance_metrics:
            recent_sharpe = performance_metrics.get('recent_sharpe', 1.0)
            recent_drawdown = performance_metrics.get('recent_drawdown', 0.05)
            
            # 表现好时可以适当放松，表现差时收紧
            performance_factor = 1.0
            if recent_sharpe > 2.0:
                performance_factor = 0.9  # 表现好，略微放松
            elif recent_sharpe < 0.5:
                performance_factor = 1.3  # 表现差，收紧控制
            
            # 回撤调整
            if recent_drawdown > 0.1:  # 回撤超过10%
                performance_factor *= 1.2
            
            self.adjustment_factors['performance_factor'] = performance_factor
        
        return self.adjustment_factors
    
    def adapt_risk_parameters(self, market_state, performance_metrics=None):
        """自适应调整风险参数"""
        
        # 计算调整因子
        factors = self.calculate_adjustment_factors(market_state, performance_metrics)
        
        # 综合调整因子
        total_factor = (
            factors['volatility_factor'] * 0.4 +
            factors['regime_factor'] * 0.3 +
            factors.get('risk_factor', 1.0) * 0.2 +
            factors.get('performance_factor', 1.0) * 0.1
        )
        
        # 调整风险参数
        self.current_params = {}
        
        # 止损调整：风险高时收紧
        self.current_params['stop_loss'] = self.base_params['stop_loss'] * total_factor
        self.current_params['stop_loss'] = np.clip(self.current_params['stop_loss'], 0.005, 0.1)
        
        # 止盈调整：风险高时也相应调整
        profit_factor = min(total_factor, 1.5)  # 止盈调整幅度较小
        self.current_params['take_profit'] = self.base_params['take_profit'] * profit_factor
        self.current_params['take_profit'] = np.clip(self.current_params['take_profit'], 0.02, 0.2)
        
        # 追踪止损调整
        self.current_params['trailing_stop'] = self.base_params['trailing_stop'] * total_factor
        self.current_params['trailing_stop'] = np.clip(self.current_params['trailing_stop'], 0.005, 0.05)
        
        # 仓位大小调整：风险高时减小仓位
        position_factor = 1 / total_factor  # 反向调整
        self.current_params['max_position_size'] = self.base_params['max_position_size'] * position_factor
        self.current_params['max_position_size'] = np.clip(self.current_params['max_position_size'], 0.05, 0.5)
        
        # 单笔风险调整
        self.current_params['risk_per_trade'] = self.base_params['risk_per_trade'] * position_factor
        self.current_params['risk_per_trade'] = np.clip(self.current_params['risk_per_trade'], 0.005, 0.03)
        
        # 记录调整历史
        adjustment_record = {
            'timestamp': datetime.now(),
            'market_state': market_state,
            'adjustment_factors': factors,
            'total_factor': total_factor,
            'old_params': self.base_params.copy(),
            'new_params': self.current_params.copy()
        }
        self.risk_adjustments_history.append(adjustment_record)
        
        return self.current_params
    
    def get_dynamic_risk_params(self, current_data, performance_metrics=None):
        """获取动态风险参数"""
        
        # 预测市场状态
        market_state = self.predict_market_state(current_data)
        
        # 自适应调整参数
        adapted_params = self.adapt_risk_parameters(market_state, performance_metrics)
        
        return {
            'risk_params': adapted_params,
            'market_state': market_state,
            'adjustment_factors': self.adjustment_factors
        }
    
    def evaluate_risk_adjustment_performance(self):
        """评估风险调整的效果"""
        if len(self.risk_adjustments_history) < 10:
            print("⚠️  调整历史不足，无法评估效果")
            return None
        
        # 分析调整频率
        adjustments_df = pd.DataFrame([
            {
                'timestamp': record['timestamp'],
                'total_factor': record['total_factor'],
                'volatility_factor': record['adjustment_factors']['volatility_factor'],
                'regime_factor': record['adjustment_factors']['regime_factor'],
                'stop_loss': record['new_params']['stop_loss'],
                'take_profit': record['new_params']['take_profit'],
                'position_size': record['new_params']['max_position_size']
            }
            for record in self.risk_adjustments_history
        ])
        
        # 计算调整统计
        stats = {
            'total_adjustments': len(adjustments_df),
            'avg_adjustment_factor': adjustments_df['total_factor'].mean(),
            'adjustment_volatility': adjustments_df['total_factor'].std(),
            'stop_loss_range': (adjustments_df['stop_loss'].min(), adjustments_df['stop_loss'].max()),
            'position_size_range': (adjustments_df['position_size'].min(), adjustments_df['position_size'].max())
        }
        
        print("📊 风险调整效果评估:")
        print(f"   总调整次数: {stats['total_adjustments']}")
        print(f"   平均调整因子: {stats['avg_adjustment_factor']:.3f}")
        print(f"   调整波动性: {stats['adjustment_volatility']:.3f}")
        print(f"   止损范围: {stats['stop_loss_range'][0]:.3f} - {stats['stop_loss_range'][1]:.3f}")
        print(f"   仓位范围: {stats['position_size_range'][0]:.3f} - {stats['position_size_range'][1]:.3f}")
        
        return stats
    
    def visualize_risk_adaptation(self):
        """可视化风险自适应过程"""
        if len(self.risk_adjustments_history) < 5:
            print("⚠️  调整历史不足，无法可视化")
            return
        
        # 准备数据
        timestamps = [record['timestamp'] for record in self.risk_adjustments_history]
        total_factors = [record['total_factor'] for record in self.risk_adjustments_history]
        stop_losses = [record['new_params']['stop_loss'] for record in self.risk_adjustments_history]
        position_sizes = [record['new_params']['max_position_size'] for record in self.risk_adjustments_history]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('自适应风险管理可视化', fontsize=16, fontweight='bold')
        
        # 1. 调整因子时间序列
        axes[0, 0].plot(timestamps, total_factors, 'b-', linewidth=2, marker='o')
        axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='基准线')
        axes[0, 0].set_title('风险调整因子变化')
        axes[0, 0].set_ylabel('调整因子')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 止损参数变化
        axes[0, 1].plot(timestamps, stop_losses, 'r-', linewidth=2, marker='s')
        axes[0, 1].axhline(y=self.base_params['stop_loss'], color='g', linestyle='--', alpha=0.7, label='基准止损')
        axes[0, 1].set_title('动态止损参数')
        axes[0, 1].set_ylabel('止损比例')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 仓位大小变化
        axes[1, 0].plot(timestamps, position_sizes, 'g-', linewidth=2, marker='^')
        axes[1, 0].axhline(y=self.base_params['max_position_size'], color='b', linestyle='--', alpha=0.7, label='基准仓位')
        axes[1, 0].set_title('动态仓位大小')
        axes[1, 0].set_ylabel('最大仓位比例')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 调整因子分布
        axes[1, 1].hist(total_factors, bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(x=np.mean(total_factors), color='red', linestyle='--', label=f'均值: {np.mean(total_factors):.3f}')
        axes[1, 1].set_title('调整因子分布')
        axes[1, 1].set_xlabel('调整因子')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('/tmp/adaptive_risk_management.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 风险自适应可视化已保存至: /tmp/adaptive_risk_management.png")

def main():
    """主函数 - 演示自适应风险管理"""
    print("🛡️  自适应风险管理系统演示")
    print("=" * 60)
    
    # 创建风险管理器
    risk_manager = AdaptiveRiskManager()
    
    # 模拟市场数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # 生成模拟价格数据
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.lognormal(15, 0.5, len(dates))
    
    market_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Volume': volumes
    }).set_index('Date')
    
    # 训练模型
    risk_manager.train_volatility_predictor(market_data)
    risk_manager.train_regime_classifier(market_data)
    risk_manager.train_anomaly_detector(market_data)
    
    # 模拟实时风险调整
    print("\n🔄 模拟实时风险调整...")
    
    for i in range(10):
        # 获取当前数据窗口
        current_window = market_data.iloc[max(0, -50-i*10):-i*10] if i > 0 else market_data.iloc[-50:]
        
        # 模拟表现指标
        performance_metrics = {
            'recent_sharpe': np.random.normal(1.5, 0.5),
            'recent_drawdown': abs(np.random.normal(0.05, 0.03))
        }
        
        # 获取动态风险参数
        risk_result = risk_manager.get_dynamic_risk_params(current_window, performance_metrics)
        
        print(f"\n调整 {i+1}:")
        print(f"   市场状态: {risk_result['market_state']['market_regime']}")
        print(f"   波动率预测: {risk_result['market_state']['volatility_forecast']:.3f}")
        print(f"   风险分数: {risk_result['market_state']['risk_score']:.3f}")
        print(f"   止损: {risk_result['risk_params']['stop_loss']:.3f}")
        print(f"   止盈: {risk_result['risk_params']['take_profit']:.3f}")
        print(f"   最大仓位: {risk_result['risk_params']['max_position_size']:.3f}")
    
    # 评估调整效果
    print("\n📊 评估风险调整效果...")
    risk_manager.evaluate_risk_adjustment_performance()
    
    # 可视化结果
    risk_manager.visualize_risk_adaptation()
    
    print("\n🚀 自适应风险管理演示完成!")

if __name__ == "__main__":
    main()