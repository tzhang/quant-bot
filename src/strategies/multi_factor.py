import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class MultiFactor:
    """
    多因子模型：结合多个技术指标和基本面因子进行量化选股
    """
    
    def __init__(
        self,
        factors: List[str] = None,
        lookback: int = 252,
        rebalance_freq: int = 21,
        top_n: int = 10,
        model_type: str = "linear"  # "linear", "random_forest"
    ):
        """
        初始化多因子模型
        
        Args:
            factors: 因子列表
            lookback: 因子计算回望期
            rebalance_freq: 再平衡频率（天）
            top_n: 选择的股票数量
            model_type: 模型类型
        """
        self.factors = factors or [
            'momentum', 'mean_reversion', 'volatility', 'volume', 
            'rsi', 'macd', 'bollinger'
        ]
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        
    def _calculate_momentum_factor(self, prices: pd.Series) -> pd.Series:
        """计算动量因子"""
        return prices.pct_change(self.lookback // 4).fillna(0)
    
    def _calculate_mean_reversion_factor(self, prices: pd.Series) -> pd.Series:
        """计算均值回归因子"""
        sma = prices.rolling(self.lookback // 2).mean()
        return ((prices - sma) / sma).fillna(0)
    
    def _calculate_volatility_factor(self, prices: pd.Series) -> pd.Series:
        """计算波动率因子"""
        returns = prices.pct_change()
        return returns.rolling(self.lookback // 4).std().fillna(0)
    
    def _calculate_volume_factor(self, volumes: pd.Series) -> pd.Series:
        """计算成交量因子"""
        avg_volume = volumes.rolling(self.lookback // 2).mean()
        return ((volumes - avg_volume) / avg_volume).fillna(0)
    
    def _calculate_rsi_factor(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI因子"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50  # 标准化到[-1, 1]
    
    def _calculate_macd_factor(self, prices: pd.Series) -> pd.Series:
        """计算MACD因子"""
        ema_fast = prices.ewm(span=12).mean()
        ema_slow = prices.ewm(span=26).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=9).mean()
        return (macd - signal_line).fillna(0)
    
    def _calculate_bollinger_factor(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """计算布林带因子"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        # 计算价格在布林带中的相对位置
        bb_position = (prices - lower_band) / (upper_band - lower_band + 1e-12)
        return (bb_position - 0.5) * 2  # 标准化到[-1, 1]
    
    def calculate_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            因子DataFrame
        """
        factor_df = pd.DataFrame(index=data.index)
        
        if 'momentum' in self.factors:
            factor_df['momentum'] = self._calculate_momentum_factor(data['Close'])
        
        if 'mean_reversion' in self.factors:
            factor_df['mean_reversion'] = self._calculate_mean_reversion_factor(data['Close'])
        
        if 'volatility' in self.factors:
            factor_df['volatility'] = self._calculate_volatility_factor(data['Close'])
        
        if 'volume' in self.factors and 'Volume' in data.columns:
            factor_df['volume'] = self._calculate_volume_factor(data['Volume'])
        
        if 'rsi' in self.factors:
            factor_df['rsi'] = self._calculate_rsi_factor(data['Close'])
        
        if 'macd' in self.factors:
            factor_df['macd'] = self._calculate_macd_factor(data['Close'])
        
        if 'bollinger' in self.factors:
            factor_df['bollinger'] = self._calculate_bollinger_factor(data['Close'])
        
        return factor_df.fillna(0)
    
    def train_model(self, factor_data: pd.DataFrame, returns: pd.Series) -> None:
        """
        训练因子模型
        
        Args:
            factor_data: 因子数据
            returns: 未来收益率
        """
        # 对齐数据
        aligned_data = pd.concat([factor_data, returns], axis=1, join='inner').dropna()
        
        if len(aligned_data) < 100:
            raise ValueError("训练数据不足")
        
        X = aligned_data.iloc[:, :-1]  # 因子数据
        y = aligned_data.iloc[:, -1]   # 收益率
        
        # 标准化因子
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        self.model.fit(X_scaled, y)
    
    def predict_returns(self, factor_data: pd.DataFrame) -> pd.Series:
        """
        预测收益率
        
        Args:
            factor_data: 因子数据
            
        Returns:
            预测收益率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler.transform(factor_data.fillna(0))
        predictions = self.model.predict(X_scaled)
        
        return pd.Series(predictions, index=factor_data.index)
    
    def get_factor_importance(self) -> Dict[str, float]:
        """
        获取因子重要性
        
        Returns:
            因子重要性字典
        """
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'coef_'):
            # 线性模型
            importance = abs(self.model.coef_)
        elif hasattr(self.model, 'feature_importances_'):
            # 随机森林
            importance = self.model.feature_importances_
        else:
            return {}
        
        return dict(zip(self.factors, importance))


class FactorPortfolio:
    """
    因子组合管理器：基于多因子模型构建投资组合
    """
    
    def __init__(
        self,
        multi_factor: MultiFactor,
        weight_method: str = "equal",  # "equal", "factor_weighted", "risk_parity"
        max_weight: float = 0.1,
        min_weight: float = 0.01
    ):
        """
        初始化因子组合管理器
        
        Args:
            multi_factor: 多因子模型实例
            weight_method: 权重分配方法
            max_weight: 最大权重
            min_weight: 最小权重
        """
        self.multi_factor = multi_factor
        self.weight_method = weight_method
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def _calculate_equal_weights(self, selected_assets: List[str]) -> Dict[str, float]:
        """计算等权重"""
        n = len(selected_assets)
        weight = 1.0 / n
        return {asset: weight for asset in selected_assets}
    
    def _calculate_factor_weights(
        self, 
        selected_assets: List[str], 
        factor_scores: pd.Series
    ) -> Dict[str, float]:
        """基于因子得分计算权重"""
        scores = factor_scores[selected_assets]
        # 将负分数转换为正数（取绝对值）
        abs_scores = abs(scores)
        weights = abs_scores / abs_scores.sum()
        
        # 应用权重约束
        weights = weights.clip(self.min_weight, self.max_weight)
        weights = weights / weights.sum()  # 重新标准化
        
        return weights.to_dict()
    
    def _calculate_risk_parity_weights(
        self, 
        selected_assets: List[str], 
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """计算风险平价权重"""
        # 简化的风险平价：基于历史波动率的倒数
        volatilities = returns_data[selected_assets].std()
        inv_vol = 1 / (volatilities + 1e-12)
        weights = inv_vol / inv_vol.sum()
        
        # 应用权重约束
        weights = weights.clip(self.min_weight, self.max_weight)
        weights = weights / weights.sum()
        
        return weights.to_dict()
    
    def construct_portfolio(
        self, 
        factor_data: pd.DataFrame, 
        returns_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        构建投资组合
        
        Args:
            factor_data: 因子数据
            returns_data: 收益率数据（用于风险平价）
            
        Returns:
            资产权重字典
        """
        # 预测收益率
        predicted_returns = self.multi_factor.predict_returns(factor_data)
        
        # 选择top N资产
        top_assets = predicted_returns.nlargest(self.multi_factor.top_n).index.tolist()
        
        # 计算权重
        if self.weight_method == "equal":
            weights = self._calculate_equal_weights(top_assets)
        elif self.weight_method == "factor_weighted":
            weights = self._calculate_factor_weights(top_assets, predicted_returns)
        elif self.weight_method == "risk_parity" and returns_data is not None:
            weights = self._calculate_risk_parity_weights(top_assets, returns_data)
        else:
            weights = self._calculate_equal_weights(top_assets)
        
        return weights
    
    def backtest_portfolio(
        self, 
        price_data: Dict[str, pd.DataFrame], 
        start_date: str = None, 
        end_date: str = None
    ) -> Dict[str, pd.Series]:
        """
        回测多资产组合
        
        Args:
            price_data: 多资产价格数据字典
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果
        """
        # 获取所有资产的日期范围
        all_dates = set()
        for asset_data in price_data.values():
            all_dates.update(asset_data.index)
        
        date_range = sorted(all_dates)
        if start_date:
            date_range = [d for d in date_range if d >= pd.to_datetime(start_date)]
        if end_date:
            date_range = [d for d in date_range if d <= pd.to_datetime(end_date)]
        
        portfolio_returns = []
        portfolio_weights = []
        rebalance_dates = []
        
        for i in range(self.multi_factor.lookback, len(date_range), self.multi_factor.rebalance_freq):
            current_date = date_range[i]
            
            # 计算每个资产的因子
            asset_factors = {}
            for asset, data in price_data.items():
                if current_date in data.index:
                    historical_data = data.loc[:current_date].tail(self.multi_factor.lookback)
                    if len(historical_data) >= self.multi_factor.lookback:
                        factors = self.multi_factor.calculate_factors(historical_data)
                        asset_factors[asset] = factors.iloc[-1]  # 最新因子值
            
            if not asset_factors:
                continue
            
            # 构建因子DataFrame
            factor_df = pd.DataFrame(asset_factors).T
            
            # 训练模型（使用历史数据）
            try:
                # 计算未来收益率作为标签
                future_returns = {}
                for asset, data in price_data.items():
                    if asset in asset_factors:
                        future_idx = data.index.get_loc(current_date)
                        if future_idx + self.multi_factor.rebalance_freq < len(data):
                            future_price = data.iloc[future_idx + self.multi_factor.rebalance_freq]['Close']
                            current_price = data.loc[current_date]['Close']
                            future_returns[asset] = (future_price / current_price - 1)
                
                if len(future_returns) >= 5:  # 至少需要5个资产
                    returns_series = pd.Series(future_returns)
                    self.multi_factor.train_model(factor_df, returns_series)
                    
                    # 构建组合
                    weights = self.construct_portfolio(factor_df)
                    portfolio_weights.append(weights)
                    rebalance_dates.append(current_date)
                    
                    # 计算组合收益
                    period_return = 0
                    for asset, weight in weights.items():
                        if asset in future_returns:
                            period_return += weight * future_returns[asset]
                    
                    portfolio_returns.append(period_return)
            
            except Exception as e:
                print(f"在日期 {current_date} 处理时出错: {e}")
                continue
        
        # 构建结果
        results = {
            'portfolio_returns': pd.Series(portfolio_returns, index=rebalance_dates),
            'rebalance_dates': rebalance_dates,
            'portfolio_weights': portfolio_weights
        }
        
        if portfolio_returns:
            cumulative_returns = (1 + pd.Series(portfolio_returns, index=rebalance_dates)).cumprod()
            results['cumulative_returns'] = cumulative_returns
        
        return results