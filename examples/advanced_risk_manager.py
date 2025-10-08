#!/usr/bin/env python3
"""
高级风险管理系统
为高频交易提供全面的风险控制、仓位管理和合规监控
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import threading
import time
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    """风险等级"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """警报类型"""
    POSITION_LIMIT = "POSITION_LIMIT"
    LOSS_LIMIT = "LOSS_LIMIT"
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    CORRELATION_RISK = "CORRELATION_RISK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"

@dataclass
class RiskAlert:
    """风险警报"""
    timestamp: datetime
    alert_type: AlertType
    risk_level: RiskLevel
    symbol: str
    message: str
    current_value: float
    threshold: float
    recommended_action: str

@dataclass
class PositionRisk:
    """持仓风险"""
    symbol: str
    quantity: int
    market_value: float
    unrealized_pnl: float
    var_1d: float  # 1日风险价值
    var_5d: float  # 5日风险价值
    beta: float    # 市场贝塔
    volatility: float  # 波动率
    liquidity_score: float  # 流动性评分
    concentration_ratio: float  # 集中度比例

@dataclass
class PortfolioRisk:
    """组合风险"""
    timestamp: datetime
    total_value: float
    total_exposure: float
    net_exposure: float
    gross_exposure: float
    var_1d: float
    var_5d: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    correlation_risk: float
    concentration_risk: float

class AdvancedRiskManager:
    """高级风险管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化风险管理器
        
        Args:
            config: 风险管理配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 风险限制配置
        self.max_portfolio_value = config.get('max_portfolio_value', 1000000)
        self.max_daily_loss = config.get('max_daily_loss', 50000)
        self.max_position_size = config.get('max_position_size', 100000)
        self.max_symbol_exposure = config.get('max_symbol_exposure', 200000)
        self.max_sector_exposure = config.get('max_sector_exposure', 300000)
        self.max_correlation_exposure = config.get('max_correlation_exposure', 500000)
        self.max_leverage = config.get('max_leverage', 3.0)
        self.var_limit = config.get('var_limit', 20000)
        
        # 风险监控参数
        self.volatility_threshold = config.get('volatility_threshold', 0.3)
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.liquidity_threshold = config.get('liquidity_threshold', 0.3)
        self.concentration_threshold = config.get('concentration_threshold', 0.2)
        
        # 数据存储
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_history: deque = deque(maxlen=1000)
        self.risk_alerts: deque = deque(maxlen=1000)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))  # 1年数据
        self.return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        
        # 市场数据
        self.market_data: Dict[str, Dict] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.beta_values: Dict[str, float] = {}
        
        # 状态管理
        self.running = False
        self.last_update = datetime.now()
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 风险计算缓存
        self.risk_cache: Dict[str, Any] = {}
        self.cache_expiry = timedelta(minutes=1)
        
        # 合规检查
        self.compliance_rules = self.load_compliance_rules()
        
        # 预警系统
        self.alert_callbacks: List[callable] = []
    
    def load_compliance_rules(self) -> Dict[str, Any]:
        """加载合规规则"""
        return {
            'max_single_position': 0.1,  # 单一持仓不超过组合的10%
            'max_sector_weight': 0.3,    # 单一行业不超过30%
            'min_diversification': 5,     # 最少持有5个不同品种
            'max_turnover_rate': 5.0,     # 最大换手率500%
            'max_intraday_loss': 0.02,    # 日内最大亏损2%
            'position_size_limits': {     # 按品种设置仓位限制
                'AAPL': 50000,
                'MSFT': 50000,
                'GOOGL': 50000,
                'TSLA': 30000,  # 高波动品种限制更严格
                'SPY': 100000
            }
        }
    
    def add_alert_callback(self, callback: callable):
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)
    
    def update_market_data(self, symbol: str, price: float, volume: int = 0, 
                          bid: float = 0, ask: float = 0):
        """更新市场数据"""
        with self.lock:
            current_time = datetime.now()
            
            # 更新价格历史
            if symbol in self.price_history and len(self.price_history[symbol]) > 0:
                prev_price = self.price_history[symbol][-1]
                if prev_price > 0:
                    return_rate = (price - prev_price) / prev_price
                    self.return_history[symbol].append(return_rate)
            
            self.price_history[symbol].append(price)
            
            # 更新市场数据
            self.market_data[symbol] = {
                'price': price,
                'volume': volume,
                'bid': bid,
                'ask': ask,
                'spread': ask - bid if ask > bid else 0,
                'timestamp': current_time
            }
            
            # 更新波动率和贝塔
            self.update_risk_metrics(symbol)
    
    def update_risk_metrics(self, symbol: str):
        """更新风险指标"""
        try:
            if len(self.return_history[symbol]) < 20:
                return
            
            returns = np.array(list(self.return_history[symbol]))
            
            # 计算波动率（年化）
            volatility = np.std(returns) * np.sqrt(252)
            
            # 计算贝塔（简化，使用SPY作为基准）
            if symbol != 'SPY' and 'SPY' in self.return_history:
                spy_returns = np.array(list(self.return_history['SPY']))
                if len(spy_returns) == len(returns) and len(returns) > 10:
                    covariance = np.cov(returns, spy_returns)[0, 1]
                    market_variance = np.var(spy_returns)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0 if symbol == 'SPY' else 1.0
            
            self.beta_values[symbol] = beta
            
            # 更新持仓风险（如果存在持仓）
            if symbol in self.positions:
                position = self.positions[symbol]
                position.volatility = volatility
                position.beta = beta
                
                # 计算VaR
                position.var_1d = abs(position.market_value) * volatility / np.sqrt(252) * 1.65  # 95%置信度
                position.var_5d = abs(position.market_value) * volatility / np.sqrt(252/5) * 1.65
                
        except Exception as e:
            self.logger.error(f"风险指标更新错误 {symbol}: {e}")
    
    def update_position(self, symbol: str, quantity: int, avg_price: float, 
                       market_value: float, unrealized_pnl: float):
        """更新持仓信息"""
        with self.lock:
            # 计算流动性评分
            liquidity_score = self.calculate_liquidity_score(symbol)
            
            # 计算集中度比例
            total_portfolio_value = sum(abs(pos.market_value) for pos in self.positions.values())
            concentration_ratio = abs(market_value) / max(total_portfolio_value, 1)
            
            # 创建或更新持仓风险
            self.positions[symbol] = PositionRisk(
                symbol=symbol,
                quantity=quantity,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                var_1d=0,  # 将在update_risk_metrics中计算
                var_5d=0,
                beta=self.beta_values.get(symbol, 1.0),
                volatility=0,  # 将在update_risk_metrics中计算
                liquidity_score=liquidity_score,
                concentration_ratio=concentration_ratio
            )
            
            # 更新风险指标
            self.update_risk_metrics(symbol)
            
            # 检查持仓风险
            self.check_position_risk(symbol)
    
    def calculate_liquidity_score(self, symbol: str) -> float:
        """计算流动性评分"""
        try:
            if symbol not in self.market_data:
                return 0.5  # 默认中等流动性
            
            market_data = self.market_data[symbol]
            
            # 基于买卖价差计算流动性
            spread = market_data.get('spread', 0) if isinstance(market_data, dict) else getattr(market_data, 'ask', 0) - getattr(market_data, 'bid', 0)
            price = market_data.get('price', 1) if isinstance(market_data, dict) else getattr(market_data, 'last', 1)
            spread_ratio = spread / price if price > 0 else 1
            
            # 基于成交量计算流动性
            volume = market_data.get('volume', 0) if isinstance(market_data, dict) else getattr(market_data, 'volume', 0)
            
            # 流动性评分（0-1，1为最高流动性）
            spread_score = max(0, 1 - spread_ratio * 100)  # 价差越小流动性越好
            volume_score = min(1, volume / 100000)  # 成交量越大流动性越好
            
            # 综合评分
            liquidity_score = (spread_score * 0.6 + volume_score * 0.4)
            
            return max(0.1, min(1.0, liquidity_score))
            
        except Exception as e:
            self.logger.error(f"流动性评分计算错误 {symbol}: {e}")
            return 0.5
    
    def check_position_risk(self, symbol: str):
        """检查单个持仓风险"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        alerts = []
        
        # 检查持仓规模限制
        max_position = self.compliance_rules['position_size_limits'].get(symbol, self.max_position_size)
        if abs(position.market_value) > max_position:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.POSITION_LIMIT,
                risk_level=RiskLevel.HIGH,
                symbol=symbol,
                message=f"持仓规模超限: {abs(position.market_value):.2f} > {max_position:.2f}",
                current_value=abs(position.market_value),
                threshold=max_position,
                recommended_action="减少持仓规模"
            ))
        
        # 检查集中度风险
        if position.concentration_ratio > self.concentration_threshold:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.CONCENTRATION_RISK,
                risk_level=RiskLevel.MEDIUM,
                symbol=symbol,
                message=f"持仓集中度过高: {position.concentration_ratio:.2%}",
                current_value=position.concentration_ratio,
                threshold=self.concentration_threshold,
                recommended_action="分散投资，降低单一品种权重"
            ))
        
        # 检查流动性风险
        if position.liquidity_score < self.liquidity_threshold:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.LIQUIDITY_RISK,
                risk_level=RiskLevel.MEDIUM,
                symbol=symbol,
                message=f"流动性不足: {position.liquidity_score:.2f}",
                current_value=position.liquidity_score,
                threshold=self.liquidity_threshold,
                recommended_action="谨慎交易，考虑减仓"
            ))
        
        # 检查波动率风险
        if position.volatility > self.volatility_threshold:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.VOLATILITY_SPIKE,
                risk_level=RiskLevel.HIGH,
                symbol=symbol,
                message=f"波动率异常: {position.volatility:.2%}",
                current_value=position.volatility,
                threshold=self.volatility_threshold,
                recommended_action="降低仓位或加强止损"
            ))
        
        # 添加警报
        for alert in alerts:
            self.add_alert(alert)
    
    def check_portfolio_risk(self) -> PortfolioRisk:
        """检查组合风险"""
        try:
            current_time = datetime.now()
            
            # 计算组合基本指标
            total_value = sum(pos.market_value for pos in self.positions.values())
            total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
            net_exposure = sum(pos.market_value for pos in self.positions.values())
            gross_exposure = total_exposure
            
            # 计算组合VaR
            portfolio_var_1d = self.calculate_portfolio_var(1)
            portfolio_var_5d = self.calculate_portfolio_var(5)
            
            # 计算预期损失
            expected_shortfall = portfolio_var_1d * 1.3  # 简化计算
            
            # 计算最大回撤
            max_drawdown = self.calculate_max_drawdown()
            
            # 计算风险调整收益指标
            sharpe_ratio = self.calculate_sharpe_ratio()
            sortino_ratio = self.calculate_sortino_ratio()
            calmar_ratio = self.calculate_calmar_ratio()
            
            # 计算相关性风险
            correlation_risk = self.calculate_correlation_risk()
            
            # 计算集中度风险
            concentration_risk = self.calculate_concentration_risk()
            
            # 创建组合风险对象
            portfolio_risk = PortfolioRisk(
                timestamp=current_time,
                total_value=total_value,
                total_exposure=total_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                var_1d=portfolio_var_1d,
                var_5d=portfolio_var_5d,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk
            )
            
            # 检查组合级别风险限制
            self.check_portfolio_limits(portfolio_risk)
            
            # 保存历史记录
            self.portfolio_history.append(portfolio_risk)
            
            return portfolio_risk
            
        except Exception as e:
            self.logger.error(f"组合风险检查错误: {e}")
            return None
    
    def calculate_portfolio_var(self, days: int) -> float:
        """计算组合VaR"""
        try:
            if not self.positions:
                return 0.0
            
            # 简化的VaR计算（假设正态分布）
            total_var = 0.0
            
            for symbol, position in self.positions.items():
                if days == 1:
                    position_var = position.var_1d
                else:
                    position_var = position.var_5d
                
                total_var += position_var ** 2
            
            # 考虑相关性（简化处理）
            correlation_adjustment = 0.8  # 假设平均相关性
            portfolio_var = np.sqrt(total_var) * correlation_adjustment
            
            return portfolio_var
            
        except Exception as e:
            self.logger.error(f"组合VaR计算错误: {e}")
            return 0.0
    
    def calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        try:
            if len(self.portfolio_history) < 2:
                return 0.0
            
            values = [p.total_value for p in self.portfolio_history]
            peak = values[0]
            max_dd = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception as e:
            self.logger.error(f"最大回撤计算错误: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        try:
            if len(self.portfolio_history) < 10:
                return 0.0
            
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_value = self.portfolio_history[i-1].total_value
                curr_value = self.portfolio_history[i].total_value
                if prev_value > 0:
                    ret = (curr_value - prev_value) / prev_value
                    returns.append(ret)
            
            if len(returns) < 2:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # 年化夏普比率（假设无风险利率为0）
            sharpe = mean_return / std_return * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            self.logger.error(f"夏普比率计算错误: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self) -> float:
        """计算索提诺比率"""
        try:
            if len(self.portfolio_history) < 10:
                return 0.0
            
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_value = self.portfolio_history[i-1].total_value
                curr_value = self.portfolio_history[i].total_value
                if prev_value > 0:
                    ret = (curr_value - prev_value) / prev_value
                    returns.append(ret)
            
            if len(returns) < 2:
                return 0.0
            
            mean_return = np.mean(returns)
            negative_returns = [r for r in returns if r < 0]
            
            if len(negative_returns) == 0:
                return float('inf')
            
            downside_deviation = np.std(negative_returns)
            
            if downside_deviation == 0:
                return 0.0
            
            # 年化索提诺比率
            sortino = mean_return / downside_deviation * np.sqrt(252)
            return sortino
            
        except Exception as e:
            self.logger.error(f"索提诺比率计算错误: {e}")
            return 0.0
    
    def calculate_calmar_ratio(self) -> float:
        """计算卡玛比率"""
        try:
            max_dd = self.calculate_max_drawdown()
            if max_dd == 0:
                return 0.0
            
            # 计算年化收益率
            if len(self.portfolio_history) < 2:
                return 0.0
            
            start_value = self.portfolio_history[0].total_value
            end_value = self.portfolio_history[-1].total_value
            
            if start_value <= 0:
                return 0.0
            
            total_return = (end_value - start_value) / start_value
            
            # 简化的年化收益率计算
            periods = len(self.portfolio_history)
            annual_return = total_return * (252 / periods) if periods > 0 else 0
            
            calmar = annual_return / max_dd if max_dd > 0 else 0
            return calmar
            
        except Exception as e:
            self.logger.error(f"卡玛比率计算错误: {e}")
            return 0.0
    
    def calculate_correlation_risk(self) -> float:
        """计算相关性风险"""
        try:
            if len(self.positions) < 2:
                return 0.0
            
            symbols = list(self.positions.keys())
            correlations = []
            
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    
                    if (symbol1 in self.return_history and symbol2 in self.return_history and
                        len(self.return_history[symbol1]) > 10 and len(self.return_history[symbol2]) > 10):
                        
                        returns1 = np.array(list(self.return_history[symbol1]))
                        returns2 = np.array(list(self.return_history[symbol2]))
                        
                        min_len = min(len(returns1), len(returns2))
                        if min_len > 10:
                            corr = np.corrcoef(returns1[-min_len:], returns2[-min_len:])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
            
            if not correlations:
                return 0.0
            
            # 平均相关性作为相关性风险指标
            avg_correlation = np.mean(correlations)
            return avg_correlation
            
        except Exception as e:
            self.logger.error(f"相关性风险计算错误: {e}")
            return 0.0
    
    def calculate_concentration_risk(self) -> float:
        """计算集中度风险"""
        try:
            if not self.positions:
                return 0.0
            
            total_value = sum(abs(pos.market_value) for pos in self.positions.values())
            if total_value == 0:
                return 0.0
            
            # 计算赫芬达尔指数
            weights = [abs(pos.market_value) / total_value for pos in self.positions.values()]
            herfindahl_index = sum(w**2 for w in weights)
            
            return herfindahl_index
            
        except Exception as e:
            self.logger.error(f"集中度风险计算错误: {e}")
            return 0.0
    
    def check_portfolio_limits(self, portfolio_risk: PortfolioRisk):
        """检查组合限制"""
        alerts = []
        
        # 检查总敞口限制
        if portfolio_risk.total_exposure > self.max_portfolio_value:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.EXPOSURE_LIMIT,
                risk_level=RiskLevel.HIGH,
                symbol="PORTFOLIO",
                message=f"总敞口超限: {portfolio_risk.total_exposure:.2f}",
                current_value=portfolio_risk.total_exposure,
                threshold=self.max_portfolio_value,
                recommended_action="减少整体仓位"
            ))
        
        # 检查VaR限制
        if portfolio_risk.var_1d > self.var_limit:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.LOSS_LIMIT,
                risk_level=RiskLevel.HIGH,
                symbol="PORTFOLIO",
                message=f"VaR超限: {portfolio_risk.var_1d:.2f}",
                current_value=portfolio_risk.var_1d,
                threshold=self.var_limit,
                recommended_action="降低风险敞口"
            ))
        
        # 检查相关性风险
        if portfolio_risk.correlation_risk > self.correlation_threshold:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.CORRELATION_RISK,
                risk_level=RiskLevel.MEDIUM,
                symbol="PORTFOLIO",
                message=f"相关性风险过高: {portfolio_risk.correlation_risk:.2f}",
                current_value=portfolio_risk.correlation_risk,
                threshold=self.correlation_threshold,
                recommended_action="增加投资品种多样性"
            ))
        
        # 添加警报
        for alert in alerts:
            self.add_alert(alert)
    
    def add_alert(self, alert: RiskAlert):
        """添加风险警报"""
        self.risk_alerts.append(alert)
        self.logger.warning(f"风险警报: {alert.message}")
        
        # 调用回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"警报回调错误: {e}")
    
    def check_trade_risk(self, symbol: str, action: str, quantity: int, price: float) -> Tuple[bool, str]:
        """检查交易风险"""
        try:
            # 计算交易价值
            trade_value = quantity * price
            
            # 检查单笔交易限制
            max_trade_size = self.compliance_rules['position_size_limits'].get(symbol, self.max_position_size) * 0.1
            if trade_value > max_trade_size:
                return False, f"单笔交易规模过大: {trade_value:.2f} > {max_trade_size:.2f}"
            
            # 检查持仓限制
            current_position = self.positions.get(symbol)
            if current_position:
                if action == 'BUY':
                    new_value = current_position.market_value + trade_value
                else:
                    new_value = current_position.market_value - trade_value
                
                max_position = self.compliance_rules['position_size_limits'].get(symbol, self.max_position_size)
                if abs(new_value) > max_position:
                    return False, f"交易后持仓将超限: {abs(new_value):.2f} > {max_position:.2f}"
            
            # 检查流动性
            if symbol in self.market_data:
                liquidity_score = self.calculate_liquidity_score(symbol)
                if liquidity_score < self.liquidity_threshold and trade_value > 10000:
                    return False, f"流动性不足，不适合大额交易: {liquidity_score:.2f}"
            
            # 检查波动率
            if symbol in self.positions:
                position = self.positions[symbol]
                if position.volatility > self.volatility_threshold and trade_value > 5000:
                    return False, f"波动率过高，限制交易规模: {position.volatility:.2%}"
            
            return True, "交易风险检查通过"
            
        except Exception as e:
            self.logger.error(f"交易风险检查错误: {e}")
            return False, f"风险检查失败: {str(e)}"
    
    def get_position_limit(self, symbol: str) -> float:
        """获取持仓限制"""
        return self.compliance_rules['position_size_limits'].get(symbol, self.max_position_size)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        portfolio_risk = self.check_portfolio_risk()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_risk.total_value if portfolio_risk else 0,
            'total_exposure': portfolio_risk.total_exposure if portfolio_risk else 0,
            'var_1d': portfolio_risk.var_1d if portfolio_risk else 0,
            'max_drawdown': portfolio_risk.max_drawdown if portfolio_risk else 0,
            'sharpe_ratio': portfolio_risk.sharpe_ratio if portfolio_risk else 0,
            'correlation_risk': portfolio_risk.correlation_risk if portfolio_risk else 0,
            'concentration_risk': portfolio_risk.concentration_risk if portfolio_risk else 0,
            'active_positions': len(self.positions),
            'recent_alerts': len([a for a in self.risk_alerts if 
                                (datetime.now() - a.timestamp).total_seconds() < 3600]),
            'risk_level': self.get_overall_risk_level()
        }
    
    def get_overall_risk_level(self) -> str:
        """获取整体风险等级"""
        try:
            # 基于多个指标评估整体风险
            risk_score = 0
            
            # 检查最近警报
            recent_alerts = [a for a in self.risk_alerts if 
                           (datetime.now() - a.timestamp).total_seconds() < 3600]
            
            critical_alerts = sum(1 for a in recent_alerts if a.risk_level == RiskLevel.CRITICAL)
            high_alerts = sum(1 for a in recent_alerts if a.risk_level == RiskLevel.HIGH)
            
            if critical_alerts > 0:
                risk_score += 40
            if high_alerts > 2:
                risk_score += 30
            
            # 检查组合指标
            portfolio_risk = self.check_portfolio_risk()
            if portfolio_risk:
                if portfolio_risk.var_1d > self.var_limit * 0.8:
                    risk_score += 20
                if portfolio_risk.correlation_risk > self.correlation_threshold:
                    risk_score += 15
                if portfolio_risk.concentration_risk > 0.5:
                    risk_score += 15
            
            # 确定风险等级
            if risk_score >= 60:
                return "CRITICAL"
            elif risk_score >= 40:
                return "HIGH"
            elif risk_score >= 20:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            self.logger.error(f"整体风险等级评估错误: {e}")
            return "UNKNOWN"
    
    def export_risk_report(self, filepath: str):
        """导出风险报告"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'risk_summary': self.get_risk_summary(),
                'positions': {symbol: asdict(pos) for symbol, pos in self.positions.items()},
                'recent_alerts': [asdict(alert) for alert in list(self.risk_alerts)[-50:]],
                'portfolio_history': [asdict(p) for p in list(self.portfolio_history)[-100:]],
                'compliance_rules': self.compliance_rules,
                'risk_limits': {
                    'max_portfolio_value': self.max_portfolio_value,
                    'max_daily_loss': self.max_daily_loss,
                    'max_position_size': self.max_position_size,
                    'var_limit': self.var_limit
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"风险报告已导出: {filepath}")
            
        except Exception as e:
            self.logger.error(f"风险报告导出错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 风险管理配置
    risk_config = {
        'max_portfolio_value': 1000000,
        'max_daily_loss': 50000,
        'max_position_size': 100000,
        'max_symbol_exposure': 200000,
        'var_limit': 20000,
        'volatility_threshold': 0.3,
        'correlation_threshold': 0.8,
        'liquidity_threshold': 0.3,
        'concentration_threshold': 0.2
    }
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建风险管理器
    risk_manager = AdvancedRiskManager(risk_config)
    
    # 添加警报回调
    def alert_handler(alert: RiskAlert):
        print(f"风险警报: {alert.message}")
    
    risk_manager.add_alert_callback(alert_handler)
    
    # 模拟数据更新
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    for i in range(100):
        for symbol in symbols:
            # 模拟价格数据
            price = 100 + np.random.normal(0, 5)
            volume = np.random.randint(1000, 10000)
            
            risk_manager.update_market_data(symbol, price, volume)
            
            # 模拟持仓更新
            if i % 10 == 0:
                quantity = np.random.randint(-500, 500)
                market_value = quantity * price
                unrealized_pnl = np.random.normal(0, 1000)
                
                risk_manager.update_position(symbol, quantity, price, market_value, unrealized_pnl)
        
        time.sleep(0.1)
    
    # 获取风险摘要
    summary = risk_manager.get_risk_summary()
    print(f"风险摘要: {summary}")
    
    # 导出风险报告
    risk_manager.export_risk_report('risk_report.json')