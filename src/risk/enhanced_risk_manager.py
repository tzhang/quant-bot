"""
增强版风险管理器

整合现有风险管理功能，提供统一接口和实时监控
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from enum import Enum
import threading
import time
from collections import deque, defaultdict
import warnings

from .risk_manager import (
    RiskManager, RiskMetric, VaRMethod, RiskMeasure,
    StressTestScenario, StressTestResult
)
from ..core.risk_manager import AdaptiveRiskManager, MarketRegimeDetector, VolatilityPredictor

warnings.filterwarnings('ignore')

# 配置日志
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """警报类型"""
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    VAR_BREACH = "var_breach"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_RISK = "correlation_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    DRAWDOWN_LIMIT = "drawdown_limit"
    LEVERAGE_LIMIT = "leverage_limit"


@dataclass
class RiskAlert:
    """风险警报"""
    timestamp: datetime
    alert_type: AlertType
    risk_level: RiskLevel
    symbol: Optional[str]
    message: str
    current_value: float
    threshold: float
    recommended_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'risk_level': self.risk_level.value,
            'symbol': self.symbol,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'recommended_action': self.recommended_action
        }


@dataclass
class RiskLimits:
    """风险限制配置"""
    # 仓位限制
    max_position_size: float = 0.1  # 单个仓位最大比例
    max_sector_exposure: float = 0.3  # 单个行业最大暴露
    max_leverage: float = 3.0  # 最大杠杆
    
    # 损失限制
    max_daily_loss: float = 0.05  # 最大日损失比例
    max_drawdown: float = 0.15  # 最大回撤比例
    stop_loss_pct: float = 0.05  # 止损百分比
    
    # 风险指标限制
    var_limit_1d: float = 0.02  # 1日VaR限制
    var_limit_5d: float = 0.05  # 5日VaR限制
    max_volatility: float = 0.3  # 最大波动率
    max_correlation: float = 0.8  # 最大相关性
    
    # 流动性限制
    min_liquidity_score: float = 0.3  # 最小流动性评分
    max_concentration: float = 0.2  # 最大集中度
    
    # 交易限制
    max_daily_trades: int = 100  # 日最大交易次数
    max_order_size: float = 50000  # 最大订单金额


@dataclass
class PortfolioRiskMetrics:
    """投资组合风险指标"""
    timestamp: datetime
    
    # 基础指标
    total_value: float
    total_pnl: float
    daily_pnl: float
    
    # 风险指标
    var_1d: float
    var_5d: float
    cvar_1d: float
    expected_shortfall: float
    volatility: float
    max_drawdown: float
    
    # 暴露指标
    leverage: float
    long_exposure: float
    short_exposure: float
    net_exposure: float
    
    # 集中度指标
    concentration_ratio: float
    largest_position: float
    sector_exposures: Dict[str, float]
    
    # 流动性指标
    avg_liquidity_score: float
    illiquid_positions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'var_1d': self.var_1d,
            'var_5d': self.var_5d,
            'cvar_1d': self.cvar_1d,
            'expected_shortfall': self.expected_shortfall,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'leverage': self.leverage,
            'long_exposure': self.long_exposure,
            'short_exposure': self.short_exposure,
            'net_exposure': self.net_exposure,
            'concentration_ratio': self.concentration_ratio,
            'largest_position': self.largest_position,
            'sector_exposures': self.sector_exposures,
            'avg_liquidity_score': self.avg_liquidity_score,
            'illiquid_positions': self.illiquid_positions
        }


class EnhancedRiskManager:
    """增强版风险管理器"""
    
    def __init__(self, 
                 risk_limits: Optional[RiskLimits] = None,
                 monitoring_interval: int = 60):
        """
        初始化增强版风险管理器
        
        Args:
            risk_limits: 风险限制配置
            monitoring_interval: 监控间隔（秒）
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.monitoring_interval = monitoring_interval
        
        # 核心风险管理器
        self.base_risk_manager = RiskManager()
        self.adaptive_risk_manager = AdaptiveRiskManager()
        self.regime_detector = MarketRegimeDetector()
        self.volatility_predictor = VolatilityPredictor()
        
        # 监控数据
        self.positions: Dict[str, Any] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.risk_metrics_history: deque = deque(maxlen=1000)
        self.alerts: deque = deque(maxlen=500)
        
        # 实时监控
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 回调函数
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []
        
        self.logger = logger
        self.logger.info("增强版风险管理器初始化完成")
    
    def start_monitoring(self):
        """启动实时监控"""
        if self._monitoring:
            self.logger.warning("风险监控已在运行")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("风险监控已启动")
    
    def stop_monitoring(self):
        """停止实时监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("风险监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                self._perform_risk_check()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"风险监控错误: {e}")
                time.sleep(self.monitoring_interval)
    
    def _perform_risk_check(self):
        """执行风险检查"""
        with self._lock:
            if not self.positions:
                return
            
            # 计算当前风险指标
            risk_metrics = self.calculate_portfolio_risk_metrics()
            
            # 检查风险限制
            alerts = self._check_risk_limits(risk_metrics)
            
            # 处理警报
            for alert in alerts:
                self._handle_alert(alert)
            
            # 保存历史记录
            self.risk_metrics_history.append(risk_metrics)
    
    def update_positions(self, positions: Dict[str, Any]):
        """更新持仓信息"""
        with self._lock:
            self.positions = positions.copy()
            self.logger.debug(f"更新持仓信息: {len(positions)} 个仓位")
    
    def update_market_data(self, symbol: str, data: pd.DataFrame):
        """更新市场数据"""
        with self._lock:
            self.market_data[symbol] = data.copy()
            self.logger.debug(f"更新市场数据: {symbol}")
    
    def calculate_portfolio_risk_metrics(self) -> PortfolioRiskMetrics:
        """计算投资组合风险指标"""
        if not self.positions:
            return self._empty_risk_metrics()
        
        try:
            # 基础计算
            total_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
            total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
            
            # 构建权重向量
            weights = pd.Series({
                symbol: pos.get('market_value', 0) / total_value if total_value > 0 else 0
                for symbol, pos in self.positions.items()
            })
            
            # 获取收益率数据
            returns_data = self._get_returns_data()
            
            if returns_data is not None and not returns_data.empty:
                # 使用基础风险管理器计算风险指标
                self.base_risk_manager.risk_models['historical'].set_returns_data(returns_data)
                risk_measures = self.base_risk_manager.calculate_portfolio_risk(weights, 'historical')
                
                var_1d = risk_measures.get('VaR_5%', RiskMeasure(RiskMetric.VAR, 0.0)).value
                volatility = risk_measures.get('Volatility', RiskMeasure(RiskMetric.VOLATILITY, 0.0)).value
                max_drawdown = risk_measures.get('Maximum_Drawdown', RiskMeasure(RiskMetric.MAXIMUM_DRAWDOWN, 0.0)).value
            else:
                var_1d = 0.0
                volatility = 0.0
                max_drawdown = 0.0
            
            # 计算暴露指标
            long_exposure = sum(max(0, pos.get('market_value', 0)) for pos in self.positions.values())
            short_exposure = sum(min(0, pos.get('market_value', 0)) for pos in self.positions.values())
            net_exposure = long_exposure + short_exposure
            leverage = (long_exposure - short_exposure) / max(total_value, 1)
            
            # 计算集中度指标
            position_values = [abs(pos.get('market_value', 0)) for pos in self.positions.values()]
            largest_position = max(position_values) / max(total_value, 1) if position_values else 0
            concentration_ratio = sum(sorted(position_values, reverse=True)[:5]) / max(total_value, 1)
            
            # 行业暴露（简化版）
            sector_exposures = self._calculate_sector_exposures()
            
            # 流动性指标
            liquidity_scores = [pos.get('liquidity_score', 0.5) for pos in self.positions.values()]
            avg_liquidity_score = np.mean(liquidity_scores) if liquidity_scores else 0.5
            illiquid_positions = [
                symbol for symbol, pos in self.positions.items()
                if pos.get('liquidity_score', 0.5) < self.risk_limits.min_liquidity_score
            ]
            
            return PortfolioRiskMetrics(
                timestamp=datetime.now(),
                total_value=total_value,
                total_pnl=total_pnl,
                daily_pnl=self._calculate_daily_pnl(),
                var_1d=var_1d,
                var_5d=var_1d * np.sqrt(5),  # 简化计算
                cvar_1d=var_1d * 1.2,  # 简化计算
                expected_shortfall=var_1d * 1.3,  # 简化计算
                volatility=volatility,
                max_drawdown=max_drawdown,
                leverage=leverage,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                net_exposure=net_exposure,
                concentration_ratio=concentration_ratio,
                largest_position=largest_position,
                sector_exposures=sector_exposures,
                avg_liquidity_score=avg_liquidity_score,
                illiquid_positions=illiquid_positions
            )
            
        except Exception as e:
            self.logger.error(f"计算风险指标错误: {e}")
            return self._empty_risk_metrics()
    
    def _empty_risk_metrics(self) -> PortfolioRiskMetrics:
        """返回空的风险指标"""
        return PortfolioRiskMetrics(
            timestamp=datetime.now(),
            total_value=0.0,
            total_pnl=0.0,
            daily_pnl=0.0,
            var_1d=0.0,
            var_5d=0.0,
            cvar_1d=0.0,
            expected_shortfall=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            leverage=0.0,
            long_exposure=0.0,
            short_exposure=0.0,
            net_exposure=0.0,
            concentration_ratio=0.0,
            largest_position=0.0,
            sector_exposures={},
            avg_liquidity_score=0.5,
            illiquid_positions=[]
        )
    
    def _get_returns_data(self) -> Optional[pd.DataFrame]:
        """获取收益率数据"""
        try:
            if not self.market_data:
                return None
            
            returns_dict = {}
            for symbol, data in self.market_data.items():
                if 'Close' in data.columns and len(data) > 1:
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_dict[symbol] = returns
            
            if returns_dict:
                return pd.DataFrame(returns_dict).fillna(0)
            return None
            
        except Exception as e:
            self.logger.error(f"获取收益率数据错误: {e}")
            return None
    
    def _calculate_daily_pnl(self) -> float:
        """计算日收益"""
        # 简化实现，实际应该基于历史数据
        return sum(pos.get('daily_pnl', 0) for pos in self.positions.values())
    
    def _calculate_sector_exposures(self) -> Dict[str, float]:
        """计算行业暴露"""
        # 简化实现，实际应该基于股票行业分类
        sector_exposures = defaultdict(float)
        total_value = sum(abs(pos.get('market_value', 0)) for pos in self.positions.values())
        
        for symbol, pos in self.positions.items():
            # 简化的行业分类逻辑
            sector = pos.get('sector', 'Unknown')
            sector_exposures[sector] += abs(pos.get('market_value', 0)) / max(total_value, 1)
        
        return dict(sector_exposures)
    
    def _check_risk_limits(self, metrics: PortfolioRiskMetrics) -> List[RiskAlert]:
        """检查风险限制"""
        alerts = []
        
        # 检查VaR限制
        if metrics.var_1d > self.risk_limits.var_limit_1d:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.VAR_BREACH,
                risk_level=RiskLevel.HIGH,
                symbol=None,
                message=f"1日VaR超限: {metrics.var_1d:.4f} > {self.risk_limits.var_limit_1d:.4f}",
                current_value=metrics.var_1d,
                threshold=self.risk_limits.var_limit_1d,
                recommended_action="减少风险暴露或调整仓位"
            ))
        
        # 检查杠杆限制
        if metrics.leverage > self.risk_limits.max_leverage:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.LEVERAGE_LIMIT,
                risk_level=RiskLevel.HIGH,
                symbol=None,
                message=f"杠杆超限: {metrics.leverage:.2f} > {self.risk_limits.max_leverage:.2f}",
                current_value=metrics.leverage,
                threshold=self.risk_limits.max_leverage,
                recommended_action="降低杠杆水平"
            ))
        
        # 检查最大回撤
        if metrics.max_drawdown > self.risk_limits.max_drawdown:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.DRAWDOWN_LIMIT,
                risk_level=RiskLevel.CRITICAL,
                symbol=None,
                message=f"最大回撤超限: {metrics.max_drawdown:.4f} > {self.risk_limits.max_drawdown:.4f}",
                current_value=metrics.max_drawdown,
                threshold=self.risk_limits.max_drawdown,
                recommended_action="立即减仓或停止交易"
            ))
        
        # 检查集中度风险
        if metrics.concentration_ratio > self.risk_limits.max_concentration:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.CONCENTRATION_RISK,
                risk_level=RiskLevel.MEDIUM,
                symbol=None,
                message=f"集中度风险: {metrics.concentration_ratio:.4f} > {self.risk_limits.max_concentration:.4f}",
                current_value=metrics.concentration_ratio,
                threshold=self.risk_limits.max_concentration,
                recommended_action="分散投资组合"
            ))
        
        # 检查流动性风险
        if metrics.avg_liquidity_score < self.risk_limits.min_liquidity_score:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.LIQUIDITY_RISK,
                risk_level=RiskLevel.MEDIUM,
                symbol=None,
                message=f"流动性不足: {metrics.avg_liquidity_score:.4f} < {self.risk_limits.min_liquidity_score:.4f}",
                current_value=metrics.avg_liquidity_score,
                threshold=self.risk_limits.min_liquidity_score,
                recommended_action="增加流动性较好的资产"
            ))
        
        return alerts
    
    def _handle_alert(self, alert: RiskAlert):
        """处理风险警报"""
        # 添加到警报历史
        self.alerts.append(alert)
        
        # 记录日志
        log_level = {
            RiskLevel.LOW: logging.INFO,
            RiskLevel.MEDIUM: logging.WARNING,
            RiskLevel.HIGH: logging.ERROR,
            RiskLevel.CRITICAL: logging.CRITICAL
        }.get(alert.risk_level, logging.WARNING)
        
        self.logger.log(log_level, f"风险警报: {alert.message}")
        
        # 调用回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"警报回调错误: {e}")
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)
        self.logger.info("已添加警报回调函数")
    
    def get_current_risk_metrics(self) -> Optional[PortfolioRiskMetrics]:
        """获取当前风险指标"""
        with self._lock:
            return self.calculate_portfolio_risk_metrics()
    
    def get_risk_history(self, hours: int = 24) -> List[PortfolioRiskMetrics]:
        """获取风险历史记录"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metrics for metrics in self.risk_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_recent_alerts(self, hours: int = 24) -> List[RiskAlert]:
        """获取最近的警报"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]
    
    def validate_trade(self, symbol: str, quantity: int, price: float) -> Tuple[bool, str]:
        """验证交易"""
        try:
            # 计算交易价值
            trade_value = abs(quantity * price)
            
            # 检查订单大小限制
            if trade_value > self.risk_limits.max_order_size:
                return False, f"订单金额超限: {trade_value} > {self.risk_limits.max_order_size}"
            
            # 检查仓位限制
            current_position = self.positions.get(symbol, {})
            current_value = current_position.get('market_value', 0)
            new_value = current_value + (quantity * price)
            
            total_portfolio_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
            if total_portfolio_value > 0:
                position_ratio = abs(new_value) / total_portfolio_value
                if position_ratio > self.risk_limits.max_position_size:
                    return False, f"仓位比例超限: {position_ratio:.4f} > {self.risk_limits.max_position_size:.4f}"
            
            return True, "交易验证通过"
            
        except Exception as e:
            self.logger.error(f"交易验证错误: {e}")
            return False, f"验证错误: {str(e)}"
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """生成风险报告"""
        current_metrics = self.get_current_risk_metrics()
        recent_alerts = self.get_recent_alerts(24)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics.to_dict() if current_metrics else None,
            'recent_alerts': [alert.to_dict() for alert in recent_alerts],
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_leverage': self.risk_limits.max_leverage,
                'var_limit_1d': self.risk_limits.var_limit_1d,
                'max_drawdown': self.risk_limits.max_drawdown,
                'max_concentration': self.risk_limits.max_concentration
            },
            'monitoring_status': self._monitoring,
            'positions_count': len(self.positions),
            'market_data_symbols': list(self.market_data.keys())
        }


# 使用示例
if __name__ == "__main__":
    # 创建风险限制配置
    risk_limits = RiskLimits(
        max_position_size=0.15,
        max_leverage=2.5,
        var_limit_1d=0.025,
        max_drawdown=0.12
    )
    
    # 创建增强版风险管理器
    risk_manager = EnhancedRiskManager(risk_limits=risk_limits)
    
    # 添加警报回调
    def alert_handler(alert: RiskAlert):
        print(f"🚨 风险警报: {alert.message}")
    
    risk_manager.add_alert_callback(alert_handler)
    
    # 模拟持仓数据
    positions = {
        'AAPL': {
            'market_value': 10000,
            'unrealized_pnl': 500,
            'daily_pnl': 200,
            'liquidity_score': 0.9,
            'sector': 'Technology'
        },
        'GOOGL': {
            'market_value': 8000,
            'unrealized_pnl': -200,
            'daily_pnl': -100,
            'liquidity_score': 0.8,
            'sector': 'Technology'
        }
    }
    
    # 更新持仓
    risk_manager.update_positions(positions)
    
    # 启动监控
    risk_manager.start_monitoring()
    
    # 计算风险指标
    metrics = risk_manager.get_current_risk_metrics()
    if metrics:
        print(f"总价值: ${metrics.total_value:,.2f}")
        print(f"1日VaR: {metrics.var_1d:.4f}")
        print(f"杠杆: {metrics.leverage:.2f}")
        print(f"集中度: {metrics.concentration_ratio:.4f}")
    
    # 验证交易
    is_valid, message = risk_manager.validate_trade('MSFT', 100, 300)
    print(f"交易验证: {message}")
    
    # 生成风险报告
    report = risk_manager.generate_risk_report()
    print(f"风险报告生成完成，监控状态: {report['monitoring_status']}")
    
    # 停止监控
    time.sleep(2)
    risk_manager.stop_monitoring()