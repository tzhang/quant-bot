#!/usr/bin/env python3
"""
Interactive Brokers 风险管理系统
提供全面的风险控制和订单管理功能
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskAction(Enum):
    """风险处理动作"""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    FORCE_CLOSE = "force_close"

@dataclass
class RiskLimit:
    """风险限制配置"""
    # 仓位限制
    max_position_value: float = 50000.0  # 最大仓位价值
    max_symbol_exposure: float = 20000.0  # 单标的最大暴露
    max_sector_exposure: float = 30000.0  # 单行业最大暴露
    max_leverage: float = 2.0  # 最大杠杆
    
    # 交易限制
    max_daily_trades: int = 100  # 日最大交易次数
    max_order_size: int = 1000  # 单笔订单最大数量
    min_order_value: float = 100.0  # 最小订单价值
    
    # 损失限制
    max_daily_loss: float = 5000.0  # 日最大损失
    max_drawdown: float = 0.15  # 最大回撤比例
    stop_loss_pct: float = 0.05  # 止损百分比
    
    # 时间限制
    trading_start_time: str = "09:30"  # 交易开始时间
    trading_end_time: str = "16:00"  # 交易结束时间
    
    # 波动率限制
    max_volatility: float = 0.5  # 最大波动率
    volatility_window: int = 20  # 波动率计算窗口

@dataclass
class RiskMetrics:
    """风险指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 仓位风险
    total_position_value: float = 0.0
    total_exposure: float = 0.0
    leverage: float = 0.0
    concentration_risk: float = 0.0
    
    # 损益风险
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0
    
    # 交易风险
    daily_trades: int = 0
    trade_frequency: float = 0.0
    avg_trade_size: float = 0.0
    
    # 市场风险
    portfolio_beta: float = 0.0
    var_1d: float = 0.0  # 1日风险价值
    var_5d: float = 0.0  # 5日风险价值
    
    # 风险等级
    overall_risk_level: RiskLevel = RiskLevel.LOW

@dataclass
class RiskAlert:
    """风险警报"""
    timestamp: datetime
    risk_type: str
    level: RiskLevel
    message: str
    symbol: Optional[str] = None
    value: Optional[float] = None
    limit: Optional[float] = None
    action: RiskAction = RiskAction.WARN

class IBRiskManager:
    """IB风险管理器"""
    
    def __init__(self, limits: RiskLimit = None):
        self.limits = limits or RiskLimit()
        
        # 风险数据
        self.positions: Dict[str, Any] = {}
        self.orders: Dict[int, Any] = {}
        self.executions: List[Any] = []
        self.market_data: Dict[str, Any] = {}
        
        # 风险指标历史
        self.risk_history: deque = deque(maxlen=1000)
        self.alerts: deque = deque(maxlen=500)
        
        # 统计数据
        self.daily_stats = {
            'trades': 0,
            'pnl': 0.0,
            'volume': 0,
            'start_time': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        }
        
        # 价格历史（用于波动率计算）
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 风险监控线程
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """启动风险监控"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("风险监控已启动")
    
    def stop_monitoring(self):
        """停止风险监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("风险监控已停止")
    
    def _monitor_loop(self):
        """风险监控循环"""
        while self._monitoring:
            try:
                # 计算风险指标
                metrics = self.calculate_risk_metrics()
                
                # 检查风险限制
                alerts = self.check_risk_limits(metrics)
                
                # 处理风险警报
                for alert in alerts:
                    self._handle_risk_alert(alert)
                
                # 记录风险指标
                with self._lock:
                    self.risk_history.append(metrics)
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"风险监控异常: {e}")
                time.sleep(5)
    
    def check_order_risk(self, symbol: str, action: str, quantity: int, price: float) -> Tuple[bool, List[RiskAlert]]:
        """检查订单风险"""
        alerts = []
        
        # 检查交易时间
        if not self._is_trading_hours():
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="trading_hours",
                level=RiskLevel.HIGH,
                message="非交易时间下单",
                action=RiskAction.BLOCK
            ))
        
        # 检查订单大小
        order_value = quantity * price
        if quantity > self.limits.max_order_size:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="order_size",
                level=RiskLevel.HIGH,
                message=f"订单数量超限: {quantity} > {self.limits.max_order_size}",
                symbol=symbol,
                value=quantity,
                limit=self.limits.max_order_size,
                action=RiskAction.BLOCK
            ))
        
        if order_value < self.limits.min_order_value:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="order_value",
                level=RiskLevel.MEDIUM,
                message=f"订单价值过小: ${order_value:.2f} < ${self.limits.min_order_value}",
                symbol=symbol,
                value=order_value,
                limit=self.limits.min_order_value,
                action=RiskAction.WARN
            ))
        
        # 检查日交易次数
        if self.daily_stats['trades'] >= self.limits.max_daily_trades:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="daily_trades",
                level=RiskLevel.HIGH,
                message=f"日交易次数超限: {self.daily_stats['trades']} >= {self.limits.max_daily_trades}",
                action=RiskAction.BLOCK
            ))
        
        # 检查仓位暴露
        current_exposure = self._calculate_symbol_exposure(symbol)
        if action.upper() == "BUY":
            new_exposure = current_exposure + order_value
        else:
            new_exposure = current_exposure - order_value
            
        if abs(new_exposure) > self.limits.max_symbol_exposure:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="symbol_exposure",
                level=RiskLevel.HIGH,
                message=f"标的暴露超限: {symbol} ${abs(new_exposure):.2f} > ${self.limits.max_symbol_exposure}",
                symbol=symbol,
                value=abs(new_exposure),
                limit=self.limits.max_symbol_exposure,
                action=RiskAction.BLOCK
            ))
        
        # 检查总仓位价值
        total_position_value = self._calculate_total_position_value()
        if total_position_value + order_value > self.limits.max_position_value:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="position_value",
                level=RiskLevel.HIGH,
                message=f"总仓位价值超限: ${total_position_value + order_value:.2f} > ${self.limits.max_position_value}",
                value=total_position_value + order_value,
                limit=self.limits.max_position_value,
                action=RiskAction.BLOCK
            ))
        
        # 检查日损失限制
        if self.daily_stats['pnl'] <= -self.limits.max_daily_loss:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="daily_loss",
                level=RiskLevel.CRITICAL,
                message=f"日损失超限: ${self.daily_stats['pnl']:.2f} <= ${-self.limits.max_daily_loss}",
                value=self.daily_stats['pnl'],
                limit=-self.limits.max_daily_loss,
                action=RiskAction.BLOCK
            ))
        
        # 检查波动率
        volatility = self._calculate_volatility(symbol)
        if volatility > self.limits.max_volatility:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="volatility",
                level=RiskLevel.MEDIUM,
                message=f"标的波动率过高: {symbol} {volatility:.2%} > {self.limits.max_volatility:.2%}",
                symbol=symbol,
                value=volatility,
                limit=self.limits.max_volatility,
                action=RiskAction.WARN
            ))
        
        # 判断是否允许订单
        blocking_alerts = [a for a in alerts if a.action == RiskAction.BLOCK]
        allow_order = len(blocking_alerts) == 0
        
        return allow_order, alerts
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """计算风险指标"""
        metrics = RiskMetrics()
        
        with self._lock:
            # 仓位风险
            metrics.total_position_value = self._calculate_total_position_value()
            metrics.total_exposure = self._calculate_total_exposure()
            metrics.leverage = self._calculate_leverage()
            metrics.concentration_risk = self._calculate_concentration_risk()
            
            # 损益风险
            metrics.daily_pnl = self.daily_stats['pnl']
            metrics.unrealized_pnl = self._calculate_unrealized_pnl()
            metrics.realized_pnl = self._calculate_realized_pnl()
            metrics.max_drawdown = self._calculate_max_drawdown()
            
            # 交易风险
            metrics.daily_trades = self.daily_stats['trades']
            metrics.trade_frequency = self._calculate_trade_frequency()
            metrics.avg_trade_size = self._calculate_avg_trade_size()
            
            # 市场风险
            metrics.portfolio_beta = self._calculate_portfolio_beta()
            metrics.var_1d = self._calculate_var(1)
            metrics.var_5d = self._calculate_var(5)
            
            # 综合风险等级
            metrics.overall_risk_level = self._assess_overall_risk(metrics)
        
        return metrics
    
    def check_risk_limits(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """检查风险限制"""
        alerts = []
        
        # 检查仓位限制
        if metrics.total_position_value > self.limits.max_position_value:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="position_value",
                level=RiskLevel.HIGH,
                message=f"总仓位价值超限: ${metrics.total_position_value:.2f}",
                value=metrics.total_position_value,
                limit=self.limits.max_position_value,
                action=RiskAction.WARN
            ))
        
        # 检查杠杆限制
        if metrics.leverage > self.limits.max_leverage:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="leverage",
                level=RiskLevel.HIGH,
                message=f"杠杆超限: {metrics.leverage:.2f}x",
                value=metrics.leverage,
                limit=self.limits.max_leverage,
                action=RiskAction.WARN
            ))
        
        # 检查回撤限制
        if metrics.max_drawdown > self.limits.max_drawdown:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="drawdown",
                level=RiskLevel.CRITICAL,
                message=f"回撤超限: {metrics.max_drawdown:.2%}",
                value=metrics.max_drawdown,
                limit=self.limits.max_drawdown,
                action=RiskAction.FORCE_CLOSE
            ))
        
        # 检查日损失
        if metrics.daily_pnl <= -self.limits.max_daily_loss:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type="daily_loss",
                level=RiskLevel.CRITICAL,
                message=f"日损失超限: ${metrics.daily_pnl:.2f}",
                value=metrics.daily_pnl,
                limit=-self.limits.max_daily_loss,
                action=RiskAction.FORCE_CLOSE
            ))
        
        return alerts
    
    def update_position(self, symbol: str, position_data: Dict):
        """更新持仓数据"""
        with self._lock:
            self.positions[symbol] = position_data
    
    def update_order(self, order_id: int, order_data: Dict):
        """更新订单数据"""
        with self._lock:
            self.orders[order_id] = order_data
    
    def update_execution(self, execution_data: Dict):
        """更新执行数据"""
        with self._lock:
            self.executions.append(execution_data)
            
            # 更新日统计
            if self._is_same_day(execution_data.get('timestamp', datetime.now())):
                self.daily_stats['trades'] += 1
                self.daily_stats['volume'] += execution_data.get('quantity', 0)
    
    def update_market_data(self, symbol: str, market_data: Dict):
        """更新市场数据"""
        with self._lock:
            self.market_data[symbol] = market_data
            
            # 更新价格历史
            if 'last' in market_data:
                self.price_history[symbol].append({
                    'price': market_data['last'],
                    'timestamp': datetime.now()
                })
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        metrics = self.calculate_risk_metrics()
        recent_alerts = list(self.alerts)[-10:]  # 最近10个警报
        
        return {
            'metrics': metrics,
            'recent_alerts': recent_alerts,
            'daily_stats': self.daily_stats,
            'limits': self.limits,
            'monitoring_status': self._monitoring
        }
    
    # ========== 私有方法 ==========
    
    def _is_trading_hours(self) -> bool:
        """检查是否在交易时间内"""
        now = datetime.now()
        
        # 检查是否为工作日
        if now.weekday() >= 5:  # 周末
            return False
        
        # 检查时间范围
        start_time = datetime.strptime(self.limits.trading_start_time, "%H:%M").time()
        end_time = datetime.strptime(self.limits.trading_end_time, "%H:%M").time()
        
        return start_time <= now.time() <= end_time
    
    def _is_same_day(self, timestamp: datetime) -> bool:
        """检查是否为同一天"""
        return timestamp.date() == self.daily_stats['start_time'].date()
    
    def _calculate_symbol_exposure(self, symbol: str) -> float:
        """计算标的暴露"""
        position = self.positions.get(symbol, {})
        quantity = position.get('quantity', 0)
        market_price = self.market_data.get(symbol, {}).get('last', 0)
        return abs(quantity * market_price)
    
    def _calculate_total_position_value(self) -> float:
        """计算总仓位价值"""
        total = 0.0
        for symbol, position in self.positions.items():
            quantity = position.get('quantity', 0)
            market_price = self.market_data.get(symbol, {}).get('last', position.get('avg_cost', 0))
            total += abs(quantity * market_price)
        return total
    
    def _calculate_total_exposure(self) -> float:
        """计算总暴露"""
        return sum(self._calculate_symbol_exposure(symbol) for symbol in self.positions.keys())
    
    def _calculate_leverage(self) -> float:
        """计算杠杆比率"""
        total_position_value = self._calculate_total_position_value()
        # 这里需要账户净值信息，暂时使用仓位价值
        return total_position_value / max(total_position_value, 1.0)
    
    def _calculate_concentration_risk(self) -> float:
        """计算集中度风险"""
        if not self.positions:
            return 0.0
            
        total_value = self._calculate_total_position_value()
        if total_value == 0:
            return 0.0
            
        # 计算最大单一持仓占比
        max_position_pct = 0.0
        for symbol in self.positions.keys():
            symbol_exposure = self._calculate_symbol_exposure(symbol)
            position_pct = symbol_exposure / total_value
            max_position_pct = max(max_position_pct, position_pct)
            
        return max_position_pct
    
    def _calculate_unrealized_pnl(self) -> float:
        """计算未实现盈亏"""
        total_pnl = 0.0
        for symbol, position in self.positions.items():
            quantity = position.get('quantity', 0)
            avg_cost = position.get('avg_cost', 0)
            market_price = self.market_data.get(symbol, {}).get('last', avg_cost)
            total_pnl += (market_price - avg_cost) * quantity
        return total_pnl
    
    def _calculate_realized_pnl(self) -> float:
        """计算已实现盈亏"""
        daily_executions = [
            exec for exec in self.executions 
            if self._is_same_day(exec.get('timestamp', datetime.now()))
        ]
        
        # 简化计算：假设所有执行都是盈利的
        return sum(exec.get('pnl', 0) for exec in daily_executions)
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.risk_history) < 2:
            return 0.0
            
        pnl_series = [metrics.daily_pnl for metrics in self.risk_history]
        peak = pnl_series[0]
        max_dd = 0.0
        
        for pnl in pnl_series[1:]:
            if pnl > peak:
                peak = pnl
            else:
                drawdown = (peak - pnl) / max(abs(peak), 1.0)
                max_dd = max(max_dd, drawdown)
                
        return max_dd
    
    def _calculate_trade_frequency(self) -> float:
        """计算交易频率"""
        hours_elapsed = (datetime.now() - self.daily_stats['start_time']).total_seconds() / 3600
        return self.daily_stats['trades'] / max(hours_elapsed, 1.0)
    
    def _calculate_avg_trade_size(self) -> float:
        """计算平均交易规模"""
        if self.daily_stats['trades'] == 0:
            return 0.0
        return self.daily_stats['volume'] / self.daily_stats['trades']
    
    def _calculate_portfolio_beta(self) -> float:
        """计算组合贝塔"""
        # 简化实现，实际需要市场数据
        return 1.0
    
    def _calculate_var(self, days: int) -> float:
        """计算风险价值"""
        # 简化实现，实际需要历史价格数据和统计模型
        return self._calculate_total_position_value() * 0.02 * (days ** 0.5)
    
    def _calculate_volatility(self, symbol: str) -> float:
        """计算波动率"""
        prices = self.price_history.get(symbol, deque())
        if len(prices) < 2:
            return 0.0
            
        price_values = [p['price'] for p in prices]
        returns = []
        
        for i in range(1, len(price_values)):
            ret = (price_values[i] - price_values[i-1]) / price_values[i-1]
            returns.append(ret)
        
        if not returns:
            return 0.0
            
        # 计算标准差
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5
    
    def _assess_overall_risk(self, metrics: RiskMetrics) -> RiskLevel:
        """评估整体风险等级"""
        risk_score = 0
        
        # 仓位风险评分
        if metrics.leverage > self.limits.max_leverage * 0.8:
            risk_score += 2
        elif metrics.leverage > self.limits.max_leverage * 0.6:
            risk_score += 1
            
        # 损失风险评分
        if metrics.daily_pnl <= -self.limits.max_daily_loss * 0.8:
            risk_score += 3
        elif metrics.daily_pnl <= -self.limits.max_daily_loss * 0.5:
            risk_score += 2
        elif metrics.daily_pnl <= -self.limits.max_daily_loss * 0.3:
            risk_score += 1
            
        # 回撤风险评分
        if metrics.max_drawdown > self.limits.max_drawdown * 0.8:
            risk_score += 3
        elif metrics.max_drawdown > self.limits.max_drawdown * 0.6:
            risk_score += 2
            
        # 集中度风险评分
        if metrics.concentration_risk > 0.5:
            risk_score += 2
        elif metrics.concentration_risk > 0.3:
            risk_score += 1
        
        # 根据评分确定风险等级
        if risk_score >= 6:
            return RiskLevel.CRITICAL
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _handle_risk_alert(self, alert: RiskAlert):
        """处理风险警报"""
        with self._lock:
            self.alerts.append(alert)
        
        # 记录警报
        level_str = alert.level.value.upper()
        logger.log(
            logging.CRITICAL if alert.level == RiskLevel.CRITICAL else
            logging.ERROR if alert.level == RiskLevel.HIGH else
            logging.WARNING if alert.level == RiskLevel.MEDIUM else
            logging.INFO,
            f"[{level_str}] {alert.message}"
        )
        
        # 根据动作类型处理
        if alert.action == RiskAction.FORCE_CLOSE:
            logger.critical(f"触发强制平仓: {alert.message}")
            # 这里应该调用强制平仓逻辑
        elif alert.action == RiskAction.BLOCK:
            logger.error(f"阻止操作: {alert.message}")
        elif alert.action == RiskAction.WARN:
            logger.warning(f"风险警告: {alert.message}")


def example_usage():
    """使用示例"""
    # 创建风险限制配置
    limits = RiskLimit(
        max_position_value=10000.0,
        max_daily_trades=50,
        max_daily_loss=1000.0,
        max_drawdown=0.10
    )
    
    # 创建风险管理器
    risk_manager = IBRiskManager(limits)
    
    # 启动监控
    risk_manager.start_monitoring()
    
    try:
        # 模拟订单风险检查
        allow, alerts = risk_manager.check_order_risk("AAPL", "BUY", 100, 150.0)
        print(f"订单风险检查: {'允许' if allow else '拒绝'}")
        for alert in alerts:
            print(f"  警报: {alert.message}")
        
        # 模拟数据更新
        risk_manager.update_position("AAPL", {
            'quantity': 100,
            'avg_cost': 150.0
        })
        
        risk_manager.update_market_data("AAPL", {
            'last': 155.0,
            'bid': 154.8,
            'ask': 155.2
        })
        
        # 获取风险摘要
        summary = risk_manager.get_risk_summary()
        print(f"\n风险摘要:")
        print(f"  总仓位价值: ${summary['metrics'].total_position_value:.2f}")
        print(f"  日盈亏: ${summary['metrics'].daily_pnl:.2f}")
        print(f"  风险等级: {summary['metrics'].overall_risk_level.value}")
        
        time.sleep(5)
        
    finally:
        risk_manager.stop_monitoring()


if __name__ == "__main__":
    example_usage()