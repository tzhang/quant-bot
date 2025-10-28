import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


class BacktestEngine:
    """
    增强版回测引擎：支持多种交易模式和风险管理
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        min_periods: int = 50,
        position_sizing: str = "fixed",  # "fixed", "volatility", "kelly"
        max_position: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        rebalance_freq: str = "daily"  # "daily", "weekly", "monthly"
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点成本
            min_periods: 最小数据周期要求
            position_sizing: 仓位管理方式
            max_position: 最大仓位比例
            stop_loss: 止损比例
            take_profit: 止盈比例
            rebalance_freq: 再平衡频率
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.min_periods = min_periods
        self.position_sizing = position_sizing
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.rebalance_freq = rebalance_freq

    def _calculate_position_size(
        self, 
        signal: float, 
        price: float, 
        volatility: float = None,
        current_capital: float = None
    ) -> float:
        """
        计算仓位大小（考虑交易成本）
        
        Args:
            signal: 交易信号强度
            price: 当前价格
            volatility: 价格波动率
            current_capital: 当前资金
            
        Returns:
            仓位比例
        """
        # 基础仓位计算
        if self.position_sizing == "fixed":
            base_position = signal * self.max_position
        elif self.position_sizing == "volatility":
            if volatility is None or volatility == 0:
                base_position = signal * self.max_position
            else:
                # 基于波动率的仓位调整
                vol_adj = min(0.02 / volatility, 1.0)  # 目标2%日波动率
                base_position = signal * self.max_position * vol_adj
        elif self.position_sizing == "kelly":
            # 简化的凯利公式（需要历史胜率和盈亏比数据）
            # 这里使用保守的固定比例
            base_position = signal * min(self.max_position * 0.5, 0.25)
        else:
            base_position = signal * self.max_position
        
        # 考虑交易成本的调整
        if abs(base_position) > 0.001 and current_capital is not None:
            # 估算交易成本比例
            cost_ratio = self.commission + self.slippage
            # 调整仓位以预留交易成本
            adjusted_position = base_position * (1 - cost_ratio)
            return max(0, adjusted_position)  # 确保不为负
        
        return base_position

    def _apply_risk_management(
        self, 
        position: float, 
        entry_price: float, 
        current_price: float
    ) -> Tuple[float, str]:
        """
        应用风险管理规则
        
        Args:
            position: 当前仓位
            entry_price: 入场价格
            current_price: 当前价格
            
        Returns:
            调整后的仓位和操作原因
        """
        if position == 0:
            return position, "no_position"
        
        pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)
        
        # 止损检查
        if self.stop_loss and pnl_pct < -self.stop_loss:
            return 0.0, "stop_loss"
        
        # 止盈检查
        if self.take_profit and pnl_pct > self.take_profit:
            return 0.0, "take_profit"
        
        return position, "hold"

    def run(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            data: 包含OHLCV数据的DataFrame
            signals: 交易信号序列
            
        Returns:
            回测结果字典
        """
        if len(data) < self.min_periods:
            raise ValueError(f"数据长度不足，至少需要 {self.min_periods} 个数据点")
        
        # 确保信号和数据对齐
        signals = signals.reindex(data.index, fill_value=0.0)
        
        # 添加调试信息
        print(f"BacktestEngine.run 开始:")
        print(f"  数据长度: {len(data)}")
        print(f"  信号长度: {len(signals)}")
        print(f"  数据索引范围: {data.index[0]} - {data.index[-1]}")
        print(f"  信号索引范围: {signals.index[0]} - {signals.index[-1]}")
        print(f"  信号前5个值: {signals.head().tolist()}")
        print(f"  信号非零数量: {(signals != 0).sum()}")
        
        # 找到第一个非零信号的位置
        first_nonzero = signals[signals != 0].index
        if len(first_nonzero) > 0:
            first_idx = signals.index.get_loc(first_nonzero[0])
            print(f"  第一个非零信号位置: {first_idx}, 值: {signals.iloc[first_idx]}")
            print(f"  信号第50-55个值: {signals.iloc[50:55].tolist()}")
        else:
            print(f"  没有找到非零信号！")
        
        # 初始化结果数组
        n = len(data)
        positions = np.zeros(n)
        cash = np.full(n, self.initial_capital)
        portfolio_value = np.full(n, self.initial_capital)
        trades = []
        
        # 计算价格波动率 - 兼容不同列名格式
        close_col = None
        for col in ['Close', 'close', 'CLOSE']:
            if col in data.columns:
                close_col = col
                break
        
        if close_col is None:
            raise ValueError("No 'Close' price column found in data")
        
        print(f"  使用价格列: {close_col}")
        print(f"  价格范围: {data[close_col].min():.2f} - {data[close_col].max():.2f}")
        
        returns = data[close_col].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        
        current_position = 0.0
        entry_price = 0.0
        shares = 0.0
        
        for i in range(1, n):
            current_price = data[close_col].iloc[i]
            signal = signals.iloc[i]
            current_vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.02
            
            # 应用风险管理
            if current_position != 0:
                current_position, action = self._apply_risk_management(
                    current_position, entry_price, current_price
                )
                if action in ["stop_loss", "take_profit"]:
                    signal = 0.0  # 强制平仓
            
            # 计算目标仓位
            target_position = self._calculate_position_size(
                signal, current_price, current_vol, cash[i-1]
            )
            
            # 仓位变化
            position_change = target_position - current_position
            
            # 添加调试信息（前10次）
            if i <= 10:
                print(f"  步骤 {i}: 信号={signal:.4f}, 目标仓位={target_position:.4f}, 当前仓位={current_position:.4f}, 仓位变化={position_change:.4f}")
            
            if abs(position_change) > 0.001:  # 最小交易阈值
                # 计算当前组合总价值
                current_portfolio_value = cash[i-1] + shares * current_price
                
                # 计算目标股票价值
                target_stock_value = target_position * current_portfolio_value
                current_stock_value = shares * current_price
                
                # 计算需要交易的股票价值
                trade_value = abs(target_stock_value - current_stock_value)
                transaction_cost = trade_value * (self.commission + self.slippage)
                
                # 添加调试日志（扩展到前20次）
                if i <= 20 or len(trades) < 5:  # 打印前20次或前5次交易的详细信息
                    print(f"交易 {i}: 信号={signals.iloc[i]:.4f}, 目标仓位={target_position:.4f}, 当前仓位={current_position:.4f}")
                    print(f"  当前价格={current_price:.2f}, 组合价值={current_portfolio_value:.2f}")
                    print(f"  目标股票价值={target_stock_value:.2f}, 当前股票价值={current_stock_value:.2f}")
                    print(f"  交易价值={trade_value:.2f}, 交易成本={transaction_cost:.2f}")
                    print(f"  现金={cash[i-1]:.2f}, 股票数量={shares:.4f}")
                
                # 更新现金和股票
                if position_change > 0:  # 买入
                    # 计算需要买入的股票数量
                    buy_value = target_stock_value - current_stock_value
                    total_cost = buy_value + transaction_cost
                    
                    if total_cost <= cash[i-1]:
                        cash[i] = cash[i-1] - total_cost
                        shares += buy_value / current_price
                        current_position = target_position
                        entry_price = current_price
                        if i <= 20 or len(trades) < 5:
                            print(f"  买入成功: 新现金={cash[i]:.2f}, 新股票数量={shares:.4f}")
                        
                        # 记录交易
                        trades.append({
                            'date': data.index[i],
                            'price': current_price,
                            'position_change': position_change,
                            'position': current_position,
                            'cost': transaction_cost
                        })
                    else:
                        # 资金不足，保持原仓位
                        cash[i] = cash[i-1]
                        current_position = positions[i-1]
                        if i <= 20 or len(trades) < 5:
                            print(f"  买入失败: 资金不足")
                else:  # 卖出
                    # 计算需要卖出的股票数量
                    sell_value = current_stock_value - target_stock_value
                    proceeds = sell_value - transaction_cost
                    
                    cash[i] = cash[i-1] + proceeds
                    shares -= sell_value / current_price
                    current_position = target_position
                    if current_position == 0:
                        entry_price = 0.0
                    if i <= 20 or len(trades) < 5:
                        print(f"  卖出成功: 新现金={cash[i]:.2f}, 新股票数量={shares:.4f}")
                    
                    # 记录交易
                    trades.append({
                        'date': data.index[i],
                        'price': current_price,
                        'position_change': position_change,
                        'position': current_position,
                        'cost': transaction_cost
                    })
            else:
                cash[i] = cash[i-1]
            
            positions[i] = current_position
            
            # 计算组合价值
            stock_value = shares * current_price
            portfolio_value[i] = cash[i] + stock_value
        
        # 计算收益率
        returns = pd.Series(portfolio_value, index=data.index).pct_change().fillna(0)
        
        # 计算换手率
        position_changes = pd.Series(positions, index=data.index).diff().abs()
        turnover = position_changes.sum() / len(data) * 252  # 年化换手率
        
        # 计算基准收益（买入持有）
        benchmark_returns = data[close_col].pct_change().fillna(0)
        
        return {
            "portfolio_value": pd.Series(portfolio_value, index=data.index),
            "positions": pd.Series(positions, index=data.index),
            "cash": pd.Series(cash, index=data.index),
            "returns": returns,
            "benchmark_returns": benchmark_returns,
            "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
            "turnover": turnover,
            "total_trades": len(trades),
            "final_value": portfolio_value[-1],
            "total_return": (portfolio_value[-1] / self.initial_capital - 1) * 100
        }

    def run_walk_forward(
        self, 
        data: pd.DataFrame, 
        strategy_class, 
        strategy_params: Dict[str, Any],
        train_period: int = 252,
        test_period: int = 63,
        step_size: int = 21
    ) -> Dict[str, Any]:
        """
        走势前进分析（Walk-Forward Analysis）
        
        Args:
            data: 历史数据
            strategy_class: 策略类
            strategy_params: 策略参数
            train_period: 训练期长度
            test_period: 测试期长度
            step_size: 步进大小
            
        Returns:
            走势前进分析结果
        """
        results = []
        start_idx = train_period
        
        while start_idx + test_period <= len(data):
            # 训练期数据
            train_data = data.iloc[start_idx - train_period:start_idx]
            
            # 测试期数据
            test_data = data.iloc[start_idx:start_idx + test_period]
            
            # 生成信号（使用训练期优化的参数）
            strategy = strategy_class(**strategy_params)
            train_signals = strategy.signal(train_data)
            test_signals = strategy.signal(test_data)
            
            # 运行回测
            test_result = self.run(test_data, test_signals)
            
            results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'test_return': test_result['total_return'],
                'test_sharpe': self._calculate_sharpe(test_result['returns']),
                'max_drawdown': self._calculate_max_drawdown(test_result['portfolio_value'])
            })
            
            start_idx += step_size
        
        return {
            'walk_forward_results': pd.DataFrame(results),
            'avg_return': np.mean([r['test_return'] for r in results]),
            'avg_sharpe': np.mean([r['test_sharpe'] for r in results]),
            'win_rate': len([r for r in results if r['test_return'] > 0]) / len(results)
        }

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if returns.std() == 0:
            return 0.0
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))

    def _calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """计算最大回撤"""
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        return drawdown.min() * 100