#!/usr/bin/env python3
"""
高频交易盯盘系统
实时监控市场数据、策略信号、交易执行和风险状况
"""

import time
import threading
import queue
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradingSignal:
    """交易信号数据结构"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float   # 信号强度 0-1
    price: float
    confidence: float
    components: Dict[str, float]  # 各组件信号贡献

@dataclass
class TradeExecution:
    """交易执行数据结构"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: int
    price: float
    order_id: str
    status: str  # 'PENDING', 'FILLED', 'CANCELLED'
    pnl: float = 0.0

@dataclass
class RiskMetrics:
    """风险指标数据结构"""
    timestamp: datetime
    portfolio_value: float
    cash: float
    total_exposure: float
    max_drawdown: float
    var_1d: float  # 1日VaR
    sharpe_ratio: float
    positions: Dict[str, float]

class HFTMonitorSystem:
    """高频交易盯盘系统"""
    
    def __init__(self):
        # 数据存储
        self.market_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.signals: deque = deque(maxlen=1000)
        self.trades: deque = deque(maxlen=1000)
        self.risk_metrics: deque = deque(maxlen=1000)
        self.performance_data: deque = deque(maxlen=1000)
        
        # 监控配置
        self.monitored_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        self.update_interval = 1.0  # 更新间隔(秒)
        
        # 状态管理
        self.running = False
        self.data_queue = queue.Queue()
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
        # GUI组件
        self.root = None
        self.setup_gui()
        
        # 图表数据
        self.price_history = {symbol: deque(maxlen=100) for symbol in self.monitored_symbols}
        self.pnl_history = deque(maxlen=100)
        self.signal_history = deque(maxlen=100)
        
        # 统计数据
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_positions': {},
            'daily_volume': 0,
            'avg_trade_size': 0.0
        }
    
    def setup_gui(self):
        """设置GUI界面"""
        self.root = tk.Tk()
        self.root.title("高频交易盯盘系统")
        self.root.geometry("1400x900")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建标签页
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 市场数据标签页
        self.market_frame = ttk.Frame(notebook)
        notebook.add(self.market_frame, text="市场数据")
        self.setup_market_tab()
        
        # 交易信号标签页
        self.signal_frame = ttk.Frame(notebook)
        notebook.add(self.signal_frame, text="交易信号")
        self.setup_signal_tab()
        
        # 交易执行标签页
        self.trade_frame = ttk.Frame(notebook)
        notebook.add(self.trade_frame, text="交易执行")
        self.setup_trade_tab()
        
        # 风险监控标签页
        self.risk_frame = ttk.Frame(notebook)
        notebook.add(self.risk_frame, text="风险监控")
        self.setup_risk_tab()
        
        # 性能分析标签页
        self.performance_frame = ttk.Frame(notebook)
        notebook.add(self.performance_frame, text="性能分析")
        self.setup_performance_tab()
        
        # 控制面板
        self.setup_control_panel(main_frame)
    
    def setup_market_tab(self):
        """设置市场数据标签页"""
        # 市场数据表格
        columns = ('Symbol', 'Last', 'Bid', 'Ask', 'Spread', 'Volume', 'Change%', 'Time')
        self.market_tree = ttk.Treeview(self.market_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.market_tree.heading(col, text=col)
            self.market_tree.column(col, width=100)
        
        # 滚动条
        market_scrollbar = ttk.Scrollbar(self.market_frame, orient=tk.VERTICAL, command=self.market_tree.yview)
        self.market_tree.configure(yscrollcommand=market_scrollbar.set)
        
        # 布局
        self.market_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        market_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 价格图表
        self.setup_price_chart()
    
    def setup_signal_tab(self):
        """设置交易信号标签页"""
        # 信号列表
        signal_columns = ('Time', 'Symbol', 'Signal', 'Strength', 'Price', 'Confidence', 'Components')
        self.signal_tree = ttk.Treeview(self.signal_frame, columns=signal_columns, show='headings', height=20)
        
        for col in signal_columns:
            self.signal_tree.heading(col, text=col)
            self.signal_tree.column(col, width=120)
        
        signal_scrollbar = ttk.Scrollbar(self.signal_frame, orient=tk.VERTICAL, command=self.signal_tree.yview)
        self.signal_tree.configure(yscrollcommand=signal_scrollbar.set)
        
        self.signal_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        signal_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_trade_tab(self):
        """设置交易执行标签页"""
        # 交易列表
        trade_columns = ('Time', 'Symbol', 'Action', 'Quantity', 'Price', 'Order ID', 'Status', 'PnL')
        self.trade_tree = ttk.Treeview(self.trade_frame, columns=trade_columns, show='headings', height=20)
        
        for col in trade_columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=100)
        
        trade_scrollbar = ttk.Scrollbar(self.trade_frame, orient=tk.VERTICAL, command=self.trade_tree.yview)
        self.trade_tree.configure(yscrollcommand=trade_scrollbar.set)
        
        self.trade_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trade_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_risk_tab(self):
        """设置风险监控标签页"""
        # 风险指标显示
        risk_info_frame = ttk.LabelFrame(self.risk_frame, text="风险指标", padding=10)
        risk_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建风险指标标签
        self.risk_labels = {}
        risk_metrics = [
            ('Portfolio Value', 'portfolio_value'),
            ('Cash', 'cash'),
            ('Total Exposure', 'total_exposure'),
            ('Max Drawdown', 'max_drawdown'),
            ('1-Day VaR', 'var_1d'),
            ('Sharpe Ratio', 'sharpe_ratio')
        ]
        
        for i, (label, key) in enumerate(risk_metrics):
            row = i // 2
            col = i % 2
            ttk.Label(risk_info_frame, text=f"{label}:").grid(row=row, column=col*2, sticky=tk.W, padx=5, pady=2)
            self.risk_labels[key] = ttk.Label(risk_info_frame, text="--", foreground="blue")
            self.risk_labels[key].grid(row=row, column=col*2+1, sticky=tk.W, padx=5, pady=2)
        
        # 持仓信息
        position_frame = ttk.LabelFrame(self.risk_frame, text="当前持仓", padding=10)
        position_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        pos_columns = ('Symbol', 'Quantity', 'Avg Price', 'Current Price', 'Market Value', 'Unrealized PnL')
        self.position_tree = ttk.Treeview(position_frame, columns=pos_columns, show='headings', height=10)
        
        for col in pos_columns:
            self.position_tree.heading(col, text=col)
            self.position_tree.column(col, width=120)
        
        self.position_tree.pack(fill=tk.BOTH, expand=True)
    
    def setup_performance_tab(self):
        """设置性能分析标签页"""
        # 统计信息
        stats_frame = ttk.LabelFrame(self.performance_frame, text="交易统计", padding=10)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_labels = {}
        stats_metrics = [
            ('Total Trades', 'total_trades'),
            ('Winning Trades', 'winning_trades'),
            ('Win Rate', 'win_rate'),
            ('Total PnL', 'total_pnl'),
            ('Avg Trade Size', 'avg_trade_size'),
            ('Daily Volume', 'daily_volume')
        ]
        
        for i, (label, key) in enumerate(stats_metrics):
            row = i // 3
            col = i % 3
            ttk.Label(stats_frame, text=f"{label}:").grid(row=row, column=col*2, sticky=tk.W, padx=5, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="--", foreground="green")
            self.stats_labels[key].grid(row=row, column=col*2+1, sticky=tk.W, padx=5, pady=2)
        
        # PnL图表
        self.setup_pnl_chart()
    
    def setup_control_panel(self, parent):
        """设置控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # 控制按钮
        self.start_button = ttk.Button(control_frame, text="启动监控", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="停止监控", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 状态显示
        self.status_label = ttk.Label(control_frame, text="状态: 未启动", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # 更新间隔设置
        ttk.Label(control_frame, text="更新间隔(秒):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.StringVar(value="1.0")
        interval_entry = ttk.Entry(control_frame, textvariable=self.interval_var, width=10)
        interval_entry.pack(side=tk.LEFT, padx=5)
    
    def setup_price_chart(self):
        """设置价格图表"""
        chart_frame = ttk.LabelFrame(self.market_frame, text="价格走势", padding=5)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.price_fig, self.price_ax = plt.subplots(figsize=(10, 4))
        self.price_canvas = FigureCanvasTkAgg(self.price_fig, chart_frame)
        self.price_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_pnl_chart(self):
        """设置PnL图表"""
        chart_frame = ttk.LabelFrame(self.performance_frame, text="PnL走势", padding=5)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pnl_fig, self.pnl_ax = plt.subplots(figsize=(10, 4))
        self.pnl_canvas = FigureCanvasTkAgg(self.pnl_fig, chart_frame)
        self.pnl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def start_monitoring(self):
        """启动监控"""
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="状态: 运行中", foreground="green")
        
        # 启动数据更新线程
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("监控系统已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="状态: 已停止", foreground="red")
        
        self.logger.info("监控系统已停止")
    
    def update_loop(self):
        """数据更新循环"""
        while self.running:
            try:
                # 更新显示
                self.update_displays()
                
                # 等待下次更新
                time.sleep(float(self.interval_var.get()))
                
            except Exception as e:
                self.logger.error(f"更新循环错误: {e}")
                time.sleep(1)
    
    def update_displays(self):
        """更新所有显示"""
        # 在主线程中更新GUI
        self.root.after(0, self._update_gui)
    
    def _update_gui(self):
        """在主线程中更新GUI"""
        try:
            self.update_market_display()
            self.update_signal_display()
            self.update_trade_display()
            self.update_risk_display()
            self.update_performance_display()
            self.update_charts()
        except Exception as e:
            self.logger.error(f"GUI更新错误: {e}")
    
    def update_market_display(self):
        """更新市场数据显示"""
        # 清空现有数据
        for item in self.market_tree.get_children():
            self.market_tree.delete(item)
        
        # 添加模拟数据（实际使用时连接真实数据源）
        for symbol in self.monitored_symbols:
            # 生成模拟数据
            last_price = np.random.uniform(100, 300)
            bid = last_price - np.random.uniform(0.01, 0.1)
            ask = last_price + np.random.uniform(0.01, 0.1)
            spread = ask - bid
            volume = np.random.randint(10000, 100000)
            change_pct = np.random.uniform(-2, 2)
            
            self.market_tree.insert('', tk.END, values=(
                symbol,
                f"{last_price:.2f}",
                f"{bid:.2f}",
                f"{ask:.2f}",
                f"{spread:.4f}",
                f"{volume:,}",
                f"{change_pct:+.2f}%",
                datetime.now().strftime('%H:%M:%S')
            ))
            
            # 更新价格历史
            self.price_history[symbol].append(last_price)
    
    def update_signal_display(self):
        """更新交易信号显示"""
        # 保持最新的20条信号
        while len(self.signal_tree.get_children()) > 20:
            self.signal_tree.delete(self.signal_tree.get_children()[0])
        
        # 随机生成信号（实际使用时连接策略信号）
        if np.random.random() < 0.1:  # 10%概率生成信号
            symbol = np.random.choice(self.monitored_symbols)
            signal_type = np.random.choice(['BUY', 'SELL', 'HOLD'])
            strength = np.random.uniform(0.5, 1.0)
            price = np.random.uniform(100, 300)
            confidence = np.random.uniform(0.6, 0.95)
            
            components = {
                'momentum': np.random.uniform(-0.5, 0.5),
                'mean_reversion': np.random.uniform(-0.5, 0.5),
                'volatility': np.random.uniform(-0.3, 0.3),
                'volume': np.random.uniform(-0.2, 0.2)
            }
            
            self.signal_tree.insert('', tk.END, values=(
                datetime.now().strftime('%H:%M:%S'),
                symbol,
                signal_type,
                f"{strength:.3f}",
                f"{price:.2f}",
                f"{confidence:.3f}",
                str(components)
            ))
    
    def update_trade_display(self):
        """更新交易执行显示"""
        # 随机生成交易（实际使用时连接交易执行器）
        if np.random.random() < 0.05:  # 5%概率生成交易
            symbol = np.random.choice(self.monitored_symbols)
            action = np.random.choice(['BUY', 'SELL'])
            quantity = np.random.randint(100, 1000)
            price = np.random.uniform(100, 300)
            order_id = f"ORD{int(time.time())}"
            status = np.random.choice(['FILLED', 'PENDING'])
            pnl = np.random.uniform(-100, 200)
            
            self.trade_tree.insert('', tk.END, values=(
                datetime.now().strftime('%H:%M:%S'),
                symbol,
                action,
                quantity,
                f"{price:.2f}",
                order_id,
                status,
                f"{pnl:+.2f}"
            ))
            
            # 更新统计
            self.stats['total_trades'] += 1
            if pnl > 0:
                self.stats['winning_trades'] += 1
            self.stats['total_pnl'] += pnl
            
            # 保持最新的50条交易
            while len(self.trade_tree.get_children()) > 50:
                self.trade_tree.delete(self.trade_tree.get_children()[0])
    
    def update_risk_display(self):
        """更新风险监控显示"""
        # 更新风险指标
        portfolio_value = 1000000 + self.stats['total_pnl']
        cash = portfolio_value * 0.3
        exposure = portfolio_value * 0.7
        max_dd = min(0, self.stats['total_pnl'] / 1000000 * 100)
        var_1d = portfolio_value * 0.02
        sharpe = self.stats['total_pnl'] / max(1, np.sqrt(self.stats['total_trades'])) if self.stats['total_trades'] > 0 else 0
        
        self.risk_labels['portfolio_value'].config(text=f"${portfolio_value:,.2f}")
        self.risk_labels['cash'].config(text=f"${cash:,.2f}")
        self.risk_labels['total_exposure'].config(text=f"${exposure:,.2f}")
        self.risk_labels['max_drawdown'].config(text=f"{max_dd:.2f}%")
        self.risk_labels['var_1d'].config(text=f"${var_1d:,.2f}")
        self.risk_labels['sharpe_ratio'].config(text=f"{sharpe:.2f}")
        
        # 更新持仓信息（模拟数据）
        for item in self.position_tree.get_children():
            self.position_tree.delete(item)
        
        for symbol in self.monitored_symbols[:3]:  # 显示前3个持仓
            quantity = np.random.randint(100, 500)
            avg_price = np.random.uniform(100, 300)
            current_price = avg_price * (1 + np.random.uniform(-0.05, 0.05))
            market_value = quantity * current_price
            unrealized_pnl = quantity * (current_price - avg_price)
            
            self.position_tree.insert('', tk.END, values=(
                symbol,
                quantity,
                f"{avg_price:.2f}",
                f"{current_price:.2f}",
                f"{market_value:,.2f}",
                f"{unrealized_pnl:+,.2f}"
            ))
    
    def update_performance_display(self):
        """更新性能分析显示"""
        win_rate = (self.stats['winning_trades'] / max(1, self.stats['total_trades'])) * 100
        avg_trade_size = self.stats['total_pnl'] / max(1, self.stats['total_trades'])
        
        self.stats_labels['total_trades'].config(text=str(self.stats['total_trades']))
        self.stats_labels['winning_trades'].config(text=str(self.stats['winning_trades']))
        self.stats_labels['win_rate'].config(text=f"{win_rate:.1f}%")
        self.stats_labels['total_pnl'].config(text=f"${self.stats['total_pnl']:+,.2f}")
        self.stats_labels['avg_trade_size'].config(text=f"${avg_trade_size:+,.2f}")
        self.stats_labels['daily_volume'].config(text=f"{self.stats['daily_volume']:,}")
    
    def update_charts(self):
        """更新图表"""
        # 更新价格图表
        self.price_ax.clear()
        for symbol in self.monitored_symbols:
            if len(self.price_history[symbol]) > 1:
                self.price_ax.plot(list(self.price_history[symbol]), label=symbol)
        
        self.price_ax.set_title('价格走势')
        self.price_ax.legend()
        self.price_ax.grid(True)
        self.price_canvas.draw()
        
        # 更新PnL图表
        self.pnl_history.append(self.stats['total_pnl'])
        
        self.pnl_ax.clear()
        if len(self.pnl_history) > 1:
            self.pnl_ax.plot(list(self.pnl_history), 'g-', linewidth=2)
            self.pnl_ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        self.pnl_ax.set_title('累计PnL')
        self.pnl_ax.set_ylabel('PnL ($)')
        self.pnl_ax.grid(True)
        self.pnl_canvas.draw()
    
    def add_market_data(self, symbol: str, data: dict):
        """添加市场数据"""
        self.market_data[symbol].append({
            'timestamp': datetime.now(),
            **data
        })
    
    def add_signal(self, signal: TradingSignal):
        """添加交易信号"""
        self.signals.append(signal)
    
    def add_trade(self, trade: TradeExecution):
        """添加交易执行"""
        self.trades.append(trade)
    
    def add_risk_metrics(self, metrics: RiskMetrics):
        """添加风险指标"""
        self.risk_metrics.append(metrics)
    
    def run(self):
        """运行监控系统"""
        self.logger.info("启动高频交易盯盘系统")
        self.root.mainloop()

# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建监控系统
    monitor = HFTMonitorSystem()
    
    # 运行系统
    monitor.run()