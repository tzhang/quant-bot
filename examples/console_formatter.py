"""
控制台输出格式化工具
提供美观、易读的控制台输出格式
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
from colorama import Fore, Back, Style, init

# 初始化colorama
init(autoreset=True)

class ConsoleFormatter:
    """控制台格式化器"""
    
    def __init__(self):
        self.width = 80
        self.separator = "=" * self.width
        self.sub_separator = "-" * self.width
        
    def print_title(self, title: str, color: str = Fore.CYAN):
        """打印主标题"""
        print(f"\n{color}{self.separator}")
        print(f"{color}{title.center(self.width)}")
        print(f"{color}{self.separator}{Style.RESET_ALL}")
        
    def print_header(self, title: str, color: str = Fore.CYAN):
        """打印标题头部"""
        print(f"\n{color}{self.separator}")
        print(f"{color}{title.center(self.width)}")
        print(f"{color}{self.separator}{Style.RESET_ALL}")
        
    def print_section(self, title: str, color: str = Fore.YELLOW):
        """打印章节标题"""
        print(f"\n{color}📊 {title}")
        print(f"{color}{self.sub_separator[:len(title)+5]}{Style.RESET_ALL}")
        
    def print_separator(self):
        """打印分隔线"""
        print(f"{Fore.CYAN}{self.sub_separator}{Style.RESET_ALL}")
        
    def print_success(self, message: str):
        """打印成功信息"""
        print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
        
    def print_key_value(self, key: str, value: Any, color: str = Fore.WHITE, 
                       value_color: str = Fore.GREEN):
        """打印键值对"""
        print(f"{color}{key:<30}: {value_color}{value}{Style.RESET_ALL}")
        
    def print_status(self, status: str, is_success: bool = True):
        """打印状态信息"""
        color = Fore.GREEN if is_success else Fore.RED
        icon = "✅" if is_success else "❌"
        print(f"{color}{icon} {status}{Style.RESET_ALL}")
        
    def print_warning(self, message: str):
        """打印警告信息"""
        print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")
        
    def print_error(self, message: str):
        """打印错误信息"""
        print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")
        
    def print_info(self, message: str):
        """打印信息"""
        print(f"{Fore.BLUE}ℹ️  {message}{Style.RESET_ALL}")
        
    def print_table(self, data: List[Dict], headers: List[str], title: str = ""):
        """打印表格"""
        if title:
            self.print_section(title)
            
        if not data:
            print(f"{Fore.YELLOW}📝 暂无数据{Style.RESET_ALL}")
            return
            
        # 计算列宽
        col_widths = {}
        for header in headers:
            col_widths[header] = max(len(header), 
                                   max(len(str(row.get(header, ''))) for row in data))
        
        # 打印表头
        header_line = "│ "
        for header in headers:
            header_line += f"{header:<{col_widths[header]}} │ "
        print(f"{Fore.CYAN}{header_line}{Style.RESET_ALL}")
        
        # 打印分隔线
        sep_line = "├─"
        for header in headers:
            sep_line += "─" * col_widths[header] + "─┼─"
        sep_line = sep_line[:-1] + "┤"
        print(f"{Fore.CYAN}{sep_line}{Style.RESET_ALL}")
        
        # 打印数据行
        for row in data:
            data_line = "│ "
            for header in headers:
                value = str(row.get(header, ''))
                data_line += f"{value:<{col_widths[header]}} │ "
            print(f"{Fore.WHITE}{data_line}{Style.RESET_ALL}")

class TradingConsoleFormatter(ConsoleFormatter):
    """交易系统专用控制台格式化器"""
    
    def print_system_status(self, status_data: Dict[str, Any]):
        """打印系统状态"""
        self.print_header("🚀 Citadel-IB 集成系统状态", Fore.CYAN)
        
        # 连接状态
        self.print_section("连接状态", Fore.BLUE)
        ib_connected = status_data.get('ib_connected', False)
        market_data_connected = status_data.get('market_data_connected', False)
        
        self.print_status(f"IB TWS 连接: {'已连接' if ib_connected else '未连接'}", ib_connected)
        self.print_status(f"市场数据流: {'已连接' if market_data_connected else '未连接'}", market_data_connected)
        
        # 账户信息
        if 'account_info' in status_data:
            self.print_section("账户信息", Fore.GREEN)
            account = status_data['account_info']
            self.print_key_value("账户余额", f"${account.get('cash', 0):,.2f}")
            self.print_key_value("总资产", f"${account.get('total_value', 0):,.2f}")
            self.print_key_value("可用资金", f"${account.get('available_funds', 0):,.2f}")
        
        # 监控的股票
        if 'symbols' in status_data:
            self.print_section("监控股票", Fore.YELLOW)
            symbols = status_data['symbols']
            print(f"{Fore.WHITE}📈 共监控 {len(symbols)} 只股票:")
            for i, symbol in enumerate(symbols, 1):
                if i % 10 == 1:
                    print(f"{Fore.CYAN}   ", end="")
                print(f"{symbol:<6}", end="")
                if i % 10 == 0 or i == len(symbols):
                    print()
    
    def print_market_data_summary(self, market_data: Dict[str, Any]):
        """打印市场数据摘要"""
        self.print_section("📊 市场数据摘要", Fore.MAGENTA)
        
        if not market_data:
            self.print_warning("暂无市场数据")
            return
            
        # 活跃股票数据
        active_symbols = len(market_data)
        self.print_key_value("活跃股票数量", active_symbols)
        
        # 价格变动统计
        price_changes = []
        for symbol, data in market_data.items():
            if 'price_change_pct' in data:
                price_changes.append(data['price_change_pct'])
        
        if price_changes:
            avg_change = sum(price_changes) / len(price_changes)
            max_gain = max(price_changes)
            max_loss = min(price_changes)
            
            self.print_key_value("平均涨跌幅", f"{avg_change:.2f}%")
            self.print_key_value("最大涨幅", f"{max_gain:.2f}%", 
                               value_color=Fore.GREEN if max_gain > 0 else Fore.RED)
            self.print_key_value("最大跌幅", f"{max_loss:.2f}%", 
                               value_color=Fore.RED if max_loss < 0 else Fore.GREEN)
    
    def print_trading_signals(self, signals: List[Dict]):
        """打印交易信号"""
        self.print_section("🎯 交易信号", Fore.YELLOW)
        
        if not signals:
            self.print_info("当前无交易信号")
            return
            
        headers = ['时间', '股票', '信号', '强度', '价格', '置信度']
        formatted_signals = []
        
        for signal in signals[-10:]:  # 显示最近10个信号
            formatted_signals.append({
                '时间': signal.get('timestamp', '').strftime('%H:%M:%S') if isinstance(signal.get('timestamp'), datetime) else str(signal.get('timestamp', '')),
                '股票': signal.get('symbol', ''),
                '信号': signal.get('signal_type', ''),
                '强度': f"{signal.get('strength', 0):.2f}",
                '价格': f"${signal.get('price', 0):.2f}",
                '置信度': f"{signal.get('confidence', 0):.1%}"
            })
            
        self.print_table(formatted_signals, headers)
    
    def print_positions(self, positions: Dict[str, Any]):
        """打印持仓信息"""
        self.print_section("💼 持仓信息", Fore.GREEN)
        
        if not positions:
            self.print_info("当前无持仓")
            return
            
        headers = ['股票', '数量', '均价', '市值', '未实现盈亏', '盈亏比例']
        formatted_positions = []
        
        total_value = 0
        total_pnl = 0
        
        for symbol, pos in positions.items():
            market_value = pos.get('market_value', 0)
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            pnl_pct = (unrealized_pnl / market_value * 100) if market_value != 0 else 0
            
            total_value += market_value
            total_pnl += unrealized_pnl
            
            formatted_positions.append({
                '股票': symbol,
                '数量': str(pos.get('quantity', 0)),
                '均价': f"${pos.get('avg_price', 0):.2f}",
                '市值': f"${market_value:,.2f}",
                '未实现盈亏': f"${unrealized_pnl:,.2f}",
                '盈亏比例': f"{pnl_pct:+.2f}%"
            })
            
        self.print_table(formatted_positions, headers)
        
        # 打印总计
        print(f"\n{Fore.CYAN}📊 持仓总计:")
        self.print_key_value("总市值", f"${total_value:,.2f}")
        pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
        self.print_key_value("总盈亏", f"${total_pnl:,.2f}", value_color=pnl_color)
        
        if total_value > 0:
            total_pnl_pct = total_pnl / total_value * 100
            self.print_key_value("总盈亏比例", f"{total_pnl_pct:+.2f}%", value_color=pnl_color)
    
    def print_performance_summary(self, performance: Dict[str, Any]):
        """打印性能摘要"""
        self.print_section("📈 性能摘要", Fore.MAGENTA)
        
        # 基本指标
        self.print_key_value("总交易次数", performance.get('total_trades', 0))
        self.print_key_value("胜率", f"{performance.get('win_rate', 0):.1%}")
        
        # 盈亏指标
        total_pnl = performance.get('total_pnl', 0)
        pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
        self.print_key_value("总盈亏", f"${total_pnl:,.2f}", value_color=pnl_color)
        
        # 风险指标
        max_drawdown = performance.get('max_drawdown', 0)
        self.print_key_value("最大回撤", f"{max_drawdown:.2%}", 
                           value_color=Fore.RED if max_drawdown > 0.05 else Fore.YELLOW)
        
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        sharpe_color = Fore.GREEN if sharpe_ratio > 1 else Fore.YELLOW if sharpe_ratio > 0 else Fore.RED
        self.print_key_value("夏普比率", f"{sharpe_ratio:.2f}", value_color=sharpe_color)
    
    def print_risk_alerts(self, alerts: List[str]):
        """打印风险警报"""
        if not alerts:
            return
            
        self.print_section("⚠️ 风险警报", Fore.RED)
        for alert in alerts:
            self.print_error(alert)

# 全局格式化器实例
console_formatter = TradingConsoleFormatter()

def setup_enhanced_logging():
    """设置增强的日志格式"""
    
    class ColoredFormatter(logging.Formatter):
        """彩色日志格式化器"""
        
        COLORS = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.MAGENTA
        }
        
        def format(self, record):
            # 添加颜色
            color = self.COLORS.get(record.levelname, Fore.WHITE)
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            
            # 格式化消息
            formatted = super().format(record)
            return formatted
    
    # 设置根日志器
    root_logger = logging.getLogger()
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    # 创建文件处理器
    file_handler = logging.FileHandler('citadel_ib_integration.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # 添加处理器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)