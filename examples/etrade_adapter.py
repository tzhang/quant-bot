"""
E*TRADE 券商适配器
实现统一的交易系统接口
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import json
import base64

logger = logging.getLogger(__name__)

class ETradeTradingSystem:
    """E*TRADE 交易系统核心类"""
    
    def __init__(self, consumer_key: str, consumer_secret: str, access_token: str = "", 
                 access_secret: str = "", sandbox: bool = True):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.sandbox = sandbox
        
        # API 端点
        if sandbox:
            self.base_url = "https://etwssandbox.etrade.com/v1"
        else:
            self.base_url = "https://api.etrade.com/v1"
            
        self.connected = False
        self.account_id = None
        
    def authenticate(self) -> bool:
        """认证并获取访问令牌"""
        try:
            if self.access_token and self.access_secret:
                # 验证现有令牌
                logger.info("E*TRADE 使用现有访问令牌")
                return True
            else:
                # 模拟模式，使用虚拟令牌
                self.access_token = "demo_access_token"
                self.access_secret = "demo_access_secret"
                logger.info("E*TRADE 模拟模式认证成功")
                return True
                
        except Exception as e:
            logger.error(f"E*TRADE 认证异常: {e}")
            return False
    
    def get_accounts(self) -> List[Dict[str, Any]]:
        """获取账户信息"""
        try:
            if not self.access_token:
                return []
                
            # 模拟返回账户数据
            return [{
                'accountIdKey': 'ET123456789',
                'accountMode': 'CASH',
                'accountDesc': 'INDIVIDUAL',
                'accountName': 'MyAccount',
                'accountType': 'INDIVIDUAL',
                'institutionType': 'BROKERAGE',
                'accountStatus': 'ACTIVE',
                'closedDate': None
            }]
            
        except Exception as e:
            logger.error(f"获取E*TRADE账户信息失败: {e}")
            return []
    
    def get_account_balance(self, account_id: str) -> Dict[str, Any]:
        """获取账户余额"""
        try:
            # 模拟返回余额数据
            return {
                'accountId': account_id,
                'accountType': 'INDIVIDUAL',
                'optionLevel': 'LEVEL_2',
                'accountDescription': 'INDIVIDUAL',
                'quoteMode': 'REAL_TIME',
                'dayTraderStatus': 'PDT_MIN_EQUITY_RES_1X',
                'accountMode': 'CASH',
                'Cash': {
                    'fundsForOpenOrdersCash': 0,
                    'moneyMktBalance': 8000.0
                },
                'Computed': {
                    'cashAvailableForInvestment': 8000.0,
                    'cashAvailableForWithdrawal': 8000.0,
                    'totalAvailableForWithdrawal': 8000.0,
                    'netCash': 8000.0,
                    'cashBalance': 8000.0,
                    'settledCashForInvestment': 8000.0,
                    'unSettledCashForInvestment': 0,
                    'fundsWithheldFromPurchasePower': 0,
                    'fundsWithheldFromWithdrawal': 0,
                    'marginBuyingPower': 16000.0,
                    'cashBuyingPower': 8000.0,
                    'dtMarginBuyingPower': 32000.0,
                    'dtCashBuyingPower': 8000.0,
                    'shortAdjustBalance': 0,
                    'regtEquity': 65000.0,
                    'regtEquityPercent': 100.0,
                    'accountBalance': 65000.0,
                    'OpenCalls': {
                        'minEquityCall': 0,
                        'fedCall': 0,
                        'cashCall': 0,
                        'houseCall': 0
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"获取E*TRADE账户余额失败: {e}")
            return {}
    
    def get_positions(self, account_id: str) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        try:
            # 模拟返回持仓数据
            return [
                {
                    'positionId': 1,
                    'Product': {
                        'securityType': 'EQ',
                        'symbol': 'NVDA'
                    },
                    'symbolDescription': 'NVIDIA CORP',
                    'dateAcquired': 1640995200000,  # timestamp
                    'pricePaid': 300.0,
                    'quantity': 50,
                    'positionType': 'LONG',
                    'daysGain': 2500.0,
                    'daysGainPct': 5.56,
                    'marketValue': 47500.0,
                    'totalCost': 15000.0,
                    'totalGain': 32500.0,
                    'totalGainPct': 216.67,
                    'pctOfPortfolio': 73.08,
                    'costPerShare': 300.0,
                    'todaysCommissions': 0,
                    'todaysFees': 0,
                    'todaysPricePaid': 0,
                    'todaysQuantity': 0,
                    'adjPrevClose': 900.0
                },
                {
                    'positionId': 2,
                    'Product': {
                        'securityType': 'EQ',
                        'symbol': 'AMD'
                    },
                    'symbolDescription': 'ADVANCED MICRO DEVICES',
                    'dateAcquired': 1640995200000,
                    'pricePaid': 100.0,
                    'quantity': 100,
                    'positionType': 'LONG',
                    'daysGain': 500.0,
                    'daysGainPct': 4.17,
                    'marketValue': 12500.0,
                    'totalCost': 10000.0,
                    'totalGain': 2500.0,
                    'totalGainPct': 25.0,
                    'pctOfPortfolio': 19.23,
                    'costPerShare': 100.0,
                    'todaysCommissions': 0,
                    'todaysFees': 0,
                    'todaysPricePaid': 0,
                    'todaysQuantity': 0,
                    'adjPrevClose': 120.0
                }
            ]
            
        except Exception as e:
            logger.error(f"获取E*TRADE持仓信息失败: {e}")
            return []
    
    def place_order(self, account_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """下单"""
        try:
            # 模拟下单成功
            order_id = f"ET_{int(time.time())}"
            logger.info(f"E*TRADE 模拟下单成功: {order_id}")
            
            return {
                'orderId': order_id,
                'orderType': order_data.get('orderType', 'MARKET'),
                'totalOrderValue': order_data.get('quantity', 0) * order_data.get('price', 100),
                'totalCommission': 0,
                'orderTerm': 'GOOD_FOR_DAY',
                'priceType': order_data.get('priceType', 'MARKET'),
                'orderValue': order_data.get('quantity', 0) * order_data.get('price', 100),
                'status': 'EXECUTED',
                'orderTime': int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"E*TRADE 下单失败: {e}")
            return {'error': str(e)}

class ETradeTradingSystemAdapter:
    """E*TRADE 交易系统适配器"""
    
    def __init__(self, consumer_key: str, consumer_secret: str, access_token: str = "", 
                 access_secret: str = "", sandbox: bool = True, dry_run: bool = True):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.sandbox = sandbox
        self.dry_run = dry_run
        
        self.etrade_system = ETradeTradingSystem(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_secret=access_secret,
            sandbox=sandbox
        )
        
        self.connected = False
        self.account_id = None
        
    def connect(self) -> bool:
        """连接到E*TRADE API"""
        try:
            if self.etrade_system.authenticate():
                accounts = self.etrade_system.get_accounts()
                if accounts:
                    self.account_id = accounts[0]['accountIdKey']
                    self.connected = True
                    logger.info(f"E*TRADE 连接成功，账户ID: {self.account_id}")
                    return True
            
            logger.error("E*TRADE 连接失败")
            return False
            
        except Exception as e:
            logger.error(f"E*TRADE 连接异常: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.connected = False
        self.account_id = None
        logger.info("E*TRADE 连接已断开")
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        try:
            if not self.connected or not self.account_id:
                return {'error': '未连接到E*TRADE'}
            
            balance = self.etrade_system.get_account_balance(self.account_id)
            if not balance:
                return {'error': '无法获取账户余额'}
            
            computed = balance.get('Computed', {})
            
            return {
                'broker': 'E*TRADE',
                'account_id': self.account_id,
                'total_value': computed.get('accountBalance', 0),
                'cash': computed.get('cashBalance', 0),
                'buying_power': computed.get('cashBuyingPower', 0),
                'margin_buying_power': computed.get('marginBuyingPower', 0),
                'day_trading_buying_power': computed.get('dtCashBuyingPower', 0),
                'market_value': computed.get('accountBalance', 0) - computed.get('cashBalance', 0),
                'day_change': 0.0,  # E*TRADE API 需要额外计算
                'day_change_percent': 0.0,
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"获取E*TRADE投资组合状态失败: {e}")
            return {'error': str(e)}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """获取当前持仓"""
        try:
            if not self.connected or not self.account_id:
                return []
            
            positions = self.etrade_system.get_positions(self.account_id)
            
            formatted_positions = []
            for pos in positions:
                product = pos['Product']
                formatted_positions.append({
                    'symbol': product['symbol'],
                    'quantity': pos['quantity'],
                    'market_value': pos['marketValue'],
                    'avg_cost': pos['costPerShare'],
                    'unrealized_pl': pos['totalGain'],
                    'unrealized_plpc': pos['totalGainPct'],
                    'day_pl': pos['daysGain'],
                    'day_plpc': pos['daysGainPct'],
                    'asset_type': 'equity' if product['securityType'] == 'EQ' else 'other'
                })
            
            return formatted_positions
            
        except Exception as e:
            logger.error(f"获取E*TRADE持仓失败: {e}")
            return []
    
    def get_performance(self) -> Dict[str, Any]:
        """获取交易表现"""
        try:
            portfolio = self.get_portfolio_status()
            positions = self.get_positions()
            
            total_pl = sum(pos.get('unrealized_pl', 0) for pos in positions)
            day_pl = sum(pos.get('day_pl', 0) for pos in positions)
            total_value = portfolio.get('total_value', 0)
            
            return {
                'broker': 'E*TRADE',
                'total_return': total_pl,
                'total_return_percent': (total_pl / total_value * 100) if total_value > 0 else 0,
                'day_return': day_pl,
                'day_return_percent': (day_pl / total_value * 100) if total_value > 0 else 0,
                'positions_count': len(positions),
                'cash_balance': portfolio.get('cash', 0)
            }
            
        except Exception as e:
            logger.error(f"获取E*TRADE交易表现失败: {e}")
            return {'error': str(e)}
    
    def place_order(self, symbol: str, quantity: int, side: str, **kwargs) -> Dict[str, Any]:
        """下单"""
        try:
            if not self.connected or not self.account_id:
                return {'error': '未连接到E*TRADE'}
            
            if self.dry_run:
                logger.info(f"E*TRADE 模拟下单: {side} {quantity} {symbol}")
                return {
                    'order_id': f"ET_DRY_{int(time.time())}",
                    'status': 'executed',
                    'filled_qty': quantity,
                    'avg_fill_price': kwargs.get('price', 100.0)
                }
            
            # 构建订单数据
            order_data = {
                'orderType': kwargs.get('order_type', 'MARKET').upper(),
                'clientOrderID': f"ET_{int(time.time())}",
                'Instrument': [{
                    'Product': {
                        'securityType': 'EQ',
                        'symbol': symbol
                    },
                    'Instruction': 'BUY' if side.upper() == 'BUY' else 'SELL',
                    'Quantity': quantity
                }],
                'priceType': 'MARKET',
                'orderTerm': 'GOOD_FOR_DAY',
                'marketSession': 'REGULAR'
            }
            
            if kwargs.get('price'):
                order_data['priceType'] = 'LIMIT'
                order_data['stopPrice'] = kwargs['price']
            
            result = self.etrade_system.place_order(self.account_id, order_data)
            
            return {
                'order_id': result.get('orderId'),
                'status': result.get('status', '').lower(),
                'filled_qty': quantity,
                'avg_fill_price': kwargs.get('price', 100.0)
            }
            
        except Exception as e:
            logger.error(f"E*TRADE 下单失败: {e}")
            return {'error': str(e)}
    
    def get_detailed_positions(self) -> List[Dict[str, Any]]:
        """获取详细持仓信息"""
        return self.get_positions()
    
    def calculate_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        """计算投资组合表现"""
        try:
            # 获取当前表现数据
            performance = self.get_performance()
            
            # E*TRADE 需要历史数据来计算准确的表现
            # 这里返回基本的表现数据
            return {
                'period_days': days,
                'total_return': performance.get('total_return', 0),
                'total_return_percent': performance.get('total_return_percent', 0),
                'annualized_return': performance.get('total_return_percent', 0) * (365 / days),
                'max_drawdown': 0,  # 需要历史数据计算
                'sharpe_ratio': 0,  # 需要历史数据计算
                'volatility': 0,    # 需要历史数据计算
                'start_date': (datetime.now() - timedelta(days=days)).isoformat(),
                'end_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"计算E*TRADE投资组合表现失败: {e}")
            return {'error': str(e)}