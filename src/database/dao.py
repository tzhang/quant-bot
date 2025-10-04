"""
数据访问对象 (DAO)

提供统一的数据库操作接口，封装CRUD操作
"""

import json
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from .models import StockData, StrategyPerformance, FactorData
from .connection import get_db_session, get_redis
from .config import get_config

logger = logging.getLogger(__name__)


class BaseDAO:
    """
    基础DAO类
    
    提供通用的数据库操作方法
    """
    
    def __init__(self, model_class):
        """
        初始化DAO
        
        Args:
            model_class: SQLAlchemy模型类
        """
        self.model_class = model_class
        self.redis_client = get_redis()
    
    def get_by_id(self, id: int) -> Optional[Any]:
        """
        根据ID获取记录
        
        Args:
            id: 记录ID
            
        Returns:
            Optional[Any]: 查询到的记录对象，如果不存在则返回None
        """
        with get_db_session() as session:
            return session.query(self.model_class).filter(self.model_class.id == id).first()
    
    def update(self, instance) -> Optional[Any]:
        """
        更新记录
        
        Args:
            instance: 要更新的模型实例
            
        Returns:
            Optional[Any]: 更新后的记录对象
        """
        with get_db_session() as session:
            # 将实例合并到当前会话
            merged_instance = session.merge(instance)
            session.flush()
            return merged_instance
    
    def delete(self, id: int) -> bool:
        """
        删除记录
        
        Args:
            id: 要删除的记录ID
            
        Returns:
            bool: 删除是否成功
        """
        with get_db_session() as session:
            instance = session.query(self.model_class).filter(self.model_class.id == id).first()
            if instance:
                session.delete(instance)
                session.flush()
                return True
            return False
    
    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            prefix: 缓存键前缀
            **kwargs: 用于生成缓存键的参数
            
        Returns:
            str: 带前缀的缓存键
        """
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, datetime):
                v = v.isoformat()
            key_parts.append(f"{k}:{v}")
        key = ":".join(key_parts)
        config = get_config()
        return config.cache.get_cache_key(key)


class StockDataDAO(BaseDAO):
    """
    股票数据访问对象
    
    提供股票数据的CRUD操作
    """
    
    def __init__(self):
        super().__init__(StockData)
    
    def create(self, symbol: str, date: datetime, open_price: float, 
               high: float, low: float, close: float, volume: int) -> StockData:
        """
        创建股票数据记录
        
        Args:
            symbol: 股票代码
            date: 交易日期
            open_price: 开盘价
            high: 最高价
            low: 最低价
            close: 收盘价
            volume: 成交量
            
        Returns:
            StockData: 创建的股票数据对象
        """
        with get_db_session() as session:
            stock_data = StockData(
                symbol=symbol,
                date=date,
                open=Decimal(str(open_price)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close)),
                volume=volume
            )
            session.add(stock_data)
            session.flush()  # 刷新到数据库但不提交
            session.refresh(stock_data)  # 刷新对象以获取数据库生成的ID
            
            # 在会话关闭前分离对象，使其可以在会话外使用
            session.expunge(stock_data)
            
            # 清除相关缓存
            self._clear_cache(symbol, date)
            
            logger.info(f"创建股票数据: {symbol} {date}")
            return stock_data
    
    def batch_create(self, data_list: List[Dict[str, Any]]) -> List[StockData]:
        """
        批量创建股票数据
        
        Args:
            data_list: 股票数据字典列表
            
        Returns:
            List[StockData]: 创建的股票数据对象列表
        """
        with get_db_session() as session:
            stock_data_objects = []
            
            for data in data_list:
                stock_data = StockData(
                    symbol=data['symbol'],
                    date=data['date'],
                    open=Decimal(str(data['open'])),
                    high=Decimal(str(data['high'])),
                    low=Decimal(str(data['low'])),
                    close=Decimal(str(data['close'])),
                    volume=data['volume']
                )
                stock_data_objects.append(stock_data)
            
            session.add_all(stock_data_objects)
            session.flush()
            
            # 清除缓存
            symbols = set(data['symbol'] for data in data_list)
            for symbol in symbols:
                self._clear_cache(symbol)
            
            logger.info(f"批量创建股票数据: {len(data_list)} 条记录")
            return stock_data_objects
    
    def get_by_symbol_and_date_range(self, symbol: str, start_date: datetime, 
                                   end_date: datetime) -> List[StockData]:
        """
        根据股票代码和日期范围获取数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[StockData]: 股票数据列表
        """
        # 尝试从缓存获取
        cache_key = self._get_cache_key(
            "stock_data", 
            symbol=symbol, 
            start=start_date, 
            end=end_date
        )
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"从缓存获取股票数据: {symbol}")
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
        
        # 从数据库查询
        with get_db_session() as session:
            query = session.query(StockData).filter(
                and_(
                    StockData.symbol == symbol,
                    StockData.date >= start_date,
                    StockData.date <= end_date
                )
            ).order_by(StockData.date)
            
            results = query.all()
            
            # 缓存结果
            try:
                cache_data = [item.to_dict() for item in results]
                config = get_config()
                self.redis_client.setex(cache_key, config.cache.ttl, json.dumps(cache_data))
            except Exception as e:
                logger.warning(f"缓存写入失败: {e}")
            
            logger.info(f"查询股票数据: {symbol}, {len(results)} 条记录")
            return results
    
    def get_latest_by_symbol(self, symbol: str) -> Optional[StockData]:
        """
        获取股票的最新数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            Optional[StockData]: 最新的股票数据
        """
        with get_db_session() as session:
            result = session.query(StockData).filter(
                StockData.symbol == symbol
            ).order_by(desc(StockData.date)).first()
            
            return result
    
    def get_symbols(self) -> List[str]:
        """
        获取所有股票代码
        
        Returns:
            List[str]: 股票代码列表
        """
        cache_key = "stock_symbols"
        
        try:
            cached_symbols = self.redis_client.get(cache_key)
            if cached_symbols:
                return json.loads(cached_symbols)
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
        
        with get_db_session() as session:
            symbols = session.query(StockData.symbol).distinct().all()
            symbol_list = [symbol[0] for symbol in symbols]
            
            # 缓存结果
            try:
                self.redis_client.setex(cache_key, 1800, json.dumps(symbol_list))
            except Exception as e:
                logger.warning(f"缓存写入失败: {e}")
            
            return symbol_list
    
    def _clear_cache(self, symbol: str, date: datetime = None):
        """
        清除相关缓存
        
        Args:
            symbol: 股票代码
            date: 日期（可选）
        """
        try:
            # 清除股票代码列表缓存
            self.redis_client.delete("stock_symbols")
            
            # 清除相关的数据缓存
            pattern = f"stock_data:symbol:{symbol}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                
        except Exception as e:
            logger.warning(f"缓存清除失败: {e}")


class StrategyPerformanceDAO(BaseDAO):
    """
    策略绩效数据访问对象
    """
    
    def __init__(self):
        super().__init__(StrategyPerformance)
    
    def create(self, strategy_name: str, date: datetime, returns: float,
               cumulative_returns: float, drawdown: float, 
               positions: Dict[str, Any]) -> StrategyPerformance:
        """
        创建策略绩效记录
        
        Args:
            strategy_name: 策略名称
            date: 日期
            returns: 当日收益率
            cumulative_returns: 累计收益率
            drawdown: 回撤
            positions: 持仓信息
            
        Returns:
            StrategyPerformance: 创建的策略绩效对象
        """
        with get_db_session() as session:
            performance = StrategyPerformance(
                strategy_name=strategy_name,
                date=date,
                returns=Decimal(str(returns)),
                cumulative_returns=Decimal(str(cumulative_returns)),
                drawdown=Decimal(str(drawdown)),
                positions=json.dumps(positions)
            )
            session.add(performance)
            session.flush()  # 刷新到数据库但不提交
            session.refresh(performance)  # 刷新对象以获取数据库生成的ID
            
            # 在会话关闭前分离对象，使其可以在会话外使用
            session.expunge(performance)
            
            # 清除相关缓存
            self._clear_cache(strategy_name)
            
            logger.info(f"创建策略绩效: {strategy_name} {date}")
            return performance
    
    def get_by_strategy_and_date_range(self, strategy_name: str, 
                                     start_date: datetime, 
                                     end_date: datetime) -> List[StrategyPerformance]:
        """
        根据策略名称和日期范围获取绩效数据
        
        Args:
            strategy_name: 策略名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[StrategyPerformance]: 策略绩效列表
        """
        with get_db_session() as session:
            query = session.query(StrategyPerformance).filter(
                and_(
                    StrategyPerformance.strategy_name == strategy_name,
                    StrategyPerformance.date >= start_date,
                    StrategyPerformance.date <= end_date
                )
            ).order_by(StrategyPerformance.date)
            
            results = query.all()
            logger.info(f"查询策略绩效: {strategy_name}, {len(results)} 条记录")
            return results
    
    def get_strategy_names(self) -> List[str]:
        """
        获取所有策略名称
        
        Returns:
            List[str]: 策略名称列表
        """
        with get_db_session() as session:
            strategies = session.query(StrategyPerformance.strategy_name).distinct().all()
            return [strategy[0] for strategy in strategies]
    
    def _clear_cache(self, strategy_name: str):
        """清除策略相关缓存"""
        try:
            pattern = f"strategy_performance:strategy:{strategy_name}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"缓存清除失败: {e}")


class FactorDataDAO(BaseDAO):
    """
    因子数据访问对象
    """
    
    def __init__(self):
        super().__init__(FactorData)
    
    def create(self, symbol: str, date: datetime, factor_name: str, 
               factor_value: float) -> FactorData:
        """
        创建因子数据记录
        
        Args:
            symbol: 股票代码
            date: 日期
            factor_name: 因子名称
            factor_value: 因子值
            
        Returns:
            FactorData: 创建的因子数据对象
        """
        with get_db_session() as session:
            factor_data = FactorData(
                symbol=symbol,
                date=date,
                factor_name=factor_name,
                factor_value=Decimal(str(factor_value))
            )
            session.add(factor_data)
            session.flush()  # 刷新到数据库但不提交
            session.refresh(factor_data)  # 刷新对象以获取数据库生成的ID
            
            # 在会话关闭前分离对象，使其可以在会话外使用
            session.expunge(factor_data)
            
            # 清除相关缓存
            self._clear_cache(symbol, factor_name)
            
            logger.info(f"创建因子数据: {symbol} {factor_name} {date}")
            return factor_data
    
    def batch_create(self, data_list: List[Dict[str, Any]]) -> List[FactorData]:
        """
        批量创建因子数据
        
        Args:
            data_list: 因子数据字典列表
            
        Returns:
            List[FactorData]: 创建的因子数据对象列表
        """
        with get_db_session() as session:
            factor_data_objects = []
            
            for data in data_list:
                factor_data = FactorData(
                    symbol=data['symbol'],
                    date=data['date'],
                    factor_name=data['factor_name'],
                    factor_value=Decimal(str(data['factor_value']))
                )
                factor_data_objects.append(factor_data)
            
            session.add_all(factor_data_objects)
            session.flush()
            
            # 清除缓存
            symbols = set(data['symbol'] for data in data_list)
            factors = set(data['factor_name'] for data in data_list)
            for symbol in symbols:
                for factor in factors:
                    self._clear_cache(symbol, factor)
            
            logger.info(f"批量创建因子数据: {len(data_list)} 条记录")
            return factor_data_objects
    
    def get_by_symbol_factor_and_date_range(self, symbol: str, factor_name: str,
                                          start_date: datetime, 
                                          end_date: datetime) -> List[FactorData]:
        """
        根据股票代码、因子名称和日期范围获取因子数据
        
        Args:
            symbol: 股票代码
            factor_name: 因子名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[FactorData]: 因子数据列表
        """
        with get_db_session() as session:
            query = session.query(FactorData).filter(
                and_(
                    FactorData.symbol == symbol,
                    FactorData.factor_name == factor_name,
                    FactorData.date >= start_date,
                    FactorData.date <= end_date
                )
            ).order_by(FactorData.date)
            
            results = query.all()
            logger.info(f"查询因子数据: {symbol} {factor_name}, {len(results)} 条记录")
            return results
    
    def get_factor_names(self) -> List[str]:
        """
        获取所有因子名称
        
        Returns:
            List[str]: 因子名称列表
        """
        with get_db_session() as session:
            factors = session.query(FactorData.factor_name).distinct().all()
            return [factor[0] for factor in factors]
    
    def _clear_cache(self, symbol: str, factor_name: str):
        """清除因子相关缓存"""
        try:
            pattern = f"factor_data:symbol:{symbol}:factor:{factor_name}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"缓存清除失败: {e}")


# 创建DAO实例
stock_data_dao = StockDataDAO()
strategy_performance_dao = StrategyPerformanceDAO()
factor_data_dao = FactorDataDAO()

__all__ = [
    'BaseDAO', 
    'StockDataDAO', 
    'StrategyPerformanceDAO', 
    'FactorDataDAO',
    'stock_data_dao',
    'strategy_performance_dao', 
    'factor_data_dao'
]