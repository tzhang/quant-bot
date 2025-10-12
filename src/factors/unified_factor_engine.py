"""
统一因子计算引擎
整合所有因子模块，提供统一的因子计算接口
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass

# 导入所有因子模块
try:
    from .technical import TechnicalFactors
except ImportError:
    from technical import TechnicalFactors

try:
    from .fundamental_factors import FundamentalFactorCalculator
except ImportError:
    from fundamental_factors import FundamentalFactorCalculator

try:
    from .ml_factors import MLFactorCalculator
except ImportError:
    from ml_factors import MLFactorCalculator

try:
    from .advanced_factors import AdvancedFactorCalculator
except ImportError:
    from advanced_factors import AdvancedFactorCalculator

try:
    from .deep_learning_factors import DeepLearningFactorCalculator
except ImportError:
    from deep_learning_factors import DeepLearningFactorCalculator

# 新增因子计算器导入
try:
    from .sentiment_factors import SentimentFactorCalculator
except ImportError:
    from sentiment_factors import SentimentFactorCalculator

try:
    from .macro_factors import MacroFactorCalculator
except ImportError:
    from macro_factors import MacroFactorCalculator

try:
    from .alternative_factors import AlternativeFactorCalculator
except ImportError:
    from alternative_factors import AlternativeFactorCalculator

try:
    from .option_factors import OptionFactorCalculator
except ImportError:
    from option_factors import OptionFactorCalculator

try:
    from .network_factors import NetworkFactorCalculator
except ImportError:
    from network_factors import NetworkFactorCalculator

try:
    from .cross_sectional_factors import CrossSectionalFactorCalculator
except ImportError:
    from cross_sectional_factors import CrossSectionalFactorCalculator

try:
    from .market_state_factors import MarketStateFactorCalculator
except ImportError:
    from market_state_factors import MarketStateFactorCalculator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

@dataclass
class FactorPerformance:
    """因子性能统计"""
    name: str
    calculation_time: float
    data_points: int
    valid_ratio: float
    memory_usage: float

class UnifiedFactorEngine:
    """统一因子引擎 - 整合所有因子计算"""
    
    def __init__(self, enable_cache: bool = True, max_workers: int = 4, cache_results: bool = True):
        """
        初始化统一因子引擎
        
        Args:
            enable_cache: 是否启用缓存
            max_workers: 最大并行工作线程数
            cache_results: 是否缓存计算结果
        """
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.enable_cache = enable_cache
        self.max_workers = max_workers
        self.cache_results = cache_results
        self.enable_parallel = True  # 添加enable_parallel属性

        # 初始化技术因子计算器
        try:
            from .technical import TechnicalFactors
            # 启用并行计算和缓存优化
            self.technical_calculator = TechnicalFactors(enable_parallel=True, max_workers=4)
            self.logger.info("技术因子计算器初始化成功（性能优化版）")
        except Exception as e:
            self.logger.error(f"技术因子计算器初始化失败: {e}")
            self.technical_calculator = None
        self.fundamental_calculator = None
        self.ml_calculator = None
        self.advanced_calculator = None
        self.deep_learning_calculator = None
        
        # 缓存
        self.factor_cache = {}
        self.performance_stats = {}
        
        # 因子分类
        self.factor_categories = {
            'technical': [],
            'fundamental': [],
            'macro': [],
            'sentiment': [],
            'ml': [],
            'advanced': [],
            'deep_learning': []
        }
        
        self._initialize_calculators()
    
    def _initialize_calculators(self):
        """初始化所有因子计算器"""
        try:
            self.technical_calculator = TechnicalFactors()
            logger.info("技术因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"技术因子计算器初始化失败: {e}")
        
        try:
            self.fundamental_calculator = FundamentalFactorCalculator()
            logger.info("基本面因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"基本面因子计算器初始化失败: {e}")
        
        try:
            self.ml_calculator = MLFactorCalculator()
            logger.info("机器学习因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"机器学习因子计算器初始化失败: {e}")
        
        try:
            self.advanced_calculator = AdvancedFactorCalculator()
            logger.info("高级因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"高级因子计算器初始化失败: {e}")
        
        try:
            self.deep_learning_calculator = DeepLearningFactorCalculator()
            logger.info("深度学习因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"深度学习因子计算器初始化失败: {e}")
        
        # 新增因子计算器初始化
        try:
            self.sentiment_calculator = SentimentFactorCalculator()
            logger.info("情绪因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"情绪因子计算器初始化失败: {e}")
            self.sentiment_calculator = None
        
        try:
            self.macro_calculator = MacroFactorCalculator()
            logger.info("宏观经济因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"宏观经济因子计算器初始化失败: {e}")
            self.macro_calculator = None
        
        try:
            self.alternative_calculator = AlternativeFactorCalculator()
            logger.info("另类数据因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"另类数据因子计算器初始化失败: {e}")
            self.alternative_calculator = None
        
        try:
            self.option_calculator = OptionFactorCalculator()
            logger.info("期权因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"期权因子计算器初始化失败: {e}")
            self.option_calculator = None
        
        try:
            self.network_calculator = NetworkFactorCalculator()
            logger.info("网络因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"网络因子计算器初始化失败: {e}")
            self.network_calculator = None
        
        try:
            self.cross_sectional_calculator = CrossSectionalFactorCalculator()
            logger.info("截面因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"截面因子计算器初始化失败: {e}")
            self.cross_sectional_calculator = None
        
        try:
            self.market_state_calculator = MarketStateFactorCalculator()
            logger.info("市场状态因子计算器初始化成功")
        except Exception as e:
            logger.warning(f"市场状态因子计算器初始化失败: {e}")
            self.market_state_calculator = None
    
    def calculate_technical_factors(self, price_data: pd.DataFrame, 
                                  volume_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算技术因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            
        Returns:
            技术因子字典
        """
        if self.technical_calculator is None:
            logger.warning("技术因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            
            # 准备数据字典格式
            data_dict = {}
            
            # 处理价格数据
            if isinstance(price_data, pd.DataFrame):
                if 'close' in price_data.columns:
                    data_dict['close'] = price_data['close']
                elif len(price_data.columns) == 1:
                    data_dict['close'] = price_data.iloc[:, 0]
                else:
                    # 如果有多列，尝试找到价格相关的列
                    for col in ['close', 'Close', 'CLOSE', 'price', 'Price']:
                        if col in price_data.columns:
                            data_dict['close'] = price_data[col]
                            break
                    else:
                        # 如果找不到，使用第一列
                        data_dict['close'] = price_data.iloc[:, 0]
                
                # 添加其他价格数据
                for col in ['high', 'High', 'HIGH']:
                    if col in price_data.columns:
                        data_dict['high'] = price_data[col]
                        break
                
                for col in ['low', 'Low', 'LOW']:
                    if col in price_data.columns:
                        data_dict['low'] = price_data[col]
                        break
                
                for col in ['open', 'Open', 'OPEN']:
                    if col in price_data.columns:
                        data_dict['open'] = price_data[col]
                        break
            
            elif isinstance(price_data, pd.Series):
                data_dict['close'] = price_data
            
            # 处理成交量数据
            if volume_data is not None:
                if isinstance(volume_data, pd.DataFrame):
                    if 'volume' in volume_data.columns:
                        data_dict['volume'] = volume_data['volume']
                    elif len(volume_data.columns) == 1:
                        data_dict['volume'] = volume_data.iloc[:, 0]
                elif isinstance(volume_data, pd.Series):
                    data_dict['volume'] = volume_data
            
            # 计算收益率
            if 'close' in data_dict:
                data_dict['returns'] = data_dict['close'].pct_change()
            
            # 调用技术因子计算器
            factors = self.technical_calculator.calculate_all_factors(data_dict)
            
            # 记录性能
            self.performance_stats['technical'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['technical'] = list(factors.keys())
            
            logger.info(f"技术因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"技术因子计算失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {}
    
    def calculate_fundamental_factors(self, fundamental_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        计算基本面因子
        
        Args:
            fundamental_data: 基本面数据字典
            
        Returns:
            基本面因子字典
        """
        if self.fundamental_calculator is None:
            logger.warning("基本面因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.fundamental_calculator.calculate_all_factors(fundamental_data)
            
            # 记录性能
            self.performance_stats['fundamental'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['fundamental'] = list(factors.keys())
            
            logger.info(f"基本面因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"基本面因子计算失败: {e}")
            return {}
    
    def calculate_ml_factors(self, price_data: pd.DataFrame, 
                           volume_data: pd.DataFrame = None,
                           fundamental_data: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        计算机器学习因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            
        Returns:
            机器学习因子字典
        """
        if self.ml_calculator is None:
            logger.warning("机器学习因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.ml_calculator.calculate_all_factors(
                price_data, volume_data, fundamental_data
            )
            
            # 记录性能
            self.performance_stats['ml'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['ml'] = list(factors.keys())
            
            logger.info(f"机器学习因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"机器学习因子计算失败: {e}")
            return {}
    
    def calculate_advanced_factors(self, price_data: pd.DataFrame,
                                 volume_data: pd.DataFrame = None,
                                 market_data: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        计算高级因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            market_data: 市场数据
            
        Returns:
            高级因子字典
        """
        if self.advanced_calculator is None:
            logger.warning("高级因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.advanced_calculator.calculate_all_advanced_factors(
                price_data, volume_data, market_data
            )
            
            # 记录性能
            self.performance_stats['advanced'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['advanced'] = list(factors.keys())
            
            logger.info(f"高级因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"高级因子计算失败: {e}")
            return {}
    
    def calculate_deep_learning_factors(self, price_data: pd.DataFrame,
                                      volume_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算深度学习因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            
        Returns:
            深度学习因子字典
        """
        if self.deep_learning_calculator is None:
            logger.warning("深度学习因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.deep_learning_calculator.calculate_all_deep_learning_factors(
                price_data, volume_data
            )
            
            # 记录性能
            self.performance_stats['deep_learning'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['deep_learning'] = list(factors.keys())
            
            logger.info(f"深度学习因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"深度学习因子计算失败: {e}")
            return {}
    
    def calculate_sentiment_factors(self, price_data: pd.DataFrame,
                                  volume_data: pd.DataFrame = None,
                                  news_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算情绪因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            news_data: 新闻数据
            
        Returns:
            情绪因子字典
        """
        if self.sentiment_calculator is None:
            logger.warning("情绪因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.sentiment_calculator.calculate_all_sentiment_factors(
                price_data, volume_data, news_data
            )
            
            # 记录性能
            self.performance_stats['sentiment'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['sentiment'] = list(factors.keys())
            
            logger.info(f"情绪因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"情绪因子计算失败: {e}")
            return {}
    
    def calculate_macro_factors(self, price_data: pd.DataFrame,
                              macro_data: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        计算宏观经济因子
        
        Args:
            price_data: 价格数据
            macro_data: 宏观经济数据
            
        Returns:
            宏观经济因子字典
        """
        if self.macro_calculator is None:
            logger.warning("宏观经济因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.macro_calculator.calculate_all_macro_factors(
                price_data, macro_data
            )
            
            # 记录性能
            self.performance_stats['macro'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['macro'] = list(factors.keys())
            
            logger.info(f"宏观经济因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"宏观经济因子计算失败: {e}")
            return {}
    
    def calculate_alternative_factors(self, price_data: pd.DataFrame,
                                    alternative_data: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        计算另类数据因子
        
        Args:
            price_data: 价格数据
            alternative_data: 另类数据
            
        Returns:
            另类数据因子字典
        """
        if self.alternative_calculator is None:
            logger.warning("另类数据因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.alternative_calculator.calculate_all_alternative_factors(
                price_data, alternative_data
            )
            
            # 记录性能
            self.performance_stats['alternative'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['alternative'] = list(factors.keys())
            
            logger.info(f"另类数据因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"另类数据因子计算失败: {e}")
            return {}
    
    def calculate_option_factors(self, price_data: pd.DataFrame,
                               option_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算期权因子
        
        Args:
            price_data: 价格数据
            option_data: 期权数据
            
        Returns:
            期权因子字典
        """
        if self.option_calculator is None:
            logger.warning("期权因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.option_calculator.calculate_all_option_factors(
                price_data, option_data
            )
            
            # 记录性能
            self.performance_stats['option'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['option'] = list(factors.keys())
            
            logger.info(f"期权因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"期权因子计算失败: {e}")
            return {}
    
    def calculate_network_factors(self, price_data: pd.DataFrame,
                                volume_data: pd.DataFrame = None,
                                correlation_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算网络因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            correlation_data: 相关性数据
            
        Returns:
            网络因子字典
        """
        if self.network_calculator is None:
            logger.warning("网络因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.network_calculator.calculate_all_network_factors(
                price_data, volume_data, correlation_data
            )
            
            # 记录性能
            self.performance_stats['network'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['network'] = list(factors.keys())
            
            logger.info(f"网络因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"网络因子计算失败: {e}")
            return {}
    
    def calculate_cross_sectional_factors(self, price_data: pd.DataFrame,
                                        volume_data: pd.DataFrame = None,
                                        fundamental_data: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        计算截面因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            
        Returns:
            截面因子字典
        """
        if self.cross_sectional_calculator is None:
            logger.warning("截面因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.cross_sectional_calculator.calculate_all_cross_sectional_factors(
                price_data, volume_data, fundamental_data
            )
            
            # 记录性能
            self.performance_stats['cross_sectional'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['cross_sectional'] = list(factors.keys())
            
            logger.info(f"截面因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"截面因子计算失败: {e}")
            return {}
    
    def calculate_market_state_factors(self, price_data: pd.DataFrame,
                                     volume_data: pd.DataFrame = None,
                                     market_data: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        计算市场状态因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            market_data: 市场数据
            
        Returns:
            市场状态因子字典
        """
        if self.market_state_calculator is None:
            logger.warning("市场状态因子计算器未初始化")
            return {}
        
        try:
            start_time = time.time()
            factors = self.market_state_calculator.calculate_all_market_state_factors(
                price_data, volume_data, market_data
            )
            
            # 记录性能
            self.performance_stats['market_state'] = {
                'calculation_time': time.time() - start_time,
                'factor_count': len(factors),
                'timestamp': datetime.now()
            }
            
            # 更新因子分类
            self.factor_categories['market_state'] = list(factors.keys())
            
            logger.info(f"市场状态因子计算完成，共{len(factors)}个因子")
            return factors
            
        except Exception as e:
            logger.error(f"市场状态因子计算失败: {e}")
            return {}
    
    def calculate_all_factors(self, 
                            price_data: pd.DataFrame,
                            volume_data: pd.DataFrame = None,
                            fundamental_data: Dict[str, pd.DataFrame] = None,
                            market_data: Dict[str, pd.Series] = None,
                            factor_types: List[str] = None) -> Dict[str, pd.Series]:
        """
        计算所有因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            market_data: 市场数据
            factor_types: 要计算的因子类型列表
            
        Returns:
            所有因子字典
        """
        if factor_types is None:
            factor_types = ['technical', 'fundamental', 'ml', 'advanced', 'deep_learning', 
                          'sentiment', 'macro', 'alternative', 'option', 'network', 
                          'cross_sectional', 'market_state']
        
        all_factors = {}
        
        # 定义计算任务
        calculation_tasks = []
        
        if 'technical' in factor_types:
            calculation_tasks.append(('technical', self.calculate_technical_factors, 
                                   (price_data, volume_data)))
        
        if 'fundamental' in factor_types and fundamental_data:
            calculation_tasks.append(('fundamental', self.calculate_fundamental_factors, 
                                   (fundamental_data,)))
        
        if 'ml' in factor_types:
            calculation_tasks.append(('ml', self.calculate_ml_factors, 
                                   (price_data, volume_data, fundamental_data)))
        
        if 'advanced' in factor_types:
            calculation_tasks.append(('advanced', self.calculate_advanced_factors, 
                                   (price_data, volume_data, market_data)))
        
        if 'deep_learning' in factor_types:
            calculation_tasks.append(('deep_learning', self.calculate_deep_learning_factors, 
                                   (price_data, volume_data)))
        
        # 新增因子类型
        if 'sentiment' in factor_types:
            calculation_tasks.append(('sentiment', self.calculate_sentiment_factors, 
                                   (price_data, volume_data, None)))
        
        if 'macro' in factor_types:
            calculation_tasks.append(('macro', self.calculate_macro_factors, 
                                   (price_data, None)))
        
        if 'alternative' in factor_types:
            calculation_tasks.append(('alternative', self.calculate_alternative_factors, 
                                   (price_data, None)))
        
        if 'option' in factor_types:
            calculation_tasks.append(('option', self.calculate_option_factors, 
                                   (price_data, None)))
        
        if 'network' in factor_types:
            calculation_tasks.append(('network', self.calculate_network_factors, 
                                   (price_data, volume_data, None)))
        
        if 'cross_sectional' in factor_types:
            calculation_tasks.append(('cross_sectional', self.calculate_cross_sectional_factors, 
                                   (price_data, volume_data, fundamental_data)))
        
        if 'market_state' in factor_types:
            calculation_tasks.append(('market_state', self.calculate_market_state_factors, 
                                   (price_data, volume_data, market_data)))
        
        # 并行或串行计算
        if self.enable_parallel and len(calculation_tasks) > 1:
            all_factors = self._calculate_factors_parallel(calculation_tasks)
        else:
            all_factors = self._calculate_factors_sequential(calculation_tasks)
        
        # 缓存结果
        if self.cache_results:
            cache_key = f"all_factors_{datetime.now().strftime('%Y%m%d_%H')}"
            self.factor_cache[cache_key] = all_factors
        
        logger.info(f"所有因子计算完成，共{len(all_factors)}个因子")
        return all_factors
    
    def _calculate_factors_parallel(self, calculation_tasks: List[Tuple]) -> Dict[str, pd.Series]:
        """并行计算因子"""
        all_factors = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_type = {}
            for factor_type, calc_func, args in calculation_tasks:
                future = executor.submit(calc_func, *args)
                future_to_type[future] = factor_type
            
            # 收集结果
            for future in as_completed(future_to_type):
                factor_type = future_to_type[future]
                try:
                    factors = future.result()
                    all_factors.update(factors)
                    logger.info(f"{factor_type}因子计算完成")
                except Exception as e:
                    logger.error(f"{factor_type}因子计算失败: {e}")
        
        return all_factors
    
    def _calculate_factors_sequential(self, calculation_tasks: List[Tuple]) -> Dict[str, pd.Series]:
        """串行计算因子"""
        all_factors = {}
        
        for factor_type, calc_func, args in calculation_tasks:
            try:
                factors = calc_func(*args)
                all_factors.update(factors)
                logger.info(f"{factor_type}因子计算完成")
            except Exception as e:
                logger.error(f"{factor_type}因子计算失败: {e}")
        
        return all_factors
    
    def clean_factors(self, factors: Dict[str, pd.Series], 
                     remove_nan_threshold: float = 0.5,
                     winsorize_quantiles: Tuple[float, float] = (0.01, 0.99),
                     standardize: bool = True) -> Dict[str, pd.Series]:
        """
        清理因子数据
        
        Args:
            factors: 原始因子数据
            remove_nan_threshold: 移除NaN比例阈值
            winsorize_quantiles: 缩尾分位数
            standardize: 是否标准化
            
        Returns:
            清理后的因子数据
        """
        cleaned_factors = {}
        
        for factor_name, factor_values in factors.items():
            try:
                # 检查NaN比例
                nan_ratio = factor_values.isna().sum() / len(factor_values)
                if nan_ratio > remove_nan_threshold:
                    logger.warning(f"因子{factor_name}的NaN比例({nan_ratio:.2%})过高，已移除")
                    continue
                
                # 移除NaN值
                clean_values = factor_values.dropna()
                if len(clean_values) < 10:
                    logger.warning(f"因子{factor_name}有效数据不足，已移除")
                    continue
                
                # 缩尾处理
                if winsorize_quantiles:
                    lower_q, upper_q = winsorize_quantiles
                    lower_bound = clean_values.quantile(lower_q)
                    upper_bound = clean_values.quantile(upper_q)
                    clean_values = clean_values.clip(lower_bound, upper_bound)
                
                # 标准化
                if standardize:
                    mean_val = clean_values.mean()
                    std_val = clean_values.std()
                    if std_val > 0:
                        clean_values = (clean_values - mean_val) / std_val
                
                cleaned_factors[factor_name] = clean_values
                
            except Exception as e:
                logger.error(f"清理因子{factor_name}时出错: {e}")
        
        logger.info(f"因子清理完成，保留{len(cleaned_factors)}个因子")
        return cleaned_factors
    
    def get_factor_statistics(self, factors: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        获取因子统计信息
        
        Args:
            factors: 因子数据
            
        Returns:
            因子统计DataFrame
        """
        stats_data = []
        
        for factor_name, factor_values in factors.items():
            try:
                clean_values = factor_values.dropna()
                
                stats = {
                    'factor_name': factor_name,
                    'count': len(clean_values),
                    'mean': clean_values.mean(),
                    'std': clean_values.std(),
                    'min': clean_values.min(),
                    'max': clean_values.max(),
                    'skew': clean_values.skew(),
                    'kurt': clean_values.kurtosis(),
                    'nan_ratio': factor_values.isna().sum() / len(factor_values)
                }
                
                stats_data.append(stats)
                
            except Exception as e:
                logger.error(f"计算因子{factor_name}统计信息时出错: {e}")
        
        return pd.DataFrame(stats_data)
    
    def generate_factor_report(self, factors: Dict[str, pd.Series]) -> str:
        """
        生成因子报告
        
        Args:
            factors: 因子数据
            
        Returns:
            格式化的报告字符串
        """
        lines = []
        lines.append("=" * 60)
        lines.append("统一因子引擎报告")
        lines.append("=" * 60)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 总体统计
        lines.append("总体统计:")
        lines.append(f"  总因子数量: {len(factors)}")
        
        # 按类别统计
        for category, factor_list in self.factor_categories.items():
            if factor_list:
                lines.append(f"  {category}因子: {len(factor_list)}个")
        lines.append("")
        
        # 性能统计
        if self.performance_stats:
            lines.append("性能统计:")
            total_time = 0
            for category, stats in self.performance_stats.items():
                calc_time = stats['calculation_time']
                total_time += calc_time
                lines.append(f"  {category}: {calc_time:.2f}秒 ({stats['factor_count']}个因子)")
            lines.append(f"  总计算时间: {total_time:.2f}秒")
            lines.append("")
        
        # 因子统计
        factor_stats = self.get_factor_statistics(factors)
        if not factor_stats.empty:
            lines.append("因子质量统计:")
            lines.append(f"  平均覆盖率: {(1 - factor_stats['nan_ratio'].mean()):.2%}")
            lines.append(f"  平均标准差: {factor_stats['std'].mean():.4f}")
            lines.append(f"  偏度范围: [{factor_stats['skew'].min():.2f}, {factor_stats['skew'].max():.2f}]")
            lines.append(f"  峰度范围: [{factor_stats['kurt'].min():.2f}, {factor_stats['kurt'].max():.2f}]")
            lines.append("")
        
        # 因子列表
        lines.append("因子列表:")
        for category, factor_list in self.factor_categories.items():
            if factor_list:
                lines.append(f"  {category}:")
                for factor in factor_list[:10]:  # 只显示前10个
                    lines.append(f"    - {factor}")
                if len(factor_list) > 10:
                    lines.append(f"    ... 还有{len(factor_list) - 10}个因子")
                lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_factors(self, factors: Dict[str, pd.Series], filename: str = None):
        """
        保存因子数据到文件
        
        Args:
            factors: 因子数据
            filename: 文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"unified_factors_{timestamp}.csv"
        
        try:
            # 转换为DataFrame
            factor_df = pd.DataFrame(factors)
            factor_df.to_csv(filename)
            logger.info(f"因子数据已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存因子数据失败: {e}")

def main():
    """示例用法"""
    # 创建统一因子引擎
    engine = UnifiedFactorEngine(
        enable_parallel=True,
        max_workers=4,
        cache_results=True
    )
    
    # 生成模拟数据
    np.random.seed(42)
    n_stocks = 100
    n_periods = 252
    
    print("开始统一因子计算测试...")
    
    # 生成模拟价格数据
    price_data = pd.DataFrame({
        'open': np.random.randn(n_periods, n_stocks).cumsum(axis=0) + 100,
        'high': np.random.randn(n_periods, n_stocks).cumsum(axis=0) + 105,
        'low': np.random.randn(n_periods, n_stocks).cumsum(axis=0) + 95,
        'close': np.random.randn(n_periods, n_stocks).cumsum(axis=0) + 100
    })
    
    # 生成模拟成交量数据
    volume_data = pd.DataFrame(
        np.random.exponential(1000000, (n_periods, n_stocks)),
        index=price_data.index,
        columns=[f'stock_{i}' for i in range(n_stocks)]
    )
    
    # 生成模拟基本面数据
    fundamental_data = {
        'revenue': pd.DataFrame(np.random.exponential(1000000, (n_periods, n_stocks))),
        'net_income': pd.DataFrame(np.random.normal(100000, 50000, (n_periods, n_stocks))),
        'total_assets': pd.DataFrame(np.random.exponential(5000000, (n_periods, n_stocks))),
        'market_cap': pd.DataFrame(np.random.exponential(1000000000, (n_periods, n_stocks)))
    }
    
    # 计算所有因子
    start_time = time.time()
    all_factors = engine.calculate_all_factors(
        price_data=price_data,
        volume_data=volume_data,
        fundamental_data=fundamental_data,
        factor_types=['technical', 'ml']  # 只计算技术和ML因子以节省时间
    )
    
    calculation_time = time.time() - start_time
    print(f"因子计算完成，耗时: {calculation_time:.2f}秒")
    
    # 清理因子
    cleaned_factors = engine.clean_factors(all_factors)
    print(f"因子清理完成，保留{len(cleaned_factors)}个因子")
    
    # 生成报告
    report = engine.generate_factor_report(cleaned_factors)
    print("\n" + report)
    
    # 保存因子数据
    engine.save_factors(cleaned_factors, "unified_factors_test.csv")
    
    print("\n统一因子引擎测试完成！")

if __name__ == "__main__":
    main()