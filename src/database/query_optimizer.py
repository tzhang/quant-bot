"""
查询优化器

实现SQL查询性能优化、执行计划分析、索引建议等功能
"""

import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque

from sqlalchemy import text, inspect, Index, Column
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Select
from sqlalchemy.dialects import postgresql, mysql, sqlite

from .connection import get_db_session, get_engine
from .models import Base

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型枚举"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"
    ALTER = "ALTER"


@dataclass
class QueryStats:
    """查询统计信息"""
    query_hash: str
    query_type: QueryType
    execution_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_executed: Optional[datetime] = None
    error_count: int = 0
    
    def update(self, execution_time: float, success: bool = True):
        """更新统计信息"""
        self.execution_count += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.execution_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.last_executed = datetime.now()
        
        if not success:
            self.error_count += 1


@dataclass
class IndexSuggestion:
    """索引建议"""
    table_name: str
    columns: List[str]
    index_type: str = "btree"
    reason: str = ""
    estimated_benefit: float = 0.0
    cost_estimate: float = 0.0


@dataclass
class QueryOptimization:
    """查询优化建议"""
    original_query: str
    optimized_query: str
    optimization_type: str
    estimated_improvement: float
    explanation: str


class QueryOptimizer:
    """
    查询优化器
    
    实现SQL查询性能分析、优化建议、索引推荐等功能
    """
    
    def __init__(self, engine: Engine = None):
        """
        初始化查询优化器
        
        Args:
            engine: 数据库引擎
        """
        self.engine = engine or get_engine()
        self.logger = logger
        
        # 查询统计
        self._query_stats: Dict[str, QueryStats] = {}
        self._stats_lock = threading.RLock()
        
        # 慢查询记录
        self._slow_queries: deque = deque(maxlen=1000)
        self._slow_query_threshold = 1.0  # 1秒
        
        # 索引使用统计
        self._index_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # 查询模式分析
        self._query_patterns: Dict[str, int] = defaultdict(int)
        
        # 获取数据库方言
        self.dialect = self.engine.dialect.name
        
        self.logger.info(f"查询优化器已初始化，数据库方言: {self.dialect}")
    
    def _generate_query_hash(self, query: str) -> str:
        """
        生成查询哈希
        
        Args:
            query: SQL查询语句
            
        Returns:
            str: 查询哈希值
        """
        # 标准化查询（去除空白、转小写等）
        normalized_query = ' '.join(query.lower().split())
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def _detect_query_type(self, query: str) -> QueryType:
        """
        检测查询类型
        
        Args:
            query: SQL查询语句
            
        Returns:
            QueryType: 查询类型
        """
        query_upper = query.strip().upper()
        
        if query_upper.startswith('SELECT'):
            return QueryType.SELECT
        elif query_upper.startswith('INSERT'):
            return QueryType.INSERT
        elif query_upper.startswith('UPDATE'):
            return QueryType.UPDATE
        elif query_upper.startswith('DELETE'):
            return QueryType.DELETE
        elif query_upper.startswith('CREATE'):
            return QueryType.CREATE
        elif query_upper.startswith('DROP'):
            return QueryType.DROP
        elif query_upper.startswith('ALTER'):
            return QueryType.ALTER
        else:
            return QueryType.SELECT  # 默认
    
    def execute_with_profiling(self, query: str, params: Dict = None) -> Tuple[Any, Dict[str, Any]]:
        """
        执行查询并进行性能分析
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            Tuple[Any, Dict[str, Any]]: 查询结果和性能信息
        """
        query_hash = self._generate_query_hash(query)
        query_type = self._detect_query_type(query)
        
        start_time = time.time()
        success = True
        error_msg = None
        result = None
        
        try:
            with get_db_session() as session:
                # 获取执行计划（如果支持）
                execution_plan = self._get_execution_plan(session, query, params)
                
                # 执行查询
                if params:
                    result = session.execute(text(query), params).fetchall()
                else:
                    result = session.execute(text(query)).fetchall()
                
                session.commit()
                
        except Exception as e:
            success = False
            error_msg = str(e)
            self.logger.error(f"查询执行失败: {e}")
            raise
        
        finally:
            execution_time = time.time() - start_time
            
            # 更新统计信息
            with self._stats_lock:
                if query_hash not in self._query_stats:
                    self._query_stats[query_hash] = QueryStats(
                        query_hash=query_hash,
                        query_type=query_type
                    )
                
                self._query_stats[query_hash].update(execution_time, success)
                
                # 记录慢查询
                if execution_time > self._slow_query_threshold:
                    self._slow_queries.append({
                        'query': query,
                        'params': params,
                        'execution_time': execution_time,
                        'timestamp': datetime.now(),
                        'error': error_msg
                    })
        
        # 构建性能信息
        perf_info = {
            'execution_time': execution_time,
            'query_hash': query_hash,
            'query_type': query_type.value,
            'success': success,
            'error': error_msg,
            'execution_plan': execution_plan if 'execution_plan' in locals() else None
        }
        
        return result, perf_info
    
    def _get_execution_plan(self, session: Session, query: str, params: Dict = None) -> Optional[Dict]:
        """
        获取查询执行计划
        
        Args:
            session: 数据库会话
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            Optional[Dict]: 执行计划信息
        """
        try:
            if self.dialect == 'postgresql':
                explain_query = f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {query}"
            elif self.dialect == 'mysql':
                explain_query = f"EXPLAIN FORMAT=JSON {query}"
            elif self.dialect == 'sqlite':
                explain_query = f"EXPLAIN QUERY PLAN {query}"
            else:
                return None
            
            if params:
                result = session.execute(text(explain_query), params).fetchall()
            else:
                result = session.execute(text(explain_query)).fetchall()
            
            if self.dialect == 'postgresql':
                return json.loads(result[0][0])[0] if result else None
            elif self.dialect == 'mysql':
                return json.loads(result[0][0]) if result else None
            elif self.dialect == 'sqlite':
                return [dict(row) for row in result] if result else None
            
        except Exception as e:
            self.logger.warning(f"获取执行计划失败: {e}")
            return None
    
    def analyze_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        分析慢查询
        
        Args:
            limit: 返回的慢查询数量限制
            
        Returns:
            List[Dict[str, Any]]: 慢查询分析结果
        """
        # 按执行时间排序
        sorted_slow_queries = sorted(
            self._slow_queries,
            key=lambda x: x['execution_time'],
            reverse=True
        )
        
        analysis_results = []
        
        for slow_query in sorted_slow_queries[:limit]:
            query = slow_query['query']
            
            # 分析查询模式
            patterns = self._analyze_query_patterns(query)
            
            # 生成优化建议
            optimizations = self._suggest_query_optimizations(query)
            
            # 生成索引建议
            index_suggestions = self._suggest_indexes(query)
            
            analysis_results.append({
                'query': query,
                'execution_time': slow_query['execution_time'],
                'timestamp': slow_query['timestamp'],
                'patterns': patterns,
                'optimizations': optimizations,
                'index_suggestions': index_suggestions,
                'error': slow_query.get('error')
            })
        
        return analysis_results
    
    def _analyze_query_patterns(self, query: str) -> List[str]:
        """
        分析查询模式
        
        Args:
            query: SQL查询语句
            
        Returns:
            List[str]: 识别的查询模式
        """
        patterns = []
        query_upper = query.upper()
        
        # 检查常见的性能问题模式
        if 'SELECT *' in query_upper:
            patterns.append('SELECT_ALL_COLUMNS')
        
        if 'WHERE' not in query_upper and 'SELECT' in query_upper:
            patterns.append('NO_WHERE_CLAUSE')
        
        if 'ORDER BY' in query_upper and 'LIMIT' not in query_upper:
            patterns.append('ORDER_WITHOUT_LIMIT')
        
        if 'LIKE' in query_upper and query.count('%') >= 2:
            patterns.append('LEADING_WILDCARD_LIKE')
        
        if 'IN (' in query_upper and query.count(',') > 100:
            patterns.append('LARGE_IN_CLAUSE')
        
        if 'JOIN' in query_upper:
            join_count = query_upper.count('JOIN')
            if join_count > 3:
                patterns.append('MULTIPLE_JOINS')
        
        if 'SUBQUERY' in query_upper or '(' in query and 'SELECT' in query:
            patterns.append('CONTAINS_SUBQUERY')
        
        if 'DISTINCT' in query_upper:
            patterns.append('USES_DISTINCT')
        
        if 'GROUP BY' in query_upper and 'HAVING' not in query_upper:
            patterns.append('GROUP_WITHOUT_HAVING')
        
        return patterns
    
    def _suggest_query_optimizations(self, query: str) -> List[QueryOptimization]:
        """
        建议查询优化
        
        Args:
            query: SQL查询语句
            
        Returns:
            List[QueryOptimization]: 优化建议列表
        """
        optimizations = []
        query_upper = query.upper()
        
        # SELECT * 优化
        if 'SELECT *' in query_upper:
            optimized_query = query.replace('SELECT *', 'SELECT column1, column2, ...')
            optimizations.append(QueryOptimization(
                original_query=query,
                optimized_query=optimized_query,
                optimization_type="COLUMN_SELECTION",
                estimated_improvement=0.2,
                explanation="避免使用SELECT *，明确指定需要的列可以减少数据传输量"
            ))
        
        # LIMIT 优化
        if 'ORDER BY' in query_upper and 'LIMIT' not in query_upper:
            optimized_query = query + ' LIMIT 1000'
            optimizations.append(QueryOptimization(
                original_query=query,
                optimized_query=optimized_query,
                optimization_type="ADD_LIMIT",
                estimated_improvement=0.5,
                explanation="为ORDER BY查询添加LIMIT子句可以避免排序大量不需要的数据"
            ))
        
        # LIKE 优化
        if 'LIKE \'%' in query and not query.count('LIKE \'%') == query.count('%\''):
            optimizations.append(QueryOptimization(
                original_query=query,
                optimized_query=query,  # 需要具体分析
                optimization_type="LIKE_OPTIMIZATION",
                estimated_improvement=0.3,
                explanation="避免在LIKE模式开头使用通配符，考虑使用全文搜索或其他索引策略"
            ))
        
        # 子查询优化
        if '(' in query and 'SELECT' in query:
            optimizations.append(QueryOptimization(
                original_query=query,
                optimized_query=query,  # 需要具体分析
                optimization_type="SUBQUERY_TO_JOIN",
                estimated_improvement=0.4,
                explanation="考虑将子查询重写为JOIN，通常可以获得更好的性能"
            ))
        
        return optimizations
    
    def _suggest_indexes(self, query: str) -> List[IndexSuggestion]:
        """
        建议索引
        
        Args:
            query: SQL查询语句
            
        Returns:
            List[IndexSuggestion]: 索引建议列表
        """
        suggestions = []
        
        try:
            # 简单的索引建议逻辑
            # 实际实现需要更复杂的查询解析
            
            # 提取WHERE子句中的列
            where_columns = self._extract_where_columns(query)
            for table, columns in where_columns.items():
                if columns:
                    suggestions.append(IndexSuggestion(
                        table_name=table,
                        columns=list(columns),
                        index_type="btree",
                        reason="WHERE子句中频繁使用的列",
                        estimated_benefit=0.6,
                        cost_estimate=0.1
                    ))
            
            # 提取ORDER BY子句中的列
            order_columns = self._extract_order_columns(query)
            for table, columns in order_columns.items():
                if columns:
                    suggestions.append(IndexSuggestion(
                        table_name=table,
                        columns=list(columns),
                        index_type="btree",
                        reason="ORDER BY子句中使用的列",
                        estimated_benefit=0.5,
                        cost_estimate=0.1
                    ))
            
            # 提取JOIN条件中的列
            join_columns = self._extract_join_columns(query)
            for table, columns in join_columns.items():
                if columns:
                    suggestions.append(IndexSuggestion(
                        table_name=table,
                        columns=list(columns),
                        index_type="btree",
                        reason="JOIN条件中使用的列",
                        estimated_benefit=0.7,
                        cost_estimate=0.1
                    ))
        
        except Exception as e:
            self.logger.warning(f"生成索引建议失败: {e}")
        
        return suggestions
    
    def _extract_where_columns(self, query: str) -> Dict[str, set]:
        """提取WHERE子句中的列"""
        # 简化实现，实际需要更复杂的SQL解析
        columns_by_table = defaultdict(set)
        
        # 这里需要实现SQL解析逻辑
        # 暂时返回空字典
        
        return dict(columns_by_table)
    
    def _extract_order_columns(self, query: str) -> Dict[str, set]:
        """提取ORDER BY子句中的列"""
        columns_by_table = defaultdict(set)
        
        # 这里需要实现SQL解析逻辑
        # 暂时返回空字典
        
        return dict(columns_by_table)
    
    def _extract_join_columns(self, query: str) -> Dict[str, set]:
        """提取JOIN条件中的列"""
        columns_by_table = defaultdict(set)
        
        # 这里需要实现SQL解析逻辑
        # 暂时返回空字典
        
        return dict(columns_by_table)
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        获取查询统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._stats_lock:
            total_queries = sum(stat.execution_count for stat in self._query_stats.values())
            total_time = sum(stat.total_time for stat in self._query_stats.values())
            avg_time = total_time / total_queries if total_queries > 0 else 0
            
            # 最慢的查询
            slowest_queries = sorted(
                self._query_stats.values(),
                key=lambda x: x.max_time,
                reverse=True
            )[:10]
            
            # 最频繁的查询
            most_frequent_queries = sorted(
                self._query_stats.values(),
                key=lambda x: x.execution_count,
                reverse=True
            )[:10]
            
            return {
                'total_queries': total_queries,
                'total_execution_time': total_time,
                'average_execution_time': avg_time,
                'slow_query_count': len(self._slow_queries),
                'slowest_queries': [
                    {
                        'query_hash': q.query_hash,
                        'max_time': q.max_time,
                        'avg_time': q.avg_time,
                        'execution_count': q.execution_count
                    }
                    for q in slowest_queries
                ],
                'most_frequent_queries': [
                    {
                        'query_hash': q.query_hash,
                        'execution_count': q.execution_count,
                        'avg_time': q.avg_time,
                        'total_time': q.total_time
                    }
                    for q in most_frequent_queries
                ]
            }
    
    def analyze_table_usage(self) -> Dict[str, Any]:
        """
        分析表使用情况
        
        Returns:
            Dict[str, Any]: 表使用分析
        """
        try:
            with get_db_session() as session:
                inspector = inspect(self.engine)
                tables = inspector.get_table_names()
                
                table_stats = {}
                
                for table in tables:
                    # 获取表大小
                    if self.dialect == 'postgresql':
                        size_query = f"""
                        SELECT pg_total_relation_size('{table}') as total_size,
                               pg_relation_size('{table}') as table_size
                        """
                    elif self.dialect == 'mysql':
                        size_query = f"""
                        SELECT 
                            ROUND(((data_length + index_length) / 1024 / 1024), 2) as total_size,
                            ROUND((data_length / 1024 / 1024), 2) as table_size
                        FROM information_schema.TABLES 
                        WHERE table_name = '{table}'
                        """
                    else:
                        # SQLite 不支持直接获取表大小
                        size_query = None
                    
                    if size_query:
                        try:
                            result = session.execute(text(size_query)).fetchone()
                            total_size = result[0] if result else 0
                            table_size = result[1] if result else 0
                        except:
                            total_size = table_size = 0
                    else:
                        total_size = table_size = 0
                    
                    # 获取行数
                    try:
                        count_result = session.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                        row_count = count_result[0] if count_result else 0
                    except:
                        row_count = 0
                    
                    # 获取索引信息
                    indexes = inspector.get_indexes(table)
                    
                    table_stats[table] = {
                        'total_size': total_size,
                        'table_size': table_size,
                        'row_count': row_count,
                        'index_count': len(indexes),
                        'indexes': [
                            {
                                'name': idx['name'],
                                'columns': idx['column_names'],
                                'unique': idx.get('unique', False)
                            }
                            for idx in indexes
                        ]
                    }
                
                return table_stats
        
        except Exception as e:
            self.logger.error(f"分析表使用情况失败: {e}")
            return {}
    
    def suggest_maintenance_tasks(self) -> List[Dict[str, Any]]:
        """
        建议维护任务
        
        Returns:
            List[Dict[str, Any]]: 维护任务建议
        """
        suggestions = []
        
        try:
            # 分析统计信息
            stats = self.get_query_statistics()
            
            # 建议更新统计信息
            if stats['total_queries'] > 1000:
                suggestions.append({
                    'task': 'UPDATE_STATISTICS',
                    'priority': 'medium',
                    'description': '更新数据库统计信息以优化查询计划',
                    'estimated_time': '5-10分钟'
                })
            
            # 建议重建索引
            table_stats = self.analyze_table_usage()
            for table, stats in table_stats.items():
                if stats['row_count'] > 10000 and stats['index_count'] > 5:
                    suggestions.append({
                        'task': 'REBUILD_INDEXES',
                        'priority': 'low',
                        'description': f'重建表 {table} 的索引以提高性能',
                        'estimated_time': '10-30分钟',
                        'table': table
                    })
            
            # 建议清理慢查询日志
            if len(self._slow_queries) > 500:
                suggestions.append({
                    'task': 'CLEANUP_SLOW_QUERY_LOG',
                    'priority': 'low',
                    'description': '清理慢查询日志以释放内存',
                    'estimated_time': '1分钟'
                })
        
        except Exception as e:
            self.logger.error(f"生成维护建议失败: {e}")
        
        return suggestions
    
    def export_performance_report(self) -> Dict[str, Any]:
        """
        导出性能报告
        
        Returns:
            Dict[str, Any]: 性能报告
        """
        return {
            'generated_at': datetime.now().isoformat(),
            'query_statistics': self.get_query_statistics(),
            'slow_queries': self.analyze_slow_queries(),
            'table_usage': self.analyze_table_usage(),
            'maintenance_suggestions': self.suggest_maintenance_tasks(),
            'configuration': {
                'slow_query_threshold': self._slow_query_threshold,
                'database_dialect': self.dialect
            }
        }


# 全局查询优化器实例
_query_optimizer = None


def get_query_optimizer() -> QueryOptimizer:
    """获取全局查询优化器实例"""
    global _query_optimizer
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()
    return _query_optimizer


def profile_query(func):
    """查询性能分析装饰器"""
    def wrapper(*args, **kwargs):
        optimizer = get_query_optimizer()
        
        # 如果函数参数中有查询语句，进行性能分析
        if args and isinstance(args[0], str) and 'SELECT' in args[0].upper():
            query = args[0]
            params = kwargs.get('params')
            
            result, perf_info = optimizer.execute_with_profiling(query, params)
            
            # 记录性能信息
            logger.info(f"查询执行时间: {perf_info['execution_time']:.3f}s")
            
            return result
        else:
            # 正常执行函数
            return func(*args, **kwargs)
    
    return wrapper


if __name__ == "__main__":
    # 测试查询优化器
    optimizer = QueryOptimizer()
    
    # 测试查询分析
    test_query = """
    SELECT * FROM stock_data 
    WHERE symbol = 'AAPL' 
    ORDER BY date DESC
    """
    
    try:
        result, perf_info = optimizer.execute_with_profiling(test_query)
        print(f"查询结果: {len(result) if result else 0} 行")
        print(f"执行时间: {perf_info['execution_time']:.3f}s")
        
        # 分析慢查询
        slow_queries = optimizer.analyze_slow_queries()
        print(f"慢查询数量: {len(slow_queries)}")
        
        # 获取统计信息
        stats = optimizer.get_query_statistics()
        print(f"查询统计: {stats}")
        
    except Exception as e:
        print(f"测试失败: {e}")