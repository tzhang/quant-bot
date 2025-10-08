"""
时区管理器
专门处理EST/EDT自动切换和夏令时处理
"""

import datetime
from datetime import date, timedelta
from typing import Optional, Dict, Any
import pytz
import logging

logger = logging.getLogger(__name__)


class TimezoneManager:
    """时区管理器，专门处理美股交易相关的时区转换"""
    
    def __init__(self):
        """初始化时区管理器"""
        # 定义常用时区
        self.eastern_tz = pytz.timezone('US/Eastern')  # 自动处理EST/EDT
        self.utc_tz = pytz.UTC
        self.pacific_tz = pytz.timezone('US/Pacific')
        self.central_tz = pytz.timezone('US/Central')
        self.mountain_tz = pytz.timezone('US/Mountain')
        
        # 亚洲时区
        self.shanghai_tz = pytz.timezone('Asia/Shanghai')
        self.tokyo_tz = pytz.timezone('Asia/Tokyo')
        self.hong_kong_tz = pytz.timezone('Asia/Hong_Kong')
        
        # 欧洲时区
        self.london_tz = pytz.timezone('Europe/London')
        self.frankfurt_tz = pytz.timezone('Europe/Berlin')
        
        # 缓存夏令时信息
        self._dst_cache = {}
    
    def get_current_eastern_time(self) -> datetime.datetime:
        """
        获取当前美东时间（自动处理EST/EDT）
        
        Returns:
            当前美东时间
        """
        return datetime.datetime.now(self.eastern_tz)
    
    def get_current_utc_time(self) -> datetime.datetime:
        """
        获取当前UTC时间
        
        Returns:
            当前UTC时间
        """
        return datetime.datetime.now(self.utc_tz)
    
    def is_daylight_saving_time(self, dt: datetime.datetime = None) -> bool:
        """
        检查指定时间是否为夏令时
        
        Args:
            dt: 要检查的时间，默认为当前时间
            
        Returns:
            是否为夏令时
        """
        if dt is None:
            dt = self.get_current_eastern_time()
        
        # 确保时间有时区信息
        if dt.tzinfo is None:
            dt = self.eastern_tz.localize(dt)
        elif dt.tzinfo != self.eastern_tz:
            dt = dt.astimezone(self.eastern_tz)
        
        # 检查是否为夏令时
        return bool(dt.dst())
    
    def get_timezone_offset(self, dt: datetime.datetime = None) -> timedelta:
        """
        获取美东时间相对于UTC的时区偏移
        
        Args:
            dt: 要检查的时间，默认为当前时间
            
        Returns:
            时区偏移量
        """
        if dt is None:
            dt = self.get_current_eastern_time()
        
        # 确保时间有时区信息
        if dt.tzinfo is None:
            dt = self.eastern_tz.localize(dt)
        elif dt.tzinfo != self.eastern_tz:
            dt = dt.astimezone(self.eastern_tz)
        
        return dt.utcoffset()
    
    def get_dst_transition_dates(self, year: int) -> Dict[str, date]:
        """
        获取指定年份的夏令时转换日期
        
        Args:
            year: 年份
            
        Returns:
            包含夏令时开始和结束日期的字典
        """
        # 检查缓存
        cache_key = f"dst_{year}"
        if cache_key in self._dst_cache:
            return self._dst_cache[cache_key]
        
        # 夏令时开始：3月第二个周日
        dst_start = self._get_nth_weekday(year, 3, 6, 2)  # 第二个周日
        
        # 夏令时结束：11月第一个周日
        dst_end = self._get_nth_weekday(year, 11, 6, 1)  # 第一个周日
        
        result = {
            'dst_start': dst_start,
            'dst_end': dst_end
        }
        
        # 缓存结果
        self._dst_cache[cache_key] = result
        
        return result
    
    def _get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> date:
        """
        获取指定月份的第n个指定星期几
        
        Args:
            year: 年份
            month: 月份
            weekday: 星期几 (0=周一, 6=周日)
            n: 第几个
            
        Returns:
            日期
        """
        first_day = date(year, month, 1)
        first_weekday = first_day.weekday()
        
        # 计算第一个指定星期几的日期
        days_ahead = weekday - first_weekday
        if days_ahead < 0:
            days_ahead += 7
        
        first_target_day = first_day + timedelta(days=days_ahead)
        target_day = first_target_day + timedelta(weeks=n-1)
        
        return target_day
    
    def convert_timezone(self, dt: datetime.datetime, 
                        from_tz: str = None, to_tz: str = 'US/Eastern') -> datetime.datetime:
        """
        时区转换
        
        Args:
            dt: 要转换的时间
            from_tz: 源时区，None表示假设为UTC
            to_tz: 目标时区
            
        Returns:
            转换后的时间
        """
        # 获取时区对象
        target_tz = pytz.timezone(to_tz)
        
        # 处理源时区
        if dt.tzinfo is None:
            if from_tz is None:
                # 假设为UTC
                dt = self.utc_tz.localize(dt)
            else:
                source_tz = pytz.timezone(from_tz)
                dt = source_tz.localize(dt)
        
        # 转换到目标时区
        return dt.astimezone(target_tz)
    
    def localize_naive_datetime(self, dt: datetime.datetime, 
                               timezone: str = 'US/Eastern') -> datetime.datetime:
        """
        为naive datetime添加时区信息
        
        Args:
            dt: naive datetime对象
            timezone: 时区名称
            
        Returns:
            带时区信息的datetime对象
        """
        if dt.tzinfo is not None:
            logger.warning("datetime对象已有时区信息，无需本地化")
            return dt
        
        tz = pytz.timezone(timezone)
        return tz.localize(dt)
    
    def get_market_timezone_info(self) -> Dict[str, Any]:
        """
        获取市场相关的时区信息
        
        Returns:
            时区信息字典
        """
        now_et = self.get_current_eastern_time()
        now_utc = self.get_current_utc_time()
        
        dst_info = self.get_dst_transition_dates(now_et.year)
        
        return {
            'current_eastern_time': now_et.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'current_utc_time': now_utc.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'is_daylight_saving': self.is_daylight_saving_time(now_et),
            'timezone_offset': str(self.get_timezone_offset(now_et)),
            'dst_start_date': dst_info['dst_start'].strftime('%Y-%m-%d'),
            'dst_end_date': dst_info['dst_end'].strftime('%Y-%m-%d'),
            'timezone_name': now_et.tzname(),
            'utc_offset_hours': self.get_timezone_offset(now_et).total_seconds() / 3600
        }
    
    def get_trading_session_times_utc(self, trading_date: date) -> Dict[str, datetime.datetime]:
        """
        获取指定交易日的各个交易时段的UTC时间
        
        Args:
            trading_date: 交易日期
            
        Returns:
            包含各个时段UTC时间的字典
        """
        # 创建美东时间的交易时段
        premarket_open_et = self.eastern_tz.localize(
            datetime.datetime.combine(trading_date, datetime.time(4, 0))
        )
        market_open_et = self.eastern_tz.localize(
            datetime.datetime.combine(trading_date, datetime.time(9, 30))
        )
        market_close_et = self.eastern_tz.localize(
            datetime.datetime.combine(trading_date, datetime.time(16, 0))
        )
        afterhours_close_et = self.eastern_tz.localize(
            datetime.datetime.combine(trading_date, datetime.time(20, 0))
        )
        
        # 转换为UTC时间
        return {
            'premarket_open_utc': premarket_open_et.astimezone(self.utc_tz),
            'market_open_utc': market_open_et.astimezone(self.utc_tz),
            'market_close_utc': market_close_et.astimezone(self.utc_tz),
            'afterhours_close_utc': afterhours_close_et.astimezone(self.utc_tz)
        }
    
    def is_time_in_trading_hours(self, check_time: datetime.datetime,
                                include_premarket: bool = False,
                                include_afterhours: bool = False) -> bool:
        """
        检查指定时间是否在交易时间内
        
        Args:
            check_time: 要检查的时间
            include_premarket: 是否包含盘前交易
            include_afterhours: 是否包含盘后交易
            
        Returns:
            是否在交易时间内
        """
        # 转换为美东时间
        if check_time.tzinfo is None:
            check_time = self.utc_tz.localize(check_time)
        
        et_time = check_time.astimezone(self.eastern_tz)
        trading_date = et_time.date()
        
        # 获取交易时段
        session_times = self.get_trading_session_times_utc(trading_date)
        check_time_utc = check_time.astimezone(self.utc_tz)
        
        # 确定检查范围
        if include_premarket and include_afterhours:
            start_time = session_times['premarket_open_utc']
            end_time = session_times['afterhours_close_utc']
        elif include_premarket:
            start_time = session_times['premarket_open_utc']
            end_time = session_times['market_close_utc']
        elif include_afterhours:
            start_time = session_times['market_open_utc']
            end_time = session_times['afterhours_close_utc']
        else:
            start_time = session_times['market_open_utc']
            end_time = session_times['market_close_utc']
        
        return start_time <= check_time_utc <= end_time
    
    def get_next_market_open_time(self, from_time: datetime.datetime = None) -> datetime.datetime:
        """
        获取下一个市场开盘时间
        
        Args:
            from_time: 起始时间，默认为当前时间
            
        Returns:
            下一个市场开盘时间（美东时间）
        """
        if from_time is None:
            from_time = self.get_current_eastern_time()
        
        # 转换为美东时间
        if from_time.tzinfo is None:
            from_time = self.eastern_tz.localize(from_time)
        elif from_time.tzinfo != self.eastern_tz:
            from_time = from_time.astimezone(self.eastern_tz)
        
        # 从明天开始查找
        check_date = from_time.date() + timedelta(days=1)
        
        # 这里需要结合市场日历来确定下一个交易日
        # 暂时简化处理，跳过周末
        while check_date.weekday() >= 5:  # 周六或周日
            check_date += timedelta(days=1)
        
        # 创建下一个开盘时间
        next_open = self.eastern_tz.localize(
            datetime.datetime.combine(check_date, datetime.time(9, 30))
        )
        
        return next_open
    
    def format_time_with_timezone(self, dt: datetime.datetime, 
                                 format_str: str = '%Y-%m-%d %H:%M:%S %Z') -> str:
        """
        格式化带时区的时间字符串
        
        Args:
            dt: 要格式化的时间
            format_str: 格式字符串
            
        Returns:
            格式化后的时间字符串
        """
        if dt.tzinfo is None:
            dt = self.utc_tz.localize(dt)
        
        return dt.strftime(format_str)
    
    def get_global_market_times(self) -> Dict[str, str]:
        """
        获取全球主要市场的当前时间
        
        Returns:
            各市场当前时间字典
        """
        now_utc = self.get_current_utc_time()
        
        return {
            'New York': now_utc.astimezone(self.eastern_tz).strftime('%H:%M:%S'),
            'London': now_utc.astimezone(self.london_tz).strftime('%H:%M:%S'),
            'Frankfurt': now_utc.astimezone(self.frankfurt_tz).strftime('%H:%M:%S'),
            'Tokyo': now_utc.astimezone(self.tokyo_tz).strftime('%H:%M:%S'),
            'Hong Kong': now_utc.astimezone(self.hong_kong_tz).strftime('%H:%M:%S'),
            'Shanghai': now_utc.astimezone(self.shanghai_tz).strftime('%H:%M:%S'),
            'UTC': now_utc.strftime('%H:%M:%S')
        }


# 创建全局实例
timezone_manager = TimezoneManager()