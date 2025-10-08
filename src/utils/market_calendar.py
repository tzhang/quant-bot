"""
美股市场日历管理器
包含完整的节假日处理和时区管理功能
"""

import datetime
from datetime import date, timedelta
from typing import List, Optional, Tuple
import pytz
from dateutil.easter import easter
import logging

from .timezone_manager import TimezoneManager

logger = logging.getLogger(__name__)


class USMarketCalendar:
    """美股市场日历管理器"""
    
    def __init__(self):
        """初始化市场日历"""
        # 集成时区管理器
        self.timezone_manager = TimezoneManager()
        self.eastern_tz = self.timezone_manager.eastern_tz
        self.utc_tz = self.timezone_manager.utc_tz
        
        # 美股正常交易时间
        self.market_open_time = datetime.time(9, 30)  # 9:30 AM
        self.market_close_time = datetime.time(16, 0)  # 4:00 PM
        
        # 盘前盘后交易时间
        self.premarket_open_time = datetime.time(4, 0)   # 4:00 AM
        self.afterhours_close_time = datetime.time(20, 0)  # 8:00 PM
        
        # 提前收盘时间（节假日前一天）
        self.early_close_time = datetime.time(13, 0)  # 1:00 PM
    
    def get_us_holidays(self, year: int) -> List[date]:
        """
        获取指定年份的美股交易所节假日
        
        Args:
            year: 年份
            
        Returns:
            节假日列表
        """
        holidays = []
        
        # 新年 (New Year's Day) - 1月1日
        new_years = date(year, 1, 1)
        if new_years.weekday() == 5:  # 周六
            new_years = date(year, 1, 3)  # 周一补休
        elif new_years.weekday() == 6:  # 周日
            new_years = date(year, 1, 2)  # 周一补休
        holidays.append(new_years)
        
        # 马丁·路德·金纪念日 (Martin Luther King Jr. Day) - 1月第三个周一
        mlk_day = self._get_nth_weekday(year, 1, 0, 3)  # 第三个周一
        holidays.append(mlk_day)
        
        # 华盛顿诞辰日/总统日 (Presidents' Day) - 2月第三个周一
        presidents_day = self._get_nth_weekday(year, 2, 0, 3)  # 第三个周一
        holidays.append(presidents_day)
        
        # 耶稣受难日 (Good Friday) - 复活节前的周五
        easter_date = easter(year)
        good_friday = easter_date - timedelta(days=2)
        holidays.append(good_friday)
        
        # 阵亡将士纪念日 (Memorial Day) - 5月最后一个周一
        memorial_day = self._get_last_weekday(year, 5, 0)  # 最后一个周一
        holidays.append(memorial_day)
        
        # 六月节 (Juneteenth) - 6月19日 (2021年开始)
        if year >= 2021:
            juneteenth = date(year, 6, 19)
            if juneteenth.weekday() == 5:  # 周六
                juneteenth = date(year, 6, 18)  # 周五
            elif juneteenth.weekday() == 6:  # 周日
                juneteenth = date(year, 6, 20)  # 周一
            holidays.append(juneteenth)
        
        # 独立日 (Independence Day) - 7月4日
        independence_day = date(year, 7, 4)
        if independence_day.weekday() == 5:  # 周六
            independence_day = date(year, 7, 3)  # 周五
        elif independence_day.weekday() == 6:  # 周日
            independence_day = date(year, 7, 5)  # 周一
        holidays.append(independence_day)
        
        # 劳动节 (Labor Day) - 9月第一个周一
        labor_day = self._get_nth_weekday(year, 9, 0, 1)  # 第一个周一
        holidays.append(labor_day)
        
        # 感恩节 (Thanksgiving Day) - 11月第四个周四
        thanksgiving = self._get_nth_weekday(year, 11, 3, 4)  # 第四个周四
        holidays.append(thanksgiving)
        
        # 圣诞节 (Christmas Day) - 12月25日
        christmas = date(year, 12, 25)
        if christmas.weekday() == 5:  # 周六
            christmas = date(year, 12, 24)  # 周五
        elif christmas.weekday() == 6:  # 周日
            christmas = date(year, 12, 26)  # 周一
        holidays.append(christmas)
        
        return sorted(holidays)
    
    def get_early_close_dates(self, year: int) -> List[date]:
        """
        获取提前收盘日期（通常是节假日前一天）
        
        Args:
            year: 年份
            
        Returns:
            提前收盘日期列表
        """
        early_close_dates = []
        
        # 感恩节后的周五（黑色星期五）
        thanksgiving = self._get_nth_weekday(year, 11, 3, 4)  # 第四个周四
        black_friday = thanksgiving + timedelta(days=1)
        early_close_dates.append(black_friday)
        
        # 圣诞节前一天（如果圣诞节是周一，则前一个周五提前收盘）
        christmas = date(year, 12, 25)
        if christmas.weekday() == 0:  # 周一
            christmas_eve = date(year, 12, 24)  # 周日
            # 找到前一个周五
            days_to_friday = (christmas_eve.weekday() - 4) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            early_close_friday = christmas_eve - timedelta(days=days_to_friday)
            early_close_dates.append(early_close_friday)
        elif christmas.weekday() in [1, 2, 3, 4]:  # 周二到周五
            christmas_eve = christmas - timedelta(days=1)
            if christmas_eve.weekday() < 5:  # 工作日
                early_close_dates.append(christmas_eve)
        
        # 独立日前一天（如果独立日是周一）
        independence_day = date(year, 7, 4)
        if independence_day.weekday() == 0:  # 周一
            july_3rd = date(year, 7, 3)  # 周日
            # 找到前一个周五
            days_to_friday = (july_3rd.weekday() - 4) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            early_close_friday = july_3rd - timedelta(days=days_to_friday)
            early_close_dates.append(early_close_friday)
        elif independence_day.weekday() in [1, 2, 3, 4]:  # 周二到周五
            july_3rd = independence_day - timedelta(days=1)
            if july_3rd.weekday() < 5:  # 工作日
                early_close_dates.append(july_3rd)
        
        return sorted(early_close_dates)
    
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
    
    def _get_last_weekday(self, year: int, month: int, weekday: int) -> date:
        """
        获取指定月份的最后一个指定星期几
        
        Args:
            year: 年份
            month: 月份
            weekday: 星期几 (0=周一, 6=周日)
            
        Returns:
            日期
        """
        # 获取下个月的第一天
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        
        # 获取当前月的最后一天
        last_day = next_month - timedelta(days=1)
        
        # 找到最后一个指定星期几
        days_back = (last_day.weekday() - weekday) % 7
        target_day = last_day - timedelta(days=days_back)
        
        return target_day
    
    def is_market_holiday(self, check_date: date) -> bool:
        """
        检查指定日期是否为市场节假日
        
        Args:
            check_date: 要检查的日期
            
        Returns:
            是否为节假日
        """
        holidays = self.get_us_holidays(check_date.year)
        return check_date in holidays
    
    def is_early_close_day(self, check_date: date) -> bool:
        """
        检查指定日期是否为提前收盘日
        
        Args:
            check_date: 要检查的日期
            
        Returns:
            是否为提前收盘日
        """
        early_close_dates = self.get_early_close_dates(check_date.year)
        return check_date in early_close_dates
    
    def is_trading_day(self, check_date: date) -> bool:
        """
        检查指定日期是否为交易日
        
        Args:
            check_date: 要检查的日期
            
        Returns:
            是否为交易日
        """
        # 检查是否为周末
        if check_date.weekday() >= 5:  # 周六或周日
            return False
        
        # 检查是否为节假日
        if self.is_market_holiday(check_date):
            return False
        
        return True
    
    def get_market_hours(self, check_date: date) -> Optional[Tuple[datetime.time, datetime.time]]:
        """
        获取指定日期的市场交易时间
        
        Args:
            check_date: 要检查的日期
            
        Returns:
            (开盘时间, 收盘时间) 或 None（如果不是交易日）
        """
        if not self.is_trading_day(check_date):
            return None
        
        if self.is_early_close_day(check_date):
            return (self.market_open_time, self.early_close_time)
        else:
            return (self.market_open_time, self.market_close_time)
    
    def is_market_open_now(self, include_premarket: bool = False, 
                          include_afterhours: bool = False) -> bool:
        """
        检查当前时间市场是否开放
        
        Args:
            include_premarket: 是否包含盘前交易时间
            include_afterhours: 是否包含盘后交易时间
            
        Returns:
            市场是否开放
        """
        now_et = self.timezone_manager.get_current_eastern_time()
        today = now_et.date()
        current_time = now_et.time()
        
        # 检查是否为交易日
        if not self.is_trading_day(today):
            return False
        
        # 获取市场交易时间
        market_hours = self.get_market_hours(today)
        if not market_hours:
            return False
        
        open_time, close_time = market_hours
        
        # 扩展交易时间范围
        if include_premarket:
            open_time = self.premarket_open_time
        
        if include_afterhours:
            close_time = self.afterhours_close_time
        
        return open_time <= current_time <= close_time
    
    def get_next_trading_day(self, from_date: date) -> date:
        """
        获取指定日期之后的下一个交易日
        
        Args:
            from_date: 起始日期
            
        Returns:
            下一个交易日
        """
        next_date = from_date + timedelta(days=1)
        
        while not self.is_trading_day(next_date):
            next_date += timedelta(days=1)
            
            # 防止无限循环
            if (next_date - from_date).days > 10:
                logger.warning(f"寻找下一个交易日超过10天，从 {from_date} 开始")
                break
        
        return next_date
    
    def get_previous_trading_day(self, from_date: date) -> date:
        """
        获取指定日期之前的上一个交易日
        
        Args:
            from_date: 起始日期
            
        Returns:
            上一个交易日
        """
        prev_date = from_date - timedelta(days=1)
        
        while not self.is_trading_day(prev_date):
            prev_date -= timedelta(days=1)
            
            # 防止无限循环
            if (from_date - prev_date).days > 10:
                logger.warning(f"寻找上一个交易日超过10天，从 {from_date} 开始")
                break
        
        return prev_date
    
    def get_trading_days_in_range(self, start_date: date, end_date: date) -> List[date]:
        """
        获取指定日期范围内的所有交易日
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日列表
        """
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def convert_to_eastern_time(self, dt: datetime.datetime) -> datetime.datetime:
        """
        将时间转换为美东时间
        
        Args:
            dt: 要转换的时间（可以是任意时区）
            
        Returns:
            美东时间
        """
        return self.timezone_manager.convert_timezone(dt, to_tz='US/Eastern')
    
    def convert_to_utc(self, dt: datetime.datetime) -> datetime.datetime:
        """
        将时间转换为UTC时间
        
        Args:
            dt: 要转换的时间（可以是任意时区）
            
        Returns:
            UTC时间
        """
        return self.timezone_manager.convert_timezone(dt, to_tz='UTC')
    
    def get_market_status(self) -> dict:
        """
        获取当前市场状态信息
        
        Returns:
            市场状态字典
        """
        now_et = self.timezone_manager.get_current_eastern_time()
        today = now_et.date()
        current_time = now_et.time()
        
        # 获取时区信息
        timezone_info = self.timezone_manager.get_market_timezone_info()
        
        status = {
            'current_time_et': now_et.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'is_trading_day': self.is_trading_day(today),
            'is_holiday': self.is_market_holiday(today),
            'is_early_close': self.is_early_close_day(today),
            'market_open': False,
            'premarket_open': False,
            'afterhours_open': False,
            'next_trading_day': None,
            'market_hours': None,
            'timezone_info': timezone_info
        }
        
        if status['is_trading_day']:
            market_hours = self.get_market_hours(today)
            if market_hours:
                open_time, close_time = market_hours
                status['market_hours'] = {
                    'open': open_time.strftime('%H:%M'),
                    'close': close_time.strftime('%H:%M')
                }
                
                # 检查各个交易时段
                status['market_open'] = open_time <= current_time <= close_time
                status['premarket_open'] = (self.premarket_open_time <= current_time < open_time)
                status['afterhours_open'] = (close_time < current_time <= self.afterhours_close_time)
        
        # 获取下一个交易日
        if not status['is_trading_day'] or current_time > self.afterhours_close_time:
            status['next_trading_day'] = self.get_next_trading_day(today).strftime('%Y-%m-%d')
        
        return status


# 创建全局实例
market_calendar = USMarketCalendar()