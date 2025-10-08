#!/usr/bin/env python3
"""
增强版市场日历演示程序
展示完整的节假日处理和时区管理功能
"""

import sys
import os
from datetime import date, datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.market_calendar import market_calendar
from src.utils.timezone_manager import timezone_manager


def demo_holiday_detection():
    """演示节假日检测功能"""
    print("=" * 60)
    print("🎄 美股节假日检测演示")
    print("=" * 60)
    
    current_year = datetime.now().year
    holidays = market_calendar.get_us_holidays(current_year)
    
    print(f"\n📅 {current_year}年美股节假日列表:")
    for i, holiday in enumerate(holidays, 1):
        weekday = holiday.strftime('%A')
        print(f"{i:2d}. {holiday.strftime('%Y-%m-%d')} ({weekday}) - {get_holiday_name(holiday)}")
    
    # 检查提前收盘日
    early_close_dates = market_calendar.get_early_close_dates(current_year)
    print(f"\n⏰ {current_year}年提前收盘日:")
    for i, early_date in enumerate(early_close_dates, 1):
        weekday = early_date.strftime('%A')
        print(f"{i}. {early_date.strftime('%Y-%m-%d')} ({weekday}) - 提前至13:00收盘")


def get_holiday_name(holiday_date: date) -> str:
    """获取节假日名称"""
    month = holiday_date.month
    day = holiday_date.day
    
    holiday_names = {
        (1, 1): "新年",
        (1, 2): "新年(补休)",
        (1, 3): "新年(补休)",
        (7, 3): "独立日(提前)",
        (7, 4): "独立日",
        (7, 5): "独立日(补休)",
        (12, 24): "圣诞节(提前)",
        (12, 25): "圣诞节",
        (12, 26): "圣诞节(补休)"
    }
    
    # 检查固定日期节假日
    if (month, day) in holiday_names:
        return holiday_names[(month, day)]
    
    # 检查浮动节假日
    if month == 1 and 15 <= day <= 21:
        return "马丁·路德·金纪念日"
    elif month == 2 and 15 <= day <= 21:
        return "总统日"
    elif month == 3 or month == 4:
        return "耶稣受难日"
    elif month == 5 and day >= 25:
        return "阵亡将士纪念日"
    elif month == 6 and day == 19:
        return "六月节"
    elif month == 9 and day <= 7:
        return "劳动节"
    elif month == 11 and 22 <= day <= 28:
        return "感恩节"
    
    return "节假日"


def demo_timezone_management():
    """演示时区管理功能"""
    print("\n" + "=" * 60)
    print("🌍 时区管理演示")
    print("=" * 60)
    
    # 获取时区信息
    tz_info = timezone_manager.get_market_timezone_info()
    
    print("\n📍 当前时区信息:")
    print(f"美东时间: {tz_info['current_eastern_time']}")
    print(f"UTC时间: {tz_info['current_utc_time']}")
    print(f"是否夏令时: {'是' if tz_info['is_daylight_saving'] else '否'}")
    print(f"时区偏移: {tz_info['timezone_offset']}")
    print(f"时区名称: {tz_info['timezone_name']}")
    print(f"UTC偏移小时: {tz_info['utc_offset_hours']}")
    
    # 夏令时转换日期
    print(f"\n🕐 夏令时信息:")
    print(f"夏令时开始: {tz_info['dst_start_date']}")
    print(f"夏令时结束: {tz_info['dst_end_date']}")
    
    # 全球市场时间
    print(f"\n🌐 全球主要市场当前时间:")
    global_times = timezone_manager.get_global_market_times()
    for market, time_str in global_times.items():
        print(f"{market:12s}: {time_str}")


def demo_trading_day_detection():
    """演示交易日检测功能"""
    print("\n" + "=" * 60)
    print("📈 交易日检测演示")
    print("=" * 60)
    
    today = date.today()
    
    print(f"\n📅 今日 ({today.strftime('%Y-%m-%d %A')}) 交易状态:")
    print(f"是否交易日: {'是' if market_calendar.is_trading_day(today) else '否'}")
    print(f"是否节假日: {'是' if market_calendar.is_market_holiday(today) else '否'}")
    print(f"是否提前收盘: {'是' if market_calendar.is_early_close_day(today) else '否'}")
    
    # 获取市场交易时间
    market_hours = market_calendar.get_market_hours(today)
    if market_hours:
        open_time, close_time = market_hours
        print(f"交易时间: {open_time.strftime('%H:%M')} - {close_time.strftime('%H:%M')}")
    else:
        print("交易时间: 非交易日")
    
    # 下一个和上一个交易日
    next_trading_day = market_calendar.get_next_trading_day(today)
    prev_trading_day = market_calendar.get_previous_trading_day(today)
    
    print(f"\n📊 相邻交易日:")
    print(f"上一个交易日: {prev_trading_day.strftime('%Y-%m-%d %A')}")
    print(f"下一个交易日: {next_trading_day.strftime('%Y-%m-%d %A')}")
    
    # 本周交易日
    monday = today - timedelta(days=today.weekday())
    friday = monday + timedelta(days=4)
    
    week_trading_days = market_calendar.get_trading_days_in_range(monday, friday)
    print(f"\n📅 本周交易日 ({monday.strftime('%m/%d')} - {friday.strftime('%m/%d')}):")
    for trading_day in week_trading_days:
        print(f"  {trading_day.strftime('%Y-%m-%d %A')}")


def demo_market_status():
    """演示市场状态功能"""
    print("\n" + "=" * 60)
    print("🔔 实时市场状态")
    print("=" * 60)
    
    status = market_calendar.get_market_status()
    
    print(f"\n⏰ 当前时间: {status['current_time_et']}")
    print(f"📊 交易状态:")
    print(f"  是否交易日: {'是' if status['is_trading_day'] else '否'}")
    print(f"  是否节假日: {'是' if status['is_holiday'] else '否'}")
    print(f"  是否提前收盘: {'是' if status['is_early_close'] else '否'}")
    
    if status['market_hours']:
        print(f"  交易时间: {status['market_hours']['open']} - {status['market_hours']['close']}")
    
    print(f"\n🚦 市场开放状态:")
    print(f"  正常交易: {'开放' if status['market_open'] else '关闭'}")
    print(f"  盘前交易: {'开放' if status['premarket_open'] else '关闭'}")
    print(f"  盘后交易: {'开放' if status['afterhours_open'] else '关闭'}")
    
    if status['next_trading_day']:
        print(f"\n📅 下一个交易日: {status['next_trading_day']}")
    
    # 显示时区详细信息
    tz_info = status['timezone_info']
    print(f"\n🌍 时区详情:")
    print(f"  夏令时状态: {'是' if tz_info['is_daylight_saving'] else '否'}")
    print(f"  UTC偏移: {tz_info['utc_offset_hours']} 小时")


def demo_time_conversion():
    """演示时间转换功能"""
    print("\n" + "=" * 60)
    print("🔄 时间转换演示")
    print("=" * 60)
    
    # 创建一个测试时间
    test_time_utc = datetime.now(timezone_manager.utc_tz)
    
    print(f"\n🕐 时间转换示例:")
    print(f"UTC时间: {test_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # 转换到美东时间
    et_time = market_calendar.convert_to_eastern_time(test_time_utc)
    print(f"美东时间: {et_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # 转换回UTC
    utc_time = market_calendar.convert_to_utc(et_time)
    print(f"转换回UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # 检查是否在交易时间内
    is_trading_hours = timezone_manager.is_time_in_trading_hours(test_time_utc)
    is_trading_hours_pre = timezone_manager.is_time_in_trading_hours(
        test_time_utc, include_premarket=True
    )
    is_trading_hours_after = timezone_manager.is_time_in_trading_hours(
        test_time_utc, include_premarket=True, include_afterhours=True
    )
    
    print(f"\n📊 交易时间检查:")
    print(f"正常交易时间: {'是' if is_trading_hours else '否'}")
    print(f"包含盘前: {'是' if is_trading_hours_pre else '否'}")
    print(f"包含盘前盘后: {'是' if is_trading_hours_after else '否'}")


def main():
    """主函数"""
    print("🚀 增强版美股市场日历系统演示")
    print("包含完整的节假日处理和时区管理功能")
    
    try:
        # 演示各个功能模块
        demo_holiday_detection()
        demo_timezone_management()
        demo_trading_day_detection()
        demo_market_status()
        demo_time_conversion()
        
        print("\n" + "=" * 60)
        print("✅ 演示完成！市场日历系统功能正常")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()