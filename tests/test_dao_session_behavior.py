from datetime import datetime
from decimal import Decimal

import pytest
import pandas as pd

from src.database.dao import StrategyPerformanceDAO, StockDataDAO
from src.database.connection import db_manager
from src.database.models import StrategyPerformance, StockData


def test_create_returns_detached_instance_and_attributes_accessible():
    sp_dao = StrategyPerformanceDAO()
    perf = sp_dao.create(
        strategy_name="test_strategy",
        date=datetime(2023, 1, 2),
        returns=0.01,
        cumulative_returns=0.02,
        drawdown=0.0,
        positions={"AAPL": 10},
    )
    # 会话关闭后对象应已分离（expunge），但属性可读
    from sqlalchemy.orm.session import object_session
    assert object_session(perf) is None
    assert perf.strategy_name == "test_strategy"
    assert float(perf.returns) == pytest.approx(0.01)


def test_update_merges_detached_instance_without_active_session():
    sp_dao = StrategyPerformanceDAO()
    perf = sp_dao.create(
        strategy_name="s1",
        date=datetime(2023, 2, 1),
        returns=0.005,
        cumulative_returns=0.010,
        drawdown=0.001,
        positions={"MSFT": 5},
    )
    # 修改分离对象的属性并进行更新（merge）
    perf.returns = Decimal("0.007")
    merged = sp_dao.update(perf)

    # 会话提交后关闭，返回对象为分离状态，但变更已写入数据库
    from sqlalchemy.orm.session import object_session
    assert object_session(merged) is None

    # 重新查询确认更新生效
    with db_manager.get_session() as session:
        q = session.query(StrategyPerformance).filter(
            StrategyPerformance.strategy_name == "s1",
            StrategyPerformance.date == datetime(2023, 2, 1),
        )
        db_obj = q.first()
        assert db_obj is not None
        assert float(db_obj.returns) == pytest.approx(0.007)


def test_stockdata_create_and_cache_calls_do_not_fail_with_dummy_redis():
    sd_dao = StockDataDAO()
    rec = sd_dao.create(
        symbol="TEST",
        date=datetime(2023, 3, 1),
        open_price=100.0,
        high=101.0,
        low=99.5,
        close=100.5,
        volume=1000,
    )
    # 分离对象
    from sqlalchemy.orm.session import object_session
    assert object_session(rec) is None
    # 查询返回列表或对象不报错（DummyRedis生效）
    res = sd_dao.get_by_symbol_and_date_range(
        symbol="TEST", start_date=datetime(2023, 3, 1), end_date=datetime(2023, 3, 2)
    )
    assert isinstance(res, list)