#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据监控面板 - 使用 IB Gateway/TWS API
完全移除模拟数据，严格以 IB 实时/账户数据作为来源
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import logging
import socket
from typing import Dict, List

# IB 适配器（实时）与数据提供者（历史）
try:
    from examples.ib_adapter import IBAdapter, IBConfig
    IB_ADAPTER_AVAILABLE = True
except Exception as e:
    IB_ADAPTER_AVAILABLE = False
    IB_ADAPTER_IMPORT_ERROR = str(e)

try:
    from src.data.ib_data_provider import IBDataProvider
    IB_PROVIDER_AVAILABLE = True
except Exception:
    IB_PROVIDER_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="量化交易监控面板 - IB 实时数据",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"
]

class RealIBDashboard:
    """使用 IBAdapter 的真实数据监控面板"""

    def __init__(self):
        # 初始化会话状态
        if "ib_connection_status" not in st.session_state:
            st.session_state.ib_connection_status = "未连接"
        if "ib_adapter" not in st.session_state:
            st.session_state.ib_adapter = None
        if "subscribed_symbols" not in st.session_state:
            st.session_state.subscribed_symbols = []
        if "price_history" not in st.session_state:
            st.session_state.price_history = {}
        if "market_data" not in st.session_state:
            st.session_state.market_data = {}
        if "account_info" not in st.session_state:
            st.session_state.account_info = {}
        if "last_update" not in st.session_state:
            st.session_state.last_update = datetime.now()

    @staticmethod
    def test_port_connectivity(host: str, port: int) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def connect(self, host: str, port: int, client_id: int) -> bool:
        if not IB_ADAPTER_AVAILABLE:
            st.error(f"❌ 无法导入 IBAdapter: {IB_ADAPTER_IMPORT_ERROR}")
            return False

        # 端口连通性检查
        if not self.test_port_connectivity(host, port):
            st.error(f"❌ 端口不可达: {host}:{port}。请检查 IB Gateway/TWS 是否监听该端口。")
            return False

        try:
            config = IBConfig(host=host, port=port, client_id=client_id)
            ib = IBAdapter(config)
            ok = ib.connect_to_ib()
            if ok:
                st.session_state.ib_adapter = ib
                st.session_state.ib_connection_status = "已连接"
                st.success(f"✅ 已连接到 IB ({host}:{port})")
                return True
            else:
                st.session_state.ib_adapter = None
                st.session_state.ib_connection_status = "未连接"
                st.error("❌ 连接失败。请核对 IB 设置与端口。")
                return False
        except Exception as e:
            st.session_state.ib_adapter = None
            st.session_state.ib_connection_status = "未连接"
            st.error(f"❌ 连接异常: {e}")
            return False

    def disconnect(self):
        ib = st.session_state.ib_adapter
        if ib:
            try:
                ib.disconnect_from_ib()
                st.info("🔌 已断开 IB 连接")
            except Exception as e:
                st.warning(f"断开时出现异常: {e}")
        st.session_state.ib_adapter = None
        st.session_state.ib_connection_status = "未连接"

    def subscribe_symbols(self, symbols: List[str]):
        ib = st.session_state.ib_adapter
        if not ib or st.session_state.ib_connection_status != "已连接":
            st.warning("未连接到 IB，无法订阅市场数据")
            return
        # 订阅新增的符号
        prev = set(st.session_state.subscribed_symbols)
        curr = set(symbols)
        for sym in curr - prev:
            try:
                ib.subscribe_market_data(sym)
                logger.info(f"订阅 {sym}")
            except Exception as e:
                logger.error(f"订阅 {sym} 失败: {e}")
        # 更新状态（取消订阅逻辑可按需添加）
        st.session_state.subscribed_symbols = list(curr)

    def refresh_data(self):
        ib = st.session_state.ib_adapter
        if ib and st.session_state.ib_connection_status == "已连接":
            # 市场数据：从 ib.market_data 字典读取已订阅的符号
            md = {}
            for sym in st.session_state.subscribed_symbols:
                data = ib.get_market_data(sym)
                if data:
                    # 兼容不同适配器结构：last/price/bid/ask
                    price = data.get("last") or data.get("price") or data.get("close")
                    change = 0.0
                    change_pct = 0.0
                    if price is not None and data.get("close"):
                        change = float(price) - float(data.get("close"))
                        if float(data.get("close")) != 0:
                            change_pct = change / float(data.get("close")) * 100
                    md[sym] = {
                        "price": float(price) if price is not None else np.nan,
                        "change": float(change),
                        "change_pct": float(change_pct),
                        "volume": int(data.get("volume", 0)),
                        "timestamp": datetime.now().isoformat(),
                    }
                    # 价格历史
                    hist = st.session_state.price_history.setdefault(sym, [])
                    if len(hist) >= 200:
                        hist.pop(0)
                    hist.append({"timestamp": md[sym]["timestamp"], "price": md[sym]["price"]})
            st.session_state.market_data = md
            # 账户信息
            acct = ib.get_account_info()
            if acct:
                st.session_state.account_info = {
                    "net_liquidation": float(getattr(acct, "net_liquidation", 0) or acct.get("net_liquidation", 0)),
                    "total_cash": float(getattr(acct, "total_cash_value", 0) or acct.get("total_cash", 0)),
                    "buying_power": float(getattr(acct, "buying_power", 0) or acct.get("buying_power", 0)),
                }
        else:
            # 未连接时不返回任何模拟数据
            st.session_state.market_data = {}
            st.session_state.account_info = {}
        st.session_state.last_update = datetime.now()

# 图表

def market_overview_chart(market_data: Dict) -> go.Figure:
    fig = go.Figure()
    if not market_data:
        fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=360)
        return fig
    symbols = list(market_data.keys())
    prices = [d.get("price", np.nan) for d in market_data.values()]
    changes = [d.get("change_pct", 0.0) for d in market_data.values()]
    colors = ["green" if (c or 0) >= 0 else "red" for c in changes]
    fig.add_trace(go.Bar(x=symbols, y=prices, marker_color=colors,
                         text=[f"{p:.2f}<br>({c:+.2f}%)" for p, c in zip(prices, changes)], textposition="auto"))
    fig.update_layout(title="市场概览", xaxis_title="股票代码", yaxis_title="价格 ($)", height=360, showlegend=False)
    return fig

def price_trend_chart(symbol: str, history: List[Dict]) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=360)
        return fig
    ts = [item["timestamp"] for item in history]
    ps = [item["price"] for item in history]
    fig.add_trace(go.Scatter(x=ts, y=ps, mode="lines+markers", name=symbol))
    fig.update_layout(title=f"{symbol} 价格趋势", xaxis_title="时间", yaxis_title="价格 ($)", height=360)
    return fig

def account_summary_chart(account: Dict) -> go.Figure:
    fig = go.Figure()
    if not account:
        fig.add_annotation(text="暂无账户数据", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=360)
        return fig
    labels = ["净值", "现金", "购买力"]
    values = [account.get("net_liquidation", 0), account.get("total_cash", 0), account.get("buying_power", 0)]
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=["blue", "orange", "green"],
                         text=[f"${v:,.2f}" for v in values], textposition="auto"))
    fig.update_layout(title="账户摘要", yaxis_title="金额 ($)", height=360, showlegend=False)
    return fig

# 主程序

def main():
    st.title("📡 量化交易监控面板 - IB 实时数据")
    st.caption("数据来源：IB Gateway/TWS API。未连接时不显示模拟数据。")
    st.markdown("---")

    dash = RealIBDashboard()

    # 侧边栏：连接配置（默认使用你提供的端口 4001）
    st.sidebar.header("🔗 IB 连接配置")
    ib_host = st.sidebar.text_input("主机地址", value="127.0.0.1", key="ib_host")
    ib_port = st.sidebar.number_input("端口", min_value=1000, max_value=9999, value=4001, key="ib_port")
    client_id = st.sidebar.number_input("客户端ID", min_value=1, max_value=100, value=1, key="client_id")
    st.sidebar.caption("常用端口: 4001/Gateway实盘, 4002/Gateway模拟, 7497/TWS实盘, 7496/TWS模拟")

    colA, colB, colC = st.sidebar.columns([1,1,1])
    with colA:
        if st.button("连接", key="btn_connect"):
            dash.connect(ib_host, ib_port, client_id)
    with colB:
        if st.button("断开", key="btn_disconnect"):
            dash.disconnect()
    with colC:
        if st.button("测试端口", key="btn_test_port"):
            ok = dash.test_port_connectivity(ib_host, ib_port)
            if ok:
                st.sidebar.success(f"端口 {ib_port} 可达")
            else:
                st.sidebar.error(f"端口 {ib_port} 不可达")

    # 订阅管理（仅在已连接时可用）
    st.sidebar.header("📈 订阅股票")
    if st.session_state.ib_connection_status == "已连接":
        selected = st.sidebar.multiselect("选择订阅的股票", DEFAULT_SYMBOLS, default=["AAPL","MSFT","GOOGL"], key="symbol_select")
        dash.subscribe_symbols(selected)
    else:
        st.sidebar.info("未连接到 IB。连接后可订阅股票")

    # 自动刷新
    st.sidebar.header("🔄 自动刷新")
    auto_refresh = st.sidebar.checkbox("启用自动刷新", value=True, key="auto_refresh")
    refresh_interval = st.sidebar.slider("刷新间隔 (秒)", 2, 30, 5, key="refresh_interval")
    if st.sidebar.button("立即刷新", key="btn_refresh"):
        dash.refresh_data()
        st.rerun()

    if auto_refresh:
        if (datetime.now() - st.session_state.last_update).seconds >= refresh_interval:
            dash.refresh_data()
            st.rerun()

    # 首次加载
    if not st.session_state.market_data and st.session_state.ib_connection_status == "已连接":
        dash.refresh_data()

    # 顶部状态
    status = st.session_state.ib_connection_status
    if status == "已连接":
        st.success("✅ 已连接到 IB，展示实时数据")
    else:
        st.warning("⚠️ 未连接到 IB。连接后将显示实时数据")

    st.caption(f"最后更新: {st.session_state.last_update.strftime('%H:%M:%S')}")

    # 数据渲染
    md = st.session_state.market_data
    acct = st.session_state.account_info

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.subheader("市场概览")
        st.plotly_chart(market_overview_chart(md), use_container_width=True)
    with col2:
        st.subheader("账户信息")
        if acct:
            st.metric("净值", f"${acct.get('net_liquidation', 0):,.2f}")
            st.metric("现金", f"${acct.get('total_cash', 0):,.2f}")
            st.metric("购买力", f"${acct.get('buying_power', 0):,.2f}")
        else:
            st.info("暂无账户数据")
    with col3:
        st.subheader("市场统计")
        if md:
            prices = [d.get("price", np.nan) for d in md.values()]
            changes = [d.get("change_pct", 0.0) for d in md.values()]
            st.metric("平均价格", f"${np.nanmean(prices):.2f}")
            st.metric("平均涨跌", f"{np.nanmean(changes):.2f}%")
            st.metric("活跃股票", len(md))
        else:
            st.info("暂无市场数据")

    st.subheader("价格趋势")
    if st.session_state.price_history:
        symbol_for_trend = st.selectbox("选择股票", list(st.session_state.price_history.keys()), key="trend_symbol_selector")
        hist = st.session_state.price_history.get(symbol_for_trend, [])
        st.plotly_chart(price_trend_chart(symbol_for_trend, hist), use_container_width=True)
    else:
        st.info("订阅股票并有数据更新后展示趋势图")

    # 账户摘要图
    st.plotly_chart(account_summary_chart(acct), use_container_width=True)

    # 原始数据
    with st.expander("🔍 原始数据"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("市场数据 (快照)")
            st.json(md)
        with c2:
            st.subheader("账户数据")
            st.json(acct)

if __name__ == "__main__":
    main()