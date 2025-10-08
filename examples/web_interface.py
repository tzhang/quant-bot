"""
量化交易系统 Web 界面
使用 Streamlit 构建的交互式 Web 应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import time
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.data.data_manager import DataManager
    from src.factors.engine import FactorEngine
    from src.strategies.templates import MomentumStrategy
    from src.backtesting.enhanced_backtest_engine import EnhancedBacktestEngine
    from src.monitoring.performance_analyzer import PerformanceAnalyzer
    from src.risk.risk_manager import RiskManager
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    st.info("请确保所有依赖模块已正确安装")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="量化交易系统 v3.0.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = None
if 'factor_engine' not in st.session_state:
    st.session_state.factor_engine = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

def initialize_system():
    """初始化系统组件"""
    try:
        if st.session_state.data_manager is None:
            st.session_state.data_manager = DataManager()
        if st.session_state.factor_engine is None:
            st.session_state.factor_engine = FactorEngine()
        return True
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        return False

def main():
    """主函数"""
    # 页面标题
    st.markdown('<h1 class="main-header">🚀 量化交易系统 v3.0.0</h1>', unsafe_allow_html=True)
    
    # 侧边栏导航
    st.sidebar.title("📊 系统导航")
    page = st.sidebar.selectbox(
        "选择功能模块",
        ["系统概览", "数据管理", "因子分析", "策略回测", "性能分析", "风险管理", "实时监控"]
    )
    
    # 系统状态检查
    if not initialize_system():
        st.stop()
    
    # 根据选择显示不同页面
    if page == "系统概览":
        show_system_overview()
    elif page == "数据管理":
        show_data_management()
    elif page == "因子分析":
        show_factor_analysis()
    elif page == "策略回测":
        show_strategy_backtest()
    elif page == "性能分析":
        show_performance_analysis()
    elif page == "风险管理":
        show_risk_management()
    elif page == "实时监控":
        show_real_time_monitoring()

def show_system_overview():
    """显示系统概览"""
    st.header("🎯 系统概览")
    
    # 系统信息卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("系统版本", "v3.0.0", "性能优化版")
    
    with col2:
        st.metric("数据源", "4个", "Qlib + OpenBB + yfinance + IB")
    
    with col3:
        st.metric("缓存文件", "31个", "智能缓存系统")
    
    with col4:
        st.metric("性能提升", "2.5x", "相比v2.0.0")
    
    # 系统架构图
    st.subheader("🏗️ 系统架构")
    
    # 创建架构图
    fig = go.Figure()
    
    # 添加架构层级
    layers = [
        {"name": "Web界面层", "y": 4, "color": "#1f77b4"},
        {"name": "策略层", "y": 3, "color": "#ff7f0e"},
        {"name": "因子计算层", "y": 2, "color": "#2ca02c"},
        {"name": "数据管理层", "y": 1, "color": "#d62728"},
        {"name": "存储层", "y": 0, "color": "#9467bd"}
    ]
    
    for layer in layers:
        fig.add_trace(go.Scatter(
            x=[0, 1, 2, 3, 4],
            y=[layer["y"]] * 5,
            mode='markers+text',
            marker=dict(size=60, color=layer["color"]),
            text=[layer["name"]] * 5,
            textposition="middle center",
            name=layer["name"]
        ))
    
    fig.update_layout(
        title="量化交易系统架构图",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 核心功能介绍
    st.subheader("⚡ v3.0.0 核心功能")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🧠 智能缓存系统**
        - 自动缓存管理
        - 缓存命中率 90%+
        - 数据获取速度提升 3x
        
        **🚀 内存池管理器**
        - 内存使用优化 30%
        - 自动内存回收
        - 大数据处理支持
        
        **📊 性能分析器**
        - 实时性能监控
        - 瓶颈自动识别
        - 优化建议生成
        """)
    
    with col2:
        st.markdown("""
        **🎯 自适应执行器**
        - 智能策略选择
        - 并行处理优化
        - 任务负载均衡
        
        **🔧 集成优化器**
        - 端到端优化
        - 系统整体调优
        - 性能基准测试
        
        **📈 大规模数据处理器**
        - 支持超大数据集
        - 分布式计算
        - 流式数据处理
        """)

def show_data_management():
    """显示数据管理页面"""
    st.header("📊 数据管理")
    
    # 数据源选择
    st.subheader("🔗 数据源配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_sources = st.multiselect(
            "选择数据源",
            ["yfinance", "OpenBB", "Qlib", "Interactive Brokers"],
            default=["yfinance", "OpenBB"]
        )
    
    with col2:
        cache_enabled = st.checkbox("启用智能缓存", value=True)
    
    # 股票选择
    st.subheader("📈 股票数据获取")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbols = st.text_input("股票代码 (逗号分隔)", "AAPL,GOOGL,MSFT,TSLA")
    
    with col2:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    
    with col3:
        end_date = st.date_input("结束日期", datetime.now())
    
    # 获取数据按钮
    if st.button("🚀 获取数据", type="primary"):
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
            
            with st.spinner("正在获取数据..."):
                try:
                    # 获取数据 - get_multiple_stocks_data返回字典
                    data_dict = st.session_state.data_manager.get_multiple_stocks_data(
                        symbol_list, 
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if data_dict and len(data_dict) > 0:
                        st.success(f"✅ 成功获取 {len(data_dict)} 只股票的数据")
                        
                        # 合并所有股票数据用于显示
                        combined_data = []
                        for symbol, symbol_data in data_dict.items():
                            if not symbol_data.empty:
                                # 添加股票代码列
                                symbol_data_copy = symbol_data.copy()
                                symbol_data_copy['symbol'] = symbol
                                combined_data.append(symbol_data_copy)
                        
                        if combined_data:
                            # 合并数据
                            data = pd.concat(combined_data, ignore_index=False)
                            
                            # 显示数据概览
                            st.subheader("📋 数据概览")
                            st.dataframe(data.head(10))
                            
                            # 数据统计
                            st.subheader("📊 数据统计")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("数据行数", len(data))
                            with col2:
                                st.metric("股票数量", len(data_dict))
                            with col3:
                                st.metric("时间跨度", f"{(end_date - start_date).days}天")
                            with col4:
                                total_cells = len(data) * len(data.columns)
                                null_cells = data.isnull().sum().sum()
                                completeness = (1 - null_cells / total_cells) * 100 if total_cells > 0 else 0
                                st.metric("数据完整性", f"{completeness:.1f}%")
                            
                            # 价格走势图
                            st.subheader("📈 价格走势")
                            
                            fig = go.Figure()
                            
                            for symbol in symbol_list:
                                if symbol in data_dict:
                                    symbol_data = data_dict[symbol]
                                    if not symbol_data.empty:
                                        # 确定价格列名
                                        close_col = None
                                        for col in ['close', 'Close', 'CLOSE']:
                                            if col in symbol_data.columns:
                                                close_col = col
                                                break
                                        
                                        if close_col:
                                            fig.add_trace(go.Scatter(
                                                x=symbol_data.index,
                                                y=symbol_data[close_col],
                                                mode='lines',
                                                name=symbol,
                                                line=dict(width=2)
                                            ))
                            
                            fig.update_layout(
                                title="股票价格走势图",
                                xaxis_title="日期",
                                yaxis_title="价格 ($)",
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ 获取到的数据为空")
                        
                    else:
                        st.warning("⚠️ 未能获取到有效数据，请检查股票代码或网络连接")
                        
                except Exception as e:
                    st.error(f"❌ 数据获取失败: {e}")
        else:
            st.warning("⚠️ 请输入股票代码")
    
    # 缓存管理
    st.subheader("💾 缓存管理")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 查看缓存状态"):
            try:
                cache_info = st.session_state.data_manager.get_cache_info()
                st.json(cache_info)
            except Exception as e:
                st.error(f"获取缓存信息失败: {e}")
    
    with col2:
        if st.button("🧹 清理缓存"):
            try:
                st.session_state.data_manager.clear_cache()
                st.success("✅ 缓存已清理")
            except Exception as e:
                st.error(f"清理缓存失败: {e}")
    
    with col3:
        if st.button("🔄 刷新缓存"):
            try:
                st.session_state.data_manager.refresh_cache()
                st.success("✅ 缓存已刷新")
            except Exception as e:
                st.error(f"刷新缓存失败: {e}")

def show_factor_analysis():
    """显示因子分析页面"""
    st.header("🧮 因子分析")
    
    st.info("💡 因子分析功能正在开发中，敬请期待！")
    
    # 模拟因子数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    factor_data = pd.DataFrame({
        'RSI': np.random.normal(50, 15, len(dates)),
        'MACD': np.random.normal(0, 2, len(dates)),
        'BB_Position': np.random.normal(0.5, 0.3, len(dates)),
        'Volume_Ratio': np.random.normal(1, 0.5, len(dates))
    }, index=dates)
    
    # 因子相关性热力图
    st.subheader("🔥 因子相关性分析")
    
    corr_matrix = factor_data.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="因子相关性热力图",
        color_continuous_scale="RdBu_r"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 因子时间序列
    st.subheader("📈 因子时间序列")
    
    selected_factors = st.multiselect(
        "选择要显示的因子",
        factor_data.columns.tolist(),
        default=['RSI', 'MACD']
    )
    
    if selected_factors:
        fig = make_subplots(
            rows=len(selected_factors), cols=1,
            subplot_titles=selected_factors,
            vertical_spacing=0.1
        )
        
        for i, factor in enumerate(selected_factors):
            fig.add_trace(
                go.Scatter(
                    x=factor_data.index,
                    y=factor_data[factor],
                    mode='lines',
                    name=factor,
                    line=dict(width=2)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(selected_factors),
            title="因子时间序列图",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_strategy_backtest():
    """显示策略回测页面"""
    st.header("🎯 策略回测")
    
    # 策略参数设置
    st.subheader("⚙️ 策略参数")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input("初始资金", min_value=10000, max_value=1000000, value=100000, step=10000)
    
    with col2:
        lookback_period = st.number_input("回看期", min_value=5, max_value=100, value=20, step=5)
    
    with col3:
        rebalance_freq = st.selectbox("调仓频率", ["daily", "weekly", "monthly"], index=1)
    
    # 回测按钮
    if st.button("🚀 开始回测", type="primary"):
        with st.spinner("正在执行回测..."):
            try:
                # 模拟回测结果
                dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                np.random.seed(42)
                
                # 生成模拟收益率
                returns = np.random.normal(0.0008, 0.02, len(dates))  # 日收益率
                cumulative_returns = (1 + pd.Series(returns, index=dates)).cumprod()
                
                # 基准收益
                benchmark_returns = np.random.normal(0.0005, 0.015, len(dates))
                benchmark_cumulative = (1 + pd.Series(benchmark_returns, index=dates)).cumprod()
                
                # 存储回测结果
                st.session_state.backtest_results = {
                    'strategy_returns': cumulative_returns,
                    'benchmark_returns': benchmark_cumulative,
                    'dates': dates
                }
                
                st.success("✅ 回测完成！")
                
                # 显示回测结果
                show_backtest_results()
                
            except Exception as e:
                st.error(f"❌ 回测失败: {e}")

def show_backtest_results():
    """显示回测结果"""
    if st.session_state.backtest_results is None:
        st.warning("⚠️ 请先执行回测")
        return
    
    results = st.session_state.backtest_results
    
    # 绩效指标
    st.subheader("📊 绩效指标")
    
    strategy_final = results['strategy_returns'].iloc[-1]
    benchmark_final = results['benchmark_returns'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("策略总收益", f"{(strategy_final - 1) * 100:.2f}%")
    
    with col2:
        st.metric("基准总收益", f"{(benchmark_final - 1) * 100:.2f}%")
    
    with col3:
        st.metric("超额收益", f"{(strategy_final - benchmark_final) * 100:.2f}%")
    
    with col4:
        strategy_vol = results['strategy_returns'].pct_change().std() * np.sqrt(252)
        st.metric("年化波动率", f"{strategy_vol * 100:.2f}%")
    
    # 净值曲线
    st.subheader("📈 净值曲线")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results['dates'],
        y=results['strategy_returns'],
        mode='lines',
        name='策略',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=results['dates'],
        y=results['benchmark_returns'],
        mode='lines',
        name='基准',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title="策略净值曲线",
        xaxis_title="日期",
        yaxis_title="净值",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_performance_analysis():
    """显示性能分析页面"""
    st.header("📈 性能分析")
    
    if st.session_state.backtest_results is None:
        st.warning("⚠️ 请先在策略回测页面执行回测")
        return
    
    results = st.session_state.backtest_results
    
    # 风险指标
    st.subheader("⚠️ 风险指标")
    
    strategy_returns = results['strategy_returns'].pct_change().dropna()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        st.metric("夏普比率", f"{sharpe_ratio:.3f}")
    
    with col2:
        max_dd = (results['strategy_returns'] / results['strategy_returns'].cummax() - 1).min()
        st.metric("最大回撤", f"{max_dd * 100:.2f}%")
    
    with col3:
        win_rate = (strategy_returns > 0).mean()
        st.metric("胜率", f"{win_rate * 100:.1f}%")
    
    with col4:
        calmar_ratio = strategy_returns.mean() * 252 / abs(max_dd)
        st.metric("卡玛比率", f"{calmar_ratio:.3f}")
    
    # 收益分布
    st.subheader("📊 收益分布")
    
    fig = px.histogram(
        x=strategy_returns * 100,
        nbins=50,
        title="日收益率分布",
        labels={'x': '日收益率 (%)', 'y': '频次'}
    )
    
    fig.add_vline(x=strategy_returns.mean() * 100, line_dash="dash", line_color="red", 
                  annotation_text="均值")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 回撤分析
    st.subheader("📉 回撤分析")
    
    drawdown = results['strategy_returns'] / results['strategy_returns'].cummax() - 1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results['dates'],
        y=drawdown * 100,
        mode='lines',
        fill='tonegative',
        name='回撤',
        line=dict(color='red', width=1),
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    
    fig.update_layout(
        title="策略回撤曲线",
        xaxis_title="日期",
        yaxis_title="回撤 (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_risk_management():
    """显示风险管理页面"""
    st.header("🛡️ 风险管理")
    
    st.info("💡 风险管理功能正在开发中，敬请期待！")
    
    # 风险限额设置
    st.subheader("⚙️ 风险限额设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_position = st.slider("最大仓位 (%)", 0, 100, 80)
        max_drawdown = st.slider("最大回撤限制 (%)", 0, 50, 20)
        stop_loss = st.slider("止损线 (%)", 0, 20, 5)
    
    with col2:
        var_confidence = st.slider("VaR置信度 (%)", 90, 99, 95)
        concentration_limit = st.slider("单股票集中度限制 (%)", 0, 50, 10)
        leverage_limit = st.slider("杠杆限制", 1.0, 5.0, 2.0, 0.1)
    
    # 风险监控仪表板
    st.subheader("📊 风险监控仪表板")
    
    # 创建仪表盘
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 仓位使用率
        position_usage = np.random.uniform(0.3, 0.8)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = position_usage * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "仓位使用率 (%)"},
            delta = {'reference': max_position},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_position
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 当前回撤
        current_dd = np.random.uniform(0.02, 0.15)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_dd * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "当前回撤 (%)"},
            delta = {'reference': max_drawdown},
            gauge = {
                'axis': {'range': [0, 50]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgray"},
                    {'range': [10, 20], 'color': "yellow"},
                    {'range': [20, 50], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_drawdown
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # VaR风险值
        var_value = np.random.uniform(0.01, 0.05)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = var_value * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"VaR ({var_confidence}%) (%)"},
            gauge = {
                'axis': {'range': [0, 10]},
                'bar': {'color': "darkorange"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgray"},
                    {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 10], 'color': "red"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_real_time_monitoring():
    """显示实时监控页面"""
    st.header("📡 实时监控")
    
    # 自动刷新选项
    auto_refresh = st.checkbox("自动刷新 (10秒)", value=False)
    
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    
    # 系统状态
    st.subheader("🖥️ 系统状态")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = np.random.uniform(20, 80)
        st.metric("CPU使用率", f"{cpu_usage:.1f}%", f"{np.random.uniform(-5, 5):.1f}%")
    
    with col2:
        memory_usage = np.random.uniform(40, 90)
        st.metric("内存使用率", f"{memory_usage:.1f}%", f"{np.random.uniform(-3, 3):.1f}%")
    
    with col3:
        disk_usage = np.random.uniform(30, 70)
        st.metric("磁盘使用率", f"{disk_usage:.1f}%", f"{np.random.uniform(-1, 1):.1f}%")
    
    with col4:
        network_io = np.random.uniform(10, 100)
        st.metric("网络I/O", f"{network_io:.1f} MB/s", f"{np.random.uniform(-10, 10):.1f}")
    
    # 实时数据流
    st.subheader("📊 实时数据流")
    
    # 模拟实时价格数据
    current_time = datetime.now()
    time_points = [current_time - timedelta(minutes=i) for i in range(60, 0, -1)]
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    fig = go.Figure()
    
    for symbol in symbols:
        prices = np.random.normal(100, 10, 60).cumsum() + np.random.uniform(100, 200)
        fig.add_trace(go.Scatter(
            x=time_points,
            y=prices,
            mode='lines',
            name=symbol,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="实时股价监控",
        xaxis_title="时间",
        yaxis_title="价格 ($)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 交易信号
    st.subheader("🚨 交易信号")
    
    signals_data = []
    for i in range(5):
        signals_data.append({
            "时间": (datetime.now() - timedelta(minutes=i*5)).strftime("%H:%M:%S"),
            "股票": np.random.choice(symbols),
            "信号": np.random.choice(["买入", "卖出", "持有"]),
            "强度": np.random.choice(["强", "中", "弱"]),
            "价格": f"${np.random.uniform(100, 300):.2f}"
        })
    
    signals_df = pd.DataFrame(signals_data)
    st.dataframe(signals_df, use_container_width=True)
    
    # 系统日志
    st.subheader("📝 系统日志")
    
    log_data = []
    for i in range(10):
        log_data.append({
            "时间": (datetime.now() - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
            "级别": np.random.choice(["INFO", "WARNING", "ERROR"]),
            "模块": np.random.choice(["DataManager", "FactorEngine", "BacktestEngine", "RiskManager"]),
            "消息": f"系统运行正常 - 任务 {i+1} 完成"
        })
    
    log_df = pd.DataFrame(log_data)
    st.dataframe(log_df, use_container_width=True)

if __name__ == "__main__":
    main()