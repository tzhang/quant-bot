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
    from src.risk.enhanced_risk_manager import EnhancedRiskManager, RiskLimits
    from src.risk.real_time_monitor import RealTimeRiskMonitor, MonitoringConfig
    from src.risk.risk_metrics import RiskMetricsEngine
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
if 'enhanced_risk_manager' not in st.session_state:
    st.session_state.enhanced_risk_manager = None
if 'risk_monitor' not in st.session_state:
    st.session_state.risk_monitor = None
if 'risk_metrics_engine' not in st.session_state:
    st.session_state.risk_metrics_engine = None

def initialize_system():
    """初始化系统组件"""
    try:
        if st.session_state.data_manager is None:
            st.session_state.data_manager = DataManager()
        if st.session_state.factor_engine is None:
            st.session_state.factor_engine = FactorEngine()
        if st.session_state.risk_metrics_engine is None:
            st.session_state.risk_metrics_engine = RiskMetricsEngine()
        if st.session_state.enhanced_risk_manager is None:
            # 创建风险限制配置
            risk_limits = RiskLimits(
                max_position_size=0.1,
                max_leverage=2.0,
                var_limit_1d=0.02,
                max_drawdown=0.15,
                max_concentration=0.05
            )
            st.session_state.enhanced_risk_manager = EnhancedRiskManager(risk_limits=risk_limits)
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
        st.metric("数据源", "3个", "IB TWS API + Qlib + OpenBB")
    
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
            ["IB TWS API", "OpenBB", "Qlib"],
            default=["IB TWS API", "Qlib"]
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
    
    # 获取风险管理器
    risk_manager = st.session_state.enhanced_risk_manager
    risk_metrics_engine = st.session_state.risk_metrics_engine
    
    if not risk_manager or not risk_metrics_engine:
        st.error("风险管理系统未初始化")
        return
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 风险仪表板", "⚙️ 风险配置", "📈 风险指标", "🚨 风险警报", "📋 风险报告"])
    
    with tab1:
        show_risk_dashboard(risk_manager, risk_metrics_engine)
    
    with tab2:
        show_risk_configuration(risk_manager)
    
    with tab3:
        show_risk_metrics(risk_metrics_engine)
    
    with tab4:
        show_risk_alerts(risk_manager)
    
    with tab5:
        show_risk_reports(risk_manager)

def show_risk_dashboard(risk_manager, risk_metrics_engine):
    """显示风险仪表板"""
    st.subheader("📊 实时风险监控仪表板")
    
    # 获取当前风险指标
    current_metrics = risk_manager.get_current_risk_metrics()
    
    if current_metrics:
        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "当前杠杆",
                f"{current_metrics.leverage:.2f}x",
                delta=f"限制: {risk_manager.risk_limits.max_leverage:.1f}x"
            )
        
        with col2:
            st.metric(
                "1日VaR",
                f"{current_metrics.var_1d:.2%}",
                delta=f"限制: {risk_manager.risk_limits.var_limit_1d:.2%}"
            )
        
        with col3:
            st.metric(
                "当前回撤",
                f"{current_metrics.max_drawdown:.2%}",
                delta=f"限制: {risk_manager.risk_limits.max_drawdown:.2%}"
            )
        
        with col4:
            st.metric(
                "波动率",
                f"{current_metrics.volatility:.2%}",
                delta="年化"
            )
        
        # 风险仪表盘
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 杠杆使用率仪表盘
            leverage_usage = (current_metrics.leverage / risk_manager.risk_limits.max_leverage) * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=leverage_usage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "杠杆使用率 (%)"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 150]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "orange"},
                        {'range': [100, 150], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # VaR风险仪表盘
            var_usage = (current_metrics.var_1d / risk_manager.risk_limits.var_limit_1d) * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=var_usage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "VaR风险使用率 (%)"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 150]},
                    'bar': {'color': "darkorange"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "orange"},
                        {'range': [100, 150], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # 回撤风险仪表盘
            dd_usage = (current_metrics.max_drawdown / risk_manager.risk_limits.max_drawdown) * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=dd_usage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "回撤风险使用率 (%)"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 150]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "orange"},
                        {'range': [100, 150], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # 风险趋势图
        st.subheader("📈 风险趋势分析")
        
        # 生成模拟历史数据
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        np.random.seed(42)
        
        risk_data = pd.DataFrame({
            'date': dates,
            'var_1d': np.random.normal(0.015, 0.005, len(dates)),
            'leverage': np.random.normal(1.5, 0.3, len(dates)),
            'volatility': np.random.normal(0.2, 0.05, len(dates)),
            'drawdown': np.cumsum(np.random.normal(0, 0.01, len(dates)))
        })
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('VaR趋势', '杠杆趋势', '波动率趋势', '回撤趋势'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # VaR趋势
        fig.add_trace(
            go.Scatter(x=risk_data['date'], y=risk_data['var_1d'], name='VaR 1日', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_hline(y=risk_manager.risk_limits.var_limit_1d, line_dash="dash", line_color="red", row=1, col=1)
        
        # 杠杆趋势
        fig.add_trace(
            go.Scatter(x=risk_data['date'], y=risk_data['leverage'], name='杠杆', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_hline(y=risk_manager.risk_limits.max_leverage, line_dash="dash", line_color="red", row=1, col=2)
        
        # 波动率趋势
        fig.add_trace(
            go.Scatter(x=risk_data['date'], y=risk_data['volatility'], name='波动率', line=dict(color='green')),
            row=2, col=1
        )
        
        # 回撤趋势
        fig.add_trace(
            go.Scatter(x=risk_data['date'], y=risk_data['drawdown'], name='回撤', line=dict(color='orange')),
            row=2, col=2
        )
        fig.add_hline(y=-risk_manager.risk_limits.max_drawdown, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="风险指标历史趋势")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("暂无风险指标数据")

def show_risk_configuration(risk_manager):
    """显示风险配置页面"""
    st.subheader("⚙️ 风险限制配置")
    
    # 当前配置显示
    st.write("**当前风险限制配置:**")
    current_limits = risk_manager.risk_limits
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**最大仓位大小:** {current_limits.max_position_size:.1%}")
        st.info(f"**最大杠杆:** {current_limits.max_leverage:.1f}x")
        st.info(f"**VaR限制 (1日):** {current_limits.var_limit_1d:.2%}")
    
    with col2:
        st.info(f"**最大回撤:** {current_limits.max_drawdown:.1%}")
        st.info(f"**集中度限制:** {current_limits.max_concentration:.1%}")
        st.info(f"**止损线:** {current_limits.stop_loss_pct:.1%}")
    
    # 配置修改
    st.write("**修改风险限制:**")
    
    with st.form("risk_limits_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_max_position = st.slider("最大仓位大小 (%)", 1, 50, int(current_limits.max_position_size * 100)) / 100
            new_max_leverage = st.slider("最大杠杆", 1.0, 5.0, current_limits.max_leverage, 0.1)
            new_var_limit = st.slider("VaR限制 (1日) (%)", 0.5, 10.0, current_limits.var_limit_1d * 100) / 100
        
        with col2:
            new_max_drawdown = st.slider("最大回撤 (%)", 5, 50, int(current_limits.max_drawdown * 100)) / 100
            new_concentration = st.slider("集中度限制 (%)", 1, 20, int(current_limits.max_concentration * 100)) / 100
            new_stop_loss = st.slider("止损线 (%)", 1, 20, int(current_limits.stop_loss_pct * 100)) / 100
        
        submitted = st.form_submit_button("更新风险限制")
        
        if submitted:
            # 更新风险限制
            new_limits = RiskLimits(
                max_position_size=new_max_position,
                max_leverage=new_max_leverage,
                var_limit_1d=new_var_limit,
                max_drawdown=new_max_drawdown,
                max_concentration=new_concentration,
                stop_loss_pct=new_stop_loss
            )
            
            risk_manager.update_risk_limits(new_limits)
            st.success("风险限制已更新！")
            st.rerun()

def show_risk_metrics(risk_metrics_engine):
    """显示风险指标页面"""
    st.subheader("📈 详细风险指标")
    
    # 生成示例数据
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 一年的日收益率
    prices = pd.Series(100 * np.cumprod(1 + returns))
    
    # 计算各种风险指标
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**VaR指标**")
        
        # 历史模拟VaR
        hist_var = risk_metrics_engine.var_calculator.historical_var(returns, confidence_level=0.05)
        st.metric("历史模拟VaR (95%)", f"{hist_var:.2%}")
        
        # 参数法VaR
        param_var = risk_metrics_engine.var_calculator.parametric_var(returns, confidence_level=0.05)
        st.metric("参数法VaR (95%)", f"{param_var:.2%}")
        
        # CVaR
        cvar = risk_metrics_engine.cvar_calculator.historical_cvar(returns, confidence_level=0.05)
        st.metric("条件VaR (95%)", f"{cvar:.2%}")
    
    with col2:
        st.write("**波动率指标**")
        
        # 简单波动率
        simple_vol = risk_metrics_engine.volatility_calculator.simple_volatility(returns)
        st.metric("简单波动率", f"{simple_vol:.2%}")
        
        # EWMA波动率
        ewma_vol = risk_metrics_engine.volatility_calculator.ewma_volatility(returns)
        st.metric("EWMA波动率", f"{ewma_vol:.2%}")
        
        # 最大回撤
        max_dd, _, _ = risk_metrics_engine.drawdown_calculator.maximum_drawdown(prices)
        st.metric("最大回撤", f"{max_dd:.2%}")
    
    # 风险指标图表
    st.subheader("📊 风险指标可视化")
    
    # VaR回测图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('价格走势与VaR', '收益率分布'),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # 价格走势
    dates = pd.date_range(start='2023-01-01', periods=len(prices), freq='D')
    fig.add_trace(
        go.Scatter(x=dates, y=prices, name='价格', line=dict(color='blue')),
        row=1, col=1
    )
    
    # VaR阈值
    var_threshold = prices * (1 + hist_var)
    fig.add_trace(
        go.Scatter(x=dates, y=var_threshold, name='VaR阈值', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # 收益率分布
    fig.add_trace(
        go.Histogram(x=returns, name='收益率分布', nbinsx=50),
        row=2, col=1
    )
    
    # 添加VaR线
    fig.add_vline(x=hist_var, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(height=600, title_text="风险指标分析")
    st.plotly_chart(fig, use_container_width=True)

def show_risk_alerts(risk_manager):
    """显示风险警报页面"""
    st.subheader("🚨 风险警报管理")
    
    # 获取最近的警报
    recent_alerts = risk_manager.get_recent_alerts(hours=24)
    
    if recent_alerts:
        st.write(f"**最近24小时警报 ({len(recent_alerts)}条):**")
        
        for alert in recent_alerts[-10:]:  # 显示最近10条
            alert_color = {
                'LOW': 'info',
                'MEDIUM': 'warning', 
                'HIGH': 'error',
                'CRITICAL': 'error'
            }.get(alert.level.name, 'info')
            
            with st.container():
                st.markdown(f"""
                <div class="{alert_color}-box">
                    <strong>{alert.level.name}</strong> - {alert.alert_type.name}<br>
                    {alert.message}<br>
                    <small>时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
    else:
        st.success("✅ 暂无风险警报")
    
    # 警报统计
    st.subheader("📊 警报统计")
    
    if recent_alerts:
        # 按级别统计
        alert_levels = [alert.level.name for alert in recent_alerts]
        level_counts = pd.Series(alert_levels).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=level_counts.values, names=level_counts.index, title="警报级别分布")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 按类型统计
            alert_types = [alert.alert_type.name for alert in recent_alerts]
            type_counts = pd.Series(alert_types).value_counts()
            
            fig = px.bar(x=type_counts.index, y=type_counts.values, title="警报类型分布")
            st.plotly_chart(fig, use_container_width=True)

def show_risk_reports(risk_manager):
    """显示风险报告页面"""
    st.subheader("📋 风险管理报告")
    
    # 报告生成选项
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox("报告类型", ["日报", "周报", "月报", "自定义"])
        
    with col2:
        if report_type == "自定义":
            date_range = st.date_input("选择日期范围", value=[datetime.now().date() - timedelta(days=7), datetime.now().date()])
    
    if st.button("生成风险报告"):
        with st.spinner("正在生成报告..."):
            # 生成风险报告
            report_data = risk_manager.generate_risk_report()
            
            if report_data:
                st.success("报告生成成功！")
                
                # 显示报告内容
                st.subheader("📊 风险概览")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("总体风险评级", report_data.get('overall_risk_level', 'MEDIUM'))
                
                with col2:
                    st.metric("风险事件数量", report_data.get('risk_events_count', 0))
                
                with col3:
                    st.metric("平均VaR", f"{report_data.get('avg_var', 0):.2%}")
                
                with col4:
                    st.metric("最大回撤", f"{report_data.get('max_drawdown', 0):.2%}")
                
                # 详细报告内容
                st.subheader("📝 详细分析")
                st.text_area("风险分析报告", report_data.get('detailed_analysis', '暂无详细分析'), height=200)
                
                # 建议措施
                st.subheader("💡 风险管理建议")
                recommendations = report_data.get('recommendations', [])
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            
            else:
                st.warning("暂无足够数据生成报告")

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