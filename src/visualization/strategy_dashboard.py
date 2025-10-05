import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class StrategyDashboard:
    """
    策略可视化仪表板：生成交互式图表和性能报告
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        初始化仪表板
        
        Args:
            theme: 图表主题
        """
        self.theme = theme
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 颜色配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_equity_curve_chart(
        self, 
        results: Dict[str, Dict[str, Any]], 
        benchmark_data: pd.DataFrame = None
    ) -> go.Figure:
        """
        创建净值曲线图表
        
        Args:
            results: 策略测试结果
            benchmark_data: 基准数据
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        # 添加策略净值曲线
        for strategy_name, result in results.items():
            portfolio_value = result['backtest_result']['portfolio_value']
            normalized_value = portfolio_value / portfolio_value.iloc[0]
            
            fig.add_trace(go.Scatter(
                x=normalized_value.index,
                y=normalized_value.values,
                mode='lines',
                name=strategy_name,
                line=dict(width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             '日期: %{x}<br>' +
                             '净值: %{y:.4f}<extra></extra>'
            ))
        
        # 添加基准线
        if benchmark_data is not None:
            benchmark_normalized = benchmark_data / benchmark_data.iloc[0]
            fig.add_trace(go.Scatter(
                x=benchmark_normalized.index,
                y=benchmark_normalized.values,
                mode='lines',
                name='基准',
                line=dict(width=2, dash='dash', color='gray'),
                hovertemplate='<b>基准</b><br>' +
                             '日期: %{x}<br>' +
                             '净值: %{y:.4f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='策略净值曲线对比',
            xaxis_title='日期',
            yaxis_title='净值（标准化）',
            template=self.theme,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_performance_radar_chart(
        self, 
        results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """
        创建性能雷达图
        
        Args:
            results: 策略测试结果
            
        Returns:
            Plotly图表对象
        """
        # 选择关键指标
        metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate', 'profit_loss_ratio']
        metric_names = ['夏普比率', '索提诺比率', '卡尔玛比率', '胜率', '盈亏比']
        
        fig = go.Figure()
        
        for strategy_name, result in results.items():
            performance_metrics = result['performance_metrics']
            
            # 标准化指标值（0-1范围）
            values = []
            for metric in metrics:
                value = performance_metrics.get(metric, 0)
                
                # 根据指标类型进行标准化
                if metric == 'win_rate':
                    normalized_value = value / 100  # 胜率已经是百分比
                elif metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
                    # 比率类指标，假设好的值在0-3之间
                    normalized_value = max(0, min(1, (value + 1) / 4))
                elif metric == 'profit_loss_ratio':
                    # 盈亏比，假设好的值在0-5之间
                    normalized_value = max(0, min(1, value / 5))
                else:
                    normalized_value = max(0, min(1, value))
                
                values.append(normalized_value)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names,
                fill='toself',
                name=strategy_name,
                line=dict(width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             '%{theta}: %{r:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='策略性能雷达图',
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def create_risk_return_scatter(
        self, 
        results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """
        创建风险收益散点图
        
        Args:
            results: 策略测试结果
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        returns = []
        volatilities = []
        sharpe_ratios = []
        names = []
        
        for strategy_name, result in results.items():
            metrics = result['performance_metrics']
            returns.append(metrics.get('annualized_return', 0))
            volatilities.append(metrics.get('annualized_volatility', 0))
            sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            names.append(strategy_name)
        
        # 根据夏普比率设置颜色
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=names,
            textposition='top center',
            marker=dict(
                size=12,
                color=sharpe_ratios,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="夏普比率"),
                line=dict(width=1, color='black')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         '年化收益率: %{y:.2f}%<br>' +
                         '年化波动率: %{x:.2f}%<br>' +
                         '夏普比率: %{marker.color:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='策略风险收益分析',
            xaxis_title='年化波动率 (%)',
            yaxis_title='年化收益率 (%)',
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def create_drawdown_chart(
        self, 
        results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """
        创建回撤分析图表
        
        Args:
            results: 策略测试结果
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('回撤时间序列', '最大回撤对比'),
            vertical_spacing=0.1
        )
        
        # 回撤时间序列
        for strategy_name, result in results.items():
            portfolio_value = result['backtest_result']['portfolio_value']
            
            # 计算回撤
            peak = portfolio_value.expanding().max()
            drawdown = (portfolio_value - peak) / peak * 100
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name=strategy_name,
                    fill='tonexty' if strategy_name == list(results.keys())[0] else None,
                    line=dict(width=1),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 '日期: %{x}<br>' +
                                 '回撤: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 最大回撤对比
        strategy_names = list(results.keys())
        max_drawdowns = [abs(result['performance_metrics'].get('max_drawdown', 0)) 
                        for result in results.values()]
        
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=max_drawdowns,
                name='最大回撤',
                marker_color='red',
                opacity=0.7,
                hovertemplate='<b>%{x}</b><br>' +
                             '最大回撤: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='策略回撤分析',
            template=self.theme,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_yaxes(title_text="回撤 (%)", row=1, col=1)
        fig.update_xaxes(title_text="策略", row=2, col=1)
        fig.update_yaxes(title_text="最大回撤 (%)", row=2, col=1)
        
        return fig
    
    def create_monthly_returns_heatmap(
        self, 
        strategy_name: str, 
        result: Dict[str, Any]
    ) -> go.Figure:
        """
        创建月度收益热力图
        
        Args:
            strategy_name: 策略名称
            result: 策略结果
            
        Returns:
            Plotly图表对象
        """
        returns = result['backtest_result']['returns']
        
        # 计算月度收益
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # 创建年月矩阵
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_data = monthly_returns.groupby([
            monthly_returns.index.year, 
            monthly_returns.index.month
        ]).first().unstack(fill_value=0)
        
        # 月份名称
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                      '7月', '8月', '9月', '10月', '11月', '12月']
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_data.values,
            x=month_names,
            y=monthly_data.index,
            colorscale='RdYlGn',
            zmid=0,
            hovertemplate='年份: %{y}<br>' +
                         '月份: %{x}<br>' +
                         '收益率: %{z:.2f}%<extra></extra>',
            colorbar=dict(title="收益率 (%)")
        ))
        
        fig.update_layout(
            title=f'{strategy_name} - 月度收益热力图',
            xaxis_title='月份',
            yaxis_title='年份',
            template=self.theme
        )
        
        return fig
    
    def create_rolling_metrics_chart(
        self, 
        strategy_name: str, 
        result: Dict[str, Any],
        window: int = 252
    ) -> go.Figure:
        """
        创建滚动指标图表
        
        Args:
            strategy_name: 策略名称
            result: 策略结果
            window: 滚动窗口
            
        Returns:
            Plotly图表对象
        """
        returns = result['backtest_result']['returns']
        
        # 计算滚动指标
        rolling_return = returns.rolling(window).mean() * 252 * 100
        rolling_volatility = returns.rolling(window).std() * np.sqrt(252) * 100
        rolling_sharpe = rolling_return / rolling_volatility * 100
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('滚动年化收益率', '滚动年化波动率', '滚动夏普比率'),
            vertical_spacing=0.08
        )
        
        # 滚动收益率
        fig.add_trace(
            go.Scatter(
                x=rolling_return.index,
                y=rolling_return.values,
                mode='lines',
                name='滚动收益率',
                line=dict(color=self.colors['primary']),
                hovertemplate='日期: %{x}<br>收益率: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 滚动波动率
        fig.add_trace(
            go.Scatter(
                x=rolling_volatility.index,
                y=rolling_volatility.values,
                mode='lines',
                name='滚动波动率',
                line=dict(color=self.colors['warning']),
                hovertemplate='日期: %{x}<br>波动率: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 滚动夏普比率
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='滚动夏普比率',
                line=dict(color=self.colors['success']),
                hovertemplate='日期: %{x}<br>夏普比率: %{y:.3f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'{strategy_name} - 滚动指标分析 ({window}日窗口)',
            template=self.theme,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="日期", row=3, col=1)
        fig.update_yaxes(title_text="收益率 (%)", row=1, col=1)
        fig.update_yaxes(title_text="波动率 (%)", row=2, col=1)
        fig.update_yaxes(title_text="夏普比率", row=3, col=1)
        
        return fig
    
    def create_comprehensive_dashboard(
        self, 
        results: Dict[str, Dict[str, Any]],
        save_path: str = None
    ) -> str:
        """
        创建综合仪表板HTML文件
        
        Args:
            results: 策略测试结果
            save_path: 保存路径
            
        Returns:
            HTML内容
        """
        # 创建各种图表
        equity_chart = self.create_equity_curve_chart(results)
        radar_chart = self.create_performance_radar_chart(results)
        scatter_chart = self.create_risk_return_scatter(results)
        drawdown_chart = self.create_drawdown_chart(results)
        
        # 生成HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>量化策略分析仪表板</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f8f9fa;
                }}
                .header {{
                    text-align: center;
                    background-color: #343a40;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .chart {{
                    width: 100%;
                    height: 500px;
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }}
                .metrics-table th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>量化策略分析仪表板</h1>
                <p>策略数量: {len(results)} | 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="chart-container">
                <h2>策略净值曲线对比</h2>
                <div id="equity-chart" class="chart"></div>
            </div>
            
            <div class="chart-container">
                <h2>策略性能雷达图</h2>
                <div id="radar-chart" class="chart"></div>
            </div>
            
            <div class="chart-container">
                <h2>风险收益分析</h2>
                <div id="scatter-chart" class="chart"></div>
            </div>
            
            <div class="chart-container">
                <h2>回撤分析</h2>
                <div id="drawdown-chart" class="chart"></div>
            </div>
            
            <script>
                // 渲染图表
                Plotly.newPlot('equity-chart', {equity_chart.to_json()});
                Plotly.newPlot('radar-chart', {radar_chart.to_json()});
                Plotly.newPlot('scatter-chart', {scatter_chart.to_json()});
                Plotly.newPlot('drawdown-chart', {drawdown_chart.to_json()});
            </script>
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"仪表板已保存到: {save_path}")
        
        return html_content
    
    def export_charts_to_images(
        self, 
        results: Dict[str, Dict[str, Any]],
        save_dir: str
    ) -> None:
        """
        导出图表为图片文件
        
        Args:
            results: 策略测试结果
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建并保存各种图表
        charts = {
            'equity_curve': self.create_equity_curve_chart(results),
            'performance_radar': self.create_performance_radar_chart(results),
            'risk_return_scatter': self.create_risk_return_scatter(results),
            'drawdown_analysis': self.create_drawdown_chart(results)
        }
        
        for chart_name, chart in charts.items():
            file_path = os.path.join(save_dir, f"{chart_name}.png")
            chart.write_image(file_path, width=1200, height=800, scale=2)
            print(f"图表已保存: {file_path}")
        
        # 为每个策略创建详细图表
        for strategy_name, result in results.items():
            strategy_dir = os.path.join(save_dir, strategy_name.replace(' ', '_'))
            os.makedirs(strategy_dir, exist_ok=True)
            
            # 月度收益热力图
            heatmap_chart = self.create_monthly_returns_heatmap(strategy_name, result)
            heatmap_path = os.path.join(strategy_dir, "monthly_returns_heatmap.png")
            heatmap_chart.write_image(heatmap_path, width=1000, height=600, scale=2)
            
            # 滚动指标图表
            rolling_chart = self.create_rolling_metrics_chart(strategy_name, result)
            rolling_path = os.path.join(strategy_dir, "rolling_metrics.png")
            rolling_chart.write_image(rolling_path, width=1200, height=900, scale=2)
            
            print(f"策略 {strategy_name} 详细图表已保存到: {strategy_dir}")