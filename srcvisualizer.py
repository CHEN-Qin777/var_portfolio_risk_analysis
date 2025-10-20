# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RiskVisualizer:
    def __init__(self):
        self.fig_size = (12, 8)
    
    def plot_returns_distribution(self, returns, var_results, save_path=None):
        """绘制收益率分布和VaR"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：收益率分布直方图
        ax1.hist(returns, bins=50, alpha=0.7, density=True, edgecolor='black')
        ax1.axvline(-var_results['historical']['var'], color='red', 
                   linestyle='--', linewidth=2, label=f"VaR {var_results['confidence_level']*100}%")
        ax1.axvline(-var_results['expected_shortfall']['es'], color='darkred', 
                   linestyle='--', linewidth=2, label='Expected Shortfall')
        ax1.set_xlabel('日收益率')
        ax1.set_ylabel('频率')
        ax1.set_title('投资组合收益率分布与风险度量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：QQ图检验正态性
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('收益率QQ图（正态性检验）')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_monte_carlo_simulations(self, mc_results, save_path=None):
        """绘制蒙特卡洛模拟结果"""
        simulations = mc_results['simulations']
        
        plt.figure(figsize=self.fig_size)
        
        # 绘制部分模拟路径
        n_paths_to_plot = 100
        for i in range(min(n_paths_to_plot, simulations.shape[1])):
            plt.plot(simulations[:, i], alpha=0.1, color='blue')
        
        # 绘制平均路径和VaR水平
        mean_path = simulations.mean(axis=1)
        initial_value = simulations[0, 0]
        var_level = initial_value * (1 - mc_results['var'])
        
        plt.plot(mean_path, color='red', linewidth=2, label='平均路径')
        plt.axhline(var_level, color='darkred', linestyle='--', 
                   linewidth=2, label=f"VaR {mc_results['confidence_level']*100}%")
        plt.axhline(initial_value, color='green', linestyle='-', 
                   linewidth=2, label='初始价值')
        
        plt.xlabel('时间（天）')
        plt.ylabel('投资组合价值')
        plt.title('蒙特卡洛模拟 - 投资组合价值路径')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_var_comparison(self, var_results, save_path=None):
        """比较不同方法的VaR结果"""
        methods = ['Historical', 'Parametric', 'Monte Carlo']
        var_values = [
            var_results['historical']['var_value'],
            var_results['parametric']['var_value'],
            var_results['monte_carlo']['var_value']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, var_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        
        plt.ylabel('VaR (金额)')
        plt.title('不同VaR计算方法比较')
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, var_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'${value:,.0f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_var_analysis(self, returns, var_results, mc_results):
        """创建交互式可视化（Plotly）"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('收益率分布与VaR', '蒙特卡洛模拟',
                          'VaR方法比较', '回测分析'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 收益率分布
        hist_data = returns
        fig.add_trace(
            go.Histogram(x=hist_data, nbinsx=50, name="收益率分布",
                        opacity=0.7, marker_color='blue'),
            row=1, col=1
        )
        
        # VaR线
        var_line = -var_results['historical']['var']
        fig.add_vline(x=var_line, line_dash="dash", line_color="red",
                     annotation_text=f"VaR 95%: {var_line:.4f}", row=1, col=1)
        
        # 蒙特卡洛模拟（显示部分路径）
        simulations = mc_results['simulations']
        for i in range(min(50, simulations.shape[1])):
            fig.add_trace(
                go.Scatter(y=simulations[:, i], mode='lines',
                          line=dict(width=1, color='lightblue'),
                          showlegend=False),
                row=1, col=2
            )
        
        # 平均路径
        fig.add_trace(
            go.Scatter(y=simulations.mean(axis=1), mode='lines',
                      line=dict(width=3, color='red'), name='平均路径'),
            row=1, col=2
        )
        
        # VaR比较
        methods = ['Historical', 'Parametric', 'Monte Carlo']
        var_values = [
            var_results['historical']['var_value'],
            var_results['parametric']['var_value'],
            var_results['monte_carlo']['var_value']
        ]
        
        fig.add_trace(
            go.Bar(x=methods, y=var_values, name='VaR比较',
                  marker_color=['skyblue', 'lightcoral', 'lightgreen']),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="投资组合风险分析仪表板")
        fig.show()