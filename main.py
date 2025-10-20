# main.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from var_calculator import VaRCalculator
from monte_carlo import MonteCarloSimulator
from visualizer import RiskVisualizer
from report_generator import ReportGenerator
import config
import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("      多资产投资组合VaR风险分析系统")
    print("=" * 60)
    
    # 初始化组件
    data_loader = DataLoader(config.Config.START_DATE, config.Config.END_DATE)
    var_calculator = VaRCalculator(config.Config.CONFIDENCE_LEVEL)
    mc_simulator = MonteCarloSimulator(
        n_simulations=config.Config.MONTE_CARLO_SIMULATIONS,
        time_horizon=config.Config.MONTE_CARLO_DAYS,
        random_seed=config.Config.RANDOM_SEED
    )
    visualizer = RiskVisualizer()
    report_generator = ReportGenerator()
    
    # 创建输出目录
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)
    
    try:
        # 步骤1: 加载数据
        print("\n1. 加载投资组合数据...")
        portfolio_data = data_loader.generate_sample_data(
            symbols=list(config.Config.DEFAULT_PORTFOLIO.keys()),
            portfolio_weights=config.Config.DEFAULT_PORTFOLIO
        )
        
        # 步骤2: 计算投资组合统计
        print("\n2. 计算投资组合统计...")
        portfolio_stats = var_calculator.calculate_portfolio_stats(
            portfolio_data['returns'],
            np.array(list(portfolio_data['weights'].values()))
        )
        
        # 步骤3: 计算VaR
        print("\n3. 计算Value-at-Risk...")
        
        # 历史法VaR
        historical_var = var_calculator.historical_var(
            portfolio_data['returns'],
            np.array(list(portfolio_data['weights'].values())),
            portfolio_stats['portfolio_value']
        )
        
        # 参数法VaR
        parametric_var = var_calculator.parametric_var(
            portfolio_data['returns'],
            np.array(list(portfolio_data['weights'].values())),
            portfolio_stats['portfolio_value']
        )
        
        # Expected Shortfall
        expected_shortfall = var_calculator.calculate_expected_shortfall(
            historical_var['portfolio_returns'],
            portfolio_stats['portfolio_value']
        )
        
        # 蒙特卡洛VaR
        print("4. 执行蒙特卡洛模拟...")
        mc_simulations = mc_simulator.simulate_gbm(
            portfolio_data['returns'],
            portfolio_data['weights'],
            portfolio_stats['portfolio_value']
        )
        
        monte_carlo_var = mc_simulator.monte_carlo_var(
            mc_simulations,
            config.Config.CONFIDENCE_LEVEL
        )
        monte_carlo_var['simulations'] = mc_simulations
        
        # 整合结果
        var_results = {
            'historical': historical_var,
            'parametric': parametric_var,
            'monte_carlo': monte_carlo_var,
            'expected_shortfall': expected_shortfall,
            'confidence_level': config.Config.CONFIDENCE_LEVEL
        }
        
        # 步骤4: 可视化结果
        print("\n5. 生成可视化图表...")
        visualizer.plot_returns_distribution(
            historical_var['portfolio_returns'], 
            var_results,
            'output/plots/returns_distribution.png'
        )
        
        visualizer.plot_monte_carlo_simulations(
            monte_carlo_var,
            'output/plots/monte_carlo.png'
        )
        
        visualizer.plot_var_comparison(
            var_results,
            'output/plots/var_comparison.png'
        )
        
        # 交互式可视化
        visualizer.plot_interactive_var_analysis(
            historical_var['portfolio_returns'],
            var_results,
            monte_carlo_var
        )
        
        # 步骤5: 生成报告
        print("\n6. 生成分析报告...")
        summary_report = report_generator.generate_summary_report(
            portfolio_data, var_results, portfolio_stats
        )
        
        # 保存报告
        report_generator.save_detailed_report(
            summary_report, 
            'output/reports/risk_analysis_report.txt'
        )
        
        report_generator.generate_latex_report(
            summary_report,
            'output/reports/risk_analysis_report.tex'
        )
        
        # 显示结果摘要
        print("\n" + "=" * 60)
        print("           分析完成 - 结果摘要")
        print("=" * 60)
        
        print(f"\n投资组合价值: ${portfolio_stats['portfolio_value']:,.2f}")
        print(f"年化波动率: {portfolio_stats['volatility'] * np.sqrt(252):.2%}")
        print(f"夏普比率: {portfolio_stats['sharpe_ratio']:.2f}")
        
        print(f"\n风险指标 ({config.Config.CONFIDENCE_LEVEL*100}% 置信水平):")
        print(f"  历史法VaR: ${historical_var['var_value']:,.2f} ({historical_var['var']:.2%})")
        print(f"  参数法VaR: ${parametric_var['var_value']:,.2f} ({parametric_var['var']:.2%})")
        print(f"  蒙特卡洛VaR: ${monte_carlo_var['var_value']:,.2f} ({monte_carlo_var['var']:.2%})")
        print(f"  Expected Shortfall: ${expected_shortfall['es_value']:,.2f} ({expected_shortfall['es']:.2%})")
        
        print(f"\n报告和图表已保存到 'output/' 目录")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()