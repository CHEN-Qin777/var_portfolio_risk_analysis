# src/report_generator.py
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ReportGenerator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_summary_report(self, portfolio_data, var_results, portfolio_stats):
        """生成风险分析摘要报告"""
        
        report = {
            'timestamp': self.timestamp,
            'portfolio_summary': {
                '资产数量': len(portfolio_data['weights']),
                '投资组合价值': f"${portfolio_stats['portfolio_value']:,.2f}",
                '年化波动率': f"{portfolio_stats['volatility'] * np.sqrt(252):.2%}",
                '夏普比率': f"{portfolio_stats['sharpe_ratio']:.2f}",
            },
            'risk_metrics': {
                '历史法VaR (95%)': f"${var_results['historical']['var_value']:,.2f}",
                '历史法VaR (%)': f"{var_results['historical']['var']:.2%}",
                '参数法VaR (95%)': f"${var_results['parametric']['var_value']:,.2f}",
                '参数法VaR (%)': f"{var_results['parametric']['var']:.2%}",
                '蒙特卡洛VaR (95%)': f"${var_results['monte_carlo']['var_value']:,.2f}",
                '蒙特卡洛VaR (%)': f"{var_results['monte_carlo']['var']:.2%}",
                'Expected Shortfall': f"${var_results['expected_shortfall']['es_value']:,.2f}",
                'Expected Shortfall (%)': f"{var_results['expected_shortfall']['es']:.2%}",
            },
            'portfolio_composition': portfolio_data['weights']
        }
        
        return report
    
    def save_detailed_report(self, report_data, filename):
        """保存详细报告到CSV"""
        # 创建DataFrame用于报告
        risk_df = pd.DataFrame.from_dict(report_data['risk_metrics'], 
                                       orient='index', columns=['Value'])
        portfolio_df = pd.DataFrame.from_dict(report_data['portfolio_composition'], 
                                            orient='index', columns=['Weight'])
        
        with open(filename, 'w') as f:
            f.write("投资组合风险分析报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {report_data['timestamp']}\n\n")
            
            f.write("投资组合摘要:\n")
            f.write("-" * 20 + "\n")
            for key, value in report_data['portfolio_summary'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n风险指标:\n")
            f.write("-" * 20 + "\n")
            for key, value in report_data['risk_metrics'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n资产配置:\n")
            f.write("-" * 20 + "\n")
            for asset, weight in report_data['portfolio_composition'].items():
                f.write(f"{asset}: {weight:.1%}\n")
        
        print(f"详细报告已保存到: {filename}")
    
    def generate_latex_report(self, report_data, filename):
        """生成LaTeX格式的报告"""
        latex_content = f"""
\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}

\\title{{投资组合风险分析报告}}
\\author{{VaR分析系统}}
\\date{{{report_data['timestamp']}}}

\\begin{{document}}

\\maketitle

\\section{{执行摘要}}
本报告提供了多资产投资组合的风险分析，使用多种方法计算了Value-at-Risk (VaR)。

\\section{{投资组合信息}}
\\begin{{itemize}}
    \\item 资产数量: {report_data['portfolio_summary']['资产数量']}
    \\item 投资组合价值: {report_data['portfolio_summary']['投资组合价值']}
    \\item 年化波动率: {report_data['portfolio_summary']['年化波动率']}
    \\item 夏普比率: {report_data['portfolio_summary']['夏普比率']}
\\end{{itemize}}

\\section{{风险指标}}
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{指标}} & \\textbf{{数值}} \\\\
\\midrule
"""
        
        for key, value in report_data['risk_metrics'].items():
            latex_content += f"{key} & {value} \\\\\n"
        
        latex_content += """
\\bottomrule
\\end{tabular}
\\caption{风险指标汇总}
\\end{table}

\\section{资产配置}
\\begin{table}[h]
\\centering
\\begin{tabular}{lr}
\\toprule
\\textbf{资产} & \\textbf{权重} \\\\
\\midrule
"""
        
        for asset, weight in report_data['portfolio_composition'].items():
            latex_content += f"{asset} & {weight:.1%} \\\\\n"
        
        latex_content += """
\\bottomrule
\\end{tabular}
\\caption{投资组合资产配置}
\\end{table}

\\end{document}
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"LaTeX报告已保存到: {filename}")