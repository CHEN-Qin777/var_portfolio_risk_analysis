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
        """Génère un rapport de synthèse de l'analyse des risques"""
        
        report = {
            'timestamp': self.timestamp,
            'portfolio_summary': {
                'Nombre d\'actifs': len(portfolio_data['weights']),
                'Valeur du portefeuille': f"${portfolio_stats['portfolio_value']:,.2f}",
                'Volatilité annualisée': f"{portfolio_stats['volatility'] * np.sqrt(252):.2%}",
                'Ratio de Sharpe': f"{portfolio_stats['sharpe_ratio']:.2f}",
            },
            'risk_metrics': {
                'VaR historique (95%)': f"${var_results['historical']['var_value']:,.2f}",
                'VaR historique (%)': f"{var_results['historical']['var']:.2%}",
                'VaR paramétrique (95%)': f"${var_results['parametric']['var_value']:,.2f}",
                'VaR paramétrique (%)': f"{var_results['parametric']['var']:.2%}",
                'VaR Monte-Carlo (95%)': f"${var_results['monte_carlo']['var_value']:,.2f}",
                'VaR Monte-Carlo (%)': f"{var_results['monte_carlo']['var']:.2%}",
                'Expected Shortfall': f"${var_results['expected_shortfall']['es_value']:,.2f}",
                'Expected Shortfall (%)': f"{var_results['expected_shortfall']['es']:.2%}",
            },
            'portfolio_composition': portfolio_data['weights']
        }
        
        return report
    
    def save_detailed_report(self, report_data, filename):
        """Enregistre le rapport détaillé au format texte"""
        # Création d'un DataFrame pour le rapport
        risk_df = pd.DataFrame.from_dict(report_data['risk_metrics'], 
                                       orient='index', columns=['Valeur'])
        portfolio_df = pd.DataFrame.from_dict(report_data['portfolio_composition'], 
                                            orient='index', columns=['Poids'])
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Rapport d'analyse des risques du portefeuille\n")
            f.write("=" * 50 + "\n")
            f.write(f"Généré le : {report_data['timestamp']}\n\n")
            
            f.write("Résumé du portefeuille :\n")
            f.write("-" * 20 + "\n")
            for key, value in report_data['portfolio_summary'].items():
                f.write(f"{key} : {value}\n")
            
            f.write("\nIndicateurs de risque :\n")
            f.write("-" * 20 + "\n")
            for key, value in report_data['risk_metrics'].items():
                f.write(f"{key} : {value}\n")
            
            f.write("\nAllocation d'actifs :\n")
            f.write("-" * 20 + "\n")
            for asset, weight in report_data['portfolio_composition'].items():
                f.write(f"{asset} : {weight:.1%}\n")
        
        print(f"Rapport détaillé enregistré dans : {filename}")
    
    def generate_latex_report(self, report_data, filename):
        """Génère un rapport au format LaTeX"""
        latex_content = f"""
\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}

\\title{{Rapport d'analyse des risques du portefeuille}}
\\author{{Système d'analyse VaR}}
\\date{{{report_data['timestamp']}}}

\\begin{{document}}

\\maketitle

\\section{{Résumé exécutif}}
Ce rapport fournit une analyse des risques pour un portefeuille multi-actifs, en calculant la Value-at-Risk (VaR) à l'aide de différentes méthodes.

\\section{{Informations sur le portefeuille}}
\\begin{{itemize}}
    \\item Nombre d'actifs : {report_data['portfolio_summary']['Nombre d\'actifs']}
    \\item Valeur du portefeuille : {report_data['portfolio_summary']['Valeur du portefeuille']}
    \\item Volatilité annualisée : {report_data['portfolio_summary']['Volatilité annualisée']}
    \\item Ratio de Sharpe : {report_data['portfolio_summary']['Ratio de Sharpe']}
\\end{{itemize}}

\\section{{Indicateurs de risque}}
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Indicateur}} & \\textbf{{Valeur}} \\\\
\\midrule
"""
        
        for key, value in report_data['risk_metrics'].items():
            latex_content += f"{key} & {value} \\\\\n"
        
        latex_content += """
\\bottomrule
\\end{tabular}
\\caption{Récapitulatif des indicateurs de risque}
\\end{table}

\\section{Allocation d'actifs}
\\begin{table}[h]
\\centering
\\begin{tabular}{lr}
\\toprule
\\textbf{Actif} & \\textbf{Poids} \\\\
\\midrule
"""
        
        for asset, weight in report_data['portfolio_composition'].items():
            latex_content += f"{asset} & {weight:.1%} \\\\\n"
        
        latex_content += """
\\bottomrule
\\end{tabular}
\\caption{Allocation du portefeuille}
\\end{table}

\\end{document}
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"Rapport LaTeX enregistré dans : {filename}")
