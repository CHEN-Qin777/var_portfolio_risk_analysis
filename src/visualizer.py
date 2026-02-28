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

# Configuration du style matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RiskVisualizer:
    def __init__(self):
        self.fig_size = (12, 8)
    
    def plot_returns_distribution(self, returns, var_results, save_path=None):
        """Trace la distribution des rendements et la VaR"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique de gauche : histogramme de la distribution des rendements
        ax1.hist(returns, bins=50, alpha=0.7, density=True, edgecolor='black')
        ax1.axvline(-var_results['historical']['var'], color='red', 
                   linestyle='--', linewidth=2, label=f"VaR {var_results['confidence_level']*100}%")
        ax1.axvline(-var_results['expected_shortfall']['es'], color='darkred', 
                   linestyle='--', linewidth=2, label='Expected Shortfall')
        ax1.set_xlabel('Rendement journalier')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution des rendements du portefeuille et mesures de risque')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique de droite : QQ-plot pour tester la normalité
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('QQ-plot des rendements (test de normalité)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_monte_carlo_simulations(self, mc_results, save_path=None):
        """Trace les résultats de la simulation Monte-Carlo"""
        simulations = mc_results['simulations']
        
        plt.figure(figsize=self.fig_size)
        
        # Tracer un échantillon de trajectoires
        n_paths_to_plot = 100
        for i in range(min(n_paths_to_plot, simulations.shape[1])):
            plt.plot(simulations[:, i], alpha=0.1, color='blue')
        
        # Tracer la trajectoire moyenne et le niveau de VaR
        mean_path = simulations.mean(axis=1)
        initial_value = simulations[0, 0]
        var_level = initial_value * (1 - mc_results['var'])
        
        plt.plot(mean_path, color='red', linewidth=2, label='Trajectoire moyenne')
        plt.axhline(var_level, color='darkred', linestyle='--', 
                   linewidth=2, label=f"VaR {mc_results['confidence_level']*100}%")
        plt.axhline(initial_value, color='green', linestyle='-', 
                   linewidth=2, label='Valeur initiale')
        
        plt.xlabel('Temps (jours)')
        plt.ylabel('Valeur du portefeuille')
        plt.title('Simulation Monte-Carlo - Trajectoires de la valeur du portefeuille')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_var_comparison(self, var_results, save_path=None):
        """Compare les résultats de VaR obtenus par différentes méthodes"""
        methods = ['Historique', 'Paramétrique', 'Monte Carlo']
        var_values = [
            var_results['historical']['var_value'],
            var_results['parametric']['var_value'],
            var_results['monte_carlo']['var_value']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, var_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        
        plt.ylabel('VaR (en valeur)')
        plt.title('Comparaison des méthodes de calcul de la VaR')
        
        # Ajouter des étiquettes de valeur sur les barres
        for bar, value in zip(bars, var_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'{value:,.0f} $', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_var_analysis(self, returns, var_results, mc_results):
        """Crée une visualisation interactive (Plotly)"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution des rendements et VaR', 'Simulation Monte-Carlo',
                          'Comparaison des méthodes VaR', 'Analyse de backtest'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Distribution des rendements
        hist_data = returns
        fig.add_trace(
            go.Histogram(x=hist_data, nbinsx=50, name="Distribution des rendements",
                        opacity=0.7, marker_color='blue'),
            row=1, col=1
        )
        
        # Ligne VaR
        var_line = -var_results['historical']['var']
        fig.add_vline(x=var_line, line_dash="dash", line_color="red",
                     annotation_text=f"VaR 95% : {var_line:.4f}", row=1, col=1)
        
        # Simulation Monte-Carlo (affichage d'un sous-ensemble de trajectoires)
        simulations = mc_results['simulations']
        for i in range(min(50, simulations.shape[1])):
            fig.add_trace(
                go.Scatter(y=simulations[:, i], mode='lines',
                          line=dict(width=1, color='lightblue'),
                          showlegend=False),
                row=1, col=2
            )
        
        # Trajectoire moyenne
        fig.add_trace(
            go.Scatter(y=simulations.mean(axis=1), mode='lines',
                      line=dict(width=3, color='red'), name='Trajectoire moyenne'),
            row=1, col=2
        )
        
        # Comparaison VaR
        methods = ['Historique', 'Paramétrique', 'Monte Carlo']
        var_values = [
            var_results['historical']['var_value'],
            var_results['parametric']['var_value'],
            var_results['monte_carlo']['var_value']
        ]
        
        fig.add_trace(
            go.Bar(x=methods, y=var_values, name='Comparaison VaR',
                  marker_color=['skyblue', 'lightcoral', 'lightgreen']),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="Tableau de bord de l'analyse des risques du portefeuille")

        fig.show()
