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
    print("      Système d'analyse des risques VaR pour portefeuille multi-actifs")
    print("=" * 60)
    
    # Initialisation des composants
    data_loader = DataLoader(config.Config.START_DATE, config.Config.END_DATE)
    var_calculator = VaRCalculator(config.Config.CONFIDENCE_LEVEL)
    mc_simulator = MonteCarloSimulator(
        n_simulations=config.Config.MONTE_CARLO_SIMULATIONS,
        time_horizon=config.Config.MONTE_CARLO_DAYS,
        random_seed=config.Config.RANDOM_SEED
    )
    visualizer = RiskVisualizer()
    report_generator = ReportGenerator()
    
    # Création des répertoires de sortie
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)
    
    try:
        # Étape 1 : Chargement des données
        print("\n1. Chargement des données du portefeuille...")
        portfolio_data = data_loader.generate_sample_data(
            symbols=list(config.Config.DEFAULT_PORTFOLIO.keys()),
            portfolio_weights=config.Config.DEFAULT_PORTFOLIO
        )
        
        # Étape 2 : Calcul des statistiques du portefeuille
        print("\n2. Calcul des statistiques du portefeuille...")
        portfolio_stats = var_calculator.calculate_portfolio_stats(
            portfolio_data['returns'],
            np.array(list(portfolio_data['weights'].values()))
        )
        
        # Étape 3 : Calcul de la Value-at-Risk (VaR)
        print("\n3. Calcul de la Value-at-Risk (VaR)...")
        
        # VaR historique
        historical_var = var_calculator.historical_var(
            portfolio_data['returns'],
            np.array(list(portfolio_data['weights'].values())),
            portfolio_stats['portfolio_value']
        )
        
        # VaR paramétrique
        parametric_var = var_calculator.parametric_var(
            portfolio_data['returns'],
            np.array(list(portfolio_data['weights'].values())),
            portfolio_stats['portfolio_value']
        )
        
        # Déficit attendu (Expected Shortfall)
        expected_shortfall = var_calculator.calculate_expected_shortfall(
            historical_var['portfolio_returns'],
            portfolio_stats['portfolio_value']
        )
        
        # VaR Monte-Carlo
        print("4. Exécution de la simulation Monte-Carlo...")
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
        
        # Agrégation des résultats
        var_results = {
            'historical': historical_var,
            'parametric': parametric_var,
            'monte_carlo': monte_carlo_var,
            'expected_shortfall': expected_shortfall,
            'confidence_level': config.Config.CONFIDENCE_LEVEL
        }
        
        # Étape 4 : Visualisation des résultats
        print("\n5. Génération des graphiques de visualisation...")
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
        
        # Visualisation interactive
        visualizer.plot_interactive_var_analysis(
            historical_var['portfolio_returns'],
            var_results,
            monte_carlo_var
        )
        
        # Étape 5 : Génération du rapport
        print("\n6. Génération du rapport d'analyse...")
        summary_report = report_generator.generate_summary_report(
            portfolio_data, var_results, portfolio_stats
        )
        
        # Sauvegarde du rapport
        report_generator.save_detailed_report(
            summary_report, 
            'output/reports/risk_analysis_report.txt'
        )
        
        report_generator.generate_latex_report(
            summary_report,
            'output/reports/risk_analysis_report.tex'
        )
        
        # Affichage du résumé des résultats
        print("\n" + "=" * 60)
        print("           Analyse terminée - Résumé des résultats")
        print("=" * 60)
        
        print(f"\nValeur du portefeuille : ${portfolio_stats['portfolio_value']:,.2f}")
        print(f"Volatilité annualisée : {portfolio_stats['volatility'] * np.sqrt(252):.2%}")
        print(f"Ratio de Sharpe : {portfolio_stats['sharpe_ratio']:.2f}")
        
        print(f"\nIndicateurs de risque (niveau de confiance {config.Config.CONFIDENCE_LEVEL*100}%) :")
        print(f"  VaR historique : ${historical_var['var_value']:,.2f} ({historical_var['var']:.2%})")
        print(f"  VaR paramétrique : ${parametric_var['var_value']:,.2f} ({parametric_var['var']:.2%})")
        print(f"  VaR Monte-Carlo : ${monte_carlo_var['var_value']:,.2f} ({monte_carlo_var['var']:.2%})")
        print(f"  Déficit attendu : ${expected_shortfall['es_value']:,.2f} ({expected_shortfall['es']:.2%})")
        
        print(f"\nLes rapports et graphiques ont été enregistrés dans le répertoire 'output/'")
        
    except Exception as e:
        print(f"\nErreur : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
