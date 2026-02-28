# tests/test_var_calculator.py
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from var_calculator import VaRCalculator

class TestVaRCalculator(unittest.TestCase):
    
    def setUp(self):
        """Configure les données de test"""
        np.random.seed(42)
        self.returns = pd.DataFrame({
            'Asset1': np.random.normal(0.001, 0.02, 1000),
            'Asset2': np.random.normal(0.0005, 0.015, 1000)
        })
        self.weights = np.array([0.6, 0.4])
        self.portfolio_value = 1000000
        self.var_calculator = VaRCalculator(confidence_level=0.95)
    
    def test_historical_var(self):
        """Teste le calcul de la VaR historique"""
        result = self.var_calculator.historical_var(
            self.returns, self.weights, self.portfolio_value
        )
        
        self.assertIn('var', result)
        self.assertIn('var_value', result)
        self.assertGreater(result['var_value'], 0)
    
    def test_parametric_var(self):
        """Teste le calcul de la VaR paramétrique"""
        result = self.var_calculator.parametric_var(
            self.returns, self.weights, self.portfolio_value
        )
        
        self.assertIn('var', result)
        self.assertIn('var_value', result)
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertGreater(result['var_value'], 0)
    
    def test_expected_shortfall(self):
        """Teste le calcul de l'Expected Shortfall"""
        historical_result = self.var_calculator.historical_var(
            self.returns, self.weights, self.portfolio_value
        )
        
        es_result = self.var_calculator.calculate_expected_shortfall(
            historical_result['portfolio_returns'], self.portfolio_value
        )
        
        self.assertIn('es', es_result)
        self.assertIn('es_value', es_result)
        # ES devrait être supérieur ou égal à la VaR
        self.assertGreaterEqual(es_result['es_value'], historical_result['var_value'])

if __name__ == '__main__':
    unittest.main()
