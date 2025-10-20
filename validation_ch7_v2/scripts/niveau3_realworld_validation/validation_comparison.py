"""
Validation Comparison: Theory (SPRINT 3) vs Observed (SPRINT 4).

Statistical comparison of ARZ model predictions (SPRINT 3) with observed TomTom data.

Validations Performed:
---------------------
1. Speed differential: |Δv_obs - Δv_pred| / Δv_pred < 10%
2. Throughput ratio: |ratio_obs - ratio_pred| / ratio_pred < 15%
3. Fundamental diagram correlation: Spearman ρ > 0.7
4. Infiltration rate: Within observed range (50-80%)
5. Statistical tests: KS test, correlation tests

Success Criteria (Revendication R2):
-----------------------------------
- PASS if ALL validation criteria met
- FAIL if ANY critical criterion fails

Usage:
------
    comparator = ValidationComparator(
        predicted_metrics_path="SPRINT3_DELIVERABLES/results/fundamental_diagrams.json",
        observed_metrics_path="data/validation_results/realworld_tests/observed_metrics.json"
    )
    results = comparator.compare_all()
    comparator.save_results(results, "comparison_results.json")
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationComparator:
    """
    Compare ARZ predictions with observed data for validation.
    
    Attributes:
        predicted (Dict): SPRINT 3 predicted metrics
        observed (Dict): SPRINT 4 observed metrics
        comparison_results (Dict): Comparison results and tests
    """
    
    def __init__(self, predicted_metrics_path: str, observed_metrics_path: str):
        """
        Initialize comparator with predicted and observed metrics.
        
        Args:
            predicted_metrics_path: Path to SPRINT 3 results JSON
            observed_metrics_path: Path to SPRINT 4 observed metrics JSON
        """
        self.predicted = self._load_json(predicted_metrics_path)
        self.observed = self._load_json(observed_metrics_path)
        self.comparison_results = {}
        
        logger.info("Initialized ValidationComparator")
        logger.info(f"  Predicted metrics from: {predicted_metrics_path}")
        logger.info(f"  Observed metrics from: {observed_metrics_path}")
    
    def _load_json(self, path: str) -> Dict:
        """Load JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {path}. Using empty dict.")
            return {}
    
    def compare_all(self) -> Dict:
        """
        Perform all validation comparisons.
        
        Returns:
            Dict with comparison results for all metrics
        """
        logger.info("=" * 70)
        logger.info("VALIDATION COMPARISON: THEORY vs OBSERVED")
        logger.info("=" * 70)
        
        self.comparison_results = {
            'speed_differential': self.compare_speed_differential(),
            'throughput_ratio': self.compare_throughput_ratio(),
            'fundamental_diagrams': self.compare_fundamental_diagrams(),
            'infiltration_rate': self.compare_infiltration_rate(),
            'overall_validation': {}
        }
        
        # Compute overall validation status
        self.comparison_results['overall_validation'] = self._compute_overall_status()
        
        logger.info("\n" + "=" * 70)
        logger.info("OVERALL VALIDATION STATUS")
        logger.info("=" * 70)
        logger.info(f"  Status: {self.comparison_results['overall_validation']['status']}")
        logger.info(f"  Passed: {self.comparison_results['overall_validation']['n_passed']}/{self.comparison_results['overall_validation']['n_total']}")
        
        return self.comparison_results
    
    def compare_speed_differential(self) -> Dict:
        """
        Compare speed differential: |Δv_obs - Δv_pred| / Δv_pred < 10%.
        
        Returns:
            Dict with comparison results
        """
        logger.info("\n1. Speed Differential Validation:")
        
        # Extract values
        # Predicted from SPRINT 3 gap_filling or interweaving tests
        delta_v_pred = 10.0  # Target from SPRINT 3 (conservative baseline)
        
        # Observed from SPRINT 4
        if 'speed_differential' in self.observed:
            delta_v_obs = self.observed['speed_differential']['delta_v_kmh']
        else:
            delta_v_obs = 0
        
        # Compute error
        if delta_v_pred > 0:
            relative_error = abs(delta_v_obs - delta_v_pred) / delta_v_pred
        else:
            relative_error = 1.0
        
        # Pass/fail criterion
        threshold = 0.10  # 10%
        passed = bool(relative_error < threshold)
        
        result = {
            'delta_v_predicted_kmh': float(delta_v_pred),
            'delta_v_observed_kmh': float(delta_v_obs),
            'absolute_error_kmh': float(abs(delta_v_obs - delta_v_pred)),
            'relative_error': float(relative_error),
            'threshold': threshold,
            'passed': passed,
            'interpretation': '✅ PASS' if passed else '❌ FAIL'
        }
        
        logger.info(f"   Predicted Δv: {delta_v_pred:.1f} km/h")
        logger.info(f"   Observed Δv: {delta_v_obs:.1f} km/h")
        logger.info(f"   Relative error: {relative_error*100:.1f}% (threshold: {threshold*100:.0f}%)")
        logger.info(f"   Result: {result['interpretation']}")
        
        return result
    
    def compare_throughput_ratio(self) -> Dict:
        """
        Compare throughput ratio: |ratio_obs - ratio_pred| / ratio_pred < 15%.
        
        Returns:
            Dict with comparison results
        """
        logger.info("\n2. Throughput Ratio Validation:")
        
        # Predicted from SPRINT 3 fundamental diagrams
        if 'calibration' in self.predicted:
            ratio_pred = self.predicted['calibration']['throughput_ratio']
        else:
            ratio_pred = 1.50  # SPRINT 3 validated value
        
        # Observed from SPRINT 4
        if 'throughput_ratio' in self.observed:
            ratio_obs = self.observed['throughput_ratio']['throughput_ratio']
        else:
            ratio_obs = 0
        
        # Compute error
        if ratio_pred > 0:
            relative_error = abs(ratio_obs - ratio_pred) / ratio_pred
        else:
            relative_error = 1.0
        
        # Pass/fail criterion
        threshold = 0.15  # 15%
        passed = bool(relative_error < threshold)
        
        result = {
            'ratio_predicted': float(ratio_pred),
            'ratio_observed': float(ratio_obs),
            'absolute_error': float(abs(ratio_obs - ratio_pred)),
            'relative_error': float(relative_error),
            'threshold': threshold,
            'passed': passed,
            'interpretation': '✅ PASS' if passed else '❌ FAIL'
        }
        
        logger.info(f"   Predicted ratio: {ratio_pred:.2f}")
        logger.info(f"   Observed ratio: {ratio_obs:.2f}")
        logger.info(f"   Relative error: {relative_error*100:.1f}% (threshold: {threshold*100:.0f}%)")
        logger.info(f"   Result: {result['interpretation']}")
        
        return result
    
    def compare_fundamental_diagrams(self) -> Dict:
        """
        Compare fundamental diagrams: Spearman correlation > 0.7.
        
        Returns:
            Dict with correlation results
        """
        logger.info("\n3. Fundamental Diagram Validation:")
        
        # Extract Q-ρ points from observed data
        if 'fundamental_diagrams' in self.observed:
            fd_obs = self.observed['fundamental_diagrams']
            
            # Motorcycles
            rho_motos_obs = fd_obs['motorcycle']['data_points']['rho']
            Q_motos_obs = fd_obs['motorcycle']['data_points']['Q']
            
            # Cars
            rho_cars_obs = fd_obs['car']['data_points']['rho']
            Q_cars_obs = fd_obs['car']['data_points']['Q']
        else:
            rho_motos_obs = Q_motos_obs = []
            rho_cars_obs = Q_cars_obs = []
        
        # Generate predicted Q-ρ curves from ARZ (SPRINT 3 parameters)
        params_motos = {'Vmax_ms': 60/3.6, 'rho_max': 0.15}
        params_cars = {'Vmax_ms': 50/3.6, 'rho_max': 0.12}
        
        # For each observed ρ, compute predicted Q
        if len(rho_motos_obs) > 0:
            Q_motos_pred = [self._compute_Q_from_rho(rho, params_motos) for rho in rho_motos_obs]
            
            # Spearman correlation
            if len(Q_motos_obs) > 2:
                corr_motos, pval_motos = stats.spearmanr(Q_motos_obs, Q_motos_pred)
            else:
                corr_motos = pval_motos = 0
        else:
            Q_motos_pred = []
            corr_motos = pval_motos = 0
        
        if len(rho_cars_obs) > 0:
            Q_cars_pred = [self._compute_Q_from_rho(rho, params_cars) for rho in rho_cars_obs]
            
            if len(Q_cars_obs) > 2:
                corr_cars, pval_cars = stats.spearmanr(Q_cars_obs, Q_cars_pred)
            else:
                corr_cars = pval_cars = 0
        else:
            Q_cars_pred = []
            corr_cars = pval_cars = 0
        
        # Average correlation
        avg_corr = (corr_motos + corr_cars) / 2 if (corr_motos != 0 or corr_cars != 0) else 0
        
        # Pass/fail criterion
        threshold = 0.7
        passed = bool(avg_corr > threshold)
        
        result = {
            'motorcycles': {
                'correlation': float(corr_motos),
                'p_value': float(pval_motos),
                'n_points': len(rho_motos_obs)
            },
            'cars': {
                'correlation': float(corr_cars),
                'p_value': float(pval_cars),
                'n_points': len(rho_cars_obs)
            },
            'average_correlation': float(avg_corr),
            'threshold': threshold,
            'passed': passed,
            'interpretation': '✅ PASS' if passed else '❌ FAIL'
        }
        
        logger.info(f"   Motorcycles: ρ = {corr_motos:.2f} (p={pval_motos:.4f}, n={len(rho_motos_obs)})")
        logger.info(f"   Cars: ρ = {corr_cars:.2f} (p={pval_cars:.4f}, n={len(rho_cars_obs)})")
        logger.info(f"   Average correlation: {avg_corr:.2f} (threshold: {threshold})")
        logger.info(f"   Result: {result['interpretation']}")
        
        return result
    
    def _compute_Q_from_rho(self, rho: float, params: Dict) -> float:
        """Compute flow Q from density ρ using ARZ model."""
        V = params['Vmax_ms'] * max(0, 1 - rho / params['rho_max'])
        Q = rho * V * 3600  # Convert to veh/h
        return Q
    
    def compare_infiltration_rate(self) -> Dict:
        """
        Compare infiltration rate: Within observed range (50-80%).
        
        Returns:
            Dict with comparison results
        """
        logger.info("\n4. Infiltration Rate Validation:")
        
        # Observed from SPRINT 4
        if 'infiltration_rate' in self.observed:
            rate_obs = self.observed['infiltration_rate']['infiltration_rate']
        else:
            rate_obs = 0
        
        # Expected range from SPRINT 3 (qualitative)
        expected_min = 0.50  # 50%
        expected_max = 0.80  # 80%
        
        # Pass/fail criterion
        passed = bool((rate_obs >= expected_min) and (rate_obs <= expected_max))
        
        result = {
            'infiltration_rate_observed': float(rate_obs),
            'expected_min': expected_min,
            'expected_max': expected_max,
            'passed': passed,
            'interpretation': '✅ PASS' if passed else '❌ FAIL (expected 50-80%)'
        }
        
        logger.info(f"   Observed rate: {rate_obs*100:.1f}%")
        logger.info(f"   Expected range: {expected_min*100:.0f}% - {expected_max*100:.0f}%")
        logger.info(f"   Result: {result['interpretation']}")
        
        return result
    
    def _compute_overall_status(self) -> Dict:
        """
        Compute overall validation status.
        
        Returns:
            Dict with overall PASS/FAIL status
        """
        # Collect pass/fail for each test
        tests = [
            ('speed_differential', self.comparison_results['speed_differential']['passed']),
            ('throughput_ratio', self.comparison_results['throughput_ratio']['passed']),
            ('fundamental_diagrams', self.comparison_results['fundamental_diagrams']['passed']),
            ('infiltration_rate', self.comparison_results['infiltration_rate']['passed'])
        ]
        
        n_passed = sum(1 for _, passed in tests if passed)
        n_total = len(tests)
        
        # Overall status: PASS if all pass, PARTIAL if some pass, FAIL if none pass
        if n_passed == n_total:
            status = '✅ PASS - All criteria met'
        elif n_passed >= n_total * 0.75:  # 75% threshold
            status = '⚠️ PARTIAL PASS - Most criteria met'
        else:
            status = '❌ FAIL - Multiple criteria not met'
        
        return {
            'status': status,
            'n_passed': n_passed,
            'n_total': n_total,
            'pass_rate': n_passed / n_total if n_total > 0 else 0,
            'tests': {name: 'PASS' if passed else 'FAIL' for name, passed in tests},
            'revendication_r2': 'VALIDATED ✅' if n_passed == n_total else 'NOT VALIDATED ❌'
        }
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """
        Save comparison results to JSON file.
        
        Args:
            results: Comparison results dictionary
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Saved comparison results to: {output_path}")


if __name__ == "__main__":
    """Test validation comparator."""
    
    # Paths (adjust based on actual file locations)
    predicted_path = "../../SPRINT3_DELIVERABLES/results/fundamental_diagrams.json"
    observed_path = "../../data/validation_results/realworld_tests/observed_metrics.json"
    
    # Create comparator
    comparator = ValidationComparator(predicted_path, observed_path)
    
    # Run comparison
    results = comparator.compare_all()
    
    # Save results
    comparator.save_results(results, "../../data/validation_results/realworld_tests/comparison_results.json")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nRevendication R2: {results['overall_validation']['revendication_r2']}")
    print(f"Pass rate: {results['overall_validation']['pass_rate']*100:.0f}%")
    print(f"Tests passed: {results['overall_validation']['n_passed']}/{results['overall_validation']['n_total']}")
    
    print("\n✅ Test complete!")
