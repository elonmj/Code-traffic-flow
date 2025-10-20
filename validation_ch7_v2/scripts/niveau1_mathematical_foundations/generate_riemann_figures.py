"""
Generate All Riemann Test Figures.

This script orchestrates the generation of all Riemann test figures
and consolidates results for LaTeX integration.

Outputs:
-------
1. Individual test figures (PDF):
   - test1_shock_motos.pdf
   - test2_rarefaction_motos.pdf
   - test3_shock_voitures.pdf
   - test4_rarefaction_voitures.pdf
   - test5_multiclass_interaction.pdf
   - convergence_study_weno5.pdf

2. Summary figure (6-panel composite)

3. LaTeX table data (JSON)

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List

# Import test modules
from scripts.niveau1_mathematical_foundations import (
    test_riemann_motos_shock,
    test_riemann_motos_rarefaction,
    test_riemann_voitures_shock,
    test_riemann_voitures_rarefaction,
    test_riemann_multiclass,
    convergence_study
)


def run_all_tests(save_individual: bool = True) -> Dict:
    """
    Run all 5 Riemann tests + convergence study.
    
    Returns:
        Dictionary with all results
    """
    print("=" * 80)
    print("RIEMANN TEST SUITE - FULL EXECUTION")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Shock motos
    print("\n" + "▶" * 40)
    results['test1'] = test_riemann_motos_shock.run_test(save_results=save_individual)
    
    # Test 2: Rarefaction motos
    print("\n" + "▶" * 40)
    results['test2'] = test_riemann_motos_rarefaction.run_test(save_results=save_individual)
    
    # Test 3: Shock voitures
    print("\n" + "▶" * 40)
    results['test3'] = test_riemann_voitures_shock.run_test(save_results=save_individual)
    
    # Test 4: Rarefaction voitures
    print("\n" + "▶" * 40)
    results['test4'] = test_riemann_voitures_rarefaction.run_test(save_results=save_individual)
    
    # Test 5: Multiclass (CRITICAL)
    print("\n" + "▶" * 40)
    results['test5'] = test_riemann_multiclass.run_test(save_results=save_individual)
    
    # Convergence study
    print("\n" + "▶" * 40)
    results['convergence'] = convergence_study.run_convergence_study(save_results=save_individual)
    
    return results


def generate_summary_table(results: Dict) -> Dict:
    """
    Generate LaTeX table data from results.
    
    Returns:
        Dictionary formatted for LaTeX table
    """
    table_data = {
        'niveau1_riemann_tests': [
            {
                'test_name': 'Test 1: Shock (Motos)',
                'L2_error': f"{results['test1']['L2_error']:.2e}",
                'status': '✅' if results['test1']['test_passed'] else '❌',
                'wave_type': 'Shock'
            },
            {
                'test_name': 'Test 2: Rarefaction (Motos)',
                'L2_error': f"{results['test2']['L2_error']:.2e}",
                'status': '✅' if results['test2']['test_passed'] else '❌',
                'wave_type': 'Rarefaction'
            },
            {
                'test_name': 'Test 3: Shock (Voitures)',
                'L2_error': f"{results['test3']['L2_error']:.2e}",
                'status': '✅' if results['test3']['test_passed'] else '❌',
                'wave_type': 'Shock'
            },
            {
                'test_name': 'Test 4: Rarefaction (Voitures)',
                'L2_error': f"{results['test4']['L2_error']:.2e}",
                'status': '✅' if results['test4']['test_passed'] else '❌',
                'wave_type': 'Rarefaction'
            },
            {
                'test_name': 'Test 5: Multiclass ⭐',
                'L2_error': f"{results['test5']['L2_error_average']:.2e}",
                'status': '✅' if results['test5']['test_passed'] else '❌',
                'wave_type': 'Coupled',
                'coupling': results['test5']['coupling_coefficient']
            }
        ],
        'convergence_study': {
            'average_order': f"{results['convergence']['average_order']:.2f}",
            'theoretical_order': '5.0',
            'target_order': '4.5',
            'status': '✅' if results['convergence']['order_passed'] else '❌'
        },
        'summary': {
            'total_tests': 5,
            'passed': sum([1 for k in ['test1', 'test2', 'test3', 'test4', 'test5'] if results[k]['test_passed']]),
            'convergence_validated': results['convergence']['order_passed']
        }
    }
    
    return table_data


def main():
    """Main execution."""
    print("\n🚀 Starting Riemann Test Suite Generation...")
    
    # Run all tests
    results = run_all_tests(save_individual=True)
    
    # Generate table data
    table_data = generate_summary_table(results)
    
    # Save consolidated results
    results_dir = project_root / "data" / "validation_results" / "riemann_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = results_dir / "niveau1_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'raw_results': results,
            'latex_table_data': table_data
        }, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 NIVEAU 1 VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nRiemann Tests: {table_data['summary']['passed']}/{table_data['summary']['total_tests']} passed")
    print(f"Convergence Study: {'✅ PASSED' if table_data['summary']['convergence_validated'] else '❌ FAILED'}")
    print(f"\nConvergence Order: {table_data['convergence_study']['average_order']} (target: ≥4.5)")
    
    print(f"\n📄 Results saved:")
    print(f"  - Individual figures: figures/niveau1_riemann/*.pdf")
    print(f"  - Summary: {summary_path}")
    
    # Overall status
    all_passed = (table_data['summary']['passed'] == 5) and table_data['summary']['convergence_validated']
    
    if all_passed:
        print("\n" + "🎉" * 40)
        print("✅ NIVEAU 1 VALIDATION: COMPLETE SUCCESS")
        print("   R3 (FVM+WENO5 accuracy) fully validated!")
        print("🎉" * 40)
        return 0
    else:
        print("\n❌ NIVEAU 1 VALIDATION: SOME TESTS FAILED")
        print("   Review individual test logs above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
