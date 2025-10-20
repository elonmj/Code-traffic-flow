"""
SPRINT 4: Real-World Data Validation with Lagos Traffic Data

This script performs validation comparison using REAL Lagos TomTom traffic observations
instead of synthetic ARZ-generated trajectories.

Data Source:
-----------
- Real observations: observed_metrics_REAL.json (4,270 Lagos traffic observations)
- Predicted metrics: SPRINT 3 ARZ model predictions

Validation Tests:
----------------
1. Speed differential: |Δv_real - Δv_pred| / Δv_pred < 10%
2. Throughput ratio: |ratio_real - ratio_pred| / ratio_pred < 15%
3. Fundamental diagram correlation: Spearman ρ > 0.7
4. Infiltration rate: Within expected range (50-80%)

Usage:
------
    python validate_with_real_data.py
"""

import json
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from validation_comparison import ValidationComparator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Execute real-world validation with Lagos traffic data."""
    
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    
    # Use REAL Lagos observations (not synthetic ARZ)
    observed_real_path = base_dir / "data" / "validation_results" / "realworld_tests" / "observed_metrics_REAL.json"
    
    # Predicted metrics from SPRINT 3 ARZ model
    predicted_path = base_dir / "SPRINT3_DELIVERABLES" / "results" / "fundamental_diagrams.json"
    
    # Output paths
    output_dir = base_dir / "data" / "validation_results" / "realworld_tests"
    comparison_output = output_dir / "comparison_results_REAL.json"
    summary_output = output_dir / "niveau3_summary_REAL.json"
    
    # Print header
    print("\n" + "=" * 80)
    print("SPRINT 4: REAL-WORLD DATA VALIDATION WITH LAGOS TRAFFIC")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Verify real data exists
    if not observed_real_path.exists():
        logger.error(f"❌ REAL data not found: {observed_real_path}")
        logger.error("   Please run real_data_adapter.py first to generate observed_metrics_REAL.json")
        return 1
    
    # Load real data to show summary
    logger.info("📊 REAL LAGOS TRAFFIC DATA:")
    with open(observed_real_path, 'r', encoding='utf-8') as f:
        real_data = json.load(f)
    
    logger.info(f"   Source: donnees_trafic_75_segments (2).csv")
    logger.info(f"   Location: Lagos, Nigeria")
    logger.info(f"   Time range: {real_data.get('metadata', {}).get('time_range', 'N/A')}")
    logger.info(f"   Total observations: {real_data.get('metadata', {}).get('total_observations', 'N/A')}")
    logger.info(f"   Motorcycles: {real_data.get('metadata', {}).get('motorcycle_count', 'N/A')}")
    logger.info(f"   Cars: {real_data.get('metadata', {}).get('car_count', 'N/A')}")
    
    # Key metrics preview
    logger.info("\n📈 KEY REAL METRICS:")
    if 'speed_differential' in real_data:
        sd = real_data['speed_differential']
        logger.info(f"   Speed differential: Δv = {sd.get('delta_v_kmh', 'N/A'):.1f} km/h")
        logger.info(f"      Motorcycles: {sd.get('motos_mean_kmh', 'N/A'):.1f} ± {sd.get('motos_std_kmh', 'N/A'):.1f} km/h")
        logger.info(f"      Cars: {sd.get('cars_mean_kmh', 'N/A'):.1f} ± {sd.get('cars_std_kmh', 'N/A'):.1f} km/h")
    
    if 'throughput_ratio' in real_data:
        tr = real_data['throughput_ratio']
        logger.info(f"\n   Throughput ratio: Q_m/Q_c = {tr.get('throughput_ratio', 'N/A'):.2f}")
        logger.info(f"      Q_motos = {tr.get('Q_motos_veh_per_h', 'N/A'):.0f} veh/h")
        logger.info(f"      Q_cars = {tr.get('Q_cars_veh_per_h', 'N/A'):.0f} veh/h")
    
    if 'infiltration_rate' in real_data:
        ir = real_data['infiltration_rate']
        logger.info(f"\n   Infiltration rate: {ir.get('infiltration_rate', 'N/A'):.1%}")
        logger.info(f"      Car-dominated segments: {ir.get('car_dominated_segments', 'N/A')}")
    
    if 'segregation_index' in real_data:
        si = real_data['segregation_index']
        logger.info(f"\n   Segregation index: {si.get('segregation_index', 'N/A'):.3f}")
        logger.info(f"      Position separation: ≈ {si.get('position_separation_m', 'N/A'):.0f} m")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPARISON: ARZ PREDICTIONS vs REAL LAGOS DATA")
    print("=" * 80 + "\n")
    
    # Initialize comparator with REAL data
    comparator = ValidationComparator(
        predicted_metrics_path=str(predicted_path),
        observed_metrics_path=str(observed_real_path)  # Use REAL data
    )
    
    # Perform all comparisons
    results = comparator.compare_all()
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    comparator.save_results(results, str(comparison_output))
    
    logger.info(f"\n✅ Saved comparison results to: {comparison_output}")
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_source': 'REAL Lagos TomTom Traffic Data',
        'validation_status': results['overall_validation']['status'],
        'tests_passed': f"{results['overall_validation']['n_passed']}/{results['overall_validation']['n_total']}",
        'revendication_R2': 'VALIDATED' if results['overall_validation']['status'] == '✅ PASS' else 'NOT VALIDATED',
        'individual_tests': {
            'speed_differential': results['speed_differential']['interpretation'],
            'throughput_ratio': results['throughput_ratio']['interpretation'],
            'fundamental_diagrams': results['fundamental_diagrams']['interpretation'],
            'infiltration_rate': results['infiltration_rate']['interpretation']
        },
        'key_findings': {
            'speed_differential_error_pct': results['speed_differential'].get('relative_error', 'N/A') * 100 if isinstance(results['speed_differential'].get('relative_error'), (int, float)) else 'N/A',
            'throughput_ratio_error_pct': results['throughput_ratio'].get('relative_error', 'N/A') * 100 if isinstance(results['throughput_ratio'].get('relative_error'), (int, float)) else 'N/A',
            'fd_correlation_avg': results['fundamental_diagrams'].get('average_correlation', 'N/A'),
            'infiltration_rate_observed': results['infiltration_rate'].get('observed_rate', 'N/A') * 100 if isinstance(results['infiltration_rate'].get('observed_rate'), (int, float)) else 'N/A'
        },
        'data_quality': {
            'total_observations': real_data.get('metadata', {}).get('total_observations', 'N/A'),
            'time_range': real_data.get('metadata', {}).get('time_range', 'N/A'),
            'unique_segments': real_data.get('metadata', {}).get('unique_segments', 'N/A')
        }
    }
    
    with open(summary_output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Saved summary to: {summary_output}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n📊 Overall Status: {results['overall_validation']['status']}")
    print(f"   Tests passed: {results['overall_validation']['n_passed']}/{results['overall_validation']['n_total']}")
    print(f"\n   Revendication R2: {summary['revendication_R2']} {'✅' if summary['revendication_R2'] == 'VALIDATED' else '❌'}")
    
    print("\n📝 Individual Test Results:")
    for test_name, test_result in summary['individual_tests'].items():
        emoji = "✅" if "PASS" in test_result else "❌"
        print(f"   {emoji} {test_name.replace('_', ' ').title()}: {test_result}")
    
    print("\n📁 Output Files:")
    print(f"   Comparison: {comparison_output.relative_to(base_dir)}")
    print(f"   Summary: {summary_output.relative_to(base_dir)}")
    
    print("\n" + "=" * 80)
    print("✅ REAL-WORLD VALIDATION COMPLETE")
    print("=" * 80 + "\n")
    
    return 0 if results['overall_validation']['status'] == '✅ PASS' else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
