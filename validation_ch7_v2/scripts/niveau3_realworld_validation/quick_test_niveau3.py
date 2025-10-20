"""
Quick Test for Niveau 3: Real-World Data Validation (SPRINT 4).

Orchestrates complete validation workflow:
1. Load TomTom trajectories (or generate synthetic)
2. Extract observed metrics
3. Compare with SPRINT 3 predictions
4. Generate validation report

Usage:
------
    python quick_test_niveau3.py

Outputs:
-------
- data/processed/trajectories_niveau3.json: Processed trajectories
- data/validation_results/realworld_tests/observed_metrics.json: Observed metrics
- data/validation_results/realworld_tests/comparison_results.json: Validation comparison
- figures/niveau3_realworld/: Comparison plots (generated separately)

Expected Duration: ~10 seconds (with synthetic data)
"""

import sys
from pathlib import Path
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tomtom_trajectory_loader import TomTomTrajectoryLoader
from feature_extractor import FeatureExtractor
from validation_comparison import ValidationComparator

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main orchestration function for Niveau 3 validation."""
    
    print("\n" + "=" * 80)
    print("SPRINT 4: REAL-WORLD DATA VALIDATION (NIVEAU 3)")
    print("=" * 80)
    print("\nObjective: Validate ARZ predictions against observed TomTom data")
    print("Revendication R2: 'The ARZ model matches observed West African traffic patterns'")
    print("=" * 80)
    
    start_time = time.time()
    
    # Configuration
    input_data_path = "../../data/raw/TomTom_trajectories.csv"
    processed_traj_path = "../../data/processed/trajectories_niveau3.json"
    observed_metrics_path = "../../data/validation_results/realworld_tests/observed_metrics.json"
    comparison_results_path = "../../data/validation_results/realworld_tests/comparison_results.json"
    predicted_metrics_path = "../../SPRINT3_DELIVERABLES/results/fundamental_diagrams.json"
    
    # Step 1: Load trajectories
    print("\n" + "=" * 80)
    print("STEP 1: LOADING TRAJECTORIES")
    print("=" * 80)
    
    loader = TomTomTrajectoryLoader(input_data_path)
    trajectories = loader.load_and_parse()
    
    print(f"\n✅ Loaded {len(trajectories)} trajectory points")
    print(f"   Vehicles: {trajectories['vehicle_id'].nunique()}")
    print(f"   Classes: {trajectories['vehicle_class'].value_counts().to_dict()}")
    
    # Save processed trajectories
    loader.save_processed(trajectories, processed_traj_path)
    
    # Step 2: Extract observed metrics
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACTING OBSERVED METRICS")
    print("=" * 80)
    
    extractor = FeatureExtractor(trajectories)
    observed_metrics = extractor.extract_all_metrics()
    
    print(f"\n✅ Extracted metrics:")
    print(f"   Speed differential: {observed_metrics['speed_differential']['delta_v_kmh']:.1f} km/h")
    print(f"   Throughput ratio: {observed_metrics['throughput_ratio']['throughput_ratio']:.2f}")
    print(f"   Infiltration rate: {observed_metrics['infiltration_rate']['infiltration_rate']*100:.1f}%")
    print(f"   Segregation index: {observed_metrics['segregation_index']['segregation_index']:.2f}")
    
    # Save observed metrics
    extractor.save_metrics(observed_metrics, observed_metrics_path)
    
    # Step 3: Compare with predictions
    print("\n" + "=" * 80)
    print("STEP 3: VALIDATION COMPARISON (THEORY vs OBSERVED)")
    print("=" * 80)
    
    comparator = ValidationComparator(predicted_metrics_path, observed_metrics_path)
    comparison_results = comparator.compare_all()
    
    # Save comparison results
    comparator.save_results(comparison_results, comparison_results_path)
    
    # Step 4: Summary
    print("\n" + "=" * 80)
    print("NIVEAU 3 VALIDATION SUMMARY")
    print("=" * 80)
    
    overall = comparison_results['overall_validation']
    
    print(f"\n📊 Overall Status: {overall['status']}")
    print(f"   Tests passed: {overall['n_passed']}/{overall['n_total']} ({overall['pass_rate']*100:.0f}%)")
    print(f"\n   Revendication R2: {overall['revendication_r2']}")
    
    print(f"\n📝 Detailed Results:")
    for test_name, result in overall['tests'].items():
        status_icon = "✅" if result == "PASS" else "❌"
        print(f"   {status_icon} {test_name.replace('_', ' ').title()}: {result}")
    
    # Execution time
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total execution time: {elapsed_time:.1f}s")
    
    # Output files
    print(f"\n📁 Output Files:")
    print(f"   Trajectories: {processed_traj_path}")
    print(f"   Observed metrics: {observed_metrics_path}")
    print(f"   Comparison results: {comparison_results_path}")
    
    # Create summary JSON
    summary = {
        'sprint': 'SPRINT4_NIVEAU3',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'execution_time_s': elapsed_time,
        'data_source': loader.metadata.get('data_source', 'unknown'),
        'n_trajectories': len(trajectories),
        'n_vehicles': trajectories['vehicle_id'].nunique(),
        'validation_status': overall['status'],
        'revendication_r2': overall['revendication_r2'],
        'pass_rate': overall['pass_rate'],
        'tests': overall['tests'],
        'key_metrics': {
            'delta_v_kmh': observed_metrics['speed_differential']['delta_v_kmh'],
            'throughput_ratio': observed_metrics['throughput_ratio']['throughput_ratio'],
            'infiltration_rate': observed_metrics['infiltration_rate']['infiltration_rate'],
            'segregation_index': observed_metrics['segregation_index']['segregation_index']
        }
    }
    
    summary_path = "../../data/validation_results/realworld_tests/niveau3_summary.json"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Summary: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ NIVEAU 3 VALIDATION COMPLETE")
    print("=" * 80)
    
    # Return status code
    return 0 if overall['pass_rate'] == 1.0 else 1


if __name__ == "__main__":
    """Execute Niveau 3 validation."""
    exit_code = main()
    sys.exit(exit_code)
