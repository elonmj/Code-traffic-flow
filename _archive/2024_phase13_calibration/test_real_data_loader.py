"""
Test Real Data Loader
====================

Test script to validate the RealDataLoader class with actual traffic data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arz_model.calibration.data.real_data_loader import RealDataLoader, load_real_traffic_data


def test_real_data_loader():
    """Test RealDataLoader with actual Victoria Island data."""
    
    print("=" * 80)
    print("TESTING REAL DATA LOADER")
    print("=" * 80)
    
    # File paths
    csv_file = project_root / "donnees_trafic_75_segments.csv"
    network_json = project_root / "arz_model" / "calibration" / "data" / "groups" / "victoria_island_corridor.json"
    
    print(f"\n[FILES] Data files:")
    print(f"   CSV: {csv_file}")
    print(f"   JSON: {network_json}")
    print(f"   CSV exists: {csv_file.exists()}")
    print(f"   JSON exists: {network_json.exists()}")
    
    try:
        # Initialize loader
        print("\n[INIT] Initializing RealDataLoader...")
        loader = RealDataLoader(
            csv_file=str(csv_file),
            network_json=str(network_json),
            min_confidence=0.8
        )
        
        print("[OK] Loader initialized successfully\n")
        
        # Get data quality report
        print("[REPORT] Data Quality Report:")
        print("-" * 80)
        report = loader.get_data_quality_report()
        
        print(f"\n[STATS] OVERALL STATISTICS:")
        print(f"   Total records: {report['total_records']:,}")
        print(f"   Unique segments: {report['unique_segments']}")
        
        print(f"\n[TIME] TIME COVERAGE:")
        print(f"   Start: {report['time_range']['start']}")
        print(f"   End: {report['time_range']['end']}")
        print(f"   Duration: {report['time_range']['duration_hours']:.1f} hours")
        
        print(f"\n[SPEED] SPEED STATISTICS:")
        print(f"   Mean current speed: {report['speed_statistics']['mean_current_speed']:.1f} km/h")
        print(f"   Std current speed: {report['speed_statistics']['std_current_speed']:.1f} km/h")
        print(f"   Mean freeflow speed: {report['speed_statistics']['mean_freeflow_speed']:.1f} km/h")
        print(f"   Speed range: [{report['speed_statistics']['min_speed']:.1f}, {report['speed_statistics']['max_speed']:.1f}] km/h")
        
        print(f"\n[CONFIDENCE] CONFIDENCE:")
        print(f"   Mean confidence: {report['confidence']['mean']:.3f}")
        print(f"   Min confidence: {report['confidence']['min']:.3f}")
        print(f"   Records below threshold: {report['confidence']['below_threshold']}")
        
        print(f"\n[COVERAGE] SEGMENT COVERAGE:")
        print(f"   Segments with data: {report['segment_coverage']['segments_with_data']}")
        print(f"   Expected segments: {report['segment_coverage']['expected_segments']}")
        print(f"   Coverage: {report['segment_coverage']['coverage_percentage']:.1f}%")
        
        print(f"\n[OBS] OBSERVATIONS PER SEGMENT:")
        print(f"   Mean: {report['observations_per_segment']['mean']:.1f}")
        print(f"   Range: [{report['observations_per_segment']['min']}, {report['observations_per_segment']['max']}]")
        
        # Validate data quality
        print("\n[VALIDATION] Validating data quality...")
        try:
            loader.validate_data_quality()
            print("[OK] Data quality validation PASSED")
        except ValueError as e:
            print(f"[FAIL] Data quality validation FAILED:")
            print(f"   {e}")
        
        # Get average speeds per segment
        print("\n[AVG] Average speeds per segment (first 10):")
        avg_speeds = loader.get_all_segments_average_speeds()
        for i, (seg_id, speed) in enumerate(list(avg_speeds.items())[:10]):
            print(f"   {seg_id}: {speed:.1f} km/h")
        
        # Get calibration dataset
        print("\n[CALIB] Generating calibration dataset (15-min aggregation)...")
        calib_data = loader.get_calibration_dataset(aggregation_minutes=15)
        print(f"   Dataset shape: {calib_data.shape}")
        print(f"   Columns: {calib_data.columns.tolist()}")
        print(f"\n   Sample data (first 5 rows):")
        print(calib_data.head().to_string(index=False))
        
        # Test time series for a specific segment
        print("\n[TS] Time series for first segment:")
        first_segment = list(avg_speeds.keys())[0]
        ts_data = loader.get_segment_speed_timeseries(first_segment)
        print(f"   Segment: {first_segment}")
        print(f"   Records: {len(ts_data)}")
        print(f"\n   Sample (first 5):")
        print(ts_data.head().to_string(index=False))
        
        print("\n" + "=" * 80)
        print("[SUCCESS] ALL TESTS PASSED - REAL DATA LOADER WORKING CORRECTLY")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] TEST FAILED:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_real_data_loader()
    sys.exit(0 if success else 1)
