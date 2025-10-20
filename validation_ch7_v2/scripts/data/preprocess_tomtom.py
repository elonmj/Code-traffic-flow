"""
Main Preprocessing Orchestrator for TomTom Data.

This script orchestrates the complete preprocessing pipeline:
1. Load raw TomTom data
2. Apply multiclass calibration (vehicle class inference)
3. Generate rush hour demand
4. Construct network topology
5. Export preprocessed data for UXsim

Usage:
    python preprocess_tomtom.py --input donnees_trafic_75_segments.csv --output preprocessed/

Author: ARZ-RL Validation Team
Date: 2025-01-17
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json

# Add parent directory to path to import preprocessing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.vehicle_class_rules import apply_multiclass_calibration, validate_class_split
from preprocessing.temporal_augmentation import generate_rush_hour_demand, validate_temporal_consistency
from preprocessing.network_topology import construct_network_from_tomtom, validate_network_topology, export_to_uxsim_format


def preprocess_tomtom_data(
    input_csv: str,
    output_dir: str,
    peak_factor: float = 2.5,
    random_seed: int = 42
) -> None:
    """
    Complete preprocessing pipeline for TomTom data.
    
    Pipeline Steps:
    --------------
    1. Load raw TomTom CSV
    2. Apply multiclass calibration (motos/voitures inference)
    3. Generate rush hour demand from midday baseline
    4. Construct network topology for UXsim
    5. Validate all outputs
    6. Export preprocessed data
    
    Args:
        input_csv: Path to raw TomTom CSV
        output_dir: Directory for preprocessed outputs
        peak_factor: Rush hour peak factor (default 2.5x)
        random_seed: Random seed for reproducibility (default 42)
    
    Outputs Created:
    ---------------
    - {output_dir}/calibrated_multiclass.csv: Augmented with vehicle classes
    - {output_dir}/rush_hour_demand.csv: Synthetic rush hour time series
    - {output_dir}/rush_hour_demand.json: Demand metadata
    - {output_dir}/network_topology.csv: UXsim network format
    - {output_dir}/network_metadata.json: Network statistics
    - {output_dir}/preprocessing_report.txt: Validation report
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("TOMTOM DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"\n📂 Input: {input_csv}")
    print(f"📂 Output: {output_dir}")
    print(f"⚙️  Peak factor: {peak_factor}x")
    print(f"🎲 Random seed: {random_seed}")
    
    # Step 1: Load raw data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING RAW TOMTOM DATA")
    print("=" * 70)
    
    try:
        # Read CSV with error handling for malformed lines
        df = pd.read_csv(input_csv, on_bad_lines='skip')
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        # Handle different datetime column names
        datetime_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
        print(f"   Date range: {df[datetime_col].min()} → {df[datetime_col].max()}")
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return
    
    # Step 2: Multiclass calibration
    print("\n" + "=" * 70)
    print("STEP 2: MULTICLASS CALIBRATION (MOTOS/VOITURES)")
    print("=" * 70)
    
    try:
        # Normalize column names (name -> street)
        if 'name' in df.columns and 'street' not in df.columns:
            df = df.rename(columns={'name': 'street'})
        
        df_calibrated = apply_multiclass_calibration(df)
        print(f"✅ Applied multiclass calibration")
        print(f"   New columns added: {[col for col in df_calibrated.columns if col not in df.columns]}")
        
        # Validate
        validation = validate_class_split(df_calibrated)
        print(f"\n📊 Validation:")
        print(f"   Valid: {'✅' if validation['valid'] else '❌'}")
        for check, passed in validation['checks'].items():
            status = '✅' if passed else '❌'
            print(f"     {status} {check}")
        
        # Export
        output_file = output_path / "calibrated_multiclass.csv"
        df_calibrated.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
        
    except Exception as e:
        print(f"❌ ERROR in multiclass calibration: {e}")
        return
    
    # Step 3: Rush hour demand generation
    print("\n" + "=" * 70)
    print("STEP 3: RUSH HOUR DEMAND GENERATION")
    print("=" * 70)
    
    try:
        # Compute average demand from calibrated data
        avg_speed = df_calibrated['current_speed'].mean()
        avg_flow_estimate = avg_speed * df_calibrated['class_split_motos'].mean()  # Simplified
        
        calibrated_demand = {
            'avg_vehicles_per_hour': avg_flow_estimate * 100,  # Scale to realistic values
            'source': 'TomTom midday calibration'
        }
        
        print(f"   Base midday demand: {calibrated_demand['avg_vehicles_per_hour']:.0f} veh/h")
        
        # Generate rush hour
        rush_demand = generate_rush_hour_demand(
            calibrated_demand,
            peak_factor=peak_factor,
            random_seed=random_seed
        )
        
        print(f"✅ Generated rush hour demand")
        print(f"   Time window: {rush_demand['time_window']}")
        print(f"   Peak demand: {rush_demand['peak_demand_rush']:.0f} veh/h")
        print(f"   Multiplier: {rush_demand['statistics']['peak_multiplier']:.2f}x")
        
        # Validate
        network_capacity = 3000  # Victoria Island estimate
        validation = validate_temporal_consistency(rush_demand, network_capacity)
        print(f"\n📊 Validation:")
        print(f"   Valid: {'✅' if validation['valid'] else '❌'}")
        for check, passed in validation['checks'].items():
            status = '✅' if passed else '❌'
            print(f"     {status} {check}")
        
        # Export time series
        output_file_csv = output_path / "rush_hour_demand.csv"
        df_rush = pd.DataFrame(rush_demand['time_series'])
        df_rush.to_csv(output_file_csv, index=False)
        print(f"\n💾 Saved: {output_file_csv}")
        
        # Export metadata
        output_file_json = output_path / "rush_hour_demand.json"
        with open(output_file_json, 'w') as f:
            json.dump({
                'time_window': rush_demand['time_window'],
                'base_demand_midday': rush_demand['base_demand_midday'],
                'peak_demand_rush': rush_demand['peak_demand_rush'],
                'justification': rush_demand['justification'],
                'statistics': rush_demand['statistics'],
                'metadata': rush_demand['metadata']
            }, f, indent=2)
        print(f"💾 Saved: {output_file_json}")
        
    except Exception as e:
        print(f"❌ ERROR in rush hour generation: {e}")
        return
    
    # Step 4: Network topology construction
    print("\n" + "=" * 70)
    print("STEP 4: NETWORK TOPOLOGY CONSTRUCTION")
    print("=" * 70)
    
    try:
        network = construct_network_from_tomtom(df_calibrated)
        
        print(f"✅ Constructed network topology")
        print(f"   Segments: {network.metadata['n_segments']}")
        print(f"   Nodes: {network.metadata['n_nodes']}")
        print(f"   Total length: {network.metadata['total_length_m']:.0f} m")
        print(f"   Avg capacity: {network.metadata['avg_capacity_veh_per_hour']:.0f} veh/h")
        print(f"   Total capacity: {network.metadata['total_network_capacity']:.0f} veh/h")
        
        # Validate
        validation = validate_network_topology(network)
        print(f"\n📊 Validation:")
        print(f"   Valid: {'✅' if validation['valid'] else '❌'}")
        for check, passed in validation['checks'].items():
            status = '✅' if passed else '❌'
            print(f"     {status} {check}")
        
        if validation['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"     - {warning}")
        
        # Export
        output_file = output_path / "network_topology.csv"
        export_to_uxsim_format(network, str(output_file))
        
        # Export metadata
        output_file_json = output_path / "network_metadata.json"
        with open(output_file_json, 'w') as f:
            json.dump(network.metadata, f, indent=2)
        print(f"💾 Saved: {output_file_json}")
        
    except Exception as e:
        print(f"❌ ERROR in network construction: {e}")
        return
    
    # Step 5: Generate preprocessing report
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING PREPROCESSING REPORT")
    print("=" * 70)
    
    report_lines = [
        "=" * 70,
        "TOMTOM DATA PREPROCESSING REPORT",
        "=" * 70,
        "",
        f"Input: {input_csv}",
        f"Output: {output_dir}",
        f"Date: {pd.Timestamp.now()}",
        "",
        "=" * 70,
        "1. MULTICLASS CALIBRATION",
        "=" * 70,
        f"Rows processed: {len(df_calibrated)}",
        f"Motos fraction (mean): {df_calibrated['class_split_motos'].mean():.2%}",
        f"Voitures fraction (mean): {df_calibrated['class_split_voitures'].mean():.2%}",
        f"Speed differential (mean): {(df_calibrated['speed_motos'] / df_calibrated['speed_voitures']).mean():.2f}x",
        "",
        "=" * 70,
        "2. RUSH HOUR DEMAND",
        "=" * 70,
        f"Time window: {rush_demand['time_window']}",
        f"Base midday demand: {rush_demand['base_demand_midday']:.0f} veh/h",
        f"Peak rush demand: {rush_demand['peak_demand_rush']:.0f} veh/h",
        f"Peak multiplier: {rush_demand['statistics']['peak_multiplier']:.2f}x",
        f"Peak factor: {peak_factor}x",
        "",
        "=" * 70,
        "3. NETWORK TOPOLOGY",
        "=" * 70,
        f"Segments: {network.metadata['n_segments']}",
        f"Nodes: {network.metadata['n_nodes']}",
        f"Total length: {network.metadata['total_length_m']:.0f} m",
        f"Average lanes: {network.metadata['avg_lanes']:.1f}",
        f"Total capacity: {network.metadata['total_network_capacity']:.0f} veh/h",
        "",
        "Street distribution:",
    ]
    
    for street, count in network.metadata['street_distribution'].items():
        report_lines.append(f"  - {street}: {count} segments")
    
    report_lines.extend([
        "",
        "=" * 70,
        "OUTPUT FILES",
        "=" * 70,
        f"✅ {output_path / 'calibrated_multiclass.csv'}",
        f"✅ {output_path / 'rush_hour_demand.csv'}",
        f"✅ {output_path / 'rush_hour_demand.json'}",
        f"✅ {output_path / 'network_topology.csv'}",
        f"✅ {output_path / 'network_metadata.json'}",
        f"✅ {output_path / 'preprocessing_report.txt'}",
        "",
        "=" * 70,
        "✅ PREPROCESSING COMPLETE",
        "=" * 70
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_file = output_path / "preprocessing_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n💾 Saved: {report_file}")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE - ALL FILES GENERATED")
    print("=" * 70)


def main():
    """
    Command-line interface for preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess TomTom data for UXsim validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python preprocess_tomtom.py --input donnees_trafic_75_segments.csv --output preprocessed/
  
  # With custom peak factor
  python preprocess_tomtom.py --input data.csv --output out/ --peak-factor 3.0
  
  # With custom random seed
  python preprocess_tomtom.py --input data.csv --output out/ --seed 123
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input TomTom CSV file'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for preprocessed data'
    )
    
    parser.add_argument(
        '--peak-factor',
        type=float,
        default=2.5,
        help='Rush hour peak factor (default: 2.5x)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocess_tomtom_data(
        input_csv=args.input,
        output_dir=args.output,
        peak_factor=args.peak_factor,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
