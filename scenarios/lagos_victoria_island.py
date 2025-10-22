"""
Lagos Victoria Island Scenario - 75 Road Segments
==================================================

Real traffic network from Victoria Island, Lagos, Nigeria.
Data source: TomTom Traffic API (September 2024)
Segments: 75 unique road segments
Streets: Akin Adesola, Adeola Odeku, Ahmadu Bello Way, Saka Tinubu, etc.

This module demonstrates the scalable architecture:
- NO YAML files required
- Parameters and topology in version-controlled Python
- Direct integration with NetworkBuilder → NetworkGrid workflow
- Ready for calibration and simulation

Usage:
    >>> from scenarios.lagos_victoria_island import create_grid
    >>> grid = create_grid()
    >>> grid.initialize()
    >>> for t in range(3600):  # 1-hour simulation
    ...     grid.step(dt=0.1)
"""

from arz_model.calibration.core.network_builder import NetworkBuilder
from arz_model.network.network_grid import NetworkGrid
from typing import Dict
import os


# Network configuration
CSV_PATH = 'donnees_trafic_75_segments (2).csv'

# Default global parameters (can be overridden by calibration)
GLOBAL_PARAMS = {
    'V0_c': 13.89,      # 50 km/h default for cars (m/s)
    'V0_m': 15.28,      # 55 km/h default for motorcycles (m/s)
    'tau_c': 18.0,      # Relaxation time cars (s)
    'tau_m': 20.0,      # Relaxation time motorcycles (s)
    'rho_max_c': 200.0, # Max density cars (veh/km)
    'rho_max_m': 150.0  # Max density motorcycles (veh/km)
}

# Example calibrated parameters for key segments
# NOTE: These are placeholders. Run actual calibration to populate.
CALIBRATED_PARAMS: Dict[str, Dict[str, float]] = {
    # Arterial roads (higher speeds)
    # 'seg_akin_adesola_main': {
    #     'V0_c': 13.89,  # 50 km/h
    #     'tau_c': 18.0
    # },
    
    # Secondary roads (medium speeds)
    # 'seg_adeola_odeku_1': {
    #     'V0_c': 11.11,  # 40 km/h
    #     'tau_c': 19.0
    # },
    
    # Tertiary roads (lower speeds)
    # 'seg_saka_tinubu_side': {
    #     'V0_c': 8.33,   # 30 km/h
    #     'tau_c': 20.0
    # },
    
    # To populate with actual calibration:
    # 1. Run: calibrator = CalibrationRunner(builder)
    # 2. Run: results = calibrator.calibrate(speed_data)
    # 3. Copy results['parameters'] here
}


def create_grid(
    csv_path: str = CSV_PATH,
    global_params: Dict[str, float] = None,
    calibrated_params: Dict[str, Dict[str, float]] = None,
    dx: float = 10.0,
    dt: float = 0.1
) -> NetworkGrid:
    """
    Create Lagos Victoria Island NetworkGrid from CSV data.
    
    This function demonstrates the direct integration workflow:
        CSV → NetworkBuilder → calibrate (optional) → NetworkGrid
    
    No YAML export/import required. All parameters in Python code.
    
    Args:
        csv_path: Path to CSV file with road segments
        global_params: Global default parameters (uses GLOBAL_PARAMS if None)
        calibrated_params: Per-segment calibrated parameters (uses CALIBRATED_PARAMS if None)
        dx: Spatial resolution (m)
        dt: Time step (s)
    
    Returns:
        NetworkGrid ready for simulation with 75 real road segments
    
    Example:
        >>> grid = create_grid()
        >>> grid.initialize()
        >>> print(f"Segments: {len(grid.segments)}, Nodes: {len(grid.nodes)}")
        >>> 
        >>> # Run simulation
        >>> for t in range(3600):
        ...     grid.step(dt=0.1)
    """
    # Use defaults if not provided
    if global_params is None:
        global_params = GLOBAL_PARAMS
    
    if calibrated_params is None:
        calibrated_params = CALIBRATED_PARAMS
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Lagos CSV not found: {csv_path}\n"
            f"Expected file: donnees_trafic_75_segments (2).csv"
        )
    
    # Build network from CSV
    builder = NetworkBuilder(global_params=global_params)
    builder.build_from_csv(csv_path)
    
    # Apply calibrated parameters (if any)
    for seg_id, params in calibrated_params.items():
        if seg_id in builder.segments:
            builder.set_segment_params(seg_id, params)
    
    # Create NetworkGrid DIRECTLY (no YAML intermediate!)
    grid = NetworkGrid.from_network_builder(builder, dx=dx, dt=dt)
    
    return grid


def get_scenario_info() -> Dict:
    """
    Get metadata about Lagos scenario.
    
    Returns:
        Dictionary with scenario information
    """
    return {
        'name': 'Lagos Victoria Island',
        'location': 'Victoria Island, Lagos, Nigeria',
        'segments': 75,
        'data_source': 'TomTom Traffic API',
        'data_period': 'September 2024',
        'streets': [
            'Akin Adesola Street (arterial)',
            'Adeola Odeku Street (secondary)',
            'Ahmadu Bello Way (arterial)',
            'Saka Tinubu Street (tertiary)',
            'Others (residential/commercial mix)'
        ],
        'network_type': 'Urban mixed (arterial + residential)',
        'calibration_status': 'Pending (placeholders in CALIBRATED_PARAMS)',
        'ready_for_simulation': True
    }


if __name__ == '__main__':
    """
    Example usage: Run this file directly to test Lagos scenario.
    """
    import sys
    
    print("=" * 70)
    print("Lagos Victoria Island Scenario - Test Run")
    print("=" * 70)
    print()
    
    # Get scenario info
    info = get_scenario_info()
    print("Scenario Information:")
    for key, value in info.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Check CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: CSV file not found: {CSV_PATH}")
        print("Please ensure the file is in the project root directory.")
        sys.exit(1)
    
    print(f"✅ CSV file found: {CSV_PATH}")
    print()
    
    # Create grid
    print("Creating NetworkGrid from CSV...")
    try:
        grid = create_grid()
        print(f"✅ NetworkGrid created successfully!")
        print(f"   - Segments: {len(grid.segments)}")
        print(f"   - Nodes: {len(grid.nodes)}")
        print(f"   - Links: {len(grid.links)}")
        print()
        
        # Initialize
        print("Initializing simulation...")
        grid.initialize()
        print("✅ Simulation initialized")
        print()
        
        # Run short test simulation (10 seconds)
        print("Running 10-second test simulation...")
        steps = int(10.0 / grid.dt)
        for i in range(steps):
            grid.step(dt=grid.dt)
            if (i + 1) % 25 == 0:
                print(f"   Step {i+1}/{steps} ({(i+1)*grid.dt:.1f}s)")
        
        print()
        print("✅ Test simulation completed successfully!")
        print()
        print("=" * 70)
        print("Lagos scenario is ready for production use!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Run calibration to populate CALIBRATED_PARAMS")
        print("  2. Validate against observed traffic data")
        print("  3. Use in research/production simulations")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
