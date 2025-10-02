#!/usr/bin/env python3
"""
Minimal test to verify NPZ saving workflow for Riemann problems
Tests only the fastest Riemann case to validate structure before Kaggle
"""

import numpy as np
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))
sys.path.insert(0, str(project_root / "validation_ch7" / "scripts"))

def test_minimal_riemann_npz():
    """Test minimal Riemann problem with NPZ saving"""
    
    print("=" * 80)
    print("MINIMAL RIEMANN TEST - NPZ VERIFICATION")
    print("=" * 80)
    
    # Import validation framework
    print("\n[1/5] Importing validation framework...")
    try:
        from validation_ch7.scripts.validation_utils import create_riemann_scenario_config, run_real_simulation
        from code.io.data_manager import save_simulation_data
        from datetime import datetime
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Setup output directories
    print("\n[2/5] Setting up output directories...")
    output_dir = Path("validation_ch7/results")
    npz_dir = output_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {npz_dir}")
    
    # Create minimal Riemann scenario
    print("\n[3/5] Creating minimal Riemann scenario...")
    scenario_path = output_dir / "test_minimal_riemann.yml"
    
    # Simple shock: high density/low velocity -> low density/high velocity
    U_L = [0.3, 10.0, 0.0, 0.0]  # Left state: [ρ_m, v_m, ρ_c, v_c]
    U_R = [0.1, 30.0, 0.0, 0.0]  # Right state
    
    create_riemann_scenario_config(
        scenario_path, 
        U_L, U_R,
        domain_length=5.0,  # Short domain
        N=100,              # Coarse grid
        t_final=0.5         # Short time
    )
    print(f"✓ Scenario created: {scenario_path.name}")
    
    # Run simulation
    print("\n[4/5] Running ARZ simulation (this may take 10-20 seconds)...")
    try:
        result = run_real_simulation(scenario_path, device='cpu')
        
        if result is None:
            print("✗ Simulation failed")
            return False
        
        # Convert states to numpy array if it's a list
        if isinstance(result['states'], list):
            result['states'] = np.array(result['states'])
        
        print(f"✓ Simulation completed: {len(result['times'])} timesteps")
        print(f"  Grid size: {result['states'].shape[2]} cells")
        print(f"  Final time: {result['times'][-1]:.3f}s")
        
    except Exception as e:
        print(f"✗ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save NPZ file
    print("\n[5/5] Saving NPZ file...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        npz_file = npz_dir / f"test_minimal_riemann_{timestamp}.npz"
        
        save_simulation_data(
            str(npz_file),
            result['times'],
            result['states'],
            result['grid'],
            result['params']
        )
        
        print(f"✓ NPZ saved: {npz_file.name}")
        print(f"  File size: {npz_file.stat().st_size / 1024:.1f} KB")
        
        # Verify NPZ can be loaded
        data = np.load(str(npz_file), allow_pickle=True)
        print(f"✓ NPZ verification:")
        print(f"  - times array: {data['times'].shape}")
        print(f"  - states array: {data['states'].shape}")
        print(f"  - Contains: {list(data.keys())}")
        
    except Exception as e:
        print(f"✗ NPZ save failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("SUCCESS - NPZ WORKFLOW VALIDATED")
    print("=" * 80)
    print(f"\nNPZ file ready: {npz_file}")
    print("\nNext step: Test on Kaggle with full Section 7.3")
    
    return True

if __name__ == "__main__":
    success = test_minimal_riemann_npz()
    sys.exit(0 if success else 1)
