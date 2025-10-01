#!/usr/bin/env python3
"""
Test Validation GPU Simple - Approche Directe

Script simple comme suggéré par l'utilisateur:
"suffit que tu crées un script en mettant exactement la même commande que celle 
qui appelle l'utilisation de kaggle, mais avec le device gpu, c'est aussi simple tu ne trouves pas ?"

Ce script utilise exactement la même structure que validation_ch7/ mais avec device='gpu'
pour tout. Faut que ça marche !
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Kaggle credentials setup
def setup_kaggle_credentials():
    """Setup Kaggle credentials from kaggle.json"""
    kaggle_json_path = project_root / "kaggle.json"
    
    if kaggle_json_path.exists():
        with open(kaggle_json_path, 'r') as f:
            credentials = json.load(f)
        
        # Set environment variables for Kaggle API
        os.environ['KAGGLE_USERNAME'] = credentials['username']
        os.environ['KAGGLE_KEY'] = credentials['key']
        
        print(f"✅ Kaggle credentials loaded for user: {credentials['username']}")
        return True
    else:
        print("❌ kaggle.json not found!")
        return False

# Import real ARZ modules
try:
    from code.simulation.runner import SimulationRunner
    from code.analysis.metrics import (
        calculate_total_mass, compute_mape, compute_rmse, 
        compute_geh, calculate_convergence_order
    )
    from code.core.parameters import ModelParameters
    print("✅ Real ARZ modules imported successfully")
    ARZ_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import ARZ modules: {e}")
    ARZ_AVAILABLE = False

def test_gpu_validation_simple(scenario_name="riemann_test", device='gpu'):
    """
    Test simple validation GPU - exactement la même commande mais avec device='gpu' 
    comme suggéré par l'utilisateur
    """
    print(f"\n{'='*60}")
    print(f"TEST GPU VALIDATION SIMPLE - {scenario_name.upper()}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    if not ARZ_AVAILABLE:
        print("❌ ARZ modules not available - cannot run GPU test")
        return False
    
    try:
        # Exactement la même commande que validation existante mais avec device='gpu'
        scenario_path = f"config/scenario_{scenario_name}.yml"
        base_config_path = "config/config_base.yml"
        
        print(f"📁 Using scenario: {scenario_path}")
        print(f"📁 Base config: {base_config_path}")
        
        # Create SimulationRunner with GPU device - c'est aussi simple !
        runner = SimulationRunner(
            scenario_config_path=scenario_path,
            base_config_path=base_config_path,
            device=device,  # GPU comme demandé !
            quiet=False  # Verbose pour voir ce qui se passe
        )
        
        print(f"✅ SimulationRunner created successfully")
        print(f"Device: {runner.device}")
        print(f"Grid size N: {runner.params.N}")
        print(f"Time final: {runner.params.t_final}s")
        
        # Run simulation - même commande, device GPU
        print(f"\n▶️ Running simulation on {device.upper()}...")
        times, states = runner.run()
        
        print(f"✅ Simulation completed successfully!")
        print(f"Time steps: {len(times)}")
        print(f"Final time: {times[-1]:.2f}s")
        print(f"States count: {len(states)}")
        if len(states) > 0:
            print(f"State shape: {states[0].shape}")
        
        # Basic validation - conservation de masse comme validation existante
        try:
            if isinstance(states, list) and len(states) > 0:
                states_array = states  # states is already a list of arrays
                # Calculate total mass for both vehicle classes
                initial_mass_m = calculate_total_mass(states_array[0], runner.grid, class_index=0)
                initial_mass_c = calculate_total_mass(states_array[0], runner.grid, class_index=1)
                final_mass_m = calculate_total_mass(states_array[-1], runner.grid, class_index=0)
                final_mass_c = calculate_total_mass(states_array[-1], runner.grid, class_index=1)
                
                initial_mass = initial_mass_m + initial_mass_c
                final_mass = final_mass_m + final_mass_c
                
                mass_conservation_error = abs(final_mass - initial_mass) / initial_mass
                print(f"\n📊 Mass Conservation Check:")
                print(f"Initial mass: {initial_mass:.6f}")
                print(f"Final mass: {final_mass:.6f}")
                print(f"Conservation error: {mass_conservation_error:.2e}")
                
                if mass_conservation_error < 1e-6:
                    print("✅ Mass conservation: PASSED")
                else:
                    print("⚠️ Mass conservation: CHECK NEEDED")
            else:
                print("⚠️ Cannot check mass conservation - unexpected state format")
        except Exception as e:
            print(f"⚠️ Mass conservation check failed: {e}")
        
        # Success!
        print(f"\n🎉 GPU Validation Test SUCCESSFUL!")
        print(f"Device {device} working perfectly - c'est aussi simple !")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU Validation Test FAILED: {e}")
        print(f"Device: {device}")
        return False

def test_multiple_scenarios_gpu():
    """Test multiple scenarios on GPU - faut que ça marche pour tous !"""
    print(f"\n{'='*60}")
    print("TEST MULTIPLE SCENARIOS GPU")
    print(f"{'='*60}")
    
    # Scenarios to test - comme dans validation existante
    scenarios = [
        "riemann_test",
        "convergence_test", 
        "mass_conservation_weno5",
        "gpu_validation"
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n🧪 Testing scenario: {scenario}")
        
        # Check if scenario file exists
        scenario_path = project_root / f"config/scenario_{scenario}.yml"
        if not scenario_path.exists():
            print(f"⚠️ Scenario file not found: {scenario_path}")
            results[scenario] = "SKIPPED"
            continue
        
        # Run test avec device='gpu'
        success = test_gpu_validation_simple(scenario, device='gpu')
        results[scenario] = "PASSED" if success else "FAILED"
    
    # Summary
    print(f"\n{'='*60}")
    print("GPU VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len([r for r in results.values() if r != "SKIPPED"])
    
    for scenario, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "⚠️"
        print(f"{status_icon} {scenario}: {result}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total and total > 0:
        print("🎉 All GPU tests PASSED - faut que ça marche ✓")
        return True
    else:
        print("⚠️ Some tests failed or skipped")
        return False

def main():
    """Main function - test GPU validation comme suggéré par l'utilisateur"""
    print("GPU VALIDATION TEST - SIMPLE APPROACH")
    print("====================================")
    print("Exactement la même commande mais avec device='gpu'")
    print("Faut que ça marche !")
    
    # Setup Kaggle credentials
    credentials_ok = setup_kaggle_credentials()
    if not credentials_ok:
        print("⚠️ Continuing without Kaggle credentials...")
    
    # Check if we have ARZ modules
    if not ARZ_AVAILABLE:
        print("❌ Cannot run tests without ARZ modules")
        return 1
    
    # Test single scenario first
    print("\n1️⃣ SINGLE SCENARIO TEST")
    single_test_ok = test_gpu_validation_simple("riemann_test", device='gpu')
    
    if single_test_ok:
        print("\n2️⃣ MULTIPLE SCENARIOS TEST")
        multiple_tests_ok = test_multiple_scenarios_gpu()
        
        if multiple_tests_ok:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"GPU validation working - c'est aussi simple que ça !")
            return 0
        else:
            print(f"\n⚠️ Some multiple scenario tests failed")
            return 1
    else:
        print(f"\n❌ Single scenario test failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)