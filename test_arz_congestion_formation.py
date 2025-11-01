#!/usr/bin/env python3
"""
Test ARZ Model: Congestion Formation Verification

This test MUST create congestion if ARZ model is working correctly.

Test Strategy:
1. HIGH INFLOW: Inject significant traffic (0.15 veh/m = 150 veh/km)
2. RED LIGHT: Block downstream flow completely for 60 seconds
3. OBSERVE: Queue MUST form if ARZ conservation laws work

Expected Result:
- Density accumulates near x=500m (red light position)
- Velocities drop below 5 m/s (queue threshold)
- Queue length > 0 vehicles

If NO congestion forms → ARZ model is BROKEN
"""

import numpy as np
import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arz_model.simulation.runner import SimulationRunner
from arz_model.core.parameters import ModelParameters

def create_congestion_test_scenario():
    """Create scenario that MUST create congestion"""
    
    # Create test config
    config = {
        'scenario_name': 'congestion_formation_test',
        
        # LONG DOMAIN to see wave propagation
        'N': 200,
        'xmin': 0.0,
        'xmax': 1000.0,
        
        # Time parameters
        't_final': 120.0,  # 2 minutes
        'dt': 0.2,
        'output_dt': 5.0,
        
        # HIGH DENSITY INFLOW (near-jam)
        'initial_conditions': {
            'type': 'uniform',
            'rho_m': 0.02,  # Light initial state
            'rho_c': 0.02,
        },
        
        # BOUNDARY CONDITIONS
        'boundary_conditions': {
            'left': {
                'type': 'inflow',
                # 🔥 ARCHITECTURAL FIX: Explicit BC state (no IC fallback)
                'state': [0.150, 1.2, 0.120, 0.72],  # [rho_m, w_m, rho_c, w_c]
                # Explanation:
                #   rho_m = 0.150 veh/m = 150 veh/km (HEAVY TRAFFIC)
                #   w_m = 0.150 * 8.0 = 1.2 kg·m/s (momentum = density × velocity)
                #   rho_c = 0.120 veh/m = 120 veh/km (CARS)
                #   w_c = 0.120 * 6.0 = 0.72 kg·m/s (car momentum)
            },
            'right': {
                'type': 'outflow'
            }
        },
        
        # UNIFORM ROAD QUALITY
        'road_quality_definition': 2,
        
        # CRITICAL: Network with traffic light
        'has_network': True,
        'enable_traffic_lights': True,
        
        # Single traffic light at middle of domain
        'nodes': [
            {
                'node_id': 0,
                'position': 0.0,
                'type': 'boundary_inflow'
            },
            {
                'node_id': 1,
                'position': 500.0,  # Middle of domain
                'type': 'signalized_intersection',
                'traffic_light': {
                    'initial_phase': 0,  # RED (blocks flow)
                    'phase_duration': 60.0,  # Stay RED for 60 seconds
                    'green_time': 30.0,
                    'red_time': 60.0
                }
            },
            {
                'node_id': 2,
                'position': 1000.0,
                'type': 'boundary_outflow'
            }
        ],
        
        # Two segments: before and after traffic light
        'network_segments': [
            {
                'segment_id': 'upstream',
                'node_start': 0,
                'node_end': 1,
                'length': 500.0
            },
            {
                'segment_id': 'downstream',
                'node_start': 1,
                'node_end': 2,
                'length': 500.0
            }
        ]
    }
    
    # Write to temp file
    config_path = Path('temp_congestion_test.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def analyze_congestion(runner, times, states):
    """Analyze if congestion formed"""
    
    print("\n" + "="*80)
    print("CONGESTION FORMATION ANALYSIS")
    print("="*80)
    
    # Get final state
    rho_m_final = states[-1][0]  # Final moto density
    rho_c_final = states[-1][1]  # Final car density
    w_m_final = states[-1][2]
    w_c_final = states[-1][3]
    
    # Calculate velocities
    v_m_final = np.where(rho_m_final > 1e-6, w_m_final / rho_m_final, 0)
    v_c_final = np.where(rho_c_final > 1e-6, w_c_final / rho_c_final, 0)
    
    # Total density
    rho_total = rho_m_final + rho_c_final
    
    # Grid positions (only physical cells)
    x = runner.grid._cell_centers[runner.grid.physical_cell_indices]
    
    # Find traffic light position (x=500m)
    light_idx = np.argmin(np.abs(x - 500.0))
    
    print(f"\n📍 TRAFFIC LIGHT AT x={x[light_idx]:.1f}m (index {light_idx})")
    print(f"   Configured: RED light for 60 seconds, blocking flow")
    
    # Analyze upstream region (400-500m)
    upstream_indices = (x >= 400) & (x <= 500)
    rho_upstream = rho_total[upstream_indices]
    v_upstream = v_m_final[upstream_indices]
    
    print(f"\n📊 UPSTREAM REGION (400-500m, before light):")
    print(f"   Max density: {np.max(rho_upstream):.4f} veh/m ({np.max(rho_upstream)*1000:.1f} veh/km)")
    print(f"   Min velocity: {np.min(v_upstream):.2f} m/s ({np.min(v_upstream)*3.6:.1f} km/h)")
    print(f"   Avg density: {np.mean(rho_upstream):.4f} veh/m")
    
    # Check for congestion
    QUEUE_VELOCITY_THRESHOLD = 6.67  # m/s (50% of free flow, from logs)
    congested_cells = v_upstream < QUEUE_VELOCITY_THRESHOLD
    num_congested = np.sum(congested_cells)
    
    print(f"\n🚦 CONGESTION DETECTION:")
    print(f"   Cells with v < {QUEUE_VELOCITY_THRESHOLD:.2f} m/s: {num_congested}/{len(v_upstream)}")
    
    if num_congested > 0:
        print(f"   ✅ CONGESTION DETECTED!")
        
        # Calculate queue length
        dx = runner.grid.dx
        queue_length_m = num_congested * dx
        queue_density = np.mean(rho_upstream[congested_cells])
        vehicles_in_queue = queue_length_m * queue_density
        
        print(f"   Queue length: {queue_length_m:.1f} meters")
        print(f"   Queue density: {queue_density:.4f} veh/m")
        print(f"   Vehicles queued: {vehicles_in_queue:.1f}")
        
    else:
        print(f"   ❌ NO CONGESTION DETECTED")
        print(f"   All velocities > {QUEUE_VELOCITY_THRESHOLD:.2f} m/s")
    
    # Analyze inflow boundary
    print(f"\n📥 INFLOW BOUNDARY (x=0):")
    print(f"   Density: {rho_total[0]:.4f} veh/m ({rho_total[0]*1000:.1f} veh/km)")
    print(f"   Velocity motos: {v_m_final[0]:.2f} m/s")
    print(f"   Velocity cars: {v_c_final[0]:.2f} m/s")
    print(f"   Expected inflow: 0.150 veh/m (150 veh/km)")
    
    inflow_ratio = rho_total[0] / 0.150 if 0.150 > 0 else 0
    print(f"   Inflow penetration: {inflow_ratio*100:.1f}%")
    
    # Final verdict
    print(f"\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    if num_congested == 0:
        print("❌ TEST FAILED: ARZ model did NOT create congestion")
        print("   Problem: Model is not accumulating density despite:")
        print("   1. High inflow (150 veh/km)")
        print("   2. RED light blocking downstream")
        print("   3. 60 seconds of accumulation time")
        print()
        print("   → ARZ MODEL IS BROKEN - Conservation laws not working")
        return False
        
    elif inflow_ratio < 0.5:
        print("⚠️  TEST PARTIAL: Congestion detected but inflow weak")
        print(f"   Only {inflow_ratio*100:.1f}% of expected inflow entering domain")
        print("   → Inflow boundary condition may be broken")
        return False
        
    else:
        print("✅ TEST PASSED: ARZ model creates congestion correctly")
        print(f"   - Congestion in {num_congested} cells")
        print(f"   - Inflow penetration: {inflow_ratio*100:.1f}%")
        print(f"   - Queue length: {queue_length_m:.1f}m")
        return True


def main():
    """Run congestion formation test"""
    
    print("="*80)
    print("ARZ MODEL VALIDATION: CONGESTION FORMATION TEST")
    print("="*80)
    print()
    print("This test verifies that ARZ model can create traffic congestion.")
    print("Setup:")
    print("  - High inflow: 150 veh/km (heavy traffic)")
    print("  - RED light at x=500m (blocks flow)")
    print("  - Duration: 120 seconds")
    print()
    print("Expected: Queue MUST form upstream of traffic light")
    print("="*80)
    
    # Create scenario
    print("\n📝 Creating test scenario...")
    config_path = create_congestion_test_scenario()
    print(f"   Config: {config_path}")
    
    # Run simulation
    print("\n▶️  Running ARZ simulation...")
    try:
        runner = SimulationRunner(
            scenario_config_path=config_path,
            base_config_path='arz_model/config/config_base.yml',
            device='cpu',  # CPU for debugging
            quiet=False
        )
        
        times, states = runner.run()
        
        print(f"\n✅ Simulation completed: {len(times)} timesteps")
        print(f"   Final time: {times[-1]:.1f}s")
        
        # Analyze results
        success = analyze_congestion(runner, times, states)
        
        # Cleanup
        config_path.unlink()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ SIMULATION CRASHED: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        if config_path.exists():
            config_path.unlink()
        
        return 2


if __name__ == "__main__":
    sys.exit(main())
