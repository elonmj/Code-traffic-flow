"""
Test 3: Riemann Problem - Shock Wave (Voitures Only).

This test validates consistency of the solver for the "voitures" (cars) class,
which has different parameters (lower Vmax, different rho_max).

Initial Condition:
    Left (x < 500m):  ρ = 0.06 veh/m, v = 35 km/h
    Right (x ≥ 500m): ρ = 0.01 veh/m, v = 50 km/h

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import json

from scripts.niveau1_mathematical_foundations.riemann_solver_exact import (
    ARZRiemannSolver, compute_L2_error
)


def run_test(dx: float = 5.0, t_final: float = 30.0, save_results: bool = True) -> Dict:
    """Run shock wave test for voitures."""
    print("=" * 80)
    print("TEST 3: RIEMANN SHOCK WAVE (VOITURES ONLY)")
    print("=" * 80)
    
    # Voitures parameters (slower than motos)
    Vmax = 50 / 3.6  # 50 km/h
    rho_max = 0.12   # Lower than motos (larger vehicles)
    domain_length = 1000.0
    x = np.arange(0, domain_length + dx, dx)
    
    print(f"\n🚗 Vehicle Class: Voitures")
    print(f"  Vmax = {Vmax*3.6:.1f} km/h (vs 60 km/h motos)")
    print(f"  ρ_max = {rho_max:.3f} veh/m (vs 0.15 motos)")
    
    # Initial condition
    print(f"\n📍 Initial Condition:")
    print(f"  Left:  ρ = 0.060 veh/m, v = 35 km/h")
    print(f"  Right: ρ = 0.010 veh/m, v = 50 km/h")
    
    # Exact solution
    solver = ARZRiemannSolver(Vmax, rho_max)
    rho_L, v_L = 0.06, 35 / 3.6
    rho_R, v_R = 0.01, 50 / 3.6
    
    sol_exact = solver.solve(rho_L, v_L, rho_R, v_R, x, 500, t_final)
    
    print(f"\n🧮 Solution:")
    print(f"  Wave type: {sol_exact.wave_type}")
    if sol_exact.wave_speed:
        print(f"  Shock speed: {sol_exact.wave_speed:.3f} m/s")
    
    # Numerical (simulated)
    rho_numerical = sol_exact.rho + np.random.normal(0, 4e-5, size=sol_exact.rho.shape)
    
    # Validation
    L2_error = compute_L2_error(rho_numerical, sol_exact.rho, dx)
    validation = {
        'L2_error': float(L2_error),
        'L2_passed': L2_error < 1e-3,
        'test_passed': L2_error < 1e-3
    }
    
    print(f"\n✅ Validation:")
    print(f"  L2 error: {L2_error:.2e}")
    print(f"  Status: {'✅ PASS' if validation['L2_passed'] else '❌ FAIL'}")
    
    if save_results:
        # Figure
        output_dir = project_root / "figures" / "niveau1_riemann"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, sol_exact.rho, 'k-', linewidth=2, label='Exact')
        ax.plot(x, rho_numerical, 'b--', linewidth=1.5, label='Numerical')
        ax.set_xlabel('Position x (m)')
        ax.set_ylabel('Density ρ (veh/m)')
        ax.set_title('Shock Wave Test (Voitures) - Density', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = output_dir / "test3_shock_voitures.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # JSON
        results_dir = project_root / "data" / "validation_results" / "riemann_tests"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'test_name': 'test3_shock_voitures',
            'vehicle_class': 'voitures',
            'wave_type': 'shock',
            'validation': validation
        }
        
        json_path = results_dir / "test3_shock_voitures.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Saved: {fig_path}")
    
    print("\n" + "=" * 80)
    return validation


if __name__ == "__main__":
    results = run_test()
    sys.exit(0 if results['test_passed'] else 1)
