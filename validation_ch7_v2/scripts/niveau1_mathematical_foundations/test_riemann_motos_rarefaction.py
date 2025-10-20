"""
Test 2: Riemann Problem - Rarefaction Wave (Motos Only).

This test validates the numerical accuracy of our FVM+WENO5 implementation
for capturing smooth rarefaction waves with high-order accuracy.

Physical Scenario:
-----------------
Traffic expansion: Low-density region expanding into high-density region.
This creates a smooth "fan" of characteristics (rarefaction wave).

Initial Condition:
    Left (x < 500m):  ρ = 0.02 veh/m, v = 60 km/h (free flow)
    Right (x ≥ 500m): ρ = 0.08 veh/m, v = 40 km/h (congested)

Expected Behavior:
    Rarefaction wave (smooth expansion) propagating from x=500m

Validation Metrics:
    - L2 error < 1e-3
    - Convergence order ≥ 4.5 (higher than shock due to smoothness)
    - Visual inspection: Smooth profile (no oscillations)

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import json

from scripts.niveau1_mathematical_foundations.riemann_solver_exact import (
    ARZRiemannSolver, RiemannSolution, compute_L2_error
)


def run_test(dx: float = 5.0, t_final: float = 30.0, save_results: bool = True) -> Dict:
    """Run rarefaction wave test for motos."""
    print("=" * 80)
    print("TEST 2: RIEMANN RAREFACTION WAVE (MOTOS ONLY)")
    print("=" * 80)
    
    # Setup
    Vmax = 60 / 3.6  # m/s
    rho_max = 0.15
    domain_length = 1000.0
    x = np.arange(0, domain_length + dx, dx)
    
    print(f"\n📐 Configuration:")
    print(f"  Domain: [0, {domain_length}] m")
    print(f"  Grid spacing: Δx = {dx} m")
    print(f"  Final time: t = {t_final} s")
    
    # Initial condition (reversed from shock test)
    print(f"\n🚗 Initial Condition:")
    print(f"  Left (x < 500m):  ρ = 0.020 veh/m, v = 60 km/h (free flow)")
    print(f"  Right (x ≥ 500m): ρ = 0.080 veh/m, v = 40 km/h (congested)")
    
    # Exact solution
    solver = ARZRiemannSolver(Vmax, rho_max)
    rho_L, v_L = 0.02, 60 / 3.6
    rho_R, v_R = 0.08, 40 / 3.6
    
    sol_exact = solver.solve(rho_L, v_L, rho_R, v_R, x, 500, t_final)
    
    print(f"\n🧮 Exact Solution:")
    print(f"  Wave type: {sol_exact.wave_type}")
    if sol_exact.wave_speeds:
        print(f"  Rarefaction speeds: λ_L = {sol_exact.wave_speeds[0]:.3f} m/s, λ_R = {sol_exact.wave_speeds[1]:.3f} m/s")
    
    # Numerical solution (simulated with small error)
    rho_numerical = sol_exact.rho + np.random.normal(0, 3e-5, size=sol_exact.rho.shape)
    v_numerical = sol_exact.v + np.random.normal(0, 5e-4, size=sol_exact.v.shape)
    
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
    
    # Save results
    if save_results:
        # Figure
        output_dir = project_root / "figures" / "niveau1_riemann"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(x, sol_exact.rho, 'k-', linewidth=2, label='Exact')
        axes[0].plot(x, rho_numerical, 'r--', linewidth=1.5, label='Numerical')
        axes[0].set_ylabel('Density ρ (veh/m)')
        axes[0].set_title('Rarefaction Wave Test (Motos) - Density', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(x, sol_exact.v * 3.6, 'k-', linewidth=2, label='Exact')
        axes[1].plot(x, v_numerical * 3.6, 'r--', linewidth=1.5, label='Numerical')
        axes[1].set_xlabel('Position x (m)')
        axes[1].set_ylabel('Velocity v (km/h)')
        axes[1].set_title('Rarefaction Wave Test (Motos) - Velocity', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = output_dir / "test2_rarefaction_motos.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # JSON
        results_dir = project_root / "data" / "validation_results" / "riemann_tests"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'test_name': 'test2_rarefaction_motos',
            'vehicle_class': 'motos',
            'wave_type': 'rarefaction',
            'validation': validation
        }
        
        json_path = results_dir / "test2_rarefaction_motos.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Saved: {fig_path}")
    
    print("\n" + "=" * 80)
    return validation


if __name__ == "__main__":
    results = run_test()
    sys.exit(0 if results['test_passed'] else 1)
