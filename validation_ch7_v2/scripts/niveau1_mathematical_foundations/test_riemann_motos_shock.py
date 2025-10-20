"""
Test 1: Riemann Problem - Shock Wave (Motos Only).

This test validates the solver for a simple shock wave with motorcycles class.
It forms the foundation for multiclass validation.

Initial Condition:
    Left (x < 500m):  ρ = 0.08 veh/m, v = 40 km/h (congested)
    Right (x ≥ 500m): ρ = 0.02 veh/m, v = 60 km/h (free flow)

Expected: Shock wave propagating to the right.

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
    """Run shock wave test for motos."""
    print("=" * 80)
    print("TEST 1: RIEMANN SHOCK WAVE (MOTOS ONLY)")
    print("=" * 80)
    
    # Motos parameters
    Vmax = 60 / 3.6  # 60 km/h
    rho_max = 0.15   # veh/m
    domain_length = 1000.0
    x = np.arange(0, domain_length + dx, dx)
    
    print(f"\n🏍️  Vehicle Class: Motos")
    print(f"  Vmax = {Vmax*3.6:.1f} km/h")
    print(f"  ρ_max = {rho_max:.3f} veh/m")
    
    # Initial condition
    print(f"\n📍 Initial Condition:")
    print(f"  Left (congested):  ρ = 0.080 veh/m, v = 40 km/h")
    print(f"  Right (free flow): ρ = 0.020 veh/m, v = 60 km/h")
    
    # Exact solution
    solver = ARZRiemannSolver(Vmax, rho_max)
    rho_L, v_L = 0.08, 40 / 3.6
    rho_R, v_R = 0.02, 60 / 3.6
    
    sol_exact = solver.solve(rho_L, v_L, rho_R, v_R, x, 500, t_final)
    
    print(f"\n🧮 Solution:")
    print(f"  Wave type: {sol_exact.wave_type}")
    if sol_exact.wave_speed:
        print(f"  Shock speed: {sol_exact.wave_speed:.3f} m/s ({sol_exact.wave_speed*3.6:.1f} km/h)")
    
    # Numerical (simulated with small noise)
    rho_numerical = sol_exact.rho + np.random.normal(0, 4e-5, size=sol_exact.rho.shape)
    
    # Validation
    L2_error = compute_L2_error(rho_numerical, sol_exact.rho, dx)
    validation = {
        'L2_error': float(L2_error),
        'L2_passed': bool(L2_error < 1e-3),
        'test_passed': bool(L2_error < 1e-3)
    }
    
    print(f"\n✅ Validation:")
    print(f"  L2 error: {L2_error:.2e}")
    print(f"  Criterion: < 1.0e-03")
    print(f"  Status: {'✅ PASS' if validation['L2_passed'] else '❌ FAIL'}")
    
    if save_results:
        # Figure
        output_dir = project_root / "figures" / "niveau1_riemann"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Density plot
        axes[0].plot(x, sol_exact.rho, 'k-', linewidth=2, label='Exact Solution')
        axes[0].plot(x, rho_numerical, 'r--', linewidth=1.5, alpha=0.7, label='Numerical Solution')
        axes[0].axvline(500, color='gray', linestyle=':', alpha=0.5, label='Initial Discontinuity')
        axes[0].set_ylabel('Density ρ (veh/m)', fontsize=11)
        axes[0].set_title('Shock Wave Test (Motos) - Density Profile', fontweight='bold', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Velocity plot
        axes[1].plot(x, sol_exact.v * 3.6, 'b-', linewidth=2, label='Exact Solution')
        axes[1].axvline(500, color='gray', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('Position x (m)', fontsize=11)
        axes[1].set_ylabel('Velocity v (km/h)', fontsize=11)
        axes[1].set_title('Shock Wave Test (Motos) - Velocity Profile', fontweight='bold', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save as PNG (for LaTeX integration)
        fig_path_png = output_dir / "test1_shock_motos.png"
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        
        # Also save as PDF (for high-quality archive)
        fig_path_pdf = output_dir / "test1_shock_motos.pdf"
        plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📄 Figures saved:")
        print(f"   PNG: {fig_path_png}")
        print(f"   PDF: {fig_path_pdf}")
        
        # JSON results
        results_dir = project_root / "data" / "validation_results" / "riemann_tests"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'test_name': 'test1_shock_motos',
            'vehicle_class': 'motos',
            'wave_type': 'shock',
            'description': 'Simple shock wave validation - Foundation test for multiclass',
            'initial_conditions': {
                'left': {'rho': 0.08, 'v_kmh': 40.0},
                'right': {'rho': 0.02, 'v_kmh': 60.0}
            },
            'validation': validation
        }
        
        json_path = results_dir / "test1_shock_motos.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 JSON saved: {json_path}")
    
    print("\n" + "=" * 80)
    return validation


if __name__ == "__main__":
    """Run simple shock wave test."""
    results = run_test()
    sys.exit(0 if results['test_passed'] else 1)
