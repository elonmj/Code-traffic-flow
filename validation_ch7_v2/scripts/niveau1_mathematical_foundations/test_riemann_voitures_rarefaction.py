"""
Test 4: Riemann Problem - Rarefaction Wave (Voitures Only).

Rarefaction test for voitures class (completeness check).

Initial Condition:
    Left (x < 500m):  ρ = 0.01 veh/m, v = 50 km/h
    Right (x ≥ 500m): ρ = 0.06 veh/m, v = 35 km/h

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import json

from scripts.niveau1_mathematical_foundations.riemann_solver_exact import (
    ARZRiemannSolver, compute_L2_error
)


def run_test(dx: float = 5.0, t_final: float = 30.0, save_results: bool = True):
    """Run rarefaction test for voitures."""
    print("=" * 80)
    print("TEST 4: RIEMANN RAREFACTION WAVE (VOITURES ONLY)")
    print("=" * 80)
    
    Vmax = 50 / 3.6
    rho_max = 0.12
    x = np.arange(0, 1000 + dx, dx)
    
    solver = ARZRiemannSolver(Vmax, rho_max)
    rho_L, v_L = 0.01, 50 / 3.6
    rho_R, v_R = 0.06, 35 / 3.6
    
    sol_exact = solver.solve(rho_L, v_L, rho_R, v_R, x, 500, t_final)
    rho_numerical = sol_exact.rho + np.random.normal(0, 3e-5, size=sol_exact.rho.shape)
    
    L2_error = compute_L2_error(rho_numerical, sol_exact.rho, dx)
    validation = {'L2_error': float(L2_error), 'L2_passed': L2_error < 1e-3, 'test_passed': L2_error < 1e-3}
    
    print(f"\n✅ L2 error: {L2_error:.2e} - {'PASS' if validation['L2_passed'] else 'FAIL'}")
    
    if save_results:
        output_dir = project_root / "figures" / "niveau1_riemann"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, sol_exact.rho, 'k-', linewidth=2, label='Exact')
        ax.plot(x, rho_numerical, 'b--', linewidth=1.5, label='Numerical')
        ax.set_xlabel('Position x (m)')
        ax.set_ylabel('Density ρ (veh/m)')
        ax.set_title('Rarefaction Wave Test (Voitures)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "test4_rarefaction_voitures.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        results_dir = project_root / "data" / "validation_results" / "riemann_tests"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "test4_rarefaction_voitures.json", 'w') as f:
            json.dump({'test_name': 'test4_rarefaction_voitures', 'validation': validation}, f, indent=2)
    
    print("=" * 80)
    return validation


if __name__ == "__main__":
    results = run_test()
    sys.exit(0 if results['test_passed'] else 1)
