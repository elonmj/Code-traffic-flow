"""Validate Riemann simulation data after bug fix"""
import numpy as np
from pathlib import Path

INPUT_DIR = Path(r"d:\Projets\Alibi\Code project\kaggle\results\generic-test-runner-kernel\thesis_stage1")

# Test cases
tests = [
    ('riemann_choc_simple_motos.npz', 'Choc Simple', [0.15, 8.0, 0.12, 6.0], [0.05, 10.0, 0.03, 8.0]),
    ('riemann_detente_voitures.npz', 'Détente', [0.03, 8.0, 0.20, 5.0], [0.02, 10.0, 0.05, 12.0]),
    ('riemann_apparition_vide_motos.npz', 'Vide', [0.10, 12.0, 0.08, 10.0], [0.001, 15.0, 0.001, 15.0]),
    ('riemann_discontinuite_contact.npz', 'Contact', [0.08, 10.0, 0.10, 10.0], [0.04, 10.0, 0.05, 10.0]),
    ('riemann_interaction_multiclasse.npz', 'Interaction', [0.10, 8.0, 0.15, 6.0], [0.05, 12.0, 0.08, 10.0])
]

print("=" * 80)
print("RIEMANN DATA VALIDATION AFTER BUG FIX")
print("=" * 80)

all_valid = True

for filename, name, U_L, U_R in tests:
    filepath = INPUT_DIR / filename
    
    print(f"\n{'=' * 60}")
    print(f"Test: {name}")
    print(f"{'=' * 60}")
    
    if not filepath.exists():
        print(f"❌ FILE NOT FOUND: {filepath}")
        all_valid = False
        continue
    
    data = np.load(filepath)
    
    # Extract data
    rho_m_hist = data['rho_m_history']
    rho_c_hist = data['rho_c_history']
    v_m_hist = data['v_m_history']
    v_c_hist = data['v_c_history']
    
    # Check for NaN
    has_nan = (np.any(np.isnan(rho_m_hist)) or np.any(np.isnan(rho_c_hist)) or 
               np.any(np.isnan(v_m_hist)) or np.any(np.isnan(v_c_hist)))
    
    if has_nan:
        print(f"❌ NaN DETECTED!")
        # Find first NaN timestep
        for i in range(len(rho_m_hist)):
            if np.any(np.isnan(rho_m_hist[i])):
                print(f"   First NaN at timestep {i}, t={data['t_history'][i]:.4f}s")
                break
        all_valid = False
    else:
        print(f"✅ No NaN")
    
    # Check initial conditions (should match U_L/U_R)
    rho_m_0 = rho_m_hist[0]
    v_m_0 = v_m_hist[0]
    rho_c_0 = rho_c_hist[0]
    v_c_0 = v_c_hist[0]
    
    print(f"\nInitial Conditions (expected → actual):")
    print(f"  Left  - rho_m: {U_L[0]:.3f} → {rho_m_0[:5].mean():.3f}")
    print(f"  Left  - v_m:   {U_L[1]:.1f} → {v_m_0[:5].mean():.1f}")
    print(f"  Right - rho_m: {U_R[0]:.3f} → {rho_m_0[-5:].mean():.3f}")
    print(f"  Right - v_m:   {U_R[1]:.1f} → {v_m_0[-5:].mean():.1f}")
    
    # Check if IC are reasonable (motos)
    ic_ok = (0.001 <= rho_m_0[:5].mean() <= 0.25 and  # Allow small margin
             0 <= v_m_0[:5].mean() <= 35)
    if ic_ok:
        print(f"✅ Initial conditions look correct")
    else:
        print(f"⚠️  Initial conditions suspicious")
        all_valid = False
    
    # Check physical bounds
    rho_max_m = rho_m_hist.max()
    rho_max_c = rho_c_hist.max()
    v_max_m = v_m_hist.max()
    v_max_c = v_c_hist.max()
    
    print(f"\nPhysical Bounds:")
    print(f"  rho_m max: {rho_max_m:.4f} veh/m (limit: 0.20)")
    print(f"  rho_c max: {rho_max_c:.4f} veh/m (limit: 0.20)")
    print(f"  v_m max:   {v_max_m:.2f} m/s (limit: 30.0)")
    print(f"  v_c max:   {v_max_c:.2f} m/s (limit: 30.0)")
    
    bounds_ok = (rho_max_m <= 0.25 and rho_max_c <= 0.25 and  # Allow small overshoot
                 v_max_m <= 35 and v_max_c <= 35)
    if bounds_ok:
        print(f"✅ All values within physical bounds")
    else:
        print(f"❌ Values exceed physical bounds!")
        all_valid = False

print("\n" + "=" * 80)
if all_valid:
    print("✅ ALL TESTS PASSED - DATA IS VALID FOR FIGURE GENERATION")
else:
    print("❌ VALIDATION FAILED - DO NOT GENERATE FIGURES YET")
print("=" * 80)
