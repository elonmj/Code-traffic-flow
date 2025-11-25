"""Quick diagnostic script to analyze NaN issue in Riemann simulations"""
import numpy as np
from pathlib import Path

# Test case configuration
test_config = {
    'name': 'choc_simple_motos',
    'U_L': [0.15, 8.0, 0.12, 6.0],  # [rho_moto_L, v_moto_L, rho_car_L, v_car_L]
    'U_R': [0.05, 10.0, 0.03, 8.0]
}

# Load simulation data
data_path = Path(r"d:\Projets\Alibi\Code project\kaggle\results\generic-test-runner-kernel\simulation_results\thesis_stage1\riemann_choc_simple_motos.npz")
data = np.load(data_path)

print("=" * 80)
print("RIEMANN SIMULATION DIAGNOSTIC: Choc Simple (Motos)")
print("=" * 80)

# Check initial conditions
print("\n1. INITIAL CONDITIONS (t=0)")
print("-" * 40)
rho_m_0 = data['rho_m_history'][0]
v_m_0 = data['v_m_history'][0]
rho_c_0 = data['rho_c_history'][0]
v_c_0 = data['v_c_history'][0]

print(f"Expected Left (motos):  rho={test_config['U_L'][0]:.3f}, v={test_config['U_L'][1]:.1f}")
print(f"Expected Right (motos): rho={test_config['U_R'][0]:.3f}, v={test_config['U_R'][1]:.1f}")
print(f"\nActual Left (first 5):  rho={rho_m_0[:5].mean():.3f}, v={v_m_0[:5].mean():.1f}")
print(f"Actual Right (last 5):  rho={rho_m_0[-5:].mean():.3f}, v={v_m_0[-5:].mean():.1f}")

# Wait - the values printed were 7.96 to 10.0, which don't match expected!
print(f"\nACTUAL range at t=0: rho_m ∈ [{rho_m_0.min():.3f}, {rho_m_0.max():.3f}]")
print(f"EXPECTED: rho_m should be 0.15 on left, 0.05 on right")
print("⚠️  ERROR: Initial conditions appear to be VELOCITY, not DENSITY!")

# Find when NaN first appears
print("\n2. NaN PROPAGATION ANALYSIS")
print("-" * 40)
t_history = data['t_history']
rho_m_hist = data['rho_m_history']
nan_timesteps = []
for i in range(len(t_history)):
    if np.any(np.isnan(rho_m_hist[i])):
        nan_timesteps.append(i)

if nan_timesteps:
    first_nan = nan_timesteps[0]
    print(f"First NaN at timestep: {first_nan}")
    print(f"Time: {t_history[first_nan]:.4f} s")
    print(f"dt ≈ {np.mean(np.diff(t_history[:10])):.6f} s")
else:
    print("No NaN detected in rho_m_history")

# Check velocity ranges
print("\n3. VELOCITY SANITY CHECK")
print("-" * 40)
print(f"v_m range at t=0: [{v_m_0.min():.2f}, {v_m_0.max():.2f}] m/s")
print(f"v_c range at t=0: [{v_c_0.min():.2f}, {v_c_0.max():.2f}] m/s")
print(f"Max velocity (v_max): 30 m/s (108 km/h)")
if v_m_0.max() > 30 or v_c_0.max() > 30:
    print("⚠️  CRITICAL: Velocity exceeds v_max! This will cause issues.")

# Check density
print("\n4. DENSITY SANITY CHECK")
print("-" * 40)
print(f"rho_m range at t=0: [{rho_m_0.min():.4f}, {rho_m_0.max():.4f}] veh/m")
print(f"rho_c range at t=0: [{rho_c_0.min():.4f}, {rho_c_0.max():.4f}] veh/m")
print(f"Max density (rho_max): 0.2 veh/m (200 veh/km)")
if rho_m_0.max() > 0.2 or rho_c_0.max() > 0.2:
    print("⚠️  CRITICAL: Density exceeds rho_max! Limiters should prevent this.")

print("\n" + "=" * 80)
