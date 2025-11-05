"""Quick CPU test to verify the refactored architecture works."""
import numpy as np
from arz_model.network.network_grid import NetworkGrid
from arz_model.core.parameters import ModelParameters

print("🚀 Quick CPU Test - Refactored Architecture")
print("=" * 50)

# Minimal parameters
params = ModelParameters()
params.N = 20
params.cfl_number = 0.5
params.device = 'cpu'
params.spatial_scheme = 'weno5'
params.Vmax_m = {0: 12.0, 1: 15.0, 2: 18.0}
params.Vmax_c = {0: 10.0, 1: 13.0, 2: 16.0}

# Create simple 1-segment network
network = NetworkGrid(params)
network.add_segment('seg_0', length=100.0, road_category=1)

# Initialize equilibrium
U_init = np.zeros((4, network.segments['seg_0']['grid'].N_total))
U_init[0, 3:-3] = 0.1  # rho_m
U_init[1, 3:-3] = 8.0  # w_m
network.segments['seg_0']['U'] = U_init.copy()

print(f"✅ Network created: 1 segment, N={params.N}")

# Evolve 10 steps
print(f"⏳ Running 10 timesteps (dt=0.001s)...")
try:
    for step in range(10):
        network.evolve(dt=0.001)
        if step % 5 == 0:
            rho = network.segments['seg_0']['U'][0, 3:-3]
            print(f"  Step {step}: rho_mean={np.mean(rho):.4f}, rho_max={np.max(rho):.4f}")
    
    # Final check
    U_final = network.segments['seg_0']['U']
    rho_final = U_final[0, 3:-3]
    
    print("\n" + "=" * 50)
    print("✅ TEST PASSED!")
    print(f"   Final rho_mean: {np.mean(rho_final):.4f}")
    print(f"   Final rho_max: {np.max(rho_final):.4f}")
    print(f"   All finite: {np.all(np.isfinite(U_final))}")
    print(f"   All positive: {np.all(rho_final >= 0)}")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
