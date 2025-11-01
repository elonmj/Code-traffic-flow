"""Debug script to test BC parsing directly"""
import numpy as np
import logging
from arz_model.network.network_grid import NetworkGrid
from arz_model.numerics.grid_1d import Grid1D
from arz_model.numerics.weno import WENO5_JS
from arz_model.numerics.config import NetworkParams, SegmentParams

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create simple network: one segment
dx = 10.0  # meters
length = 100.0  # meters  
nx = int(length / dx)

grid = Grid1D(nx, dx, weno=WENO5_JS())

# Network params with BC
params = NetworkParams(
    junction_capacity=np.inf,
    boundary_conditions={
        'seg_0': {
            'left': {
                'type': 'inflow',
                'rho_m': 0.05,
                'v_m': 10.0
            }
        }
    }
)

# Create network
network = NetworkGrid(params=params)

# Add segment
seg_params = SegmentParams()
network.add_segment('seg_0', grid, seg_params)

# Initialize
network.initialize()

# Check initial state (should have BC applied from initialize)
seg_0 = network.segments['seg_0']
print("\n=== INITIAL STATE (after initialize) ===")
print(f"seg_0 U[:,:4] (left ghost + physical):\n{seg_0['U'][:,:4]}")
print(f"Expected: rho_m=0.05 in ghost cells [:2]")
print(f"Actual  : rho_m={seg_0['U'][0,:2]}")

# Now step once and check again
print("\n=== STEPPING ONCE ===")
network.step(dt=0.1, current_time=0.0)

print(f"\nseg_0 U[:,:4] (after step):\n{seg_0['U'][:,:4]}")
print(f"Expected: rho_m=0.05 in ghost cells [:2]")
print(f"Actual  : rho_m={seg_0['U'][0,:2]}")

# Check if BC was even detected
print("\n=== BC DETECTION CHECK ===")
print(f"params.boundary_conditions = {network.params.boundary_conditions}")
print(f"'seg_0' in params.boundary_conditions = {'seg_0' in network.params.boundary_conditions}")
