"""Test network with inflow BC but NO junction blocking"""
from arz_model.network.network_grid import NetworkGrid
from arz_model.core.parameters import ModelParameters
from arz_model.core.traffic_lights import TrafficLightController, Phase
from arz_model.core import physics
from arz_model.numerics import cfl as cfl_module
import numpy as np

params = ModelParameters()
params.spatial_scheme = "weno5"
params.device = "cpu"
params.ghost_cells = 3
params.red_light_factor = 1.0  # NO BLOCKING (100% flow)
params.L_ref = 5.0
params.v_m = 3.0  # 10.8 km/h
params.v_c = 4.0  # 14.4 km/h
params.alpha = 0.9
params.rho_jam = 0.2
params.epsilon = 1e-6
params.K_m = 1.0
params.gamma_m = 2.0
params.K_c = 0.5
params.gamma_c = 2.0
params.Vmax_m = {0: 3.0, 1: 3.0, 2: 3.0}
params.Vmax_c = {0: 4.0, 1: 4.0, 2: 4.0}
params.V_creeping = 0.0

network = NetworkGrid(params)
network.add_segment("seg_0", xmin=0, xmax=100, N=50, start_node=None, end_node="node_1")
network.add_segment("seg_1", xmin=100, xmax=200, N=50, start_node="node_1", end_node=None)

phases = [Phase(duration=60.0, green_segments=["seg_1"])]  
traffic_light = TrafficLightController(cycle_time=60.0, phases=phases, offset=0.0)

network.add_node(
    node_id="node_1", position=(100.0, 0.0),
    incoming_segments=["seg_0"], outgoing_segments=["seg_1"],
    node_type="signalized_intersection", traffic_lights=traffic_light
)

network.add_link(from_segment="seg_0", to_segment="seg_1", via_node="node_1")

network.params.boundary_conditions = {
    "seg_0": {"left": {"type": "inflow", "rho_m": 0.15, "v_m": 3.0}}
}

network.initialize()

# Warm start
for seg_id, segment in network.segments.items():
    grid = segment["grid"]
    U = segment["U"]
    rho_m_init = 0.12
    rho_m_arr = np.full(grid.N_physical, rho_m_init)
    rho_c_arr = np.zeros(grid.N_physical)
    R_local = grid.road_quality[grid.physical_cell_indices]
    v_m_eq, v_c_eq = physics.calculate_equilibrium_speed(
        rho_m_arr, rho_c_arr, R_local, params
    )
    p_m, p_c = physics.calculate_pressure(
        rho_m_arr, rho_c_arr,
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    U[0, grid.physical_cell_indices] = rho_m_init
    U[1, grid.physical_cell_indices] = v_m_eq + p_m

# Run simulation
print("[TEST: Network with inflow BC, NO junction blocking (red_light_factor=1.0)]")
t, dt_old = 0.0, 0.001

while t < 15.0:
    current_U = network.segments["seg_0"]["U"]
    grid = network.segments["seg_0"]["grid"]
    U_physical = current_U[:, grid.num_ghost_cells:grid.num_ghost_cells + grid.N_physical]
    dt = cfl_module.calculate_cfl_dt(U_physical, grid, params)
    dt = min(dt, 2.0 * dt_old)
    dt_old = dt
    if t + dt > 15.0:
        dt = 15.0 - t
    network.step(dt, t)
    t += dt
    if abs(t - int(t)) < 0.01:
        v_m = current_U[1, grid.physical_cell_indices] / (current_U[0, grid.physical_cell_indices] + 1e-10)
        print(f"t={t:.1f}s: v_max={v_m.max():.2f} m/s")

U_final = network.segments["seg_0"]["U"]
v_m_final = np.mean(U_final[1, 2:-2] / (U_final[0, 2:-2] + 1e-10))
print(f"\nFinal v_m={v_m_final:.2f} m/s")
print("TEST:", "PASS ✅" if v_m_final < 20 else "FAIL ❌")
