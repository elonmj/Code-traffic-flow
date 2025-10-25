from arz_model.simulation.runner import SimulationRunner
import numpy as np

# Create runner with V0 overrides
runner = SimulationRunner(
    base_config_path='arz_model/config/config_base.yml',
    scenario_config_path='section_7_6_rl_performance/data/scenarios/traffic_light_control.yml',
    device='cpu',
    quiet=True
)

print(f'=== SimulationRunner initialized ===')
print(f'V0_m_override = {getattr(runner.params, "_V0_m_override", None)}')
print(f'V0_c_override = {getattr(runner.params, "_V0_c_override", None)}')

# Check initial velocities in simulation (before running)
U = runner.U
v_m = U[1, runner.grid.physical_cell_indices]
v_c = U[3, runner.grid.physical_cell_indices]

print(f'\n=== Initial state ===')
print(f'Mean v_m (motorcycles) = {np.mean(v_m):.3f} m/s ({np.mean(v_m)*3.6:.1f} km/h)')
print(f'Mean v_c (cars) = {np.mean(v_c):.3f} m/s ({np.mean(v_c)*3.6:.1f} km/h)')
print(f'Expected velocities from Lagos config:')
print(f'  v_m ~ 8.889 m/s (32 km/h)')
print(f'  v_c ~ 7.778 m/s (28 km/h)')

# Test equilibrium calculation directly
from arz_model.core import physics
test_densities = [
    (0.001, 0.001, "Very light traffic"),
    (0.05, 0.03, "Light traffic"),
    (0.15, 0.08, "Medium traffic"),
    (0.25, 0.12, "Heavy traffic")
]
R = np.array([2])

print(f'\n=== Direct equilibrium calculation tests ===')
print(f'rho_jam = {runner.params.rho_jam:.3f} veh/m')
print(f'V_creeping = {runner.params.V_creeping:.3f} m/s')
print(f'Vmax_m[2] = {runner.params.Vmax_m[2]:.3f} m/s')
print(f'Vmax_c[2] = {runner.params.Vmax_c[2]:.3f} m/s')
print()

for rho_m, rho_c, desc in test_densities:
    rho_total = rho_m + rho_c
    g = max(0, 1 - rho_total/runner.params.rho_jam)
    
    print(f'{desc}: rho_m={rho_m:.3f}, rho_c={rho_c:.3f}, rho_total={rho_total:.3f}, g={g:.3f}')
    
    # Without override (should use Vmax[R=2])
    Ve_m_no, Ve_c_no = physics.calculate_equilibrium_speed(
        np.array([rho_m]), np.array([rho_c]), R, runner.params
    )
    print(f'  WITHOUT override: Ve_m={Ve_m_no[0]:.3f} m/s ({Ve_m_no[0]*3.6:.1f} km/h), Ve_c={Ve_c_no[0]:.3f} m/s ({Ve_c_no[0]*3.6:.1f} km/h)')
    
    # With override (should use V0_m=8.889, V0_c=7.778)
    Ve_m_yes, Ve_c_yes = physics.calculate_equilibrium_speed(
        np.array([rho_m]), np.array([rho_c]), R, runner.params,
        V0_m_override=8.889,
        V0_c_override=7.778
    )
    print(f'  WITH override: Ve_m={Ve_m_yes[0]:.3f} m/s ({Ve_m_yes[0]*3.6:.1f} km/h), Ve_c={Ve_c_yes[0]:.3f} m/s ({Ve_c_yes[0]*3.6:.1f} km/h)')
    print()


