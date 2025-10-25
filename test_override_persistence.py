from arz_model.simulation.runner import SimulationRunner
from arz_model.core import physics
import numpy as np

# Create runner
runner = SimulationRunner(
    base_config_path='arz_model/config/config_base.yml',
    scenario_config_path='section_7_6_rl_performance/data/scenarios/traffic_light_control.yml',
    device='cpu',
    quiet=True
)

print('=== After initialization ===')
print(f'params._V0_m_override = {getattr(runner.params, "_V0_m_override", "NOT SET")}')
print(f'params._V0_c_override = {getattr(runner.params, "_V0_c_override", "NOT SET")}')

# Simulate calling calculate_equilibrium_speed the way _ode_rhs does it
print('\n=== Simulating _ode_rhs call pattern ===')
rho_m = np.array([0.05])
rho_c = np.array([0.03])
R = np.array([2])

V0_m_override = getattr(runner.params, '_V0_m_override', None)
V0_c_override = getattr(runner.params, '_V0_c_override', None)

print(f'Retrieved from params: V0_m_override={V0_m_override}, V0_c_override={V0_c_override}')

Ve_m, Ve_c = physics.calculate_equilibrium_speed(
    rho_m, rho_c, R, runner.params,
    V0_m_override=V0_m_override,
    V0_c_override=V0_c_override
)

print(f'Result: Ve_m={Ve_m[0]:.3f} m/s ({Ve_m[0]*3.6:.1f} km/h)')
print(f'Result: Ve_c={Ve_c[0]:.3f} m/s ({Ve_c[0]*3.6:.1f} km/h)')
print(f'Expected (with Lagos overrides): Ve_m ~ 6.5 m/s, Ve_c ~ 5.3 m/s (light traffic)')
