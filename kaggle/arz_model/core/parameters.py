import yaml
import copy
import os
from typing import List, Dict, Optional

# Conversion factors
KMH_TO_MS = 1000.0 / 3600.0
VEH_KM_TO_VEH_M = 1.0 / 1000.0

def _deep_merge_dicts(base, update):
    """
    Recursively merges update dict into base dict.
    Update values overwrite base values.
    """
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

class ModelParameters:
    """
    Loads, stores, and provides access to model parameters, handling unit conversions.
    This class is being refactored to act as a bridge between the new Pydantic
    configuration system and the legacy simulation engine code.

    NEW (Pydantic-based initialization):
        params = ModelParameters(config=my_network_simulation_config)
    
    LEGACY (YAML-based initialization):
        params = ModelParameters()
        params.load_from_yaml(...)

    Internal units are SI: meters (m), seconds (s), vehicles/meter (veh/m).
    """
    def __init__(self, config: Optional['NetworkSimulationConfig'] = None, **kwargs):
        """
        Initializes parameters. If a Pydantic config object is provided,
        it populates the parameters from it. Otherwise, initializes with defaults.
        """
        # Initialize all attributes to None or default values
        self._initialize_defaults()

        if config:
            self.load_from_pydantic(config)
        else:
            # Allow legacy kwargs like num_lanes, Vmax_c_kmh for backward compatibility
            if 'num_lanes' in kwargs:
                self.num_lanes = kwargs['num_lanes']
            if 'Vmax_c_kmh' in kwargs:
                self.Vmax_c = {1: kwargs['Vmax_c_kmh'] * KMH_TO_MS}
            if 'Vmax_m_kmh' in kwargs:
                self.Vmax_m = {1: kwargs['Vmax_m_kmh'] * KMH_TO_MS}

    def _initialize_defaults(self):
        """Sets all parameter attributes to their default state before loading."""
        # Physical Parameters (SI units)
        self.alpha: float = None
        self.V_creeping: float = None
        self.rho_jam: float = None
        self.gamma_m: float = None
        self.gamma_c: float = None
        self.K_m: float = None
        self.K_c: float = None
        self.tau_m: float = None
        self.tau_c: float = None
        self.Vmax_c: dict = {1: 50.0 * KMH_TO_MS}
        self.Vmax_m: dict = {1: 50.0 * KMH_TO_MS}
        self.flux_composition: dict = {}
        self.num_lanes = 1

        # Numerical Parameters
        self.cfl_number: float = 0.8
        self.ghost_cells: int = 3
        self.num_ghost_cells: int = 3
        self.spatial_scheme: str = 'weno5'
        self.numerical_flux: str = 'godunov'
        self.time_scheme: str = 'ssprk3'
        self.ode_solver: str = 'RK45'
        self.ode_rtol: float = 1e-3
        self.ode_atol: float = 1e-6
        self.epsilon: float = 1e-6

        # Scenario specific
        self.scenario_name: str = "default"
        self.N: int = None
        self.xmin: float = None
        self.xmax: float = None
        self.t_final: float = 3600.0
        self.output_dt: float = 1.0
        self.initial_conditions: dict = {}
        self.boundary_conditions: dict = {}
        self.road_quality_definition: list | str = None

        # Network parameters
        self.has_network: bool = False
        self.nodes: Optional[List[Dict]] = []
        self.network_segments: Optional[List[Dict]] = []
        self.enable_traffic_lights: bool = True
        self.enable_creeping: bool = True
        self.enable_queue_management: bool = True
        self.max_queue_length: Optional[float] = 100.0
        self.red_light_factor: Optional[float] = 0.1
        self.rho_eq_m: Optional[float] = 0.01
        self.rho_eq_c: Optional[float] = 0.01
        
        # Behavioral coupling parameters
        self.theta_moto_insertion: Optional[float] = 0.5
        self.theta_moto_circulation: Optional[float] = 0.5
        self.theta_moto_signalized: Optional[float] = 0.5
        self.theta_car_signalized: Optional[float] = 0.5
        self.theta_moto_priority: Optional[float] = 0.5
        self.theta_car_priority: Optional[float] = 0.5
        self.theta_moto_secondary: Optional[float] = 0.5
        self.theta_car_secondary: Optional[float] = 0.5

    def load_from_pydantic(self, config: 'NetworkSimulationConfig'):
        """
        Populates parameters from a Pydantic NetworkSimulationConfig object.
        """
        # Physics parameters
        phys = config.physics
        self.alpha = phys.alpha
        self.V_creeping = phys.v_creeping_kmh * KMH_TO_MS
        self.rho_jam = phys.rho_max
        self.gamma_m = phys.gamma_m
        self.gamma_c = phys.gamma_c
        self.K_m = phys.k_m * KMH_TO_MS
        self.K_c = phys.k_c * KMH_TO_MS
        self.tau_m = phys.tau_m
        self.tau_c = phys.tau_c
        self.red_light_factor = phys.red_light_factor
        self.enable_creeping = phys.enable_creeping
        self.enable_queue_management = phys.enable_queue_management
        self.max_queue_length = phys.max_queue_length_m
        
        self.Vmax_c = {1: phys.v_max_c_kmh * KMH_TO_MS}
        self.Vmax_m = {1: phys.v_max_m_kmh * KMH_TO_MS}

        # Numerical parameters
        grid_cfg = config.grid
        self.num_ghost_cells = grid_cfg.num_ghost_cells
        self.ghost_cells = grid_cfg.num_ghost_cells
        self.spatial_scheme = grid_cfg.spatial_scheme
        self.numerical_flux = grid_cfg.numerical_flux
        self.time_scheme = grid_cfg.time_scheme
        self.epsilon = phys.epsilon

        # Simulation parameters
        time_cfg = config.time
        self.t_final = time_cfg.t_final
        self.output_dt = time_cfg.output_dt
        self.cfl_number = time_cfg.cfl_factor
        
        # Network flag
        self.has_network = True
        
        # These are now handled at the segment level
        self.initial_conditions = {}
        self.boundary_conditions = {}

    def load_from_yaml(self, base_config_path, scenario_config_path=None):
        """
        Loads parameters from base YAML and optionally merges a scenario YAML.
        Performs unit conversions to internal SI units.
        """
        if not os.path.exists(base_config_path):
            raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")

        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

        if scenario_config_path:
            if not os.path.exists(scenario_config_path):
                raise FileNotFoundError(f"Scenario configuration file not found: {scenario_config_path}")
            with open(scenario_config_path, 'r') as f:
                scenario_config = yaml.safe_load(f) if f else {} # Handle empty file
            config = _deep_merge_dicts(config, scenario_config)
            # Prioritize scenario_name from inside the scenario file, fallback to filename
            self.scenario_name = scenario_config.get('scenario_name', os.path.splitext(os.path.basename(scenario_config_path))[0])

        # --- Assign Physical Parameters (with unit conversion) ---
        self.alpha = float(config['alpha'])
        self.V_creeping = float(config['V_creeping_kmh']) * KMH_TO_MS
        self.rho_jam = float(config['rho_jam_veh_km']) * VEH_KM_TO_VEH_M

        pressure_params = config['pressure']
        self.gamma_m = float(pressure_params['gamma_m'])
        self.gamma_c = float(pressure_params['gamma_c'])
# --- DEBUG: Print K values before assignment ---
        print(f"DEBUG PARAMS: Reading K_m_kmh = {pressure_params.get('K_m_kmh')}")
        print(f"DEBUG PARAMS: Reading K_c_kmh = {pressure_params.get('K_c_kmh')}")
        # --- END DEBUG ---
        self.K_m = float(pressure_params['K_m_kmh']) * KMH_TO_MS
        self.K_c = float(pressure_params['K_c_kmh']) * KMH_TO_MS

# --- DEBUG: Print K values after assignment (SI units) ---
        print(f"DEBUG PARAMS: Assigned self.K_m = {self.K_m}")
        print(f"DEBUG PARAMS: Assigned self.K_c = {self.K_c}")
        # --- END DEBUG ---
        relaxation_params = config['relaxation']
        self.tau_m = float(relaxation_params['tau_m_sec'])
        self.tau_c = float(relaxation_params['tau_c_sec'])

        vmax_params = config['Vmax_kmh']
        self.Vmax_c = {int(k): float(v) * KMH_TO_MS for k, v in vmax_params['c'].items()}
        self.Vmax_m = {int(k): float(v) * KMH_TO_MS for k, v in vmax_params['m'].items()}

        self.flux_composition = config['flux_composition']

        # --- Assign Numerical Parameters ---
        self.cfl_number = float(config['cfl_number'])
        self.ghost_cells = int(config['ghost_cells'])
        self.num_ghost_cells = self.ghost_cells  # Alias for compatibility
        self.spatial_scheme = str(config.get('spatial_scheme', 'first_order'))  # Default to 'first_order' if not present
        self.time_scheme = str(config.get('time_scheme', 'euler'))     # Default to 'euler' if not present
        self.ode_solver = str(config['ode_solver'])
        self.ode_rtol = float(config['ode_rtol'])
        self.ode_atol = float(config['ode_atol'])
        self.epsilon = float(config['epsilon'])
        
        # Validate spatial_scheme
        valid_spatial_schemes = ['first_order', 'weno5', 'godunov']
        if self.spatial_scheme not in valid_spatial_schemes:
            raise ValueError(
                f"Invalid spatial_scheme='{self.spatial_scheme}'. "
                f"Must be one of: {valid_spatial_schemes}"
            )

        # --- Assign Scenario Parameters (if present in merged config) ---
        # --- Assign Scenario Parameters (if present in merged config) ---
        # Access nested dictionaries safely using .get('key', {}) to avoid errors if keys are missing
        numerical_config = config.get('numerical', {})
        grid_config = config.get('grid', {})
        simulation_config = config.get('simulation', {})

        # Get values from nested structures first, then check top-level as fallback
        self.N = grid_config.get('N', config.get('N')) # Look in grid_config first
        self.xmin = grid_config.get('xmin', config.get('xmin'))
        self.xmax = grid_config.get('xmax', config.get('xmax'))
        self.t_final = simulation_config.get('t_final_sec', config.get('t_final'))
        self.output_dt = simulation_config.get('output_dt_sec', config.get('output_dt'))

        # These are typically top-level in the scenario config or base config
        # Load initial conditions and perform unit conversion for state arrays
        raw_initial_conditions = config.get('initial_conditions', {})
        self.initial_conditions = copy.deepcopy(raw_initial_conditions)
        
        # BUG #17 FIX: Convert IC density values from veh/km (config) to veh/m (SI units)
        # Similar to BC conversion below, IC state arrays must be converted
        # uniform_state() docstring explicitly expects densities in veh/m
        ic_type = self.initial_conditions.get('type', '').lower()
        
        if ic_type == 'uniform':
            state = self.initial_conditions.get('state')
            if state is not None and len(state) == 4:
                # Convert from [veh/km, m/s, veh/km, m/s] to [veh/m, m/s, veh/m, m/s]
                # Velocities are already in m/s, only densities need conversion
                self.initial_conditions['state'] = [
                    state[0] * VEH_KM_TO_VEH_M,  # rho_m (veh/km → veh/m)
                    state[1],                     # w_m (already m/s)
                    state[2] * VEH_KM_TO_VEH_M,  # rho_c (veh/km → veh/m)
                    state[3]                      # w_c (already m/s)
                ]
        
        elif ic_type == 'riemann':
            # Convert U_L (left state)
            U_L = self.initial_conditions.get('U_L')
            if U_L is not None and len(U_L) == 4:
                self.initial_conditions['U_L'] = [
                    U_L[0] * VEH_KM_TO_VEH_M,  # rho_m
                    U_L[1],                     # w_m
                    U_L[2] * VEH_KM_TO_VEH_M,  # rho_c
                    U_L[3]                      # w_c
                ]
            
            # Convert U_R (right state)
            U_R = self.initial_conditions.get('U_R')
            if U_R is not None and len(U_R) == 4:
                self.initial_conditions['U_R'] = [
                    U_R[0] * VEH_KM_TO_VEH_M,
                    U_R[1],
                    U_R[2] * VEH_KM_TO_VEH_M,
                    U_R[3]
                ]
        
        # Note: uniform_equilibrium IC is handled in runner.py with explicit conversion
        # density_hump and sine_wave_perturbation don't use state arrays

        # Load boundary conditions and perform unit conversion for inflow states
        raw_boundary_conditions = config.get('boundary_conditions', {})
        self.boundary_conditions = {}
        for boundary_side, bc_config in raw_boundary_conditions.items():
            processed_bc_config = copy.deepcopy(bc_config) # Work on a copy
            if processed_bc_config.get('type', '').lower() == 'inflow':
                state = processed_bc_config.get('state')
                if state is not None and len(state) == 4:
                    # Convert state values from [veh/km, km/h, veh/km, km/h] to [veh/m, m/s, veh/m, m/s]
                    processed_bc_config['state'] = [
                        state[0] * VEH_KM_TO_VEH_M, # rho_m
                        state[1] * KMH_TO_MS,      # w_m (assuming w is in same units as v in config)
                        state[2] * VEH_KM_TO_VEH_M, # rho_c
                        state[3] * KMH_TO_MS       # w_c (assuming w is in same units as v in config)
                    ]
                    # DEBUG print to verify conversion
                    # print(f"DEBUG PARAMS: Converted {boundary_side} inflow state: {state} -> {processed_bc_config['state']}")

            self.boundary_conditions[boundary_side] = processed_bc_config

        # Store the entire 'road' dictionary from the config
        self.road = config.get('road', {}) # Store the dict itself

        # Get mass conservation check config if present (nested or top-level)
        self.mass_conservation_check = config.get('mass_conservation_check')

        # Load network configuration
        network_config = config.get('network', {})
        self.has_network = network_config.get('has_network', False)
        self.nodes = network_config.get('nodes', [])
        self.network_segments = network_config.get('segments', [])

        # Load network-specific parameters (with defaults)
        self.enable_traffic_lights = config.get('enable_traffic_lights', True)
        self.enable_creeping = config.get('enable_creeping', True)
        self.enable_queue_management = config.get('enable_queue_management', True)
        self.max_queue_length = config.get('max_queue_length', 100.0)
        self.red_light_factor = config.get('red_light_factor', 0.1)
        self.rho_eq_m = config.get('rho_eq_m', 0.01)
        self.rho_eq_c = config.get('rho_eq_c', 0.01)

        # Load behavioral coupling parameters (θ_k)
        if 'behavioral_coupling' in config:
            bc = config['behavioral_coupling']
            self.theta_moto_insertion = float(bc.get('theta_moto_insertion', 0.2))
            self.theta_moto_circulation = float(bc.get('theta_moto_circulation', 0.8))
            self.theta_moto_signalized = float(bc.get('theta_moto_signalized', 0.8))
            self.theta_car_signalized = float(bc.get('theta_car_signalized', 0.5))
            self.theta_moto_priority = float(bc.get('theta_moto_priority', 0.9))
            self.theta_car_priority = float(bc.get('theta_car_priority', 0.9))
            self.theta_moto_secondary = float(bc.get('theta_moto_secondary', 0.1))
            self.theta_car_secondary = float(bc.get('theta_car_secondary', 0.1))

        # --- Validation (Optional but recommended) ---
        self._validate_parameters()

    def _validate_parameters(self):
        """ Basic validation of loaded parameters. """
        # Add checks here, e.g., ensure required scenario params are loaded
        if self.N is not None and self.N <= 0:
            raise ValueError("Number of grid cells N must be positive.")
        if self.rho_jam <= 0:
            raise ValueError("Jam density rho_jam must be positive.")
        if not (0 <= self.alpha < 1):
             raise ValueError("Alpha must be in the range [0, 1).")
        
        # Validate behavioral coupling parameters (θ_k ∈ [0,1])
        theta_params = [
            ('theta_moto_insertion', self.theta_moto_insertion),
            ('theta_moto_circulation', self.theta_moto_circulation),
            ('theta_moto_signalized', self.theta_moto_signalized),
            ('theta_car_signalized', self.theta_car_signalized),
            ('theta_moto_priority', self.theta_moto_priority),
            ('theta_car_priority', self.theta_car_priority),
            ('theta_moto_secondary', self.theta_moto_secondary),
            ('theta_car_secondary', self.theta_car_secondary)
        ]
        
        for name, value in theta_params:
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {value}")
        # ... add more checks as needed

    def __str__(self):
        """ String representation for easy printing. """
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"ModelParameters({attrs})"

# Example Usage (can be removed or put under if __name__ == '__main__':)
# if __name__ == '__main__':
#     # Assumes config/config_base.yml exists relative to this script
#     # You might need to adjust the path depending on where you run it from
#     script_dir = os.path.dirname(__file__)
#     base_config_file = os.path.join(script_dir, '..', '..', 'config', 'config_base.yml')
#
#     params = ModelParameters()
#     try:
#         params.load_from_yaml(base_config_file)
#         print("Base Parameters Loaded Successfully:")
#         print(f"Alpha: {params.alpha}")
#         print(f"Rho Jam (veh/m): {params.rho_jam}")
#         print(f"Vmax_c[1] (m/s): {params.Vmax_c.get(1)}")
#         print(f"Tau_m (s): {params.tau_m}")
#         print(f"CFL: {params.cfl_number}")
#
#         # Example loading a scenario (assuming a dummy scenario file exists)
#         # scenario_file = os.path.join(script_dir, '..', '..', 'config', 'scenario_test.yml')
#         # with open(scenario_file, 'w') as f:
#         #     yaml.dump({'N': 100, 't_final': 60.0}, f)
#         # params_scenario = ModelParameters()
#         # params_scenario.load_from_yaml(base_config_file, scenario_file)
#         # print("\nScenario Parameters Loaded:")
#         # print(f"N: {params_scenario.N}")
#         # print(f"t_final: {params_scenario.t_final}")
#         # print(f"Alpha (from base): {params_scenario.alpha}")
#
#     except FileNotFoundError as e:
#         print(f"Error loading config: {e}")
#     except Exception as e:
#         print(f"An error occurred: {e}")