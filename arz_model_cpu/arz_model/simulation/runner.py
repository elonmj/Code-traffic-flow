import numpy as np
import time
import copy # For deep merging overrides
import os
import yaml # To load road quality if defined directly in scenario
from tqdm import tqdm # For progress bar
from numba import cuda # Import cuda for device arrays
from typing import Union, Optional

from ..analysis import metrics
from ..io import data_manager
from ..core.parameters import ModelParameters, VEH_KM_TO_VEH_M # Import the constant
from ..grid.grid1d import Grid1D
from ..numerics import boundary_conditions, cfl, time_integration
from . import initial_conditions # Import the initial conditions module

# NEW: Import Pydantic config system
try:
    from ..config import (
        SimulationConfig, GridConfig, UniformIC, 
        BoundaryConditionsConfig, PeriodicBC, PhysicsConfig
    )
    from ..config.network_simulation_config import NetworkSimulationConfig
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    SimulationConfig = None
    NetworkSimulationConfig = None

from .execution.network_simulator import NetworkSimulator


model_config = {"extra": "forbid"}


class SimulationRunner:
    """
    Orchestrates the execution of a single simulation scenario.

    Supports two initialization modes:
    1. NEW (Pydantic): Pass SimulationConfig directly
    2. LEGACY (YAML): Pass scenario/base config paths (backward compatible)

    Initializes the grid, parameters, and initial state, then runs the
    time loop, applying numerical methods and storing results.
    """

    def __init__(self,
                 simulation_config: Optional[NetworkSimulationConfig] = None,
                 scenario_config_path: Optional[str] = None,
                 base_config_path: str = 'config/config_base.yml',
                 override_params: dict = None,
                 quiet: bool = False,
                 device: Optional[str] = None,
                 network_grid: Optional['NetworkGrid'] = None):
        """
        Initializes the simulation runner.

        MODES:
        1. Network Simulation:
            runner = SimulationRunner(network_grid=my_network_grid)
        
        2. Single-Segment Simulation (Pydantic):
            runner = SimulationRunner(config=ConfigBuilder.section_7_6())
        
        3. Single-Segment Simulation (Legacy YAML):
            runner = SimulationRunner(scenario_config_path='scenarios/test.yml')

        Args:
            network_grid: A fully built NetworkGrid object for multi-segment simulation.
            config: Pydantic SimulationConfig for single-segment simulation.
            scenario_config_path: Path to legacy YAML scenario file.
            base_config_path: Path to legacy base YAML file.
            override_params: Dict of parameter overrides for legacy mode.
            quiet: Suppress print statements.
            device: Override device ('cpu' or 'gpu').
        """
        # ====================================================================
        # DETECT INITIALIZATION MODE
        # ====================================================================
        
        # Case 1: Network Simulation Mode
        if network_grid is not None:
            if not isinstance(simulation_config, NetworkSimulationConfig):
                raise TypeError("Network mode requires a `NetworkSimulationConfig` object.")
            self._init_from_network_grid(network_grid, simulation_config, quiet, device)
            return

        # Case 2: Single-Segment Pydantic Mode
        if PYDANTIC_AVAILABLE and isinstance(simulation_config, SimulationConfig):
            self._init_from_pydantic(simulation_config, quiet, device)
            return
        
        # Case 3: LEGACY Single-Segment YAML Mode
        if scenario_config_path is not None:
            self._init_from_yaml(scenario_config_path, base_config_path, override_params, quiet, device)
            return
        
        # Case 4: ERROR - No valid initialization mode
        raise ValueError(
            "SimulationRunner requires one of:\n"
            "  1. network_grid: A built NetworkGrid object\n"
            "  2. config: A Pydantic SimulationConfig object\n"
            "  3. scenario_config_path: Path to a legacy YAML file"
        )

    def _init_from_network_grid(self, network_grid: 'NetworkGrid', config: 'NetworkSimulationConfig', quiet: bool, device: Optional[str]):
        """Initializes the runner for a network simulation."""
        self.mode = 'network'
        self.is_network_simulation = True
        self.quiet = quiet
        self.network_grid = network_grid
        
        # Use the provided Pydantic config
        self.config = config
        # This is a critical change: ModelParameters now gets its values from the Pydantic config
        self.params = ModelParameters(config=config)
        
        # The `simulation_config` attribute is also required for other parts of the runner
        self.simulation_config = config
        
        self.device = self._resolve_device(device, self.params, self.quiet)
        
        if not self.quiet:
            print(f"   - Mode: Network Simulation")
            print(f"   - Device: {self.device.upper()}")
            print(f"   - Segments: {len(self.network_grid.segments)}")
            print(f"   - Nodes: {len(self.network_grid.nodes)}")

        # The NetworkSimulator will handle the time loop
        self.network_simulator = NetworkSimulator(
            network=self.network_grid,
            config=self.config,
            quiet=self.quiet
        )

    def _init_from_pydantic(self, config: 'SimulationConfig', quiet: bool, device: Optional[str]):
        """Initialize from Pydantic SimulationConfig (single-segment)"""
        self.is_network_simulation = False
        self.config = config
        self.quiet = quiet if quiet is not None else config.quiet
        self.device = device if device is not None else config.device
        
        if not self.quiet:
            print(f"✅ Initializing simulation with Pydantic config")
            print(f"   Using device: {self.device}")
        
        # Create legacy params object for backward compatibility
        # TODO Phase 3: Remove this after extracting classes
        self.params = self._create_legacy_params_from_config(config)
        self.params.device = self.device
        
        # Continue with common initialization
        self._common_initialization()
    
    @staticmethod
    def _resolve_device(device_override: Optional[str], params: 'ModelParameters', quiet: bool) -> str:
        """
        Determines the computation device ('cpu' or 'gpu') to use.

        Priority order:
        1. `device_override` argument.
        2. `device` from the parameters object.
        3. Default to 'cpu'.

        If 'gpu' is chosen, it verifies CUDA availability and falls back to 'cpu'
        with a warning if not available.
        """
        chosen_device = 'cpu' # Default
        
        if device_override:
            chosen_device = device_override.lower()
        elif hasattr(params, 'device') and params.device:
            chosen_device = params.device.lower()

        if chosen_device == 'gpu':
            if not cuda.is_available():
                if not quiet:
                    print("⚠️ WARNING: GPU device requested, but CUDA is not available. Falling back to CPU.")
                return 'cpu'
            return 'gpu'
        
        return 'cpu'

    def _init_from_network_config(self, config: 'NetworkSimulationConfig', quiet: bool, device: Optional[str]):
        """Initialize from Pydantic NetworkSimulationConfig (NEW MODE - multi-segment)"""
        self.config = config
        self.quiet = quiet
        self.device = device if device is not None else 'cpu'
        
        if not self.quiet:
            print(f"✅ Initializing NETWORK simulation with Pydantic config")
            print(f"   Segments: {len(config.segments)}")
            print(f"   Nodes: {len(config.nodes)}")
            print(f"   Using device: {self.device}")

        # The builder should have already created the network object.
        # Here, we just need to instantiate the simulator.
        # This part of the code assumes the `config` might be a standalone
        # object without a pre-built network, which is not our current workflow.
        # For now, we'll assume the network is passed in or built separately.
        
        # This method should receive the fully built NetworkGrid object.
        # Let's adjust the logic to expect that.
        # The runner's __init__ will need to be adapted.
        
        # For now, let's assume the network is part of the config object for simplicity
        # This will be refactored.
        if not hasattr(config, 'network_grid'):
             raise ValueError("NetworkSimulationConfig must have a 'network_grid' attribute containing the built network.")

        self.simulator = NetworkSimulator(
            network=config.network_grid,
            config=config,
            quiet=self.quiet
        )
        
    def _init_from_network_config(self, network: 'NetworkGrid', quiet: bool, device: Optional[str]):
        """Initialize from a pre-built NetworkGrid object."""
        self.network = network
        self.config = network.simulation_config
        self.quiet = quiet
        self.device = device or self.config.device
        
        # The simulator is the specific execution engine
        self.simulator = NetworkSimulator(self.network, self.config, self.quiet)

    def _init_from_yaml(self, scenario_config_path: str, base_config_path: str,
                       override_params: dict, quiet: bool, device: Optional[str]):
        """Initialize from YAML files (LEGACY MODE - backward compatible)"""
        self.quiet = quiet
        self.device = device if device is not None else 'cpu'
        
        if not self.quiet:
            print(f"Initializing simulation from scenario: {scenario_config_path}")
            print(f"Using device: {self.device}")
        
        # Load parameters
        self.params = ModelParameters()
        self.params.load_from_yaml(base_config_path, scenario_config_path)
        
        # ✅ ARCHITECTURAL FIX (2025-10-24): Load V0 overrides from network config if present
        # This allows scenarios with network segments to specify V0_m/V0_c per segment
        # For single-segment scenarios (like RL traffic light), use first segment's V0 parameters
        self._load_network_v0_overrides(scenario_config_path)
        
        # Apply overrides if provided
        if override_params:
            if not self.quiet:
                print(f"Applying parameter overrides: {override_params}")
            for key, value in override_params.items():
                # Simple override for top-level attributes
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
                else:
                    # Handle potential nested overrides if needed in the future
                    # For now, just warn if the key doesn't exist directly
                    if not self.quiet:
                        print(f"Warning: Override key '{key}' not found as a direct attribute of ModelParameters.")
        
        # --- Add the device setting to the parameters object ---
        self.params.device = self.device
        # -------------------------------------------------------
        
        if not self.quiet:
            print(f"Parameters loaded for scenario: {getattr(self.params, 'scenario_name', 'unknown')}")
        
        # Continue with common initialization
        self._common_initialization()
    
    def _create_legacy_params_from_config(self, config: 'SimulationConfig') -> ModelParameters:
        """
        Create legacy ModelParameters from Pydantic SimulationConfig
        
        This is a temporary adapter for Phase 2. Will be removed in Phase 3.
        """
        params = ModelParameters()
        
        # Grid parameters
        params.N = config.grid.N
        params.xmin = config.grid.xmin
        params.xmax = config.grid.xmax
        params.ghost_cells = config.grid.ghost_cells
        
        # Time parameters
        params.t_final = config.t_final
        params.output_dt = config.output_dt
        params.cfl_number = config.cfl_number
        params.max_iterations = config.max_iterations
        
        # Physics parameters
        params.lambda_m = config.physics.lambda_m
        params.lambda_c = config.physics.lambda_c
        params.V_max_m = config.physics.V_max_m
        params.V_max_c = config.physics.V_max_c
        params.alpha = config.physics.alpha
        
        # Road quality (default to uniform quality from physics config)
        params.road = {
            'quality_type': 'uniform',
            'quality_value': config.physics.default_road_quality
        }
        
        # Additional required physics parameters (use defaults from literature)
        params.rho_jam = 0.2  # veh/m (200 veh/km) - typical jam density
        params.gamma_m = 2.0  # Pressure exponent motorcycles
        params.gamma_c = 2.0  # Pressure exponent cars
        params.K_m = 20.0 / 3.6  # m/s (20 km/h converted)
        params.K_c = 20.0 / 3.6  # m/s
        params.tau_m = 1.0 / config.physics.lambda_m  # Relaxation time = 1/lambda
        params.tau_c = 1.0 / config.physics.lambda_c
        params.V_creeping = 0.1  # m/s (slow creep speed)
        params.epsilon = 1e-10  # Small number for numerical stability
        
        # Velocity tables (simplified - use single value for all road qualities)
        params.Vmax_m = {i: config.physics.V_max_m / 3.6 for i in range(1, 11)}  # Convert km/h to m/s
        params.Vmax_c = {i: config.physics.V_max_c / 3.6 for i in range(1, 11)}
        
        # Numerical parameters
        params.spatial_scheme = 'first_order'
        params.time_scheme = 'euler'
        params.ode_solver = 'RK45'  # Use scipy's RK45 method
        params.ode_rtol = 1e-6
        params.ode_atol = 1e-8
        
        # Convert Pydantic IC config to legacy dict format
        params.initial_conditions = self._convert_ic_to_legacy(config.initial_conditions)
        
        # Convert Pydantic BC config to legacy dict format
        params.boundary_conditions = self._convert_bc_to_legacy(config.boundary_conditions)
        
        # Network system
        params.has_network = config.has_network
        
        # Device
        params.device = config.device
        
        # Scenario name (for logging)
        params.scenario_name = "pydantic_config"
        
        return params
    
    def _convert_ic_to_legacy(self, ic_config) -> dict:
        """Convert Pydantic IC config to legacy dict format"""
        ic_type = str(ic_config.type).replace('ICType.', '').lower()
        
        if ic_type == 'uniform_equilibrium':
            return {
                'type': 'uniform_equilibrium',
                'rho_m': ic_config.rho_m,
                'rho_c': ic_config.rho_c,
                'R_val': ic_config.R_val
            }
        elif ic_type == 'uniform':
            return {
                'type': 'uniform',
                'rho_m': ic_config.rho_m,
                'w_m': ic_config.w_m,
                'rho_c': ic_config.rho_c,
                'w_c': ic_config.w_c
            }
        elif ic_type == 'riemann':
            return {
                'type': 'riemann',
                'x_discontinuity': ic_config.x_discontinuity,
                'rho_m_left': ic_config.rho_m_left,
                'w_m_left': ic_config.w_m_left,
                'rho_c_left': ic_config.rho_c_left,
                'w_c_left': ic_config.w_c_left,
                'rho_m_right': ic_config.rho_m_right,
                'w_m_right': ic_config.w_m_right,
                'rho_c_right': ic_config.rho_c_right,
                'w_c_right': ic_config.w_c_right
            }
        else:
            raise ValueError(f"Unsupported IC type for legacy conversion: {ic_type}")
    
    def _convert_bc_to_legacy(self, bc_config) -> dict:
        """Convert Pydantic BC config to legacy dict format"""
        legacy_bc = {}
        
        # Convert left BC
        left_type = str(bc_config.left.type).replace('BCType.', '').lower()
        legacy_bc['left'] = {'type': left_type}
        if hasattr(bc_config.left, 'state'):
            legacy_bc['left']['state'] = bc_config.left.state.to_array()
            if hasattr(bc_config.left, 'schedule') and bc_config.left.schedule:
                legacy_bc['left']['schedule'] = [
                    {'time': item.time, 'phase_id': item.phase_id}
                    for item in bc_config.left.schedule
                ]
        
        # Convert right BC
        right_type = str(bc_config.right.type).replace('BCType.', '').lower()
        legacy_bc['right'] = {'type': right_type}
        if hasattr(bc_config.right, 'state'):
            legacy_bc['right']['state'] = bc_config.right.state.to_array()
        
        # Traffic signal phases (for RL control)
        if bc_config.traffic_signal_phases:
            legacy_bc['traffic_signal_phases'] = {}
            for side, phases in bc_config.traffic_signal_phases.items():
                legacy_bc['traffic_signal_phases'][side] = {
                    phase_id: state.to_array()
                    for phase_id, state in phases.items()
                }
        
        return legacy_bc
    
    def _common_initialization(self):
        """Common initialization logic for both Pydantic and YAML modes"""
        # Validate required scenario parameters
        if self.params.N is None or self.params.xmin is None or self.params.xmax is None:
            raise ValueError("Grid parameters (N, xmin, xmax) must be defined in the configuration.")
        if self.params.t_final is None or self.params.output_dt is None:
             raise ValueError("Simulation time parameters (t_final, output_dt) must be defined.")
        if not self.params.initial_conditions:
             raise ValueError("Initial conditions must be defined in the configuration.")
        if not self.params.boundary_conditions:
             raise ValueError("Boundary conditions must be defined in the configuration.")
        
        # Initialize grid
        self.grid = Grid1D(
            N=self.params.N,
            xmin=self.params.xmin,
            xmax=self.params.xmax,
            num_ghost_cells=self.params.ghost_cells
        )
        if not self.quiet:
            print(f"Grid initialized: {self.grid}")

        # Load road quality R(x)
        self._load_road_quality()
        if not self.quiet:
            print("Road quality loaded.")

        # Create initial state U^0
        self.U = self._create_initial_state()
        if not self.quiet:
            print("Initial state created.")

        # --- Transfer initial state and road quality to GPU if needed ---
        self.d_U = None # Handle for GPU state array
        self.d_R = None # Handle for GPU road quality array
        if self.device == 'gpu':
            if not self.quiet:
                print("Transferring initial state and road quality to GPU...")
            try:
                self.d_U = cuda.to_device(self.U)
                if self.grid.road_quality is not None:
                    self.d_R = cuda.to_device(self.grid.road_quality)
                else:
                    # Should not happen if _load_road_quality succeeded, but handle defensively
                    raise ValueError("Road quality not loaded, cannot transfer to GPU.")
                if not self.quiet:
                    print("GPU data transfer complete.")
            except Exception as e:
                print(f"Error transferring data to GPU: {e}")
                # Fallback to CPU or raise error? For now, raise.
                raise RuntimeError(f"Failed to initialize GPU data: {e}") from e
        # ----------------------------------------------------------------

        # --- Initialize Boundary Condition Schedules and Current State ---
        # This needs to happen *before* applying initial BCs so current_bc_params is ready
        self._initialize_boundary_conditions()
        # -------------------------------------------------------------

        # --- Apply initial boundary conditions ---
        # Apply BCs *after* potential GPU transfer and *after* initializing BC schedules
        initial_U_array = self.d_U if self.device == 'gpu' else self.U
        # Use the initialized current_bc_params which has the correct type for t=0
        # Pass both params (for device, physics constants) and current_bc_params (for BC types/states), and t_current
        boundary_conditions.apply_boundary_conditions(initial_U_array, self.grid, self.params, self.current_bc_params, t_current=0.0)
        if not self.quiet:
            print("Initial boundary conditions applied.")
        # -----------------------------------------

        # Initialize time and results storage
        self.t = 0.0
        self.times = [self.t]
        # Store only physical cells (always store CPU copy)
        self.states = [np.copy(self.U[:, self.grid.physical_cell_indices])]
        self.step_count = 0

        # --- Mass Conservation Check Initialization (uses CPU data) ---
        self.mass_check_config = getattr(self.params, 'mass_conservation_check', None)
        if self.mass_check_config:
            if not self.quiet:
                print("Initializing mass conservation check...")
            self.mass_times = []
            self.mass_m_data = []
            self.mass_c_data = []
            # Initial mass calculation
            U_phys_initial = self.U[:, self.grid.physical_cell_indices]
            try:
                self.initial_mass_m = metrics.calculate_total_mass(U_phys_initial, self.grid, class_index=0)
                self.initial_mass_c = metrics.calculate_total_mass(U_phys_initial, self.grid, class_index=2)
                self.mass_times.append(0.0)
                self.mass_m_data.append(self.initial_mass_m)
                self.mass_c_data.append(self.initial_mass_c)
                if not self.quiet:
                    print(f"  Initial Mass (Motos): {self.initial_mass_m:.6e}")
                    print(f"  Initial Mass (Cars):  {self.initial_mass_c:.6e}")
            except Exception as e:
                if not self.quiet:
                    print(f"Error calculating initial mass: {e}")
                # Decide how to handle this - maybe disable the check?
                self.mass_check_config = None # Disable check if initial calc fails

        # --- Initialize Network System ---
        if self.params.has_network:
            if not self.quiet:
                print("Initializing network system...")
            self._initialize_network()
        else:
            self.nodes = None
            self.network_coupling = None
    
    def _initialize_network(self):
        """Initialize the network nodes and coupling system."""
        from ..core.intersection import create_intersection_from_config
        from ..numerics.network_coupling_corrected import NetworkCouplingCorrected

        self.nodes = []
        if self.params.nodes:
            for node_config in self.params.nodes:
                intersection = create_intersection_from_config(node_config)
                self.nodes.append(intersection)

        self.network_coupling = NetworkCouplingCorrected(self.nodes, self.params)

        if not self.quiet:
            print(f"  Initialized {len(self.nodes)} network nodes")
            print(f"  Network coupling system ready")

    def _load_road_quality(self):
        """ Loads road quality data based on the definition in params. """
        # Check if 'road' config exists and is a dictionary
        road_config = getattr(self.params, 'road', None)
        if not isinstance(road_config, dict):
            # Fallback or error? Let's try the old way for backward compatibility or raise error
            # For now, let's raise an error if 'road' dict is missing or not a dict
            # --- Check if the old attribute exists for backward compatibility ---
            old_definition = getattr(self.params, 'road_quality_definition', None)
            if old_definition is not None:
                if not self.quiet:
                    print("Warning: Using deprecated 'road_quality_definition'. Define road quality under 'road: {quality_type: ...}' instead.")
                if isinstance(old_definition, list):
                    road_config = {'quality_type': 'list', 'quality_values': old_definition}
                elif isinstance(old_definition, str):
                    road_config = {'quality_type': 'from_file', 'quality_file': old_definition}
                elif isinstance(old_definition, int):
                    road_config = {'quality_type': 'uniform', 'quality_value': old_definition}
                else:
                     raise TypeError("Invalid legacy 'road_quality_definition' type. Use list, file path (str), or uniform int.")
            else:
                raise ValueError("Configuration missing 'road' dictionary defining quality_type, and legacy 'road_quality_definition' not found.")
            # --- End backward compatibility check ---


        quality_type = road_config.get('quality_type', 'uniform').lower()
        if not self.quiet:
            print(f"  Loading road quality type: {quality_type}") # Debug print

        if quality_type == 'uniform':
            R_value = road_config.get('quality_value', 1) # Default to 1 if uniform but no value given
            if not isinstance(R_value, int):
                 raise ValueError(f"'quality_value' must be an integer for uniform road type, got {R_value}")
            if not self.quiet:
                print(f"  Uniform road quality value: {R_value}") # Debug print
            R_array = np.full(self.grid.N_physical, R_value, dtype=int)
            self.grid.load_road_quality(R_array)

        elif quality_type == 'from_file':
            file_path = road_config.get('quality_file')
            if not file_path or not isinstance(file_path, str):
                raise ValueError("'quality_file' path (string) is required for 'from_file' road type.")
            if not self.quiet:
                print(f"  Loading road quality from file: {file_path}") # Debug print

            # Assume file_path is relative to the project root (where the script is run)
            # TODO: Consider resolving path relative to config file location or project root robustly
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"Road quality file not found: {file_path}")

            try:
                R_array = np.loadtxt(file_path, dtype=int)
                if R_array.ndim == 0: # Handle case of single value file
                    R_array = np.full(self.grid.N_physical, int(R_array))
                elif R_array.ndim > 1:
                     raise ValueError("Road quality file should contain a 1D list of integers.")
                # Check length after potential expansion from single value
                if len(R_array) != self.grid.N_physical:
                     raise ValueError(f"Road quality file '{file_path}' length ({len(R_array)}) must match N_physical ({self.grid.N_physical}).")
                self.grid.load_road_quality(R_array)
            except Exception as e:
                raise ValueError(f"Error loading road quality file '{file_path}': {e}") from e

        elif quality_type == 'list': # Added option for direct list
            value_list = road_config.get('quality_values')
            if not isinstance(value_list, list):
                 raise ValueError("'quality_values' (list) is required for 'list' road type.")
            R_array = np.array(value_list, dtype=int)
            if len(R_array) != self.grid.N_physical:
                 raise ValueError(f"Road quality list length ({len(R_array)}) must match N_physical ({self.grid.N_physical}).")
            self.grid.load_road_quality(R_array)

        # Add elif for 'piecewise_constant' here if needed later

        else:
            raise ValueError(f"Unsupported road quality type: '{quality_type}'")

    def _load_network_v0_overrides(self, scenario_config_path: str):
        """
        Load V0_m/V0_c overrides from scenario config if present.
        
        For scenarios with network config (e.g., RL traffic light control), reads
        V0 parameters from the global 'parameters' section of the scenario YAML.
        This allows scenarios to specify Lagos speeds (32 km/h, 28 km/h) without
        modifying config_base.yml.
        
        Architectural Note (2025-10-24):
            This bridges the gap between NetworkGrid (multi-segment) and SimulationRunner
            (single-segment). For RL environments, the scenario has network config but
            SimulationRunner treats it as a single domain. We extract V0 from the global
            parameters section to honor the scenario's intended speeds.
            
        Supports two config formats:
            1. Global parameters: config['parameters']['V0_m'] (Lagos format)
            2. Segment parameters: config['network']['segments'][0]['parameters']['V0_m'] (NetworkGrid format)
        """
        if not os.path.exists(scenario_config_path):
            return  # Scenario file doesn't exist, skip
        
        try:
            with open(scenario_config_path, 'r') as f:
                scenario_config = yaml.safe_load(f)
            
            V0_m = None
            V0_c = None
            source = None
            
            # Strategy 1: Check global 'parameters' section (Lagos config format)
            if 'parameters' in scenario_config:
                params = scenario_config['parameters']
                V0_m = params.get('V0_m')
                V0_c = params.get('V0_c')
                if V0_m is not None or V0_c is not None:
                    source = "global parameters"
            
            # Strategy 2: Check network.segments[0].parameters (NetworkGrid format)
            if V0_m is None and V0_c is None:
                if 'network' in scenario_config:
                    network_config = scenario_config['network']
                    if 'segments' in network_config:
                        segments = network_config['segments']
                        
                        # Handle both list and dict formats
                        if isinstance(segments, list) and len(segments) > 0:
                            first_segment = segments[0]
                            if 'parameters' in first_segment:
                                V0_m = first_segment['parameters'].get('V0_m')
                                V0_c = first_segment['parameters'].get('V0_c')
                                source = f"segment '{first_segment.get('id', '0')}'"
                        elif isinstance(segments, dict) and len(segments) > 0:
                            first_segment_id = list(segments.keys())[0]
                            first_segment = segments[first_segment_id]
                            if 'parameters' in first_segment:
                                V0_m = first_segment['parameters'].get('V0_m')
                                V0_c = first_segment['parameters'].get('V0_c')
                                source = f"segment '{first_segment_id}'"
            
            # Apply overrides if found
            if V0_m is not None or V0_c is not None:
                if V0_m is not None:
                    self.params._V0_m_override = float(V0_m)
                if V0_c is not None:
                    self.params._V0_c_override = float(V0_c)
                
                if not self.quiet:
                    print(f"[NETWORK V0 OVERRIDE] Loaded from {source}:")
                    if V0_m is not None:
                        print(f"  V0_m = {V0_m:.3f} m/s ({V0_m*3.6:.1f} km/h)")
                    if V0_c is not None:
                        print(f"  V0_c = {V0_c:.3f} m/s ({V0_c*3.6:.1f} km/h)")
        
        except Exception as e:
            if not self.quiet:
                print(f"Warning: Could not load network V0 overrides: {e}")
            # Don't fail - just continue without overrides


    def _create_initial_state(self) -> np.ndarray:
        """ Creates the initial state array U based on config. """
        ic_config = self.params.initial_conditions
        ic_type = ic_config.get('type', '').lower()
        
        # 🔥 ARCHITECTURAL FIX: REMOVE IC→BC COUPLING
        # Initial conditions define domain state at t=0 ONLY
        # Boundary conditions are COMPLETELY INDEPENDENT
        # DO NOT store any "equilibrium state" for BC reuse
        # self.initial_equilibrium_state = None  # ❌ REMOVED - was causing IC→BC coupling
        
        # ✅ ARCHITECTURAL FIX: Extract BC state for traffic signal control
        # Traffic signal modulates BOUNDARY CONDITIONS, not initial conditions
        # This must come from BC config ONLY, never from IC
        self.traffic_signal_base_state = None
        bc_config = self.params.boundary_conditions
        if bc_config and 'left' in bc_config:
            left_bc = bc_config['left']
            if left_bc.get('type') == 'inflow' and 'state' in left_bc:
                self.traffic_signal_base_state = left_bc['state']  # [rho_m, w_m, rho_c, w_c]
                if not self.quiet:
                    print(f"  ✅ ARCHITECTURE: traffic_signal_base_state from BC = {self.traffic_signal_base_state}")

        if ic_type == 'uniform':
            state_vals = ic_config.get('state')
            if state_vals is None or len(state_vals) != 4:
                raise ValueError("Uniform IC requires 'state': [rho_m, rho_c, w_m, w_c]")
            U_init = initial_conditions.uniform_state(self.grid, *state_vals)
            # 🔥 ARCHITECTURAL FIX: Do NOT store IC state for BC use
            # IC is for t=0 ONLY - BC are independent
            # self.initial_equilibrium_state = state_vals  # ❌ REMOVED - caused IC→BC coupling
        elif ic_type == 'uniform_equilibrium':
            rho_m = ic_config.get('rho_m')
            rho_c = ic_config.get('rho_c')
            R_val = ic_config.get('R_val') # Assumes uniform R for equilibrium calc
            if rho_m is None or rho_c is None or R_val is None:
                 raise ValueError("Uniform Equilibrium IC requires 'rho_m', 'rho_c', 'R_val'.")

            # Convert densities from veh/km (config) to veh/m (SI units)
            rho_m_si = rho_m * VEH_KM_TO_VEH_M # Use imported constant
            rho_c_si = rho_c * VEH_KM_TO_VEH_M # Use imported constant

            # Capture both the initial state array and the equilibrium state vector
            U_init, eq_state_vector = initial_conditions.uniform_state_from_equilibrium(
                self.grid, rho_m_si, rho_c_si, R_val, self.params
            )
            # 🔥 ARCHITECTURAL FIX: Do NOT store IC equilibrium state for BC use
            # IC is for t=0 ONLY - BC are independent
            # self.initial_equilibrium_state = eq_state_vector  # ❌ REMOVED - caused IC→BC coupling
        elif ic_type == 'riemann':
            U_L = ic_config.get('U_L')
            U_R = ic_config.get('U_R')
            split_pos = ic_config.get('split_pos')
            if U_L is None or U_R is None or split_pos is None:
                raise ValueError("Riemann IC requires 'U_L', 'U_R', 'split_pos'.")
            U_init = initial_conditions.riemann_problem(self.grid, U_L, U_R, split_pos)
            # 🔥 ARCHITECTURAL FIX: Do NOT store IC Riemann state for BC use
            # IC is for t=0 ONLY - BC are independent
            # self.initial_equilibrium_state = U_L  # ❌ REMOVED - caused IC→BC coupling
        elif ic_type == 'density_hump':
             bg_state = ic_config.get('background_state')
             center = ic_config.get('center')
             width = ic_config.get('width')
             rho_m_max = ic_config.get('rho_m_max')
             rho_c_max = ic_config.get('rho_c_max')
             if None in [bg_state, center, width, rho_m_max, rho_c_max] or len(bg_state)!=4:
                  raise ValueError("Density Hump IC requires 'background_state' [rho_m, rho_c, w_m, w_c], 'center', 'width', 'rho_m_max', 'rho_c_max'.")
             U_init = initial_conditions.density_hump(self.grid, *bg_state, center, width, rho_m_max, rho_c_max)
        elif ic_type == 'sine_wave_perturbation':
            # Access nested dictionaries
            bg_state_config = ic_config.get('background_state', {})
            perturbation_config = ic_config.get('perturbation', {})

            rho_m_bg = bg_state_config.get('rho_m')
            rho_c_bg = bg_state_config.get('rho_c')
            epsilon_rho_m = perturbation_config.get('amplitude') # Use 'amplitude' key from YAML
            wave_number = perturbation_config.get('wave_number')

            # R_val should be present if road_quality_definition is int, or explicitly defined
            # This logic seems okay, assuming road_quality_definition is loaded correctly now
            R_val = ic_config.get('R_val', getattr(self.params, 'road_quality_definition', None) if isinstance(getattr(self.params, 'road_quality_definition', None), int) else None)

            if None in [rho_m_bg, rho_c_bg, epsilon_rho_m, wave_number, R_val]:
                raise ValueError("Sine Wave Perturbation IC requires nested 'background_state' (with 'rho_m', 'rho_c'), 'perturbation' (with 'amplitude', 'wave_number'), and 'R_val' (or global int road_quality_definition).")
            U_init = initial_conditions.sine_wave_perturbation(self.grid, self.params, rho_m_bg, rho_c_bg, R_val, epsilon_rho_m, wave_number)
        else:
            raise ValueError(f"Unknown initial condition type: '{ic_type}'")

        # Return the raw initial state without BCs applied yet
        return U_init

    def _initialize_boundary_conditions(self):
        """Initializes boundary condition schedules and current state."""
        self.left_bc_schedule = None
        self.right_bc_schedule = None
        self.left_bc_schedule_idx = -1 # Index of the currently active schedule entry
        self.right_bc_schedule_idx = -1

        # Make a working copy of BC params from the main params object
        self.current_bc_params = copy.deepcopy(self.params.boundary_conditions)

        # 🔥 ARCHITECTURAL FIX: BC MUST be explicitly configured - NO IC fallback
        # ========================================================================
        # Validate that inflow BCs have explicit state configuration
        # This enforces separation between IC (t=0) and BC (all t)
        
        # Validate left boundary
        if self.current_bc_params.get('left', {}).get('type') == 'inflow':
            if 'state' not in self.current_bc_params['left'] or self.current_bc_params['left']['state'] is None:
                raise ValueError(
                    "❌ ARCHITECTURAL ERROR: Inflow BC requires explicit 'state' configuration.\n"
                    "Boundary conditions must be independently specified, not derived from initial conditions.\n"
                    "\n"
                    "Add to your YAML config:\n"
                    "  boundary_conditions:\n"
                    "    left:\n"
                    "      type: inflow\n"
                    "      state: [rho_m, w_m, rho_c, w_c]  # Example: [0.150, 8.0, 0.120, 6.0]\n"
                    "\n"
                    "IC (initial_conditions) defines domain at t=0.\n"
                    "BC (boundary_conditions) defines flux for all t≥0.\n"
                    "These are INDEPENDENT concepts."
                )
            if not self.quiet:
                print(f"  ✅ ARCHITECTURE: Left inflow BC explicitly configured: {self.current_bc_params['left']['state']}")
        
        # Validate right boundary (if using inflow)
        if self.current_bc_params.get('right', {}).get('type') == 'inflow':
            if 'state' not in self.current_bc_params['right'] or self.current_bc_params['right']['state'] is None:
                raise ValueError(
                    "❌ ARCHITECTURAL ERROR: Inflow BC requires explicit 'state' configuration.\n"
                    "Boundary conditions must be independently specified, not derived from initial conditions.\n"
                    "\n"
                    "Add to your YAML config:\n"
                    "  boundary_conditions:\n"
                    "    right:\n"
                    "      type: inflow\n"
                    "      state: [rho_m, w_m, rho_c, w_c]  # Example: [0.150, 8.0, 0.120, 6.0]\n"
                )
            if not self.quiet:
                print(f"  ✅ ARCHITECTURE: Right inflow BC explicitly configured: {self.current_bc_params['right']['state']}")
        # ========================================================================

        # --- Parse schedules for time-dependent BCs ---
        if self.current_bc_params.get('left', {}).get('type') == 'time_dependent':
            self.left_bc_schedule = self.current_bc_params['left'].get('schedule')
            if not isinstance(self.left_bc_schedule, list) or not self.left_bc_schedule:
                raise ValueError("Left 'time_dependent' BC requires a non-empty 'schedule' list.")
            # Validate schedule format? (e.g., time ordering, content) - Optional
            self._update_bc_from_schedule('left', 0.0) # Set initial state from schedule

        if self.current_bc_params.get('right', {}).get('type') == 'time_dependent':
            self.right_bc_schedule = self.current_bc_params['right'].get('schedule')
            if not isinstance(self.right_bc_schedule, list) or not self.right_bc_schedule:
                raise ValueError("Right 'time_dependent' BC requires a non-empty 'schedule' list.")
            # Validate schedule format? - Optional
            self._update_bc_from_schedule('right', 0.0) # Set initial state from schedule
        # ---------------------------------------------

    def _update_bc_from_schedule(self, side: str, current_time: float):
        """Updates the current_bc_params for a given side based on the schedule."""
        schedule = self.left_bc_schedule if side == 'left' else self.right_bc_schedule
        current_idx = self.left_bc_schedule_idx if side == 'left' else self.right_bc_schedule_idx

        if not schedule: return # No schedule for this side

        new_idx = -1
        for idx, entry in enumerate(schedule):
            # Unpack schedule entry
            t_start_raw, t_end_raw, bc_type, *bc_state_info = entry

            # --- Explicitly cast times to float to handle potential loading issues ---
            try:
                t_start = float(t_start_raw)
                t_end = float(t_end_raw)
            except (ValueError, TypeError) as e:
                print(f"\nERROR: Could not convert schedule time to float: entry={entry}, error={e}")
                # Decide how to handle: skip entry, raise error? Skipping for now.
                continue
            # -----------------------------------------------------------------------

            if t_start <= current_time < t_end:
                new_idx = idx
                break

        if new_idx != -1 and new_idx != current_idx:
            # Active schedule entry has changed
            t_start_raw, t_end_raw, bc_type, *bc_state_info = schedule[new_idx] # Retrieve raw values

            # --- Ensure times are float before using in f-string ---
            try:
                t_start = float(t_start_raw)
                t_end = float(t_end_raw)
            except (ValueError, TypeError) as e:
                 # Log error but try to continue with raw values for bc_type, state
                 # Or raise?
                 print(f"\nERROR: Could not convert schedule time for printing: entry={schedule[new_idx]}, error={e}")
                 t_start, t_end = t_start_raw, t_end_raw # Keep raw for message formatting attempt
            # ------------------------------------------------------

            new_bc_config = {'type': bc_type}
            if bc_state_info: # If state information is provided (e.g., for inflow)
                # Assume state info is the state list/array itself
                new_bc_config['state'] = bc_state_info[0]

            self.current_bc_params[side] = new_bc_config
            if side == 'left':
                self.left_bc_schedule_idx = new_idx
            else:
                self.right_bc_schedule_idx = new_idx

            if self.pbar is not None:
                pbar_message = f"\nBC Change ({side.capitalize()}): Switched to type '{bc_type}' at t={current_time:.4f}s (Scheduled for [{t_start:.1f}, {t_end:.1f}))"
                # Try to write using tqdm's method if available, otherwise print
                try:
                    self.pbar.write(pbar_message)
                except AttributeError:
                    print(pbar_message)


    def run(self, t_final: Optional[float] = None):
        """
        Runs the simulation loop until t_final.

        Args:
            t_final (float, optional): Simulation end time. Overrides config if provided.

        Returns:
            A history object containing the simulation results.
        """
        # NetworkGrid mode: delegate to network_simulator
        if self.is_network_simulation:
            # Use the t_final from the runner's config if not overridden
            sim_t_final = t_final if t_final is not None else self.simulation_config.time.t_final
            return self.network_simulator.run(t_final=sim_t_final)

        # --- LEGACY/SINGLE-SEGMENT SIMULATION ---
        # This part remains for backward compatibility with single-segment models
        sim_t_final = t_final if t_final is not None else self.config.t_final
        if not self.quiet:
            print(f"Running single-segment simulation until t={sim_t_final}s...")
        
        start_time = time.time()
        
        # Initialize history storage
        history = data_manager.initialize_history(self.config, self.U)
        
        # Main time loop
        pbar = tqdm(total=sim_t_final, desc="Simulating", disable=self.quiet)
        
        while self.current_time < sim_t_final:
            self.step()
            
            # Store results at specified intervals
            if self.time_step % self.config.output_frequency == 0:
                data_manager.store_history_data(history, self)
            
            pbar.update(self.dt)

        pbar.close()
        
        end_time = time.time()
        if not self.quiet:
            print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
            
        return history


    def step(self):
        """
        Advances the simulation by one time step (dt).
        
        This method performs the core numerical integration:
        1. Calculates the stable time step (dt) using CFL condition.
        2. Updates boundary conditions if they are time-dependent.
        3. Applies the chosen time integration scheme (e.g., SSP-RK3).
        4. Updates the current time and step count.
        """
        # Ensure we don't overshoot t_final
        if self.t >= self.params.t_final:
            return

        # --- Determine dt using CFL condition ---
        # Use GPU data if available, otherwise CPU
        U_for_cfl = self.d_U if self.device == 'gpu' else self.U
        self.dt = cfl.get_dt(U_for_cfl, self.grid, self.params)
        
        # Adjust last step to hit t_final exactly
        if self.t + self.dt > self.params.t_final:
            self.dt = self.params.t_final - self.t
        # ----------------------------------------

        # --- Update time-dependent boundary conditions ---
        self._update_bc_from_schedule('left', self.t)
        self._update_bc_from_schedule('right', self.t)
        # ---------------------------------------------

        # --- Perform time integration step ---
        # The time integration function will handle device-specific kernels
        time_integration.step(
            self.U, self.grid, self.params, self.dt, self.t,
            self.current_bc_params, self.device, self.d_U, self.d_R,
            self.network_coupling # Pass the network coupling system
        )
        # -----------------------------------

        # --- Update time and step count ---
        self.t += self.dt
        self.step_count += 1
        # ----------------------------------

        # --- Store mass conservation data if enabled ---
        if self.mass_check_config and self.step_count % self.mass_check_config.get('frequency', 10) == 0:
            # Always calculate from CPU data for consistency
            U_phys = self.U[:, self.grid.physical_cell_indices]
            mass_m = metrics.calculate_total_mass(U_phys, self.grid, class_index=0)
            mass_c = metrics.calculate_total_mass(U_phys, self.grid, class_index=2)
            self.mass_times.append(self.t)
            self.mass_m_data.append(mass_m)
            self.mass_c_data.append(mass_c)
        # ---------------------------------------------

    def get_results(self):
        """
        Returns the simulation results.

        Returns:
            A dictionary containing the time points and state history.
        """
        results = {
            'times': self.times,
            'states': self.states,
            'grid': self.grid,
            'params': self.params
        }
        # Add mass conservation data if it was collected
        if hasattr(self, 'mass_times') and self.mass_times:
            results['mass_conservation'] = {
                'times': self.mass_times,
                'mass_m': self.mass_m_data,
                'mass_c': self.mass_c_data,
                'initial_mass_m': self.initial_mass_m,
                'initial_mass_c': self.initial_mass_c
            }
        return results

    def save_results(self, filename: str):
        """
        Saves the simulation results to a file.

        Args:
            filename (str): The path to the output file.
        """
        results = self.get_results()
        data_manager.save_results(results, filename)
        if not self.quiet:
            print(f"Results saved to {filename}")

    def plot_results(self, t_indices: list = None, show: bool = True, save_path: str = None):
        """
        Generates and displays plots of the simulation results.

        Args:
            t_indices (list, optional): A list of time indices to plot.
                                        If None, plots a few snapshots.
            show (bool): Whether to display the plot.
            save_path (str, optional): Path to save the plot image.
        """
        from ..visualization import plotting # Lazy import
        
        if t_indices is None:
            # Default to plotting a few snapshots
            num_snapshots = min(5, len(self.times))
            t_indices = np.linspace(0, len(self.times) - 1, num_snapshots, dtype=int)

        plotting.plot_simulation_snapshots(
            self.get_results(),
            t_indices,
            show=show,
            save_path=save_path
        )

    def animate_results(self, save_path: str = 'simulation.mp4', interval: int = 50):
        """
        Creates an animation of the simulation results.

        Args:
            save_path (str): The path to save the animation file.
            interval (int): The delay between frames in milliseconds.
        """
        from ..visualization import plotting # Lazy import
        plotting.animate_simulation(self.get_results(), save_path=save_path, interval=interval)
        if not self.quiet:
            print(f"Animation saved to {save_path}")

    def __repr__(self):
        scenario = getattr(self.params, 'scenario_name', 'N/A')
        return f"SimulationRunner(scenario='{scenario}', t={self.t:.2f}/{self.params.t_final}, device='{self.device}')"

