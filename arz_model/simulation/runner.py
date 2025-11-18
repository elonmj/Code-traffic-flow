import numpy as np
import time
import copy # For deep merging overrides
import os
from tqdm import tqdm # For progress bar
from numba import cuda # Import cuda for device arrays
from typing import Union, Optional

# from ..analysis import metrics
from ..io import data_manager
from ..core.parameters import ModelParameters, VEH_KM_TO_VEH_M # Import the constant
from ..grid.grid1d import Grid1D
from ..numerics import boundary_conditions, cfl, time_integration


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

    Supports GPU-only execution via Pydantic configuration objects.
    Initializes the grid, parameters, and initial state, then runs the
    time loop, applying numerical methods and storing results.
    """

    def __init__(self,
                 simulation_config: Optional[Union[SimulationConfig, NetworkSimulationConfig]] = None,
                 quiet: bool = False,
                 network_grid: Optional['NetworkGrid'] = None,
                 device: str = 'gpu',
                 debug: bool = False):
        """
        Initializes the simulation runner.

        MODES:
        1. Network Simulation (Pydantic):
            runner = SimulationRunner(network_grid=my_network_grid, simulation_config=network_config)

        2. Single-Segment Simulation (Pydantic):
            runner = SimulationRunner(simulation_config=ConfigBuilder.section_7_6())

        Args:
            network_grid: A fully built NetworkGrid object for multi-segment simulation.
            simulation_config: Pydantic SimulationConfig or NetworkSimulationConfig instance.
            quiet: Suppress print statements.
        """
        # ====================================================================
        # DETECT INITIALIZATION MODE
        # ====================================================================
        
        # Case 1: Network Simulation Mode
        self.debug = debug

        if network_grid is not None:
            if not isinstance(simulation_config, NetworkSimulationConfig):
                raise TypeError("Network mode requires a `NetworkSimulationConfig` object.")
            self._init_from_network_grid(network_grid, simulation_config, quiet, device)
            return

        # Case 2: Single-Segment Pydantic Mode - DEPRECATED
        # This mode is no longer supported in the GPU-only architecture.
        # All simulations, including single-segment ones, should be run
        # as a network simulation with one segment.
        if PYDANTIC_AVAILABLE and isinstance(simulation_config, SimulationConfig):
            raise NotImplementedError(
                "Single-segment simulation mode is deprecated. "
                "Please use the network simulation mode with a single segment."
            )

        # Case 3: ERROR - No valid initialization mode
        raise ValueError(
            "SimulationRunner requires one of:\n"
            "  1. network_grid: A built NetworkGrid object\n"
            "  2. simulation_config: A Pydantic SimulationConfig object"
        )

    def _init_from_network_grid(self, network_grid: 'NetworkGrid', config: 'NetworkSimulationConfig', quiet: bool, device: str):
        """Initializes the runner for a network simulation."""
        self.mode = 'network'
        self.is_network_simulation = True
        self.quiet = quiet
        self.network_grid = network_grid
        self.device = device # Set the device here
        
        # Use the provided Pydantic config
        self.config = config
        # This is a critical change: ModelParameters now gets its values from the Pydantic config
        self.params = ModelParameters(config=config)
        
        # The `simulation_config` attribute is also required for other parts of the runner
        self.simulation_config = config
        
        # GPU-only validation and architecture check
        if self.device == 'gpu':
            self._validate_gpu_availability()
        else:
            self.cc = (0,0) # Default compute capability for CPU mode
            if not self.quiet:
                print("Running in CPU mode, skipping GPU validation.")
        
        if not self.quiet:
            print(f"   - Mode: Network Simulation")
            print(f"   - Device: {self.device.upper()} (Compute Capability: {self.cc})")
            print(f"   - Segments: {len(self.network_grid.segments)}")
            print(f"   - Nodes: {len(self.network_grid.nodes)}")

        # The NetworkSimulator will handle the time loop, including the GPU pool
        self.network_simulator = NetworkSimulator(
            network=self.network_grid,
            config=self.config,
            quiet=self.quiet,
            device=self.device, # Pass device down
            debug=self.debug
        )
        
        # Initialize time tracking for step() method compatibility
        # In network mode, step() delegates to network_simulator but needs self.t for boundary checks
        self.t = 0.0
        self.step_count = 0
    
    @staticmethod
    def _validate_gpu_architecture():
        """
        DEPRECATED: Replaced by _validate_gpu_availability which also stores compute capability.
        Validates that CUDA is available for GPU-only execution.
        
        Raises:
            RuntimeError: If CUDA is not available
        """
        print("--- CUDA Availability Check ---")
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA not available. This GPU-only build requires an NVIDIA GPU with CUDA support.\n"
                "Please ensure your drivers and CUDA toolkit are correctly installed."
            )
        
        # Log GPU info for user awareness
        try:
            device = cuda.get_current_device()
            cc = device.compute_capability
            print(f"✅ GPU Detected: {device.name.decode('utf-8')}")
            print(f"   - Compute Capability: {cc}")
            print(f"   - Memory: {device.memory.total / (1024**3):.1f} GB")
            if cc[0] < 6:
                print("   - ⚠️ WARNING: Compute Capability is below the recommended 6.0. Some performance features may be limited.")
        except Exception:
            print(f"✅ CUDA Available (device info unavailable)")

    
    def _validate_gpu_availability(self):
        """
        Validates that CUDA is available and checks the GPU's compute capability.
        
        Raises:
            RuntimeError: If CUDA is not available or the GPU is unsupported.
        """
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA not available. This GPU-only build requires an NVIDIA GPU with CUDA support.\n"
                "Please ensure your drivers and CUDA toolkit are correctly installed."
            )
        
        try:
            device = cuda.get_current_device()
            self.cc = device.compute_capability
        except Exception as e:
            raise RuntimeError(f"Could not retrieve GPU device information: {e}") from e

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
        params.ghost_cells = config.grid.num_ghost_cells
        
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
        """Common initialization logic for Pydantic-based configurations"""
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

        # Road quality is now managed by the GPUMemoryPool.
        # We need to load it from the grid if it exists and pass it during initialization.
        self._load_road_quality()
        if not self.quiet:
            print("Road quality loaded from configuration.")

        # Create initial state U^0
        self.U = self._create_initial_state()
        if not self.quiet:
            print("Initial state created.")

        # --- Initialize GPU Memory Pool (GPU-Only Architecture) ---
        if not self.quiet:
            print("Creating GPU memory pool...")
        
        # Import GPUMemoryPool
        from ..numerics.gpu.memory_pool import GPUMemoryPool
        
        # Check if this is a network simulation
        if hasattr(self, 'is_network_simulation') and self.is_network_simulation and hasattr(self, 'network_grid'):
            # Network simulation: create pool from network grid
            segment_ids = list(self.network_grid.segments.keys())
            N_per_segment = {seg_id: segment.grid.N for seg_id, segment in self.network_grid.segments.items()}
            
            if not self.quiet:
                print(f"  Network simulation: {len(segment_ids)} segments")
        else:
            # Single-segment simulation: create simple pool
            segment_ids = ['main_segment']
            N_per_segment = {'main_segment': self.params.N}
            
            if not self.quiet:
                print(f"  Single-segment simulation")
        
        try:
            self.gpu_pool = GPUMemoryPool(
                segment_ids=segment_ids,
                N_per_segment=N_per_segment,
                ghost_cells=self.params.ghost_cells
            )
            
            # Initialize state and road quality in GPU pool
            if hasattr(self, 'is_network_simulation') and self.is_network_simulation:
                # Network simulation: initialize all segments
                for seg_id, segment in self.network_grid.segments.items():
                    # Get initial conditions for this segment
                    U_seg = segment.get_initial_state()  # This method needs to exist
                    R_seg = segment.grid.road_quality if hasattr(segment.grid, 'road_quality') else None
                    
                    self.gpu_pool.initialize_segment_state(seg_id, U_seg, R_seg)
                    
                # Legacy interface points to first segment for backward compatibility
                first_seg_id = segment_ids[0]
                self.d_U = self.gpu_pool.get_segment_state(first_seg_id)
                self.d_R = self.gpu_pool.get_road_quality(first_seg_id)
            else:
                # Single-segment simulation
                self.gpu_pool.initialize_segment_state(
                    'main_segment',
                    self.U,  # Initial state
                    self.grid.road_quality if hasattr(self.grid, 'road_quality') else None
                )
                
                # Legacy interface for backward compatibility
                self.d_U = self.gpu_pool.get_segment_state('main_segment')
                self.d_R = self.gpu_pool.get_road_quality('main_segment')
            
            if not self.quiet:
                print("GPU memory pool initialized successfully.")
                stats = self.gpu_pool.get_memory_stats()
                print(f"  GPU memory allocated: {stats['allocated_mb']:.2f} MB")
                
        except Exception as e:
            print(f"Error initializing GPU memory pool: {e}")
            raise RuntimeError(f"Failed to initialize GPU memory pool: {e}") from e
        # ----------------------------------------------------------------

        # --- Initialize Boundary Condition Schedules and Current State ---
        # This needs to happen *before* applying initial BCs so current_bc_params is ready
        self._initialize_boundary_conditions()
        # -------------------------------------------------------------

        # --- Apply initial boundary conditions ---
        # GPU-only: always use device array from GPU pool
        initial_U_array = self.d_U
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

        # --- Mass Conservation Check Initialization (REMOVED) ---
        # This legacy CPU-based check is obsolete and will be replaced by a
        # GPU-native implementation as per Task 5.3.
        self.mass_check_config = None
        # -------------------------------------------------------------

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
        """
        Loads road quality data into the grid object based on configuration.
        
        This is a necessary step before the GPUMemoryPool is initialized,
        as the pool will pull this data from the grid object.
        """
        road_config = self.params.road
        quality_type = road_config.get('quality_type', 'uniform')

        if quality_type == 'uniform':
            quality_value = road_config.get('quality_value', 10)
            self.grid.set_road_quality(quality_value)
            if not self.quiet:
                print(f"  - Uniform road quality set to: {quality_value}")
        
        elif quality_type == 'from_file':
            filepath = road_config.get('filepath')
            if not filepath:
                raise ValueError("Road quality type is 'from_file' but no filepath is provided.")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Road quality file not found: {filepath}")
            
            # Assuming the file contains a single column of quality values
            # matching the grid size N.
            quality_data = np.loadtxt(filepath)
            if quality_data.shape[0] != self.grid.N:
                raise ValueError(
                    f"Road quality data size ({quality_data.shape[0]}) does not match "
                    f"grid size ({self.grid.N})."
                )
            self.grid.set_road_quality(quality_data)
            if not self.quiet:
                print(f"  - Road quality loaded from: {filepath}")
        
        else:
            raise ValueError(f"Unsupported road quality type: {quality_type}")

        


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
            epsilon_rho_m = perturbation_config.get('amplitude') # Use 'amplitude' key from config
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
                    "Add to your simulation config:\n"
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
                    "Add to your simulation config:\n"
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

    def set_boundary_phase(
        self,
        segment_id: int,
        phase: str,
        validate: bool = True
    ) -> None:
        """
        Change traffic signal phase at runtime for a single segment.
        
        This method provides runtime control of traffic signals by modifying
        the boundary condition parameters during simulation execution. The phase
        change takes effect on the next timestep.
        
        Args:
            segment_id: Segment ID (integer node ID from topology)
            phase: Phase name (must exist in segment's traffic_signal_phases config)
            validate: Enable validation checks (default: True)
            
        Raises:
            KeyError: If segment_id not found or doesn't have traffic signal BC
            ValueError: If phase name not found in configuration
            
        Examples:
            >>> # Single phase change
            >>> runner.set_boundary_phase(31674708, 'green_NS')
            >>> runner.step(dt=1.0)  # Phase active in this step
            >>>
            >>> # Fast switching (disable validation for performance)
            >>> runner.set_boundary_phase(31674708, 'yellow_NS', validate=False)
            
        Notes:
            - No CPU-GPU transfers (updates dict on CPU only)
            - Action latency typically <0.5ms
            - Phase change affects next BC application in simulation step
            - For bulk updates, use set_boundary_phases_bulk() for better performance
            
        See Also:
            set_boundary_phases_bulk: Update multiple signals atomically
            config.network_simulation_config.NodeConfig: Traffic light configuration
        """
        # Convert to single-item dict and use bulk method
        self.set_boundary_phases_bulk({segment_id: phase}, validate=validate)
    
    def set_boundary_phases_bulk(
        self,
        phase_updates: dict,
        validate: bool = True
    ) -> None:
        """
        Update multiple traffic signal phases atomically.
        
        More efficient than calling set_boundary_phase() repeatedly. All updates
        are validated first (fail-fast), then applied together.
        
        Args:
            phase_updates: Dict mapping segment_id -> phase_name
            validate: Enable validation checks (default: True)
            
        Raises:
            KeyError: If any segment_id invalid
            ValueError: If any phase name invalid
            
        Examples:
            >>> # Synchronized signal control (all green NS)
            >>> runner.set_boundary_phases_bulk({
            ...     31674708: 'green_NS',
            ...     31674712: 'green_NS',
            ...     36240967: 'green_NS'
            ... })
            >>>
            >>> # Green wave pattern
            >>> runner.set_boundary_phases_bulk({
            ...     31674708: 'green_NS',
            ...     31674712: 'yellow_NS',
            ...     36240967: 'red_all'
            ... })
            
        Notes:
            - Fail-fast: If any validation fails, no updates are applied
            - Atomic: All updates applied together or none at all
            - Use validate=False for performance (after initial validation)
        """
        if validate:
            # Validate all updates first (fail-fast)
            for segment_id, phase in phase_updates.items():
                self._validate_segment_phase(segment_id, phase)
        
        # Apply all updates (already validated or validation disabled)
        for segment_id, phase in phase_updates.items():
            # Network mode: segments stored by node_id as keys
            if segment_id not in self.network_grid.segments:
                if validate:
                    # Should never reach here if validation passed
                    raise KeyError(f"Segment {segment_id} not found")
                else:
                    # Skip invalid segment when validation disabled
                    continue
            
            segment = self.network_grid.segments[segment_id]
            
            # Update current_bc_params for this segment
            # Traffic signals are typically on 'left' boundary (inflow)
            if 'current_bc_params' not in segment:
                segment['current_bc_params'] = {}
            
            if 'left' not in segment['current_bc_params']:
                segment['current_bc_params']['left'] = {}
            
            # Set the new phase
            segment['current_bc_params']['left']['current_phase'] = phase
            
            if self.debug:
                print(f"[RL CONTROL] t={self.current_time:.1f}s: Segment {segment_id} -> Phase '{phase}'")
    
    def _validate_segment_phase(self, segment_id: int, phase: str) -> None:
        """
        Validate that segment exists and phase is configured.
        
        Args:
            segment_id: Segment ID to validate
            phase: Phase name to validate
            
        Raises:
            KeyError: If segment not found or no traffic signal BC
            ValueError: If phase not in configuration
        """
        # Check segment exists
        if segment_id not in self.network_grid.segments:
            raise KeyError(
                f"Segment {segment_id} not found. "
                f"Available segments: {list(self.network_grid.segments.keys())[:10]}..."
            )
        
        segment = self.network_grid.segments[segment_id]
        
        # Check segment has traffic signal BC configuration
        bc_config = segment.get('bc_config')
        if not bc_config:
            raise KeyError(
                f"Segment {segment_id} has no boundary condition configuration"
            )
        
        # Check for traffic_signal_phases in left BC
        left_bc = bc_config.get('left', {})
        if left_bc.get('type') != 'traffic_signal':
            raise KeyError(
                f"Segment {segment_id} does not have traffic_signal BC type. "
                f"Current type: {left_bc.get('type')}"
            )
        
        traffic_signal_phases = left_bc.get('traffic_signal_phases', {})
        if not traffic_signal_phases:
            raise KeyError(
                f"Segment {segment_id} has no traffic_signal_phases configuration"
            )
        
        # Check phase exists in configuration
        if phase not in traffic_signal_phases:
            available_phases = list(traffic_signal_phases.keys())
            raise ValueError(
                f"Phase '{phase}' not found for segment {segment_id}. "
                f"Available phases: {available_phases}"
            )


    def run(self, t_final: Optional[float] = None, timeout: Optional[float] = None):
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
            return self.network_simulator.run(t_final=sim_t_final, timeout=timeout)

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
        # Network mode: delegate to network_simulator's time-stepping logic
        if self.is_network_simulation:
            # Network simulator doesn't expose a step() method but handles stepping internally
            # We need to manually perform one time step using its components
            from arz_model.numerics.cfl import cfl_condition_gpu_native
            from arz_model.numerics.time_integration import strang_splitting_step_gpu_native
            
            # Access network simulator's components
            ns = self.network_simulator
            
            # Check if we've reached t_final
            if ns.t >= ns.config.time.t_final:
                return
            
            # 1. Calculate dt using CFL condition
            stable_dt = cfl_condition_gpu_native(
                gpu_pool=ns.gpu_pool,
                network=ns.network,
                params=ns.config.physics,
                cfl_max=ns.config.time.cfl_factor,
                return_diagnostics=False
            )
            
            # 2. Evolve each segment on the GPU using Strang splitting
            for seg_id, segment_data in ns.network.segments.items():
                d_U_in = ns.gpu_pool.get_segment_state(seg_id)
                grid = segment_data['grid']
                
                # Perform one full time step for the segment
                d_U_out = strang_splitting_step_gpu_native(
                    d_U_n=d_U_in,
                    dt=stable_dt,
                    grid=grid,
                    params=ns.config.physics,
                    gpu_pool=ns.gpu_pool,
                    seg_id=seg_id,
                    current_time=ns.t
                )
                
                # The output d_U_out is a new array; update the pool to point to it.
                ns.gpu_pool.update_segment_state(seg_id, d_U_out)

            # 3. Apply network coupling on the GPU
            ns.network_coupling.apply_coupling(ns.config.physics)
            
            # 4. Update time tracking
            ns.t += stable_dt
            ns.time_step += 1
            self.t = ns.t  # Keep SimulationRunner's t in sync
            self.step_count = ns.time_step
            
            return
        
        # Single-segment mode (legacy)
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

