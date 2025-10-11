import numpy as np
import time
import copy # For deep merging overrides
import os
import yaml # To load road quality if defined directly in scenario
from tqdm import tqdm # For progress bar
from numba import cuda # Import cuda for device arrays

from ..analysis import metrics
from ..io import data_manager
from ..core.parameters import ModelParameters, VEH_KM_TO_VEH_M # Import the constant
from ..grid.grid1d import Grid1D
from ..numerics import boundary_conditions, cfl, time_integration
from . import initial_conditions # Import the initial conditions module

class SimulationRunner:
    """
    Orchestrates the execution of a single simulation scenario.

    Initializes the grid, parameters, and initial state, then runs the
    time loop, applying numerical methods and storing results.
    """

    def __init__(self, scenario_config_path: str,
                 base_config_path: str = 'config/config_base.yml',
                 override_params: dict = None,
                 quiet: bool = False,
                 device: str = 'cpu'): # Add device parameter
        """
        Initializes the simulation runner.

        Args:
            scenario_config_path (str): Path to the scenario-specific YAML configuration file.
            base_config_path (str): Path to the base YAML configuration file.
            override_params (dict, optional): Dictionary of parameters to override
                                              values loaded from config files. Defaults to None.
            quiet (bool, optional): If True, suppress most print statements. Defaults to False.
        """
        self.quiet = quiet
        self.device = device # Store the device parameter
        if not self.quiet:
            print(f"Initializing simulation from scenario: {scenario_config_path}")
            print(f"Using device: {self.device}") # Indicate which device is being used
        # Load parameters
        self.params = ModelParameters()
        self.params.load_from_yaml(base_config_path, scenario_config_path) # Load base and scenario

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
            # Re-validate after overrides if necessary
            # self.params._validate_parameters()

        # --- Add the device setting to the parameters object ---
        # This makes it accessible to functions like time_integration
        self.params.device = self.device
        # -------------------------------------------------------

        if not self.quiet:
            print(f"Parameters loaded for scenario: {self.params.scenario_name}")

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
        from ..numerics.network_coupling import NetworkCoupling

        self.nodes = []
        if self.params.nodes:
            for node_config in self.params.nodes:
                intersection = create_intersection_from_config(node_config)
                self.nodes.append(intersection)

        self.network_coupling = NetworkCoupling(self.nodes, self.params)

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


    def _create_initial_state(self) -> np.ndarray:
        """ Creates the initial state array U based on config. """
        ic_config = self.params.initial_conditions
        ic_type = ic_config.get('type', '').lower()
        self.initial_equilibrium_state = None # Initialize attribute
        
        # ✅ BUG #15 FIX: Store INFLOW boundary state for traffic signal control
        # Traffic signal should modulate the INFLOW (boundary condition), not the IC
        # Extract left boundary inflow state if available
        self.traffic_signal_base_state = None
        bc_config = self.params.boundary_conditions
        if bc_config and 'left' in bc_config:
            left_bc = bc_config['left']
            if left_bc.get('type') == 'inflow' and 'state' in left_bc:
                self.traffic_signal_base_state = left_bc['state']  # [rho_m, w_m, rho_c, w_c]

        if ic_type == 'uniform':
            state_vals = ic_config.get('state')
            if state_vals is None or len(state_vals) != 4:
                raise ValueError("Uniform IC requires 'state': [rho_m, w_m, rho_c, w_c]")
            U_init = initial_conditions.uniform_state(self.grid, *state_vals)
            # BUG #12 FIX: Store uniform IC state for traffic signal boundary conditions
            # Traffic signal needs base_state for phase transitions (red/green)
            # NOTE: This is used as FALLBACK if traffic_signal_base_state not available
            self.initial_equilibrium_state = state_vals  # [rho_m, w_m, rho_c, w_c]
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
            # Store the equilibrium state vector for potential BC use
            self.initial_equilibrium_state = eq_state_vector
        elif ic_type == 'riemann':
            U_L = ic_config.get('U_L')
            U_R = ic_config.get('U_R')
            split_pos = ic_config.get('split_pos')
            if U_L is None or U_R is None or split_pos is None:
                raise ValueError("Riemann IC requires 'U_L', 'U_R', 'split_pos'.")
            U_init = initial_conditions.riemann_problem(self.grid, U_L, U_R, split_pos)
            # Store U_L as equilibrium state for inflow BC (traffic signal control)
            # This allows green phase to impose the high-density left state
            self.initial_equilibrium_state = U_L
        elif ic_type == 'density_hump':
             bg_state = ic_config.get('background_state')
             center = ic_config.get('center')
             width = ic_config.get('width')
             rho_m_max = ic_config.get('rho_m_max')
             rho_c_max = ic_config.get('rho_c_max')
             if None in [bg_state, center, width, rho_m_max, rho_c_max] or len(bg_state)!=4:
                  raise ValueError("Density Hump IC requires 'background_state' [rho_m, w_m, rho_c, w_c], 'center', 'width', 'rho_m_max', 'rho_c_max'.")
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

        # --- Reuse initial equilibrium state for inflow BC if applicable ---
        if self.initial_equilibrium_state is not None:
            if not self.quiet:
                print(f"  DEBUG BC Init: Calculated initial_equilibrium_state = {self.initial_equilibrium_state}") # Debug print

            if self.current_bc_params.get('left', {}).get('type') == 'inflow':
                if 'state' not in self.current_bc_params['left'] or self.current_bc_params['left']['state'] is None:
                    self.current_bc_params['left']['state'] = self.initial_equilibrium_state
                    if not self.quiet:
                        print("  Populated left inflow BC state from initial equilibrium.")
                        print(f"  DEBUG BC Init: Final left inflow BC state = {self.current_bc_params['left']['state']}") # Debug print
            # Could add similar logic for right BC if needed
        # --------------------------------------------------------------------

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
                 # Log error but try to continue with raw values for BC config? Or raise?
                 # For now, log and keep raw type for bc_type, state
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


    def run(self, t_final: float = None, output_dt: float = None, max_steps: int = None) -> tuple[list[float], list[np.ndarray]]:
        """
        Runs the simulation loop until t_final.

        Args:
            t_final (float, optional): Simulation end time. Overrides config if provided. Defaults to None.
            output_dt (float, optional): Time interval for storing results. Overrides config if provided. Defaults to None.

        Returns:
            tuple[list[float], list[np.ndarray]]: List of times and list of corresponding state arrays (physical cells only).
        """
        t_final = t_final if t_final is not None else self.params.t_final
        output_dt = output_dt if output_dt is not None else self.params.output_dt

        if t_final <= self.t:
            print("Warning: t_final is less than or equal to current time. No steps taken.")
            return self.times, self.states

        if output_dt <= 0:
            raise ValueError("output_dt must be positive.")

        if not self.quiet:
            print(f"Running simulation until t = {t_final:.2f} s, outputting every {output_dt:.2f} s")
        start_time = time.time()
        last_output_time = self.t

        # Initialize tqdm progress bar, disable if quiet
        # If quiet, set pbar to None to avoid encoding issues with tqdm.write
        if self.quiet:
            self.pbar = None
        else:
            pbar = tqdm(total=t_final, desc="Running Simulation", unit="s", initial=self.t, leave=True, disable=self.quiet)
            self.pbar = pbar # Store pbar instance for writing messages

        try: # Ensure pbar is closed even if errors occur
            while self.t < t_final and (max_steps is None or self.step_count < max_steps):

                # --- Select state array based on device ---
                current_U = self.d_U if self.device == 'gpu' else self.U
                # -----------------------------------------

                # 1. Update Time-Dependent Boundary Conditions (if any)
                self._update_bc_from_schedule('left', self.t)
                self._update_bc_from_schedule('right', self.t)

                # 2. Apply Boundary Conditions
                # Ensures ghost cells are up-to-date before CFL calc and time step
                # Use the potentially updated current_bc_params
                # Pass both params (for device, physics constants) and current_bc_params (for BC types/states)
                # --- DEBUG PRINT: Check BC type passed (Commented out) ---
                # if self.t < 61.0 and not self.quiet: # Print for the first 60s + a bit
                #     right_bc_type_passed = self.current_bc_params.get('right', {}).get('type', 'N/A')
                #     print(f"DEBUG RUNNER @ t={self.t:.4f}: Passing right BC type '{right_bc_type_passed}' to apply_boundary_conditions")
                # -----------------------------------------
                # Pass both params (for device, physics constants) and current_bc_params (for BC types/states), and t_current
                boundary_conditions.apply_boundary_conditions(current_U, self.grid, self.params, self.current_bc_params, t_current=self.t)

                # 3. Calculate Stable Timestep
                # NOTE: calculate_cfl_dt now handles GPU arrays directly
                # Pass the appropriate array slice based on device
                if self.device == 'gpu':
                    # GPU function expects the full array (including ghosts)
                    dt = cfl.calculate_cfl_dt(current_U, self.grid, self.params)
                else: # CPU
                    # CPU function expects only physical cells
                    U_physical = current_U[:, self.grid.physical_cell_indices]
                    dt = cfl.calculate_cfl_dt(U_physical, self.grid, self.params)

                # 3. Adjust dt to not overshoot t_final or next output time
                time_to_final = t_final - self.t
                time_to_next_output = (last_output_time + output_dt) - self.t
                # Ensure dt doesn't step over the next output time by more than a small tolerance
                dt = min(dt, time_to_final, time_to_next_output + 1e-9) # Add tolerance for float comparison

                # Prevent excessively small dt near the end
                if dt < self.params.epsilon:
                     # Add newline to avoid overwriting pbar
                     if self.pbar is not None:
                         self.pbar.write(f"\nTime step too small ({dt:.2e}), ending simulation slightly early at t={self.t:.4f}.")
                     else:
                         if not self.quiet:
                             print(f"\nTime step too small ({dt:.2e}), ending simulation slightly early at t={self.t:.4f}.")
                     break


                # 4. Perform Time Step using Strang Splitting
                # NOTE: strang_splitting_step will need modification to handle/return GPU arrays
                if self.params.has_network:
                    # Use network-aware time integration
                    if self.device == 'gpu':
                        self.d_U = time_integration.strang_splitting_step_with_network(
                            self.d_U, dt, self.grid, self.params, self.nodes, self.network_coupling
                        )
                        current_U = self.d_U
                    else:
                        self.U = time_integration.strang_splitting_step_with_network(
                            self.U, dt, self.grid, self.params, self.nodes, self.network_coupling
                        )
                        current_U = self.U
                else:
                    # Standard time integration
                    if self.device == 'gpu':
                        self.d_U = time_integration.strang_splitting_step(self.d_U, dt, self.grid, self.params, d_R=self.d_R)
                        current_U = self.d_U
                    else:
                        self.U = time_integration.strang_splitting_step(self.U, dt, self.grid, self.params)
                        current_U = self.U


                # 5. Update Time
                self.t += dt
                self.step_count += 1
                
                # Print périodique pour suivre l'avancement (tous les 50 steps)
                if self.step_count % 50 == 0 and not self.quiet:
                    progress_percent = (self.t / t_final) * 100 if t_final > 0 else 0
                    print(f"[PROGRESS] Step {self.step_count}: t={self.t:.2f}/{t_final:.2f}s ({progress_percent:.1f}%) | dt={dt:.4f}s")
                
                # Update progress bar display
                if self.pbar is not None:
                    self.pbar.n = min(self.t, t_final) # Set current progress
                    # self.pbar.refresh() # Let tqdm handle refresh automatically

                # --- Mass Conservation Check ---
                if self.mass_check_config and (self.step_count % self.mass_check_config['frequency_steps'] == 0):
                    try:
                        # If GPU, copy back physical cells temporarily for mass calc
                        if self.device == 'gpu':
                            U_phys_current = current_U[:, self.grid.physical_cell_indices].copy_to_host()
                        else:
                            U_phys_current = current_U[:, self.grid.physical_cell_indices]

                        current_mass_m = metrics.calculate_total_mass(U_phys_current, self.grid, class_index=0)
                        current_mass_c = metrics.calculate_total_mass(U_phys_current, self.grid, class_index=2)
                        self.mass_times.append(self.t)
                        self.mass_m_data.append(current_mass_m)
                        self.mass_c_data.append(current_mass_c)
                    except Exception as e:
                        if self.pbar is not None:
                            self.pbar.write(f"Warning: Error calculating mass at t={self.t:.4f}: {e}")
                        else:
                            if not self.quiet:
                                print(f"Warning: Error calculating mass at t={self.t:.4f}: {e}")

                # 6. Check for Numerical Issues (Positivity handled in hyperbolic step)
                # If GPU, copy back temporarily to check for NaNs on CPU
                if self.device == 'gpu':
                    U_cpu_check = current_U.copy_to_host()
                    nan_check_array = U_cpu_check
                else:
                    nan_check_array = current_U

                if np.isnan(nan_check_array).any():
                    error_msg = f"Error: NaN detected in state vector at t = {self.t:.4f}, step {self.step_count}."
                    if self.pbar is not None:
                        self.pbar.write(error_msg)
                    else:
                        if not self.quiet:
                            print(error_msg)
                    # Optionally save the state just before NaN for debugging
                    # Consider saving the GPU state if possible, or the CPU copy
                    # io.data_manager.save_simulation_data("nan_state.npz", self.times, self.states, self.grid, self.params) # Saves CPU states list
                    raise ValueError("Simulation failed due to NaN values.")

                # 7. Store Results if Output Time Reached
                # Use a small tolerance for floating point comparison
                if self.t >= last_output_time + output_dt - 1e-9 or abs(self.t - t_final) < 1e-9 :
                    self.times.append(self.t)
                    # If GPU, copy back only physical cells for storage
                    if self.device == 'gpu':
                        state_cpu = current_U[:, self.grid.physical_cell_indices].copy_to_host()
                        self.states.append(state_cpu)
                    else:
                        self.states.append(np.copy(current_U[:, self.grid.physical_cell_indices]))
                    last_output_time = self.t
                    # Use pbar.write to print messages without breaking the bar
                    # Check if pbar is available before writing
                    if self.pbar is not None:
                        self.pbar.write(f"  Stored output at t = {self.t:.4f} s (Step {self.step_count})")
                    else:
                        # In quiet mode or if pbar is None, we don't print this specific message
                        # unless there's a specific need. For now, let's keep it quiet if quiet=True.
                        if not self.quiet:
                            print(f"  Stored output at t = {self.t:.4f} s (Step {self.step_count})")

        finally:
            if self.pbar is not None:
                self.pbar.close() # Close the progress bar

        end_time = time.time()
        # Add newline before final summary prints
        if not self.quiet:
            print(f"\nSimulation finished at t = {self.t:.4f} s after {self.step_count} steps.")
            print(f"Total runtime: {end_time - start_time:.2f} seconds.")

        # --- Save Mass Conservation Data ---
        if self.mass_check_config and self.mass_times:
            try:
                filename_pattern = self.mass_check_config['output_file_pattern']
                output_filename = filename_pattern.format(N=self.grid.N_physical)
                data_manager.save_mass_data(
                    filename=output_filename,
                    times=self.mass_times,
                    mass_m_list=self.mass_m_data,
                    mass_c_list=self.mass_c_data
                )
            except KeyError:
                if not self.quiet:
                    print("Error: 'output_file_pattern' not found in mass_conservation_check config.")
            except Exception as e:
                if not self.quiet:
                    print(f"Error saving mass conservation data: {e}")

        return self.times, self.states

    # ========================================================================
    # REINFORCEMENT LEARNING EXTENSIONS
    # ========================================================================

    def set_traffic_signal_state(self, intersection_id: str, phase_id: int) -> None:
        """
        Sets the traffic signal state for a specific intersection.
        
        This method is designed for RL environment integration, allowing
        direct control of traffic signals by updating boundary conditions.
        
        Args:
            intersection_id (str): Identifier for the intersection/boundary
                                  (e.g., 'left', 'right', 'intersection_1')
            phase_id (int): Traffic signal phase identifier
                           (0 = red/stop, 1 = green/free flow, etc.)
                           
        Raises:
            ValueError: If intersection_id is invalid or phase_id is out of bounds
            
        Example:
            >>> runner.set_traffic_signal_state('left', phase_id=1)  # Green phase
            >>> runner.set_traffic_signal_state('right', phase_id=0)  # Red phase
        """
        # Validate intersection_id
        valid_ids = ['left', 'right']
        if intersection_id not in valid_ids:
            raise ValueError(
                f"Invalid intersection_id '{intersection_id}'. "
                f"Must be one of {valid_ids}"
            )
        
        # Validate phase_id (basic range check)
        if not isinstance(phase_id, int) or phase_id < 0:
            raise ValueError(
                f"Invalid phase_id {phase_id}. Must be non-negative integer."
            )
        
        # Map phase_id to boundary condition configuration
        # BUG #4 FIX: ALWAYS use inflow at upstream boundary
        # Traffic always arrives from upstream - phase controls inflow characteristics, not BC type
        # Phase 0 = red (reduced velocity inflow - models queue formation)
        # Phase 1 = green (normal inflow - free flow)
        
        # ✅ BUG #15 FIX: Use INFLOW BC state (traffic_signal_base_state), not IC state
        # Traffic signal modulates the INFLOW boundary, which may differ from initial conditions
        # Example: IC = light congestion (40 veh/km), Inflow = heavy demand (120 veh/km)
        # Fallback to initial_equilibrium_state if traffic_signal_base_state not set
        base_state = self.traffic_signal_base_state if hasattr(self, 'traffic_signal_base_state') and self.traffic_signal_base_state else self.initial_equilibrium_state
        
        if phase_id == 0:
            # Red phase: Congested inflow (traffic backs up)
            # Reduce velocity by 50% to model queue formation upstream of signal
            if base_state:
                red_state = [
                    base_state[0],           # rho_m (maintain density)
                    base_state[1] * 0.5,     # w_m (reduce velocity 50%)
                    base_state[2],           # rho_c (maintain density)
                    base_state[3] * 0.5      # w_c (reduce velocity 50%)
                ]
                bc_config = {'type': 'inflow', 'state': red_state}
            else:
                bc_config = {'type': 'inflow', 'state': None}
        elif phase_id == 1:
            # Green phase: Normal inflow (free flow)
            bc_config = {
                'type': 'inflow',
                'state': base_state if base_state else None
            }
        else:
            # For phase_id > 1, treat as variations
            # Default to reduced inflow (like red phase)
            if hasattr(self, 'initial_equilibrium_state'):
                base_state = self.initial_equilibrium_state
                reduced_state = [
                    base_state[0],
                    base_state[1] * 0.5,
                    base_state[2],
                    base_state[3] * 0.5
                ]
                bc_config = {'type': 'inflow', 'state': reduced_state}
            else:
                bc_config = {'type': 'inflow', 'state': None}
        
        # Update current boundary condition parameters
        if not hasattr(self, 'current_bc_params'):
            self.current_bc_params = copy.deepcopy(self.params.boundary_conditions)
        
        self.current_bc_params[intersection_id] = bc_config
        
        # SENSITIVITY FIX: Enhanced logging to verify BC updates
        if not self.quiet:
            phase_name = "RED (reduced inflow)" if phase_id == 0 else "GREEN (normal inflow)"
            print(f"[BC UPDATE] {intersection_id} à phase {phase_id} {phase_name}", flush=True)
            if bc_config['type'] == 'inflow' and bc_config.get('state') is not None:
                state = bc_config['state']
                print(f"  └─ Inflow state: rho_m={state[0]:.4f}, w_m={state[1]:.1f}, "
                      f"rho_c={state[2]:.4f}, w_c={state[3]:.1f}", flush=True)
            elif bc_config['type'] == 'outflow':
                print(f"  └─ Outflow: zero-order extrapolation", flush=True)

    def get_segment_observations(self, segment_indices: list) -> dict:
        """
        Extracts traffic state observations from specified road segments.
        
        This method is designed for RL environment integration, providing
        normalized traffic state variables for agent observation.
        
        Args:
            segment_indices (list): List of cell indices to observe
                                   (must be within physical cells range)
                                   
        Returns:
            dict: Dictionary with keys:
                - 'rho_m': motorcycle densities (veh/m) - ndarray shape (len(segment_indices),)
                - 'q_m': motorcycle momenta (veh/s) - ndarray
                - 'rho_c': car densities (veh/m) - ndarray
                - 'q_c': car momenta (veh/s) - ndarray
                - 'v_m': motorcycle velocities (m/s) - ndarray
                - 'v_c': car velocities (m/s) - ndarray
                
        Raises:
            ValueError: If segment_indices are out of bounds
            
        Example:
            >>> obs = runner.get_segment_observations([10, 11, 12])
            >>> print(obs['rho_m'])  # Motorcycle densities for cells 10-12
        """
        # Validate indices
        if not segment_indices:
            raise ValueError("segment_indices cannot be empty")
        
        segment_indices = np.array(segment_indices, dtype=int)
        
        # Check bounds (physical cells only)
        physical_start = self.grid.num_ghost_cells
        physical_end = self.grid.num_ghost_cells + self.grid.N_physical
        
        if np.any(segment_indices < physical_start) or np.any(segment_indices >= physical_end):
            raise ValueError(
                f"Segment indices {segment_indices} out of bounds. "
                f"Must be in range [{physical_start}, {physical_end})"
            )
        
        # Extract state from appropriate array (CPU or GPU)
        if self.device == 'gpu':
            # CUDA arrays don't support fancy indexing - must loop
            # Allocate output array
            U_obs = np.zeros((4, len(segment_indices)), dtype=np.float64)
            for i, seg_idx in enumerate(segment_indices):
                U_obs[:, i] = self.d_U[:, seg_idx].copy_to_host()
        else:
            U_obs = self.U[:, segment_indices]
        
        # Extract components (U = [rho_m, q_m, rho_c, q_c])
        rho_m = U_obs[0, :]
        q_m = U_obs[1, :]
        rho_c = U_obs[2, :]
        q_c = U_obs[3, :]
        
        # Calculate velocities with epsilon to avoid division by zero
        epsilon = 1e-10
        v_m = q_m / (rho_m + epsilon)
        v_c = q_c / (rho_c + epsilon)
        
        # Return as dictionary
        return {
            'rho_m': rho_m,
            'q_m': q_m,
            'rho_c': rho_c,
            'q_c': q_c,
            'v_m': v_m,
            'v_c': v_c
        }

