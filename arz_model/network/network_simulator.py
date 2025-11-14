"""
Network Grid Simulator for RL Environment

Provides simulator interface compatible with ARZEndpointClient for use
in the RL training environment. Wraps NetworkGrid to provide consistent
API while enabling multi-segment traffic simulation.

Author: ARZ Research Team
Date: 2025-01
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm

from .network_grid import NetworkGrid
from ..core.parameters import ModelParameters


logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """
    State representation compatible with ARZEndpointClient interface.
    
    Attributes:
        timestamp: Current simulation time (seconds)
        branches: Dict mapping segment_id to traffic state
                  Each branch has: {rho_m, v_m, rho_c, v_c, queue_len, flow}
        phase_id: Current traffic light phase (optional)
    """
    timestamp: float
    branches: Dict[str, Dict[str, Any]]
    phase_id: Optional[int] = None


class NetworkGridSimulator:
    """
    Simulator wrapper for NetworkGrid compatible with RL environment.
    
    Implements the ARZEndpointClient interface while using NetworkGrid
    as the underlying traffic simulator. Provides:
    - reset(): Initialize network from scenario
    - step(): Advance simulation
    - set_signal(): Update traffic light plans
    - get_metrics(): Extract network-wide performance metrics
    
    This enables seamless integration with TrafficSignalEnv for RL training.
    """
    
    def __init__(
        self,
        params: ModelParameters,
        scenario_config: Dict[str, Any],
        dt_sim: float = 0.5
    ):
        """
        Initialize network simulator.
        
        Args:
            params: Model parameters (physics, numerics, θ_k)
            scenario_config: Network scenario specification
                {
                    'segments': [...],  # List of segment configs
                    'nodes': [...],     # List of junction configs
                    'links': [...],     # List of link configs
                    'initial_conditions': {...}  # Initial traffic state
                }
            dt_sim: Simulation timestep (seconds)
        """
        self.params = params
        self.params.is_network_mode = True  # Add flag for network context
        self.scenario_config = scenario_config
        self.dt_sim = dt_sim
        
        print(f"[DEBUG] NetworkGridSimulator.__init__: scenario_config keys = {list(scenario_config.keys())}")
        if 'initial_conditions' in scenario_config:
            print(f"[DEBUG]   IC present with {len(scenario_config['initial_conditions'])} segments")
        else:
            print(f"[DEBUG]   NO IC in scenario_config!")
        
        # Simulation state
        self.network: Optional[NetworkGrid] = None
        self.current_time: float = 0.0
        self.is_initialized: bool = False
        self.history: Dict[str, Dict[str, Any]] = {}
        
        # Tracked segment IDs for observations
        self.observed_segment_ids: List[str] = []
        
        logger.info(f"NetworkGridSimulator initialized with dt={dt_sim}s")
    
    def reset(
        self,
        scenario: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Tuple[SimulationState, float]:
        """
        Reset simulation and return initial state.
        
        Args:
            scenario: Scenario name (optional, uses default if None)
            seed: Random seed for reproducibility
            
        Returns:
            (initial_state, timestamp) tuple
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Build network from scenario configuration (using factory pattern)
        # This now uses YAML or NetworkBuilder factories instead of manual building
        if isinstance(self.scenario_config, dict):
            # If config has 'network_config' key, use YAML factory
            if 'network_config' in self.scenario_config:
                self.network = NetworkGrid.from_yaml_config(
                    config_file=self.scenario_config['network_config'],
                    traffic_file=self.scenario_config.get('traffic_control')
                )
            else:
                # Fallback: still build manually if legacy config format
                self.network = NetworkGrid(self.params, self.scenario_config)
                self._build_network_from_config_simple(self.scenario_config)
        
        # Set initial conditions (if present)
        if 'initial_conditions' in self.scenario_config:
            self._apply_initial_conditions(self.scenario_config['initial_conditions'])
        
        # Initialize network
        self.network.initialize()
        self.is_initialized = True
        
        # Reset time
        self.current_time = 0.0
        
        # Build initial state
        initial_state = self._build_state(self.current_time)
        
        logger.info(f"Network reset with {len(self.network.segments)} segments, "
                   f"{len(self.network.nodes)} nodes")
        
        return initial_state, self.current_time
    
    def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """
        Update traffic signal configuration.
        
        Args:
            signal_plan: Signal timing configuration
                {
                    'node_id': str,
                    'phase_id': int,
                    'green_times': List[float],  # Per-phase green durations
                    'yellow_time': float,
                    'all_red_time': float
                }
                
        Returns:
            True if signal updated successfully
        """
        if not self.is_initialized:
            logger.warning("Cannot set signal - network not initialized")
            return False
        
        try:
            node_id = signal_plan.get('node_id')
            if node_id not in self.network.nodes:
                logger.warning(f"Node {node_id} not found in network")
                return False
            
            node = self.network.nodes[node_id]
            
            # Update traffic light controller if present
            if node.traffic_lights is not None:
                # Extract phase timings
                green_times = signal_plan.get('green_times', [])
                yellow_time = signal_plan.get('yellow_time', 3.0)
                all_red_time = signal_plan.get('all_red_time', 2.0)
                
                # Update controller phases
                for i, green_duration in enumerate(green_times):
                    if i < len(node.traffic_lights.phases):
                        phase = node.traffic_lights.phases[i]
                        phase['green_time'] = green_duration
                        phase['yellow_time'] = yellow_time
                        phase['all_red_time'] = all_red_time
                
                logger.debug(f"Updated signal plan for node {node_id}")
                return True
            else:
                logger.warning(f"Node {node_id} has no traffic lights")
                return False
                
        except Exception as e:
            logger.error(f"Error setting signal plan: {e}")
            return False
    
    def step(
        self,
        dt: float,
    ) -> Tuple[SimulationState, float]:
        """
        Advance simulation by dt seconds.
        
        Args:
            dt: Single timestep duration (seconds)
            
        Returns:
            (new_state, new_timestamp) tuple
        """
        if not self.is_initialized:
            raise RuntimeError("Network not initialized - call reset() first")
        
        # Execute single timestep
        self.network.step(dt, current_time=self.current_time)
        self.current_time += dt
        
        # Build state observation
        new_state = self._build_state(self.current_time)
        
        # Store history for each segment
        for seg_id, segment_data in self.network.segments.items():
            if seg_id not in self.history:
                self.history[seg_id] = {'times': [], 'states': [], 'grid': segment_data['grid'], 'params': self.params}
            
            self.history[seg_id]['times'].append(self.current_time)
            # Store a copy of the physical state
            self.history[seg_id]['states'].append(segment_data['U'][:, segment_data['grid'].physical_cell_indices].copy())

        return new_state, self.current_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get network-wide performance metrics.
        
        Returns:
            Dict with metrics: total_vehicles, avg_speed, total_flux, etc.
        """
        if not self.is_initialized:
            return {}
        
        metrics = self.network.get_network_metrics()
        
        # Add timestamp
        metrics['timestamp'] = self.current_time
        
        return metrics
    
    def health(self) -> Dict[str, Any]:
        """
        Check simulator health status.
        
        Returns:
            Dict with status information
        """
        return {
            'status': 'healthy' if self.is_initialized else 'not_initialized',
            'current_time': self.current_time,
            'num_segments': len(self.network.segments) if self.network else 0,
            'num_nodes': len(self.network.nodes) if self.network else 0
        }
    
    def compute_adaptive_dt(self):
        """
        Calcule le pas de temps adaptatif (dt) pour l'ensemble du réseau en se basant sur la condition CFL.

        Cette méthode parcourt tous les segments du réseau, calcule la vitesse d'onde maximale
        pour chaque segment, puis détermine le dt global qui garantit la stabilité pour l'ensemble
        du système.

        Returns:
            float: Le pas de temps (dt) stable calculé.
        """
        global_max_lambda = 0.0
        global_dx_min = float('inf')
        
        # Itérer sur tous les segments pour trouver la vitesse d'onde maximale globale
        for seg_id, segment_data in self.network.segments.items():
            grid = segment_data['grid']
            U = segment_data['U']
            
            # Extraire les cellules physiques pour le calcul CFL
            U_physical = U[:, grid.physical_cell_indices]
            
            # Utiliser la fonction CFL existante pour calculer les vitesses d'onde
            # Note: calculate_cfl_dt retourne dt, mais on peut extraire max_lambda
            from ..core import physics
            
            rho_m = U_physical[0]
            w_m = U_physical[1]
            rho_c = U_physical[2]
            w_c = U_physical[3]
            
            # Assurer densités non-négatives
            rho_m_calc = np.maximum(rho_m, 0.0)
            rho_c_calc = np.maximum(rho_c, 0.0)
            
            # Calculer pression et vitesse
            p_m, p_c = physics.calculate_pressure(
                rho_m_calc, rho_c_calc,
                self.params.alpha, self.params.rho_jam, self.params.epsilon,
                self.params.K_m, self.params.gamma_m,
                self.params.K_c, self.params.gamma_c
            )
            v_m, v_c = physics.calculate_physical_velocity(w_m, w_c, p_m, p_c)
            
            # Calculer valeurs propres pour toutes les cellules
            all_eigenvalues_list = physics.calculate_eigenvalues(
                rho_m_calc, v_m, rho_c_calc, v_c, self.params
            )
            
            # Trouver la vitesse d'onde maximale pour ce segment
            max_abs_lambda_segment = np.max(np.abs(np.asarray(all_eigenvalues_list)))
            global_max_lambda = max(global_max_lambda, max_abs_lambda_segment)
            
            # Suivre le dx minimum pour le calcul CFL
            global_dx_min = min(global_dx_min, grid.dx)

        # Si aucune vitesse d'onde n'est détectée (réseau vide ou statique), retourner un dt par défaut.
        if global_max_lambda < self.params.epsilon:
            print(f"[DEBUG DT] global_max_lambda ({global_max_lambda}) is near zero. Using fallback dt.", flush=True)
            # Utilise dt_sim comme fallback ou une valeur par défaut si non défini
            return getattr(self.params, 'dt_sim', 0.1)

        # Calculer le dt stable en utilisant la condition CFL avec le dx minimum
        # (le dx minimum impose la contrainte la plus stricte)
        stable_dt = self.params.cfl_number * global_dx_min / global_max_lambda
        
        print(f"[DEBUG DT] cfl={self.params.cfl_number}, dx_min={global_dx_min}, max_lambda={global_max_lambda} -> stable_dt={stable_dt}", flush=True)

        return stable_dt

    # === Private Helper Methods ===
    
    def _build_network_from_config_simple(self, config: Dict[str, Any]):
        """Minimal fallback: build network from legacy config format (rarely used)."""
        # Segments
        for seg_cfg in config.get('segments', []):
            self.network.add_segment(
                segment_id=seg_cfg['id'],
                xmin=seg_cfg.get('xmin', 0),
                xmax=seg_cfg.get('xmax', 100),
                N=seg_cfg.get('N', 20),
                start_node=seg_cfg.get('start_node'),  # ✅ FIX: Pass node info for BC detection
                end_node=seg_cfg.get('end_node')       # ✅ FIX: Essential for boundary detection
            )
            self.observed_segment_ids.append(seg_cfg['id'])
        
        # Nodes
        for node_cfg in config.get('nodes', []):
            # ✅ FIX: Create TrafficLightController from config if present
            traffic_lights = None
            tl_config = node_cfg.get('traffic_light_config')
            print(f"[NODE_TL_DEBUG] node={node_cfg['id']}, traffic_light_config present: {tl_config is not None}")
            if tl_config is not None:
                print(f"[NODE_TL_DEBUG]   config keys: {list(tl_config.keys())}")
                if 'phases' in tl_config:
                    print(f"[NODE_TL_DEBUG]   phases: {tl_config['phases']}")
                from ..core.traffic_lights import create_traffic_light_from_config
                traffic_lights = create_traffic_light_from_config(tl_config)
                print(f"[NODE_TL_DEBUG]   Created traffic_lights: {traffic_lights is not None}")
            
            self.network.add_node(
                node_id=node_cfg['id'],
                position=tuple(node_cfg.get('position', [0, 0])),
                incoming_segments=node_cfg.get('incoming', []),
                outgoing_segments=node_cfg.get('outgoing', []),
                node_type=node_cfg.get('type', 'unsignalized'),
                traffic_lights=traffic_lights  # ✅ FIX: Pass traffic light controller
            )
        
        # Links
        for link_cfg in config.get('links', []):
            self.network.add_link(
                from_segment=link_cfg['from'],
                to_segment=link_cfg['to'],
                via_node=link_cfg['via']
            )
    
    def _apply_initial_conditions(self, ic_config: Dict[str, Any]):
        """Apply initial traffic conditions to segments."""
        from ..core.physics import calculate_pressure
        
        print(f"[DEBUG] _apply_initial_conditions called with {len(ic_config)} segments")
        for seg_id, ic in ic_config.items():
            if seg_id in self.network.segments:
                segment = self.network.segments[seg_id]
                U = segment['U']
                
                print(f"[DEBUG]   Applying IC to segment {seg_id}: rho_m={ic.get('rho_m', 'N/A')}, "
                      f"v_m={ic.get('v_m', 'N/A')}")
                
                # Set densities first
                if 'rho_m' in ic:
                    U[0, :] = ic['rho_m']
                if 'rho_c' in ic:
                    U[2, :] = ic['rho_c']
                
                # For ARZ model, w = v + p (Lagrangian variable)
                # We must calculate pressure first before setting momentum
                if 'v_m' in ic or 'w_m' in ic:
                    # Calculate pressure using current densities
                    p_m, p_c = calculate_pressure(
                        U[0, :], U[2, :],
                        self.params.alpha, self.params.rho_jam, self.params.epsilon,
                        self.params.K_m, self.params.gamma_m,
                        self.params.K_c, self.params.gamma_c
                    )
                    
                    if 'w_m' in ic:
                        # w_m provided directly (already in Lagrangian form)
                        U[1, :] = ic['w_m']
                    elif 'v_m' in ic:
                        # Convert physical velocity to Lagrangian momentum
                        # w_m = v_m + p_m (ARZ model definition)
                        U[1, :] = ic['v_m'] + p_m
                        print(f"[DEBUG]   Converted v_m={ic['v_m']:.4f} -> w_m={U[1,5]:.4f} (added p_m={p_m[5]:.4f})")
                
                if 'v_c' in ic or 'w_c' in ic:
                    if 'w_c' in ic:
                        U[3, :] = ic['w_c']
                    elif 'v_c' in ic:
                        # Same for cars: w_c = v_c + p_c
                        if 'v_m' not in ic and 'w_m' not in ic:  # Calculate p if not done yet
                            p_m, p_c = calculate_pressure(
                                U[0, :], U[2, :],
                                self.params.alpha, self.params.rho_jam, self.params.epsilon,
                                self.params.K_m, self.params.gamma_m,
                                self.params.K_c, self.params.gamma_c
                            )
                        U[3, :] = ic['v_c'] + p_c
                        print(f"[DEBUG]   Converted v_c={ic['v_c']:.4f} -> w_c={U[3,5]:.4f} (added p_c={p_c[5]:.4f})")
                
                print(f"[DEBUG]   After IC: U[0,5]={U[0,5]:.4f}, U[1,5]={U[1,5]:.4f}")
            else:
                print(f"[DEBUG]   Segment {seg_id} not found in network, skipping IC")
    
    def _build_state(self, timestamp: float) -> SimulationState:
        """
        Build SimulationState from current NetworkGrid state.
        
        Extracts state for observed segments and computes derived quantities
        (velocities, queue lengths, flows) compatible with RL environment.
        """
        branches = {}
        
        # In the new architecture, NetworkGrid holds the full state.
        # We iterate through its segments to build the observation.
        for seg_id, segment_data in self.network.segments.items():
            if seg_id not in self.observed_segment_ids:
                continue

            U = segment_data['U'] # Full state array with ghost cells
            grid = segment_data['grid']
            
            # Extract physical state
            state_physical = U[:, grid.physical_cell_indices]
            rho_m, w_m, rho_c, w_c = state_physical

            # Compute physical velocities
            from ..core.physics import calculate_pressure, calculate_physical_velocity
            p_m, p_c = calculate_pressure(rho_m, rho_c, self.params.alpha, self.params.rho_jam, self.params.epsilon, self.params.K_m, self.params.gamma_m, self.params.K_c, self.params.gamma_c)
            v_m, v_c = calculate_physical_velocity(w_m, w_c, p_m, p_c)

            # Average over segment
            rho_m_avg = np.mean(rho_m)
            v_m_avg = np.mean(v_m)
            rho_c_avg = np.mean(rho_c)
            v_c_avg = np.mean(v_c)

            # Estimate queue length (simple version)
            v_low_threshold = 5.0 / 3.6  # 5 km/h in m/s
            queue_cells = (v_m < v_low_threshold) | (v_c < v_low_threshold)
            queue_len = np.sum(queue_cells * (rho_m + rho_c) * grid.dx)

            # Flow at downstream boundary (last physical cell)
            flow_m = rho_m[-1] * v_m[-1]
            flow_c = rho_c[-1] * v_c[-1]
            flow_total = flow_m + flow_c
            
            branches[seg_id] = {
                'rho_m': float(rho_m_avg) / (1.0 / 1000.0), # veh/km
                'v_m': float(v_m_avg) * 3.6,  # km/h
                'rho_c': float(rho_c_avg) / (1.0 / 1000.0), # veh/km
                'v_c': float(v_c_avg) * 3.6,  # km/h
                'queue_len': float(queue_len),
                'flow': float(flow_total)
            }

        # Get current phase from first signalized node (if any)
        phase_id = None
        for node_id, node in self.network.nodes.items():
            if node.traffic_lights is not None:
                phase_id = node.traffic_lights.current_phase_index
                break
        
        return SimulationState(
            timestamp=timestamp,
            branches=branches,
            phase_id=phase_id
        )
