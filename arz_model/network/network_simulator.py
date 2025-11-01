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
                self.network = NetworkGrid(self.params)
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
        repeat_k: int = 1
    ) -> Tuple[SimulationState, float]:
        """
        Advance simulation by dt * repeat_k seconds.
        
        Args:
            dt: Single timestep duration (seconds)
            repeat_k: Number of timesteps to execute
            
        Returns:
            (new_state, new_timestamp) tuple
        """
        if not self.is_initialized:
            raise RuntimeError("Network not initialized - call reset() first")
        
        # Execute k timesteps
        for _ in range(repeat_k):
            self.network.step(dt, current_time=self.current_time)
            self.current_time += dt
        
        # Build state observation
        new_state = self._build_state(self.current_time)
        
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
        network_state = self.network.get_network_state()
        branches = {}
        
        print(f"[DEBUG] _build_state: observed_segment_ids = {self.observed_segment_ids}")
        print(f"[DEBUG]   network_state keys = {list(network_state.keys())}")
        
        for seg_id in self.observed_segment_ids:
            if seg_id not in network_state:
                print(f"[DEBUG]   WARNING: {seg_id} not in network_state!")
                continue
            
            U = network_state[seg_id]  # Direct array: (4, N_total)
            print(f"[DEBUG]   Processing {seg_id}: U.shape={U.shape}, U[0,5]={U[0,5]:.4f}, U[1,5]={U[1,5]:.4f}")
            
            # Extract state variables
            rho_m = U[0, :]
            w_m = U[1, :]
            rho_c = U[2, :]
            w_c = U[3, :]
            
            # Compute velocities (avoid division by zero)
            v_m = np.where(rho_m > 1e-6, w_m / rho_m, 0.0)
            v_c = np.where(rho_c > 1e-6, w_c / rho_c, 0.0)
            
            # Average over segment (exclude ghost cells)
            N_real = len(rho_m) - 4  # 2 ghost cells on each side
            rho_m_avg = np.mean(rho_m[2:-2])
            v_m_avg = np.mean(v_m[2:-2])
            rho_c_avg = np.mean(rho_c[2:-2])
            v_c_avg = np.mean(v_c[2:-2])
            
            # Estimate queue length (vehicles with low speed near downstream end)
            # Simple heuristic: count cells with v < 5 km/h in last 30% of segment
            queue_start_idx = int(0.7 * N_real) + 2
            v_low_threshold = 5.0 / 3.6  # 5 km/h in m/s
            
            queue_m = np.sum((v_m[queue_start_idx:-2] < v_low_threshold) * rho_m[queue_start_idx:-2])
            queue_c = np.sum((v_c[queue_start_idx:-2] < v_low_threshold) * rho_c[queue_start_idx:-2])
            queue_len = queue_m + queue_c
            
            # Flow at downstream boundary
            flow_m = rho_m[-3] * v_m[-3]  # Cell before ghost
            flow_c = rho_c[-3] * v_c[-3]
            flow_total = flow_m + flow_c
            
            branches[seg_id] = {
                'rho_m': float(rho_m_avg),
                'v_m': float(v_m_avg) * 3.6,  # Convert m/s to km/h for compatibility
                'rho_c': float(rho_c_avg),
                'v_c': float(v_c_avg) * 3.6,
                'queue_len': float(queue_len),
                'flow': float(flow_total)
            }
        
        # Get current phase from first signalized node (if any)
        phase_id = None
        for node_id, node in self.network.nodes.items():
            if node.traffic_lights is not None:
                # TrafficLightController.get_current_phase() returns Phase object
                # Phase doesn't have an 'id' attribute, use phase index instead
                phase_id = node.traffic_lights.current_phase_index
                break
        
        return SimulationState(
            timestamp=timestamp,
            branches=branches,
            phase_id=phase_id
        )
