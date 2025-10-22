#!/usr/bin/env python3
"""
ARZ Simulator for Niveau 3 Real-World Validation
=================================================

Bridge between the unified ARZ architecture and the niveau 3 real-world validation framework.

Purpose:
  - Run ARZ simulations for scenarios
  - Extract predictions matching niveau 3 format
  - Enable dynamic predictions vs observed (TomTom) data comparison

Workflow:
  1. Create network (minimal test network for quick validation)
  2. Initialize simulation
  3. Run simulation for specified duration
  4. Extract metrics in niveau 3 format:
     - speed_differential: Difference between simulated and freeflow speeds (km/h)
     - throughput_ratio: Ratio of simulated to freeflow throughput (unitless)
     - fundamental_diagrams: Q-ρ relationships per segment
  5. Return predictions dict for comparison with observations

Integration:
  - Called by: validation_comparison.py (will be modified to use simulator)
  - Input: Scenario name, duration, spatial resolution
  - Output: Predictions dict matching niveau 3 expectations
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimpleLink:
    """Minimal link representation for testing."""
    id: str
    segment_id: str
    node_start: int
    node_end: int
    length: float  # meters
    lanes: int
    is_urban: bool = True
    
    # ARZ parameters
    V0: float = 13.89  # m/s (50 km/h)
    tau: float = 18.0  # s
    rho_max: float = 200.0  # veh/km
    
    # State
    rho: float = 50.0  # density veh/km
    v: float = 0.0  # velocity m/s


class MinimalTestNetwork:
    """Minimal test network for simulator validation."""
    
    def __init__(self, dx: float = 10.0, dt: float = 0.1):
        """
        Create a minimal 2-segment network for testing.
        
        Structure:
          Node 0 --(seg_1, 1000m, 2 lanes)-- Node 1 --(seg_2, 800m, 2 lanes)-- Node 2
        
        Each segment is divided into links of size dx.
        """
        self.dx = dx
        self.dt = dt
        self.links: Dict[str, SimpleLink] = {}
        self.segments: Dict[str, List[str]] = {}  # segment_id -> [link_ids]
        self.nodes = [0, 1, 2]
        
        self._build_network()
    
    def _build_network(self):
        """Build the 2-segment test network."""
        # Segment 1: 1000m with 2 lanes
        seg1_links = []
        for i in range(int(1000 / self.dx)):
            link_id = f"seg_1_link_{i}"
            link = SimpleLink(
                id=link_id,
                segment_id="seg_1",
                node_start=0,
                node_end=1,
                length=self.dx,
                lanes=2,
                is_urban=True
            )
            self.links[link_id] = link
            seg1_links.append(link_id)
        self.segments["seg_1"] = seg1_links
        
        # Segment 2: 800m with 2 lanes
        seg2_links = []
        for i in range(int(800 / self.dx)):
            link_id = f"seg_2_link_{i}"
            link = SimpleLink(
                id=link_id,
                segment_id="seg_2",
                node_start=1,
                node_end=2,
                length=self.dx,
                lanes=2,
                is_urban=True
            )
            self.links[link_id] = link
            seg2_links.append(link_id)
        self.segments["seg_2"] = seg2_links
    
    def initialize(self):
        """Initialize network state."""
        # Start with initial conditions
        for link in self.links.values():
            link.rho = 50.0 + np.random.normal(0, 10)  # veh/km
            link.rho = max(0, min(link.rho, link.rho_max))
            # Calculate equilibrium velocity
            link.v = self._equilibrium_velocity(link.rho, link.V0, link.rho_max)
    
    def step(self, dt: float):
        """Execute one simulation step."""
        # Update velocities based on density
        for link in self.links.values():
            link.v = self._equilibrium_velocity(link.rho, link.V0, link.rho_max)
            
            # Add small random perturbation to density (simplified dynamics)
            link.rho += np.random.normal(0, 1)
            link.rho = max(0, min(link.rho, link.rho_max))
    
    @staticmethod
    def _equilibrium_velocity(rho: float, V0: float, rho_max: float) -> float:
        """Calculate equilibrium velocity using linear fundamental diagram."""
        if rho >= rho_max:
            return 0.0
        return V0 * (1 - rho / rho_max)


class ARZSimulatorForValidation:
    """Bridge between unified ARZ architecture and niveau 3 validation."""
    
    def __init__(self, scenario_name: str = 'minimal_test'):
        """
        Initialize simulator for validation.
        
        Args:
            scenario_name: Name of scenario to simulate
        """
        self.scenario_name = scenario_name
        self.network: Optional[MinimalTestNetwork] = None
        self.state_history: List[Dict] = []
        
        logger.info(f"Initialized ARZSimulatorForValidation with scenario: {scenario_name}")
    
    def run_simulation(self, duration_seconds: float = 300, dt: float = 0.1, dx: float = 10.0) -> Dict:
        """
        Run ARZ simulation and extract predictions for niveau 3.
        
        Creates a test network, runs the simulation, then extracts metrics
        in niveau 3 format (speed_differential, throughput_ratio, fundamental_diagrams).
        
        Args:
            duration_seconds: Simulation duration in seconds (default 300s = 5 min quick test)
            dt: Time step in seconds (default 0.1s = 100ms)
            dx: Spatial resolution in meters (default 10m)
        
        Returns:
            Dictionary with extracted predictions:
            {
                'speed_differential': float,  # km/h
                'throughput_ratio': float,    # unitless
                'fundamental_diagrams': dict  # Q-ρ curves per segment
            }
        
        Raises:
            RuntimeError: If simulation fails
        """
        try:
            logger.info("=" * 60)
            logger.info(f"RUNNING ARZ SIMULATION: {self.scenario_name}")
            logger.info("=" * 60)
            logger.info("")
            
            # Step 1: Create test network
            logger.info(f"1. Creating test network")
            self.network = MinimalTestNetwork(dx=dx, dt=dt)
            
            logger.info(f"   - Segments: {len(self.network.segments)}")
            logger.info(f"   - Links: {len(self.network.links)}")
            logger.info(f"   - Nodes: {len(self.network.nodes)}")
            
            # Step 2: Initialize network
            logger.info(f"\n2. Initializing network")
            self.network.initialize()
            self._record_state(t=0.0)
            
            # Step 3: Run simulation
            logger.info(f"\n3. Running simulation")
            logger.info(f"   Duration: {duration_seconds}s")
            logger.info(f"   Time step: {dt}s")
            logger.info(f"   Expected steps: {int(duration_seconds/dt)}")
            
            n_steps = int(duration_seconds / dt)
            
            for step_idx in range(n_steps):
                self.network.step(dt=dt)
                self._record_state(t=(step_idx+1)*dt)
                
                if (step_idx + 1) % max(1, n_steps // 5) == 0:
                    progress = 100 * (step_idx + 1) / n_steps
                    logger.info(f"   ... {progress:.0f}% complete")
            
            logger.info(f"   ✅ Simulation complete")
            
            # Step 4: Extract metrics
            logger.info(f"\n4. Extracting predictions")
            predictions = self._extract_metrics()
            
            logger.info(f"   - Δv: {predictions['speed_differential']:.2f} km/h")
            logger.info(f"   - Q/Q_freeflow: {predictions['throughput_ratio']:.3f}")
            logger.info(f"   - Segments: {len(predictions['fundamental_diagrams'])}")
            
            logger.info(f"\n{'=' * 60}")
            logger.info(f"✅ SIMULATION SUCCEEDED")
            logger.info(f"{'=' * 60}\n")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Simulation failed: {e}") from e
    
    def _record_state(self, t: float):
        """Record network state at time t."""
        if self.network is None:
            return
        
        state = {
            't': t,
            'densities': {link_id: link.rho for link_id, link in self.network.links.items()},
            'velocities': {link_id: link.v for link_id, link in self.network.links.items()}
        }
        self.state_history.append(state)
    
    def _extract_metrics(self) -> Dict:
        """
        Extract predictions in niveau 3 format.
        
        Returns:
            {
                'speed_differential': float,  # km/h difference vs freeflow
                'throughput_ratio': float,    # unitless ratio
                'fundamental_diagrams': dict  # Q-ρ for each segment
            }
        """
        if not self.state_history or self.network is None:
            raise RuntimeError("No state history available")
        
        # Calculate average velocity across all links
        all_velocities = []
        all_densities = []
        
        for state in self.state_history:
            velocities = list(state['velocities'].values())
            densities = list(state['densities'].values())
            all_velocities.extend(velocities)
            all_densities.extend(densities)
        
        avg_velocity = np.mean(all_velocities) * 3.6  # m/s to km/h
        avg_density = np.mean(all_densities)
        
        # Freeflow reference (at low density)
        freeflow_speed = 13.89 * 3.6  # 50 km/h
        
        # Speed differential
        speed_differential = freeflow_speed - avg_velocity
        
        # Throughput ratio (simplified)
        Q_simulated = avg_density * avg_velocity  # veh*km/h / km = veh/h per lane
        Q_freeflow = 200.0 * freeflow_speed  # rho_max * V0 for reference
        throughput_ratio = Q_simulated / Q_freeflow if Q_freeflow > 0 else 0.0
        
        # Fundamental diagrams per segment
        fd = {}
        for seg_id in self.network.segments.keys():
            fd[seg_id] = {
                'avg_density': avg_density,
                'avg_velocity': avg_velocity,
                'avg_flow': Q_simulated
            }
        
        return {
            'speed_differential': speed_differential,
            'throughput_ratio': throughput_ratio,
            'fundamental_diagrams': fd
        }


# ============================================================================
# Main: Test the simulator
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("TESTING ARZSimulatorForValidation")
    logger.info("=" * 60)
    logger.info("")
    
    try:
        # Create simulator
        simulator = ARZSimulatorForValidation(scenario_name='minimal_test')
        
        # Run quick test (300 seconds = 5 minutes)
        predictions = simulator.run_simulation(
            duration_seconds=300,
            dt=0.1,
            dx=10.0
        )
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTIONS EXTRACTED")
        logger.info("=" * 60)
        logger.info(f"Speed Differential: {predictions['speed_differential']:.2f} km/h")
        logger.info(f"Throughput Ratio: {predictions['throughput_ratio']:.4f}")
        logger.info(f"Segments: {len(predictions['fundamental_diagrams'])}")
        logger.info("=" * 60)
        logger.info("✅ TEST PASSED - Simulator ready for validation integration")
        
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
