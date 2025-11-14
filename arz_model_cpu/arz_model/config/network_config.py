"""
Network Configuration Loader for Multi-Segment Networks

Supports 2-file YAML architecture:
- network.yml: Topology (segments, nodes, links) + local parameter overrides
- traffic_control.yml: Traffic signal timing and coordination

This enables:
- Separation of concerns (topology ≠ control)
- Reusability (test different signal plans on same network)
- Collaboration (non-programmers can edit YAML)
- Parameter heterogeneity (arterial ≠ residential speeds)

Author: ARZ Research Team
Date: 2025-10-21 (Phase 6 Pragmatic Implementation)
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class NetworkConfigError(Exception):
    """Raised when network configuration is invalid."""
    pass


class NetworkConfig:
    """
    Load and validate network configuration from YAML files.
    
    Architecture:
        network.yml:
            - Segments (x_min, x_max, N, parameters)
            - Nodes (junctions, boundaries)
            - Links (connections via nodes)
            
        traffic_control.yml:
            - Traffic lights (cycle_time, phases)
            - Signal coordination (offsets)
    
    Usage:
        >>> net_cfg, traffic_cfg = NetworkConfig.load_from_files(
        ...     'config/examples/phase6/network.yml',
        ...     'config/examples/phase6/traffic_control.yml'
        ... )
        >>> print(net_cfg['network']['segments'])
    """
    
    def __init__(self):
        self.network_data: Optional[Dict] = None
        self.traffic_control_data: Optional[Dict] = None
    
    @classmethod
    def load_from_files(
        cls,
        network_path: str,
        traffic_control_path: Optional[str] = None
    ) -> Tuple[Dict, Dict]:
        """
        Load network configuration from YAML files.
        
        Args:
            network_path: Path to network.yml (topology + local parameters)
            traffic_control_path: Path to traffic_control.yml (optional)
            
        Returns:
            (network_config, traffic_control_config) tuple
            
        Raises:
            NetworkConfigError: If configuration is invalid
            FileNotFoundError: If files don't exist
        """
        config = cls()
        
        # Load network topology
        network_path = Path(network_path)
        if not network_path.exists():
            raise FileNotFoundError(f"Network config not found: {network_path}")
        
        try:
            with open(network_path) as f:
                config.network_data = yaml.safe_load(f)
            logger.info(f"Loaded network config from {network_path}")
        except yaml.YAMLError as e:
            raise NetworkConfigError(f"Invalid YAML in {network_path}: {e}")
        
        # Load traffic control (optional)
        if traffic_control_path:
            traffic_path = Path(traffic_control_path)
            if not traffic_path.exists():
                raise FileNotFoundError(f"Traffic control config not found: {traffic_path}")
            
            try:
                with open(traffic_path) as f:
                    config.traffic_control_data = yaml.safe_load(f)
                logger.info(f"Loaded traffic control from {traffic_path}")
            except yaml.YAMLError as e:
                raise NetworkConfigError(f"Invalid YAML in {traffic_path}: {e}")
        else:
            config.traffic_control_data = {}
            logger.info("No traffic control file provided - using defaults")
        
        # Validate configurations
        config._validate_network_schema()
        if config.traffic_control_data:
            config._validate_traffic_control_schema()
        
        logger.info(f"Configuration loaded successfully: "
                   f"{len(config.network_data.get('network', {}).get('segments', {}))} segments, "
                   f"{len(config.network_data.get('network', {}).get('nodes', {}))} nodes")
        
        return config.network_data, config.traffic_control_data
    
    def _validate_network_schema(self):
        """
        Validate network.yml structure.
        
        Required structure:
            network:
              segments:
                seg_id:
                  x_min: float
                  x_max: float
                  N: int
                  start_node: str
                  end_node: str
                  parameters: dict (optional)
              nodes:
                node_id:
                  type: str (boundary/signalized/stop_sign)
                  position: [x, y]
                  incoming_segments: list (optional)
                  outgoing_segments: list (optional)
              links:
                - from_segment: str
                  to_segment: str
                  via_node: str
                  coupling_type: str (optional)
        """
        if not self.network_data:
            raise NetworkConfigError("Network data is empty")
        
        if 'network' not in self.network_data:
            raise NetworkConfigError("Missing top-level 'network' key")
        
        network = self.network_data['network']
        
        # Validate segments
        if 'segments' not in network:
            raise NetworkConfigError("Missing 'segments' in network config")
        
        segments = network['segments']
        if not isinstance(segments, dict):
            raise NetworkConfigError("'segments' must be a dictionary")
        
        for seg_id, seg_data in segments.items():
            self._validate_segment(seg_id, seg_data)
        
        # Validate nodes
        if 'nodes' not in network:
            raise NetworkConfigError("Missing 'nodes' in network config")
        
        nodes = network['nodes']
        if not isinstance(nodes, dict):
            raise NetworkConfigError("'nodes' must be a dictionary")
        
        for node_id, node_data in nodes.items():
            self._validate_node(node_id, node_data)
        
        # Validate links (optional)
        if 'links' in network:
            links = network['links']
            if not isinstance(links, list):
                raise NetworkConfigError("'links' must be a list")
            
            for i, link_data in enumerate(links):
                self._validate_link(i, link_data, segments, nodes)
        
        logger.debug("Network schema validation passed")
    
    def _validate_segment(self, seg_id: str, seg_data: Dict):
        """Validate individual segment configuration."""
        required_keys = ['x_min', 'x_max', 'N', 'start_node', 'end_node']
        missing = [k for k in required_keys if k not in seg_data]
        
        if missing:
            raise NetworkConfigError(f"Segment '{seg_id}' missing required keys: {missing}")
        
        # Type checks
        if not isinstance(seg_data['x_min'], (int, float)):
            raise NetworkConfigError(f"Segment '{seg_id}': x_min must be numeric")
        if not isinstance(seg_data['x_max'], (int, float)):
            raise NetworkConfigError(f"Segment '{seg_id}': x_max must be numeric")
        if not isinstance(seg_data['N'], int):
            raise NetworkConfigError(f"Segment '{seg_id}': N must be integer")
        
        # Value checks
        if seg_data['x_max'] <= seg_data['x_min']:
            raise NetworkConfigError(f"Segment '{seg_id}': x_max must be > x_min")
        if seg_data['N'] < 1:
            raise NetworkConfigError(f"Segment '{seg_id}': N must be >= 1")
        
        # Validate local parameters if present
        if 'parameters' in seg_data:
            if not isinstance(seg_data['parameters'], dict):
                raise NetworkConfigError(f"Segment '{seg_id}': parameters must be dictionary")
    
    def _validate_node(self, node_id: str, node_data: Dict):
        """Validate individual node configuration."""
        required_keys = ['type', 'position']
        missing = [k for k in required_keys if k not in node_data]
        
        if missing:
            raise NetworkConfigError(f"Node '{node_id}' missing required keys: {missing}")
        
        # Type check
        valid_types = ['boundary', 'signalized', 'stop_sign']
        if node_data['type'] not in valid_types:
            raise NetworkConfigError(
                f"Node '{node_id}': type must be one of {valid_types}, got '{node_data['type']}'"
            )
        
        # Position check
        position = node_data['position']
        if not isinstance(position, list) or len(position) != 2:
            raise NetworkConfigError(f"Node '{node_id}': position must be [x, y] list")
        
        if not all(isinstance(p, (int, float)) for p in position):
            raise NetworkConfigError(f"Node '{node_id}': position values must be numeric")
    
    def _validate_link(self, link_idx: int, link_data: Dict, segments: Dict, nodes: Dict):
        """Validate individual link configuration."""
        required_keys = ['from_segment', 'to_segment', 'via_node']
        missing = [k for k in required_keys if k not in link_data]
        
        if missing:
            raise NetworkConfigError(f"Link {link_idx} missing required keys: {missing}")
        
        # Check references exist
        from_seg = link_data['from_segment']
        to_seg = link_data['to_segment']
        via_node = link_data['via_node']
        
        if from_seg not in segments:
            raise NetworkConfigError(f"Link {link_idx}: from_segment '{from_seg}' not found in segments")
        if to_seg not in segments:
            raise NetworkConfigError(f"Link {link_idx}: to_segment '{to_seg}' not found in segments")
        if via_node not in nodes:
            raise NetworkConfigError(f"Link {link_idx}: via_node '{via_node}' not found in nodes")
        
        # Optional coupling_type validation
        if 'coupling_type' in link_data:
            valid_types = ['behavioral', 'priority', 'proportional']
            if link_data['coupling_type'] not in valid_types:
                raise NetworkConfigError(
                    f"Link {link_idx}: coupling_type must be one of {valid_types}"
                )
    
    def _validate_traffic_control_schema(self):
        """
        Validate traffic_control.yml structure.
        
        Required structure:
            traffic_control:
              traffic_lights:
                node_id:
                  cycle_time: float
                  offset: float (optional)
                  phases:
                    - id: int
                      duration: float
                      green_segments: list
                      yellow_segments: list (optional)
        """
        if not self.traffic_control_data:
            return  # Optional file
        
        if 'traffic_control' not in self.traffic_control_data:
            raise NetworkConfigError("Missing top-level 'traffic_control' key")
        
        traffic_control = self.traffic_control_data['traffic_control']
        
        if 'traffic_lights' not in traffic_control:
            logger.warning("No 'traffic_lights' in traffic_control.yml")
            return
        
        traffic_lights = traffic_control['traffic_lights']
        if not isinstance(traffic_lights, dict):
            raise NetworkConfigError("'traffic_lights' must be a dictionary")
        
        for junction_id, tl_data in traffic_lights.items():
            self._validate_traffic_light(junction_id, tl_data)
        
        logger.debug("Traffic control schema validation passed")
    
    def _validate_traffic_light(self, junction_id: str, tl_data: Dict):
        """Validate traffic light configuration."""
        required_keys = ['cycle_time', 'phases']
        missing = [k for k in required_keys if k not in tl_data]
        
        if missing:
            raise NetworkConfigError(f"Traffic light '{junction_id}' missing: {missing}")
        
        # Cycle time check
        if not isinstance(tl_data['cycle_time'], (int, float)):
            raise NetworkConfigError(f"Traffic light '{junction_id}': cycle_time must be numeric")
        if tl_data['cycle_time'] <= 0:
            raise NetworkConfigError(f"Traffic light '{junction_id}': cycle_time must be > 0")
        
        # Phases check
        phases = tl_data['phases']
        if not isinstance(phases, list):
            raise NetworkConfigError(f"Traffic light '{junction_id}': phases must be list")
        if len(phases) == 0:
            raise NetworkConfigError(f"Traffic light '{junction_id}': must have at least 1 phase")
        
        for i, phase in enumerate(phases):
            if 'duration' not in phase:
                raise NetworkConfigError(
                    f"Traffic light '{junction_id}' phase {i}: missing 'duration'"
                )
            if not isinstance(phase['duration'], (int, float)):
                raise NetworkConfigError(
                    f"Traffic light '{junction_id}' phase {i}: duration must be numeric"
                )


# Convenience function for quick loading
def load_network_config(
    network_yml: str,
    traffic_control_yml: Optional[str] = None
) -> Tuple[Dict, Dict]:
    """
    Convenience function to load network configuration.
    
    Args:
        network_yml: Path to network.yml
        traffic_control_yml: Path to traffic_control.yml (optional)
        
    Returns:
        (network_config, traffic_control_config) tuple
    """
    return NetworkConfig.load_from_files(network_yml, traffic_control_yml)
