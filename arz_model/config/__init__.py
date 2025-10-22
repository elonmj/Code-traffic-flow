"""
Configuration module for ARZ traffic model.

Provides network configuration loading from YAML files with 2-file architecture:
- network.yml: Topology + local parameters
- traffic_control.yml: Signal timing

Author: ARZ Research Team
Date: 2025-10-21
"""

from .network_config import (
    NetworkConfig,
    NetworkConfigError,
    load_network_config
)

__all__ = [
    'NetworkConfig',
    'NetworkConfigError',
    'load_network_config'
]

__version__ = '0.1.0'
