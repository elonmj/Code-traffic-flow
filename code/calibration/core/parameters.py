"""
Minimal Parameter Set for ARZ Calibration
=========================================

Temporary implementation until the full parameter system is available.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelParameters:
    """Basic model parameters for ARZ simulation"""

    # Default parameters
    max_speed: float = 50.0
    capacity: float = 1500.0
    lane_change_prob: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'max_speed': self.max_speed,
            'capacity': self.capacity,
            'lane_change_prob': self.lane_change_prob
        }

    def update_from_dict(self, params: Dict[str, Any]):
        """Update parameters from dictionary"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
