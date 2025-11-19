"""
Traffic Light Controller

This module provides the TrafficLightController class, which manages the state
of traffic lights at a node.
"""

from typing import List, Dict, Any, Optional

class TrafficLightController:
    """
    Manages traffic light phases and timing for a node.
    """
    def __init__(self, node_id: str, config: Dict[str, Any], incoming_segments: List[str]):
        self.node_id = node_id
        self.config = config
        self.incoming_segments = incoming_segments
        
        self.cycle_time = float(config.get('cycle_time', 90.0))
        self.green_time = float(config.get('green_time', 35.0))
        self.amber_time = float(config.get('amber_time', 3.0))
        self.red_time = float(config.get('red_time', 52.0))
        self.offset = float(config.get('offset', 0.0))
        self.initial_phase = config.get('initial_phase', 'green')
        
        # Manual control mode for RL
        self.manual_mode = False
        self.current_green_segments: List[str] = []
        
    def set_manual_phase(self, green_segments: List[str]):
        """
        Sets the traffic light to a manual phase (overriding the timer).
        
        Args:
            green_segments: List of segment IDs that should be green.
        """
        self.manual_mode = True
        self.current_green_segments = green_segments

    def get_current_green_segments(self, t: float) -> List[str]:
        """
        Returns a list of segment IDs that currently have a green light.
        """
        # If in manual mode (RL control), return the manually set state
        if self.manual_mode:
            return self.current_green_segments

        # Calculate time in cycle
        t_cycle = (t + self.offset) % self.cycle_time
        
        # Simple phase logic:
        # 0 -> green_time: GREEN
        # green_time -> green_time + amber_time: AMBER (treated as RED for safety)
        # rest: RED
        
        # If initial phase is red, we shift the cycle
        if self.initial_phase == 'red':
            t_cycle = (t_cycle + self.red_time) % self.cycle_time

        if t_cycle < self.green_time:
            return self.incoming_segments # All green
        else:
            return [] # All red
