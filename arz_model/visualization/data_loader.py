"""
Data Loader Module for ARZ Traffic Simulation Results

This module provides a clean interface for loading and validating
simulation results from pickle files, following the Separation of
Concerns principle (Dijkstra, 1974).

Responsibility: Data Loading (Concern 1)
- Load pickle files containing simulation results
- Validate data structure integrity
- Provide accessor methods for time arrays and segment data

Usage:
    loader = SimulationDataLoader('network_simulation_results.pkl')
    loader.load()
    time_array = loader.get_time_array()
    segment_data = loader.get_segment_data('seg_0')
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class SimulationDataLoader:
    """
    Load and validate ARZ traffic simulation results from pickle files.
    
    This class handles all data loading operations, ensuring that the
    pickle file structure is valid before providing access to the data.
    
    Attributes:
        results_file (Path): Path to the pickle file
        results (dict): Loaded simulation results
    """
    
    def __init__(self, results_file: str):
        """
        Initialize the data loader.
        
        Args:
            results_file: Path to the simulation results pickle file
        """
        self.results_file = Path(results_file)
        self.results: Optional[Dict[str, Any]] = None
        
    def load(self) -> 'SimulationDataLoader':
        """
        Load simulation results from pickle file and validate structure.
        
        Returns:
            self: For method chaining
            
        Raises:
            FileNotFoundError: If results file doesn't exist
            ValueError: If results structure is invalid
        """
        if not self.results_file.exists():
            raise FileNotFoundError(
                f"Results file not found: {self.results_file}"
            )
            
        with open(self.results_file, 'rb') as f:
            self.results = pickle.load(f)
            
        self._validate()
        return self
        
    def _validate(self) -> None:
        """
        Validate the structure of loaded results.
        
        Raises:
            ValueError: If required keys are missing or data is malformed
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load() first.")
            
        # Normalize structure: if 'history' is missing but 'time' is present,
        # assume the root dictionary IS the history (common in RL training dumps).
        if 'history' not in self.results and 'time' in self.results:
            self.results = {'history': self.results}

        # Check for required top-level keys
        if 'history' not in self.results:
            raise ValueError("Results missing 'history' key")
            
        history = self.results['history']
        
        # Check for required history keys
        required_keys = ['time', 'segments']
        for key in required_keys:
            if key not in history:
                raise ValueError(f"History missing required key: '{key}'")
                
        # Validate time array
        if not isinstance(history['time'], (list, np.ndarray)):
            raise ValueError("Time data must be list or numpy array")
            
        # Validate segments structure
        if not isinstance(history['segments'], dict):
            raise ValueError("Segments data must be a dictionary")
            
        if len(history['segments']) == 0:
            raise ValueError("No segments found in results")
            
        # Validate at least one segment has required data
        first_seg = next(iter(history['segments'].values()))
        if 'density' not in first_seg:
            raise ValueError("Segment missing 'density' data")
            
        print(f"✓ Validation successful: {len(history['segments'])} segments, "
              f"{len(history['time'])} time steps")
        
    def get_time_array(self) -> np.ndarray:
        """
        Get the time array from simulation results.
        
        Returns:
            numpy array of time steps
            
        Raises:
            ValueError: If results haven't been loaded
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load() first.")
            
        return np.array(self.results['history']['time'])
        
    def get_segment_data(self, seg_id: str) -> Dict[str, np.ndarray]:
        """
        Get data for a specific segment.
        
        Args:
            seg_id: Segment identifier (e.g., 'seg_0', 'seg_1')
            
        Returns:
            Dictionary containing segment data (density, speed, etc.)
            
        Raises:
            ValueError: If results haven't been loaded or segment not found
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load() first.")
            
        segments = self.results['history']['segments']
        
        if seg_id not in segments:
            raise ValueError(
                f"Segment '{seg_id}' not found. "
                f"Available segments: {list(segments.keys())}"
            )
            
        return segments[seg_id]
        
    def get_all_segments(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get data for all segments.
        
        Returns:
            Dictionary mapping segment IDs to their data
            
        Raises:
            ValueError: If results haven't been loaded
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load() first.")
            
        return self.results['history']['segments']
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get simulation metadata (final time, total steps, etc.).
        
        Returns:
            Dictionary containing metadata
            
        Raises:
            ValueError: If results haven't been loaded
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load() first.")
            
        metadata = {}
        
        # Extract metadata fields if they exist
        if 'final_time' in self.results:
            metadata['final_time'] = self.results['final_time']
            
        if 'total_steps' in self.results:
            metadata['total_steps'] = self.results['total_steps']
            
        # Add computed metadata
        metadata['num_segments'] = len(self.results['history']['segments'])
        metadata['num_timesteps'] = len(self.results['history']['time'])
        
        return metadata
        
    def get_simulated_segment_ids(self) -> list:
        """
        Get the list of segment IDs that were actually simulated.
        
        This is crucial for scenario-based visualization where only a subset
        of the network may be simulated. The visualization should highlight
        only these active segments.
        
        Returns:
            List of segment IDs (e.g., ['seg_0', 'seg_1'])
            
        Raises:
            ValueError: If results haven't been loaded
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load() first.")
            
        return list(self.results['history']['segments'].keys())
