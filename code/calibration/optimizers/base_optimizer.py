"""
Base Optimizer for ARZ Calibration
=================================

This module defines the base interface for optimization algorithms
used in ARZ model calibration.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.

    Defines the interface that all optimization algorithms must implement
    for ARZ model calibration.
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize optimizer.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.objective_function = None
        self.bounds = None
        self.history = []

    def set_objective_function(self, func: Callable[[np.ndarray], float]):
        """
        Set the objective function to minimize.

        Args:
            func: Function that takes parameter vector and returns scalar score
        """
        self.objective_function = func

    def set_bounds(self, bounds: np.ndarray):
        """
        Set parameter bounds for optimization.

        Args:
            bounds: Array of (min, max) bounds for each parameter
        """
        self.bounds = bounds

    @abstractmethod
    def optimize(self, initial_params: np.ndarray) -> Dict[str, Any]:
        """
        Run optimization algorithm.

        Args:
            initial_params: Initial parameter vector

        Returns:
            Dictionary with optimization results
        """
        pass

    def _check_convergence(self, current_score: float, previous_score: float) -> bool:
        """Check if optimization has converged"""
        if previous_score is None:
            return False
        return abs(current_score - previous_score) < self.tolerance

    def _validate_inputs(self, initial_params: np.ndarray):
        """Validate optimization inputs"""
        if self.objective_function is None:
            raise ValueError("Objective function not set")

        if self.bounds is not None and len(initial_params) != len(self.bounds):
            raise ValueError(f"Parameter length {len(initial_params)} doesn't match bounds length {len(self.bounds)}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return self.history.copy()

    def clear_history(self):
        """Clear optimization history"""
        self.history = []
