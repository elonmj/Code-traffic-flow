"""
Calibration Module for ARZ Traffic Simulator
===========================================

This module provides comprehensive calibration capabilities for the ARZ traffic flow model
using real-world data from TomTom API and corridor geometry.

Architecture:
- core/: Core calibration classes and interfaces
- data/: Data loading and processing components
- optimizers/: Optimization algorithms for parameter calibration
- metrics/: Calibration quality assessment and validation
- config/: Configuration management for calibration scenarios

Main Components:
- NetworkBuilder: Transforms CSV corridor data into ARZ network objects
- DataMapper: Associates speed measurements with network segments
- CalibrationRunner: Orchestrates the complete calibration process
- ParameterOptimizer: Optimizes model parameters against real data
- CalibrationMetrics: Evaluates calibration quality and convergence
"""

from .core.network_builder import NetworkBuilder
from .core.data_mapper import DataMapper
from .core.calibration_runner import CalibrationRunner
from .core.parameter_set import ParameterSet
from .data.corridor_loader import CorridorLoader
from .data.speed_processor import SpeedDataProcessor
from .optimizers.base_optimizer import BaseOptimizer
from .optimizers.gradient_optimizer import GradientOptimizer
from .metrics.calibration_metrics import CalibrationMetrics
from .metrics.validation import CalibrationValidator

__version__ = "1.0.0"
__all__ = [
    'NetworkBuilder',
    'DataMapper',
    'CalibrationRunner',
    'ParameterSet',
    'CorridorLoader',
    'SpeedDataProcessor',
    'BaseOptimizer',
    'GradientOptimizer',
    'CalibrationMetrics',
    'CalibrationValidator'
]
