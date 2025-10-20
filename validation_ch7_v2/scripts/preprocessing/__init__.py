"""
Preprocessing Module for Validation Framework.

This module provides data augmentation and transformation utilities:
- vehicle_class_rules: Infer motos/voitures from aggregate TomTom data
- temporal_augmentation: Generate synthetic rush hour demand
- network_topology: Define simplified Victoria Island topology
"""

from .vehicle_class_rules import (
    infer_motos_fraction,
    infer_voitures_fraction,
    compute_class_specific_speeds,
    apply_multiclass_calibration,
    validate_class_split
)

from .temporal_augmentation import (
    generate_rush_hour_demand,
    validate_temporal_consistency,
    export_to_csv
)

from .network_topology import (
    construct_network_from_tomtom,
    validate_network_topology,
    export_to_uxsim_format,
    infer_segment_length,
    infer_lane_count,
    compute_segment_capacity
)

__all__ = [
    # Vehicle class inference
    'infer_motos_fraction',
    'infer_voitures_fraction',
    'compute_class_specific_speeds',
    'apply_multiclass_calibration',
    'validate_class_split',
    
    # Temporal augmentation
    'generate_rush_hour_demand',
    'validate_temporal_consistency',
    'export_to_csv',
    
    # Network topology
    'construct_network_from_tomtom',
    'validate_network_topology',
    'export_to_uxsim_format',
    'infer_segment_length',
    'infer_lane_count',
    'compute_segment_capacity'
]
