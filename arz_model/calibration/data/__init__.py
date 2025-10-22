"""
Data loading and processing components

LEGACY STATUS: October 2025 Cleanup
This package was primarily used for 2024 calibration phase (Phase 1.3).
Most modules have been archived to _archive/2024_phase13_calibration/.

MAINTAINED COMPONENTS:
- real_data_loader.py: Required by test_section_7_4_calibration.py
- groups/: Network configuration for Victoria Island corridor

See .audit/CALIBRATION_DATA_OBSOLESCENCE_AUDIT.md for detailed analysis.
"""

from .real_data_loader import RealDataLoader

__all__ = [
    'RealDataLoader',
]
