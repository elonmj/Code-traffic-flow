"""
Global Debug Configuration

Centralized debug flag for all ARZ model modules.
Set DEBUG_LOGS_ENABLED = True to enable verbose logging.
Set DEBUG_LOGS_ENABLED = False to disable all debug logs for production/testing.
"""

# Global debug flag - SET TO False TO DISABLE ALL DEBUG LOGS
DEBUG_LOGS_ENABLED = False

# Boundary condition mode
# AGGRESSIVE: Apply BC directly to ghost cells (may cause instability with high velocities)
# CONSERVATIVE: Let Riemann solver handle BC naturally (more stable)
BC_APPLICATION_MODE = 'CONSERVATIVE'  # Options: 'AGGRESSIVE', 'CONSERVATIVE'
