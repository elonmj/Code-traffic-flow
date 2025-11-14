"""
Logging utilities for numerical solvers.
Provides time-based logging control to reduce verbose output.
"""

# ===== LOGGING CONTROL =====
_last_log_time = -1000.0  # Track last time we logged (start with large negative to log at t=0)
_log_interval = 50.0  # Log every 50 seconds of simulation time

def should_log(current_time: float) -> bool:
    """
    Check if we should log at this simulation time.
    Logs at t=0 and then every _log_interval seconds.
    
    Args:
        current_time (float): Current simulation time in seconds
        
    Returns:
        bool: True if we should log, False otherwise
    """
    global _last_log_time
    if current_time - _last_log_time >= _log_interval:
        _last_log_time = current_time
        return True
    return False

def set_log_interval(interval: float):
    """
    Set the logging interval.
    
    Args:
        interval (float): Time interval in seconds between logs
    """
    global _log_interval
    _log_interval = interval

def reset_log_timer():
    """Reset the logging timer (useful for new simulations)."""
    global _last_log_time
    _last_log_time = -1000.0
