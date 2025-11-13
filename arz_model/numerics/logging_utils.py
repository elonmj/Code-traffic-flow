"""
Logging utilities for frequency-based debug output.
"""

_log_counters = {}

def should_log(time: float, interval: float = 50.0, key: str = 'default') -> bool:
    """
    Determines if a log message should be printed based on time interval.
    
    Args:
        time (float): Current simulation time.
        interval (float): Logging interval in seconds.
        key (str): Unique key for the log type to track intervals independently.
        
    Returns:
        bool: True if it's time to log, False otherwise.
    """
    global _log_counters
    
    if key not in _log_counters:
        _log_counters[key] = {'last_log_time': -interval}
        
    last_log_time = _log_counters[key]['last_log_time']
    
    if time >= last_log_time + interval:
        _log_counters[key]['last_log_time'] = time
        return True
        
    return False
