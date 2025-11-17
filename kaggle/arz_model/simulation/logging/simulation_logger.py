"""
SimulationLogger: Centralized logging system for GPU network simulations.

This module provides a structured logging system with multiple severity levels
to reduce log verbosity while maintaining diagnostic capabilities when needed.

Log Levels (in order of increasing severity):
    - DEBUG: Detailed diagnostic information (includes all CFL details, state dumps)
    - INFO: Standard progress updates (initialization, checkpoints, completion)
    - WARNING: Important events that may need attention (CFL warnings, throttled)
    - ERROR: Critical errors that stop execution
"""

from enum import IntEnum
from typing import Optional


class LogLevel(IntEnum):
    """Logging severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


class SimulationLogger:
    """
    Centralized logger for simulation runs.
    
    Provides structured logging with severity levels to control output verbosity.
    By default, uses INFO level which provides clean progress updates without
    overwhelming detail. Switch to DEBUG for full diagnostic output.
    
    Example:
        >>> logger = SimulationLogger(level=LogLevel.INFO)
        >>> logger.info("Simulation started")  # Printed
        >>> logger.debug("Detailed state dump")  # Not printed (below INFO threshold)
        >>> logger.warning("CFL dt collapsed")  # Printed (above INFO threshold)
    """
    
    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize logger with specified severity level.
        
        Args:
            level: Minimum severity level to print. Messages below this level are ignored.
        """
        self.level = level
    
    def set_level(self, level: LogLevel):
        """Change the logging level."""
        self.level = level
    
    def debug(self, message: str, flush: bool = False):
        """
        Log DEBUG level message (detailed diagnostics).
        
        Args:
            message: Message to log
            flush: If True, force immediate output (useful for crash debugging)
        """
        if self.level <= LogLevel.DEBUG:
            print(f"[DEBUG] {message}", flush=flush)
    
    def info(self, message: str, flush: bool = False):
        """
        Log INFO level message (standard progress).
        
        Args:
            message: Message to log
            flush: If True, force immediate output
        """
        if self.level <= LogLevel.INFO:
            print(message, flush=flush)
    
    def warning(self, message: str, flush: bool = True):
        """
        Log WARNING level message (important events).
        
        Args:
            message: Message to log
            flush: If True, force immediate output (default True for warnings)
        """
        if self.level <= LogLevel.WARNING:
            print(f"⚠️  {message}", flush=flush)
    
    def error(self, message: str, flush: bool = True):
        """
        Log ERROR level message (critical errors).
        
        Args:
            message: Message to log
            flush: If True, force immediate output (default True for errors)
        """
        if self.level <= LogLevel.ERROR:
            print(f"❌ ERROR: {message}", flush=flush)
    
    def section(self, title: str, char: str = "=", width: int = 70, flush: bool = False):
        """
        Print a section header (respects INFO level).
        
        Args:
            title: Section title
            char: Character to use for separator lines
            width: Width of separator line
            flush: If True, force immediate output
        """
        if self.level <= LogLevel.INFO:
            print(f"\n{char * width}", flush=flush)
            print(f"{title}", flush=flush)
            print(f"{char * width}", flush=flush)
    
    def subsection(self, title: str, char: str = "-", width: int = 70, flush: bool = False):
        """
        Print a subsection header (respects INFO level).
        
        Args:
            title: Subsection title
            char: Character to use for separator line
            width: Width of separator line
            flush: If True, force immediate output
        """
        if self.level <= LogLevel.INFO:
            print(f"\n{title}", flush=flush)
            print(f"{char * width}", flush=flush)


def create_logger_from_flags(quiet: bool = False, debug: bool = False) -> SimulationLogger:
    """
    Create a logger from legacy quiet/debug flags for backward compatibility.
    
    Args:
        quiet: If True, only show errors (ERROR level)
        debug: If True, show all diagnostics (DEBUG level)
    
    Returns:
        SimulationLogger configured according to flags
    
    Precedence:
        - If debug=True: DEBUG level (most verbose)
        - If quiet=True: ERROR level (least verbose)
        - Otherwise: INFO level (default)
    """
    if debug:
        return SimulationLogger(level=LogLevel.DEBUG)
    elif quiet:
        return SimulationLogger(level=LogLevel.ERROR)
    else:
        return SimulationLogger(level=LogLevel.INFO)
