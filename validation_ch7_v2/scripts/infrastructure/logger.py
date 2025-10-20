"""
Centralized logging for the validation system.

This module provides:
- setup_logger(): Configure loggers for different components
- get_logger(): Retrieve or create a logger
- Standard logging patterns for structured output

PHILOSOPHY: DRY principle - logging setup in one place, used everywhere.
This replaces the old system where each test had its own logging configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Standard logging patterns for structured logs
# These allow easy filtering of debug output (e.g., grep for [DEBUG_CHECKPOINT])
DEBUG_BC_RESULT = "[DEBUG_BC_RESULT]"
DEBUG_PRIMITIVES = "[DEBUG_PRIMITIVES]"
DEBUG_FLUXES = "[DEBUG_FLUXES]"
DEBUG_CACHE = "[DEBUG_CACHE]"
DEBUG_CHECKPOINT = "[DEBUG_CHECKPOINT]"
DEBUG_ORCHESTRATION = "[DEBUG_ORCHESTRATION]"
DEBUG_SIMULATION = "[DEBUG_SIMULATION]"


# Global registry of loggers (prevent duplicate handlers)
_loggers = {}


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    force_new: bool = False
) -> logging.Logger:
    """
    Setup and return a logger with the given configuration.
    
    Features:
    - Console handler (stdout) for immediate feedback
    - Optional file handler for persistence
    - Flushes immediately (Kaggle stdout buffering issue)
    - Prevents duplicate loggers
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        force_new: If True, create new logger even if exists
    
    Returns:
        Configured logging.Logger instance
    
    Example:
        ```python
        from validation_ch7_v2.scripts.infrastructure.logger import setup_logger, DEBUG_CHECKPOINT
        
        logger = setup_logger("rl_validation", log_file=Path("debug.log"))
        logger.info(f"{DEBUG_CHECKPOINT} Loading checkpoint from {path}")
        ```
    """
    
    # Check if logger already exists
    if name in _loggers and not force_new:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format: timestamp - name - level - message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Store in registry
    _loggers[name] = logger
    
    return logger


def get_logger(name: str, default_level: int = logging.INFO) -> logging.Logger:
    """
    Retrieve an existing logger or create a default one.
    
    This is the preferred way to get a logger in most code.
    
    Args:
        name: Logger name (typically __name__)
        default_level: Level if logger doesn't exist yet
    
    Returns:
        logging.Logger instance
    
    Example:
        ```python
        from validation_ch7_v2.scripts.infrastructure.logger import get_logger
        
        logger = get_logger(__name__)
        logger.debug("Detailed debugging info")
        ```
    """
    
    if name not in _loggers:
        setup_logger(name, level=default_level)
    
    return _loggers[name]


def clear_loggers() -> None:
    """
    Clear all loggers (useful for testing).
    
    This removes all handlers and clears the logger registry.
    """
    
    for logger in _loggers.values():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    
    _loggers.clear()


# Root logger configuration (for any code using standard logging)
def setup_root_logger(level: int = logging.WARNING) -> None:
    """
    Setup the root logger (for third-party libraries).
    
    Args:
        level: Logging level for root logger
    """
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Add console handler if not present
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        root_logger.addHandler(handler)
