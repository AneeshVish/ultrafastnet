"""
Logging configuration for UltrafastNet.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for UltrafastNet.
    
    Parameters
    ----------
    level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    format_string : Optional[str], default=None
        Custom format string for log messages.
    include_timestamp : bool, default=True
        Whether to include timestamp in log messages.
    log_file : Optional[str], default=None
        Path to log file. If None, logs to stdout.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("ultrafastnet")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Create handler
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stdout)
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str = "ultrafastnet") -> logging.Logger:
    """
    Get a logger instance.
    
    Parameters
    ----------
    name : str, default="ultrafastnet"
        Logger name.
        
    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logging.getLogger(name)


# Create default logger
default_logger = get_logger()
