"""
Custom exceptions for UltrafastNet.
"""

class UltrafastNetError(Exception):
    """Base exception class for UltrafastNet."""
    pass

class ConfigError(UltrafastNetError):
    """
    Raised when there's an error in network configuration.
    
    Examples
    --------
    >>> from ultrafastnet.utils.exceptions import ConfigError
    >>> raise ConfigError("input_dim must be positive")
    """
    pass

class ValidationError(UltrafastNetError):
    """
    Raised when input validation fails.
    
    Examples
    --------
    >>> from ultrafastnet.utils.exceptions import ValidationError
    >>> raise ValidationError("Input shape mismatch")
    """
    pass

class RoutingError(UltrafastNetError):
    """
    Raised when there's an error in packet routing.
    
    Examples
    --------
    >>> from ultrafastnet.utils.exceptions import RoutingError
    >>> raise RoutingError("Invalid routing mode")
    """
    pass

class AggregationError(UltrafastNetError):
    """
    Raised when there's an error in output aggregation.
    
    Examples
    --------
    >>> from ultrafastnet.utils.exceptions import AggregationError
    >>> raise AggregationError("Deduplication failed")
    """
    pass
