"""
Input validation utilities for UltrafastNet.
"""

import numpy as np
from typing import Any, Dict, Optional, Union
from numpy.typing import NDArray

from .exceptions import ValidationError, ConfigError


def validate_input_shape(
    x: NDArray[np.float32], 
    expected_features: int,
    min_batch_size: int = 1,
    max_batch_size: Optional[int] = None
) -> None:
    """
    Validate input tensor shape and properties.
    """
    if not isinstance(x, np.ndarray):
        raise ValidationError(f"Input must be numpy array, got {type(x)}")
    
    if x.ndim != 2:
        raise ValidationError(f"Input must be 2D tensor (batch_size, features), got {x.ndim}D")
    
    batch_size, features = x.shape
    
    if batch_size < min_batch_size:
        raise ValidationError(f"Batch size {batch_size} is less than minimum {min_batch_size}")
    
    if max_batch_size is not None and batch_size > max_batch_size:
        raise ValidationError(f"Batch size {batch_size} exceeds maximum {max_batch_size}")
    
    if features != expected_features:
        raise ValidationError(f"Expected {expected_features} features, got {features}")
    
    if not np.issubdtype(x.dtype, np.floating):
        raise ValidationError(f"Input must be floating point, got {x.dtype}")
    
    if not np.isfinite(x).all():
        raise ValidationError("Input contains non-finite values (NaN or Inf)")


def validate_config(**config: Any) -> None:
    """
    Validate network configuration parameters.
    """
    # Validate positive integers
    positive_int_params = [
        'input_dim', 'n_blocks', 'n_neurons', 'hidden_dim', 
        'neuron_output_dim', 'final_output_dim'
    ]
    
    for param in positive_int_params:
        if param in config:
            value = config[param]
            if not isinstance(value, int) or value <= 0:
                raise ConfigError(f"{param} must be a positive integer, got {value}")
    
    # Validate probabilities (0 to 1)
    prob_params = ['dedup_threshold', 'dropout_rate']
    
    for param in prob_params:
        if param in config:
            value = config[param]
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ConfigError(f"{param} must be in range [0, 1], got {value}")
    
    # Validate specific relationships
    if 'input_dim' in config and 'n_blocks' in config:
        input_dim = config['input_dim']
        n_blocks = config['n_blocks']
        if input_dim % n_blocks != 0:
            raise ConfigError(f"input_dim ({input_dim}) must be divisible by n_blocks ({n_blocks})")
    
    # Validate routing modes
    if 'routing_mode' in config:
        routing_mode = config['routing_mode']
        valid_modes = ['frequency_hash']
        if routing_mode not in valid_modes:
            raise ConfigError(f"routing_mode must be one of {valid_modes}, got {routing_mode}")
    
    # Validate activation functions
    if 'activation' in config:
        activation = config['activation']
        valid_activations = ['relu', 'tanh', 'sigmoid']
        if activation not in valid_activations:
            raise ConfigError(f"activation must be one of {valid_activations}, got {activation}")
    
    # Validate initialization methods
    if 'init_method' in config:
        init_method = config['init_method']
        valid_methods = ['he', 'xavier', 'normal']
        if init_method not in valid_methods:
            raise ConfigError(f"init_method must be one of {valid_methods}, got {init_method}")


def validate_routing_probs(probs: NDArray[np.float32]) -> None:
    """
    Validate routing probabilities.
    """
    if not isinstance(probs, np.ndarray):
        raise ValidationError(f"Probabilities must be numpy array, got {type(probs)}")
    
    if probs.ndim != 2:
        raise ValidationError(f"Probabilities must be 2D tensor, got {probs.ndim}D")
    
    if not np.issubdtype(probs.dtype, np.floating):
        raise ValidationError(f"Probabilities must be floating point, got {probs.dtype}")
    
    if not np.all(probs >= 0):
        raise ValidationError("Probabilities must be non-negative")
    
    if not np.all(probs <= 1):
        raise ValidationError("Probabilities must be <= 1")
    
    # Check if rows sum to approximately 1 (allowing for small numerical errors)
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-5):
        raise ValidationError("Probability rows must sum to 1")


def validate_expert_outputs(outputs: NDArray[np.float32], expected_shape: tuple) -> None:
    """
    Validate expert outputs tensor.
    """
    if not isinstance(outputs, np.ndarray):
        raise ValidationError(f"Outputs must be numpy array, got {type(outputs)}")
    
    if outputs.ndim != 3:
        raise ValidationError(f"Outputs must be 3D tensor, got {outputs.ndim}D")
    
    if outputs.shape != expected_shape:
        raise ValidationError(f"Expected shape {expected_shape}, got {outputs.shape}")
    
    if not np.issubdtype(outputs.dtype, np.floating):
        raise ValidationError(f"Outputs must be floating point, got {outputs.dtype}")
    
    if not np.isfinite(outputs).all():
        raise ValidationError("Outputs contain non-finite values (NaN or Inf)")


def validate_tensor_dtype(tensor: NDArray, expected_dtype: type = np.float32) -> None:
    """
    Validate tensor data type.
    """
    if tensor.dtype != expected_dtype:
        raise ValidationError(f"Expected dtype {expected_dtype}, got {tensor.dtype}")


def validate_batch_consistency(*tensors: NDArray) -> None:
    """
    Validate that multiple tensors have consistent batch sizes.
    """
    if len(tensors) < 2:
        return
    
    batch_sizes = [tensor.shape[0] for tensor in tensors]
    
    if not all(bs == batch_sizes[0] for bs in batch_sizes):
        raise ValidationError(f"Inconsistent batch sizes: {batch_sizes}")


def validate_parameter_range(
    param_name: str, 
    value: Union[int, float], 
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    inclusive: bool = True
) -> None:
    """
    Validate parameter is within specified range.
    """
    if min_val is not None:
        if inclusive and value < min_val:
            raise ConfigError(f"{param_name} must be >= {min_val}, got {value}")
        elif not inclusive and value <= min_val:
            raise ConfigError(f"{param_name} must be > {min_val}, got {value}")
    
    if max_val is not None:
        if inclusive and value > max_val:
            raise ConfigError(f"{param_name} must be <= {max_val}, got {value}")
        elif not inclusive and value >= max_val:
            raise ConfigError(f"{param_name} must be < {max_val}, got {value}")


def validate_array_properties(
    array: NDArray,
    check_finite: bool = True,
    check_positive: bool = False,
    check_non_negative: bool = False,
    check_normalized: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> None:
    """
    Validate various properties of numpy arrays.
    """
    if check_finite and not np.isfinite(array).all():
        raise ValidationError("Array contains non-finite values")
    
    if check_positive and not np.all(array > 0):
        raise ValidationError("Array must have all positive values")
    
    if check_non_negative and not np.all(array >= 0):
        raise ValidationError("Array must have all non-negative values")
    
    if check_normalized:
        if array.ndim == 1:
            if not np.isclose(array.sum(), 1.0, rtol=rtol, atol=atol):
                raise ValidationError("Array must be normalized (sum to 1)")
        else:
            # Check last dimension normalization
            sums = array.sum(axis=-1)
            if not np.allclose(sums, 1.0, rtol=rtol, atol=atol):
                raise ValidationError("Array must be normalized along last dimension")
