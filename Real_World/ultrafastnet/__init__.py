"""
UltrafastNet: Ultra-Fast Packet-Switching Neural Network Library

A production-ready implementation of packet-switching neural networks
with vectorized operations for high-performance inference.
"""

from .core.network import PacketNet
from .core.router import Router
from .core.expert import Expert
from .core.aggregator import Aggregator
from .utils.exceptions import UltrafastNetError, ConfigError, ValidationError

__version__ = "0.1.0"
__author__ = "Aneesh H Vishwamitra"
__email__ = "aneeshvish4@gmail.com"
__license__ = "MIT"

__all__ = [
    "PacketNet",
    "Router", 
    "Expert",
    "Aggregator",
    "UltrafastNetError",
    "ConfigError",
    "ValidationError",
    "__version__"
]
