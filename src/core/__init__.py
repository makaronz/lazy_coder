"""
LazyCoder Core Module
Contains core functionality and configuration management.
"""

from .config import LazyCoderConfig, load_config, get_config

__all__ = [
    "LazyCoderConfig",
    "load_config", 
    "get_config"
]