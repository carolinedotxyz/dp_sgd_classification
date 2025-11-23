"""
Configuration management.

This module provides configuration classes and utilities for:
- Training hyperparameters (baseline and DP-SGD)
- Notebook configuration (YAML-based)
- Platform-specific settings and detection
"""

from .training import BaselineConfig, DPConfig, DPConfigV2
from .notebook import (
    NotebookConfig,
    get_config,
    reload_config,
    get_config_path,
    config,
)
from .platform import (
    PlatformConfig,
    get_platform_config,
    get_data_loader_config,
    get_worker_init_function,
    get_torch_config,
    log_platform_info,
    platform_config,
)

__all__ = [
    # Training configs
    "BaselineConfig",
    "DPConfig",
    "DPConfigV2",
    # Notebook config
    "NotebookConfig",
    "get_config",
    "reload_config",
    "get_config_path",
    "config",
    # Platform utilities
    "PlatformConfig",
    "get_platform_config",
    "get_data_loader_config",
    "get_worker_init_function",
    "get_torch_config",
    "log_platform_info",
    "platform_config",
]

