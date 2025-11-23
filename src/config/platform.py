"""
Platform detection and configuration utilities.

This module handles platform-specific workarounds and configurations,
providing a clean interface for platform-agnostic code.
"""

import os
import platform
import sys
from typing import Dict, Any, Optional
from pathlib import Path


class PlatformConfig:
    """Platform-specific configuration and workarounds."""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.is_macos = self.system == "Darwin"
        self.is_linux = self.system == "Linux"
        self.is_windows = self.system == "Windows"
        self.is_m1_mac = self.is_macos and self.machine == "arm64"
        self.is_intel_mac = self.is_macos and self.machine == "x86_64"
        
        # Detect Python multiprocessing start method
        self.multiprocessing_start_method = self._detect_multiprocessing_method()
        
        # Platform-specific configurations
        self._apply_platform_workarounds()
    
    def _detect_multiprocessing_method(self) -> str:
        """Detect the appropriate multiprocessing start method for this platform."""
        if self.is_macos:
            return "spawn"  # macOS default
        elif self.is_linux:
            return "fork"   # Linux default
        elif self.is_windows:
            return "spawn"  # Windows default
        else:
            return "spawn"  # Conservative default
    
    def _apply_platform_workarounds(self) -> None:
        """Apply platform-specific workarounds."""
        if self.is_m1_mac:
            # Fix OpenMP issues on M1 Macs (multiple libomp.dylib copies)
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    def get_data_loader_config(self) -> Dict[str, Any]:
        """Get platform-appropriate data loader configuration."""
        if self.is_macos:
            # macOS has multiprocessing issues, disable workers
            return {
                "num_workers": 0,
                "pin_memory": False,
                "multiprocessing_context": None,
                "reason": "macOS multiprocessing compatibility"
            }
        elif self.is_linux:
            # Linux can use multiprocessing safely
            return {
                "num_workers": 4,  # Reasonable default
                "pin_memory": True,
                "multiprocessing_context": None,
                "reason": "Linux multiprocessing support"
            }
        elif self.is_windows:
            # Windows can use multiprocessing but with caution
            return {
                "num_workers": 2,  # Conservative default
                "pin_memory": False,
                "multiprocessing_context": None,
                "reason": "Windows multiprocessing compatibility"
            }
        else:
            # Unknown platform, conservative settings
            return {
                "num_workers": 0,
                "pin_memory": False,
                "multiprocessing_context": None,
                "reason": "Unknown platform, conservative settings"
            }
    
    def get_worker_init_function(self, base_seed: int) -> Optional[callable]:
        """Get platform-appropriate worker initialization function."""
        if self.multiprocessing_start_method == "spawn":
            # For spawn method, we need a top-level function
            def worker_init_fn(worker_id: int):
                import random
                import numpy as np
                seed = (base_seed + worker_id * 9973) % (2**32 - 1)
                random.seed(seed)
                np.random.seed(seed)
            return worker_init_fn
        else:
            # For fork method, worker initialization is handled differently
            return None
    
    def get_torch_config(self) -> Dict[str, Any]:
        """Get platform-appropriate PyTorch configuration."""
        config = {
            "deterministic": True,
            "benchmark": False,
        }
        
        if self.is_macos:
            config.update({
                "mps_fallback": True,  # Enable MPS fallback on macOS
            })
        
        return config
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        return {
            "system": self.system,
            "machine": self.machine,
            "python_version": sys.version,
            "platform": platform.platform(),
            "is_macos": self.is_macos,
            "is_linux": self.is_linux,
            "is_windows": self.is_windows,
            "is_m1_mac": self.is_m1_mac,
            "is_intel_mac": self.is_intel_mac,
            "multiprocessing_start_method": self.multiprocessing_start_method,
        }
    
    def log_platform_info(self, logger) -> None:
        """Log platform information for debugging."""
        env_info = self.get_environment_info()
        logger.info(f"Platform: {env_info['system']} ({env_info['machine']})")
        logger.info(f"Multiprocessing method: {env_info['multiprocessing_start_method']}")
        
        if self.is_m1_mac:
            logger.info("M1 Mac detected - OpenMP workaround applied")
        
        data_config = self.get_data_loader_config()
        logger.info(f"Data loader config: {data_config['num_workers']} workers ({data_config['reason']})")


# Global platform configuration instance
platform_config = PlatformConfig()


def get_platform_config() -> PlatformConfig:
    """Get the global platform configuration instance."""
    return platform_config


def get_data_loader_config() -> Dict[str, Any]:
    """Get platform-appropriate data loader configuration."""
    return platform_config.get_data_loader_config()


def get_worker_init_function(base_seed: int) -> Optional[callable]:
    """Get platform-appropriate worker initialization function."""
    return platform_config.get_worker_init_function(base_seed)


def get_torch_config() -> Dict[str, Any]:
    """Get platform-appropriate PyTorch configuration."""
    return platform_config.get_torch_config()


def log_platform_info(logger) -> None:
    """Log platform information for debugging."""
    platform_config.log_platform_info(logger)
