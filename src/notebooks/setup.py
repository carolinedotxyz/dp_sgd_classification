"""
Common setup utilities for notebook cells.

This module provides reusable setup functions to eliminate duplicate code
across multiple notebook cells.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

from ..config.notebook import get_config
from ..config.platform import get_platform_config, get_worker_init_function, log_platform_info
from ..core.utils import get_device, set_seed
from ..core.data import load_stats


class NotebookSetup:
    """Centralized setup utilities for notebook cells."""
    
    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or get_config()
        self.platform_config = get_platform_config()
        self.logger = logging.getLogger("notebook_setup")
        
        # Cached values
        self._device = None
        self._seed = None
        self._stats = None
        self._worker_init_fn = None
    
    def setup_device_and_seed(self, seed: Optional[int] = None) -> Tuple[torch.device, int]:
        """Setup device and seed with logging."""
        if self._device is None:
            self._device = get_device()
        
        if self._seed is None:
            self._seed = seed or self.config.random_seed
        
        # Apply seeding
        set_seed(self._seed)
        
        # Log setup information
        self.logger.info(f"Device: {self._device}")
        self.logger.info(f"Seed: {self._seed}")
        
        return self._device, self._seed
    
    def setup_training_environment(self, processed_root: str) -> Dict[str, Any]:
        """Setup complete training environment with device, seed, and stats."""
        # Setup device and seed
        device, seed = self.setup_device_and_seed()
        
        # Setup stats
        stats_path = str(Path(processed_root) / "stats" / "stats.json")
        if not Path(stats_path).exists():
            raise FileNotFoundError(f"Missing stats file: {stats_path}")
        
        if self._stats is None:
            self._stats = load_stats(stats_path)
        
        mean, std = self._stats
        
        # Setup worker initialization function
        if self._worker_init_fn is None:
            self._worker_init_fn = get_worker_init_function(seed)
        
        # Get platform-appropriate data loader config
        data_config = self.platform_config.get_data_loader_config()
        
        # Log platform information
        log_platform_info(self.logger)
        
        # Log training setup
        self.logger.info(f"Processed root: {processed_root}")
        self.logger.info(f"Stats loaded: mean={mean}, std={std}")
        self.logger.info(f"Data loader workers: {data_config['num_workers']}")
        
        return {
            "device": device,
            "seed": seed,
            "processed_root": processed_root,
            "stats_path": stats_path,
            "mean": mean,
            "std": std,
            "worker_init_fn": self._worker_init_fn,
            "num_workers": data_config["num_workers"],
            "pin_memory": data_config["pin_memory"],
        }
    
    def setup_imports(self, cell_name: str, imports: Dict[str, list]) -> None:
        """Setup imports for a specific cell with logging."""
        self.logger.info(f"Setting up imports for {cell_name}")
        
        for module, items in imports.items():
            if module == "stdlib":
                for item in items:
                    exec(f"import {item}")
            elif module == "third_party":
                for item in items:
                    exec(f"import {item}")
            elif module.startswith("src."):
                module_name = module
                for item in items:
                    exec(f"from {module_name} import {item}")
        
        self.logger.info(f"Imports completed for {cell_name}")
    
    def log_cell_start(self, cell_name: str, description: str = "") -> None:
        """Log the start of a cell execution."""
        self.logger.info(f"Starting {cell_name}")
        if description:
            self.logger.info(f"Description: {description}")
    
    def log_cell_complete(self, cell_name: str, summary: str = "") -> None:
        """Log the completion of a cell execution."""
        self.logger.info(f"Completed {cell_name}")
        if summary:
            self.logger.info(f"Summary: {summary}")
    
    def print_setup_summary(self, setup_info: Dict[str, Any]) -> None:
        """Print a formatted setup summary."""
        print("=" * 60)
        print("🔧 SETUP SUMMARY")
        print("=" * 60)
        print(f"Device: {setup_info['device']}")
        print(f"Seed: {setup_info['seed']}")
        print(f"Processed Root: {setup_info['processed_root']}")
        print(f"Data Workers: {setup_info['num_workers']}")
        print(f"Pin Memory: {setup_info['pin_memory']}")
        print("=" * 60)


# Global setup instance
setup = NotebookSetup()


def get_setup() -> NotebookSetup:
    """Get the global setup instance."""
    return setup


def setup_training_cell(processed_root: str, cell_name: str = "Training Setup") -> Dict[str, Any]:
    """Complete training setup for a cell."""
    setup.log_cell_start(cell_name, "Setting up training environment")
    
    training_env = setup.setup_training_environment(processed_root)
    
    setup.print_setup_summary(training_env)
    setup.log_cell_complete(cell_name, "Training environment ready")
    
    return training_env


def setup_analysis_cell(cell_name: str, description: str = "") -> None:
    """Setup for analysis cells."""
    setup.log_cell_start(cell_name, description)


def setup_workflow_cell(cell_name: str, description: str = "") -> None:
    """Setup for workflow cells."""
    setup.log_cell_start(cell_name, description)


def log_cell_completion(cell_name: str, summary: str = "") -> None:
    """Log cell completion."""
    setup.log_cell_complete(cell_name, summary)
