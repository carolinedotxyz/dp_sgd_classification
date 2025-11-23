"""
Centralized configuration for the CelebA Eyeglasses workflow notebook.

This module loads configuration from a YAML file and provides a clean interface
for accessing all hyperparameters, environment settings, and data configurations.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from .platform import get_platform_config, get_data_loader_config


class NotebookConfig:
    """Configuration loader for the CelebA Eyeglasses workflow notebook.
    
    This class loads configuration from a YAML file and provides convenient
    access methods for different configuration sections.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file. If None, uses default location
                under the repository root (``<repo_root>/notebooks/config.yaml``).
        """
        if config_path is None:
            # Default to config.yaml in the notebooks directory at repo root
            # __file__ = <repo_root>/src/config/notebook.py → parents[2] == <repo_root>
            # Find repo root by looking for the 'src' directory
            current = Path(__file__).resolve()
            repo_root = None
            for parent in [current] + list(current.parents):
                if (parent / "src").exists() and (parent / "notebooks").exists():
                    repo_root = parent
                    break
            if repo_root is None:
                # Fallback: use parents[2] as before
                repo_root = current.parents[2]
            config_path = repo_root / "notebooks" / "config.yaml"
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()
    
    # === Environment & Setup ===
    @property
    def random_seed(self) -> int:
        return self._config["environment"]["random_seed"]
    
    @property
    def figure_dpi(self) -> int:
        return self._config["environment"]["figure_dpi"]
    
    @property
    def matplotlib_style(self) -> str:
        return self._config["environment"]["matplotlib_style"]
    
    @property
    def matplotlib_style_fallback(self) -> str:
        return self._config["environment"]["matplotlib_style_fallback"]
    
    # === Dataset Configuration ===
    @property
    def attribute(self) -> str:
        return self._config["dataset"]["attribute"]
    
    @property
    def dataset_name(self) -> str:
        return self._config["dataset"]["dataset_name"]
    
    @property
    def processed_dataset_name(self) -> str:
        return self._config["dataset"]["processed_dataset_name"]
    
    # === Subset Configuration ===
    @property
    def max_train_per_class(self) -> int:
        return self._config["subset"]["max_train_per_class"]
    
    @property
    def max_val_per_class(self) -> int:
        return self._config["subset"]["max_val_per_class"]
    
    @property
    def max_test_per_class(self) -> int:
        return self._config["subset"]["max_test_per_class"]
    
    @property
    def link_mode(self) -> str:
        return self._config["subset"]["link_mode"]
    
    # === Preprocessing Configuration ===
    @property
    def preprocess_size(self) -> int:
        return self._config["preprocessing"]["size"]
    
    @property
    def preprocess_center_crop(self) -> bool:
        return self._config["preprocessing"]["center_crop"]
    
    @property
    def preprocess_normalize_01(self) -> bool:
        return self._config["preprocessing"]["normalize_01"]
    
    @property
    def preprocess_compute_stats(self) -> bool:
        return self._config["preprocessing"]["compute_stats"]
    
    # === Analysis Configuration ===
    @property
    def plot_top_n_attrs(self) -> int:
        return self._config["analysis"]["plot_top_n_attrs"]
    
    @property
    def size_sample_max(self) -> int:
        return self._config["analysis"]["size_sample_max"]
    
    @property
    def channel_stats_split(self) -> str:
        return self._config["analysis"]["channel_stats_split"]
    
    @property
    def proc_pixel_hist_bins(self) -> int:
        return self._config["analysis"]["proc_pixel_hist_bins"]
    
    # === Diagnostics Configuration ===
    @property
    def diag_target_size(self) -> int:
        return self._config["diagnostics"]["target_size"]
    
    @property
    def diag_visual_sample(self) -> int:
        return self._config["diagnostics"]["visual_sample"]
    
    # === Training Configuration ===
    @property
    def matched_sweep_grid(self) -> List[Dict[str, Any]]:
        """Get the matched hyperparameter sweep grid (same for baseline and DP-SGD)."""
        return self._config["training"]["matched_sweep_grid"]
    
    @property
    def epochs_quick(self) -> int:
        """Get number of epochs for quick run."""
        return self._config["training"]["epochs_quick"]
    
    @property
    def epochs_long(self) -> int:
        """Get number of epochs for longer run."""
        return self._config["training"]["epochs_long"]
    
    @property
    def training_aug_flip(self) -> bool:
        return self._config["training"]["aug_flip"]
    
    @property
    def training_aug_jitter(self) -> bool:
        return self._config["training"]["aug_jitter"]
    
    @property
    def training_aug_rotation(self) -> bool:
        return self._config["training"].get("aug_rotation", False)
    
    @property
    def training_rotation_degrees(self) -> float:
        return self._config["training"].get("rotation_degrees", 10.0)
    
    @property
    def sgd_momentum(self) -> float:
        return self._config["training"]["sgd_momentum"]
    
    @property
    def sgd_nesterov(self) -> bool:
        return self._config["training"]["sgd_nesterov"]
    
    @property
    def sweep_id_quick(self) -> str:
        return self._config["training"]["sweep_id_quick"]
    
    @property
    def sweep_id_long(self) -> str:
        return self._config["training"]["sweep_id_long"]
    
    @property
    def base_run_dir(self) -> str:
        return self._config["training"]["base_run_dir"]
    
    # === Platform-Aware Properties ===
    @property
    def training_num_workers(self) -> int:
        """Get platform-appropriate number of workers."""
        platform_config = get_platform_config()
        data_config = platform_config.get_data_loader_config()
        
        # Check if user has overridden the setting
        force_workers = self._config.get("platform", {}).get("force_num_workers")
        if force_workers is not None:
            return force_workers
        
        return data_config["num_workers"]
    
    @property
    def training_pin_memory(self) -> bool:
        """Get platform-appropriate pin_memory setting."""
        platform_config = get_platform_config()
        data_config = platform_config.get_data_loader_config()
        
        # Check if user has overridden the setting
        force_pin_memory = self._config.get("platform", {}).get("force_pin_memory")
        if force_pin_memory is not None:
            return force_pin_memory
        
        return data_config["pin_memory"]
    
    # === Path Configuration ===
    def get_paths(self, project_root: Path) -> Dict[str, Path]:
        """Generate all required paths based on project root."""
        paths_config = self._config["dataset"]["paths"]
        return {
            "archive_dir": project_root / paths_config["archive_dir"],
            "images_root": project_root / paths_config["images_root"],
            "output_dir": project_root / paths_config["subsets_dir"] / self.dataset_name,
            "out_root": project_root / paths_config["processed_dir"] / self.processed_dataset_name,
        }
    
    # === Configuration Section Accessors ===
    def get_display_config(self) -> Dict[str, Any]:
        """Get configuration for display and logging."""
        return self._config["environment"].copy()
    
    def get_subset_config(self) -> Dict[str, Any]:
        """Get configuration for subset building."""
        subset_config = self._config["subset"].copy()
        subset_config.update({
            "attribute": self.attribute,
            "dataset_name": self.dataset_name,
        })
        return subset_config
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get configuration for preprocessing."""
        return self._config["preprocessing"].copy()
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get configuration for analysis."""
        return self._config["analysis"].copy()
    
    def get_training_config(self, quick: bool = True) -> Dict[str, Any]:
        """Get configuration for training.
        
        Args:
            quick: If True, use quick run settings (fewer epochs). If False, use longer run.
        """
        training_config = self._config["training"].copy()
        training_config["sweep_grid"] = self.matched_sweep_grid
        training_config["epochs"] = self.epochs_quick if quick else self.epochs_long
        training_config["sweep_id"] = self.sweep_id_quick if quick else self.sweep_id_long
        training_config["num_workers"] = self.training_num_workers
        training_config["pin_memory"] = self.training_pin_memory
        return training_config
    
    def get_dp_sgd_config(self, quick: bool = True) -> Dict[str, Any]:
        """Get configuration for DP-SGD training.
        
        Args:
            quick: If True, use quick run settings (fewer epochs). If False, use longer run.
        """
        dp_config = self._config["dp_sgd"].copy()
        # Use the same matched sweep grid from training section
        dp_config["sweep_grid"] = self.matched_sweep_grid
        dp_config["epochs"] = self.epochs_quick if quick else self.epochs_long
        dp_config["sweep_id"] = self._config["dp_sgd"]["sweep_id_quick"] if quick else self._config["dp_sgd"]["sweep_id_long"]
        dp_config["num_workers"] = self.training_num_workers
        dp_config["pin_memory"] = self.training_pin_memory
        return dp_config


# Global configuration instance
config = NotebookConfig()


def get_config(config_path: Optional[Path] = None) -> NotebookConfig:
    """Get the global configuration instance.
    
    Args:
        config_path: Optional path to custom config file
        
    Returns:
        NotebookConfig instance
    """
    global config
    if config_path is not None:
        config = NotebookConfig(config_path)
    return config


def reload_config() -> None:
    """Reload configuration from file."""
    global config
    config.reload()


def get_config_path() -> Path:
    """Get the path to the current configuration file."""
    return config.config_path
