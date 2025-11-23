# Configuration System

This directory contains YAML configuration files for the CelebA Eyeglasses workflow notebook.

## Files

- `config.yaml` - Main configuration file with default settings
- `config_quick_experiment.yaml` - Example configuration for quick experiments

## Usage

### Default Configuration
```python
from src.config import get_config
config = get_config()
print(f"Training epochs: {config.training_epochs}")
```

### Custom Configuration
```python
from src.config import get_config
from pathlib import Path

# Load custom config
config = get_config(Path("notebooks/config_quick_experiment.yaml"))
print(f"Custom epochs: {config.training_epochs}")
```

### Reload Configuration
```python
from src.config import reload_config

# After modifying config.yaml, reload it
reload_config()
```

## Configuration Sections

### Environment
- `random_seed`: Random seed for reproducibility
- `figure_dpi`: Matplotlib figure DPI
- `matplotlib_style`: Primary matplotlib style
- `matplotlib_style_fallback`: Fallback matplotlib style

### Dataset
- `attribute`: CelebA attribute to classify
- `dataset_name`: Name for subset dataset
- `processed_dataset_name`: Name for processed dataset
- `paths`: Relative paths to data directories

### Subset
- `max_train_per_class`: Maximum training samples per class
- `max_val_per_class`: Maximum validation samples per class
- `max_test_per_class`: Maximum test samples per class
- `link_mode`: How to create subset ("copy" or "link")

### Preprocessing
- `size`: Target image size (width/height)
- `center_crop`: Whether to center crop before resize
- `normalize_01`: Whether to normalize to [0,1]
- `compute_stats`: Whether to compute channel statistics

### Analysis
- `plot_top_n_attrs`: Number of top attributes to plot
- `size_sample_max`: Maximum samples for size analysis
- `channel_stats_split`: Which split to use for channel stats
- `proc_pixel_hist_bins`: Number of bins for pixel histograms

### Diagnostics
- `target_size`: Target size for crop diagnostics
- `visual_sample`: Number of samples for visual diagnostics

### Training
- `sgd_sweep_grid`: List of hyperparameter combinations
- `epochs`: Number of training epochs
- `num_workers`: Number of data loader workers
- `aug_flip`: Enable horizontal flip augmentation
- `aug_jitter`: Enable color jitter augmentation
- `sgd_momentum`: SGD momentum parameter
- `sgd_nesterov`: Enable Nesterov momentum
- `sweep_id`: Identifier for hyperparameter sweep
- `base_run_dir`: Base directory for saving runs

### DP-SGD
- `max_grad_norm`: Gradient clipping norm
- `noise_multiplier`: Noise multiplier for privacy
- `target_delta`: Target delta for privacy accounting
- Additional training parameters for DP-SGD

## Creating Custom Configurations

1. Copy `config.yaml` to a new file (e.g., `config_my_experiment.yaml`)
2. Modify the values you want to change
3. Load the custom config in your notebook:
   ```python
   config = get_config(Path("notebooks/config_my_experiment.yaml"))
   ```

## Benefits

- **Easy Experimentation**: Change hyperparameters without touching code
- **Reproducibility**: Save exact configuration with each experiment
- **Version Control**: Track configuration changes in git
- **Documentation**: YAML comments explain each parameter
- **Flexibility**: Support for multiple experiment configurations
