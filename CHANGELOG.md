# Changelog

## Unreleased
- Refactor: Moved reusable workflow functions from `notebook_as_script.py` into `src/celeba_workflow.py` (`review_archive`, `analyze_original_subset`, `analyze_processed`, `compute_channel_stats_for_paths`).
- Notebook: Added `notebooks/celeba_eyeglasses_workflow.ipynb` — a classroom‑ready, top‑to‑bottom workflow using `src/*` helpers.
- Legacy: Relocated `celeba_workflow_notebook.ipynb` to `notebooks/_legacy/celeba_workflow_notebook_legacy.ipynb` (preserved unmodified).
- Docs: Added `docs/notebooks/celeba_eyeglasses_workflow.md` and README link to the classroom notebook.

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
[09/27/2025]
- Data processing scripts: `scripts/celeba_preprocess.py`, `scripts/celeba_build_subset.py`, `scripts/celeba_centering.py`, `scripts/celeba_analyze.py`.
- Test suite for data pipeline: `tests/test_celeba_preprocess.py`, `tests/test_celeba_build_subset.py`, `tests/test_celeba_centering.py`, `tests/test_celeba_analyze.py`, and `tests/conftest.py`.
- Data documentation (moved and expanded): `docs/data_docs/preprocessing.md`, `docs/data_docs/subset_strategy.md`.
- Conda environment file at `envs/environment.yml` (Python 3.12; CPU-friendly defaults).

[09/28/2025]
- src scripts: `src/data.py`, `src/model.py`, `src/train_baseline.py`, `src/utils_py`, `src/sweeps.py`, `src/train_dp.py`
- Test suite for data pipeline: `tests/test_data.py`, `tests/test_train_baseline.py`, `test/test_utils.py`, `test/test_train_dp.py`
- Unit test documentation: `docs/test_docs/test_utils.md`, `docs/test_docs/test_data.md`, `docs/test_docs/test_train_dp.md`, `docs/test_docs/test_train_dp.md`

[09/30/2025]


### Changed
[09/27/2025]
- Update `.gitignore` to ignore `data/celeba/`, Python caches, and notebook checkpoints.

[09/28/2025]
- - docstrings for `scripts/celeba_analyze.py`, `scripts/celeba_build_subset.py`, `scripts/celeba_centering.py`, `scripts/celeba_preprocesing.py`
- docstrings for `src/config.py`, `src/data.py`, `src/model.py`, `src/sweeps.py`, `src/train_baseline.py`

[10/01/2025]
- Refactor workflow and enhance documentation: Moved reusable functions to `src/celeba_workflow.py`, added classroom notebook `notebooks/celeba_eyeglasses_workflow.ipynb`, and created accompanying documentation. Relocated legacy notebook and updated README 

### Deprecated

### Removed


### Fixed

### Security


## [0.1.0] - 2025-09-28

### Added
- Initial project scaffolding.




