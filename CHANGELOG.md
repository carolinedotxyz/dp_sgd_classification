# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

<!--
How to use this file (developer guidance):
- For every user-facing change, add an entry under Unreleased using the right category.
- Prefer one entry per pull request; keep entries short and imperative (e.g., "Add X").
- When creating a release, move Unreleased entries under a new version heading
  (e.g., [1.0.0] - YYYY-MM-DD), then reset Unreleased with empty categories.
-->

## [Unreleased]

### Added
(09/28/2025)
- Data processing scripts: `scripts/celeba_preprocess.py`, `scripts/celeba_build_subset.py`, `scripts/celeba_centering.py`, `scripts/celeba_analyze.py`.
- Test suite for data pipeline: `tests/test_celeba_preprocess.py`, `tests/test_celeba_build_subset.py`, `tests/test_celeba_centering.py`, `tests/test_celeba_analyze.py`, and `tests/conftest.py`.
- Data documentation (moved and expanded): `docs/data_docs/preprocessing.md`, `docs/data_docs/subset_strategy.md`.
- Conda environment file at `envs/environment.yml` (Python 3.12; CPU-friendly defaults).
- src scripts: `src/data.py`, `src/model.py`, `src/data.py`, `src/train_baseline.py`, `src/utils_py`, `src/sweeps.py`
- Test suite for data pipeline: `tests/test_data.py`, `tests/test_train_baseline.py`
- docstrings for `scripts/celeba_analyze.py`, `scripts/celeba_build_subset.py`, `scripts/celeba_centering.py`, `scripts/celeba_preprocesing.py`
- docstrings for `src/config.py`, `src/data.py`, `src/model.py`, `src/sweeps.py`, `src/train_baseline.py`

(09/29/2025)

### Changed
(09/28/2025)
- Update `.gitignore` to ignore `data/celeba/`, Python caches, and notebook checkpoints.

(09/29/2025)

### Deprecated

### Removed


### Fixed

### Security


## [0.1.0] - 2025-09-28

### Added
- Initial project scaffolding.




