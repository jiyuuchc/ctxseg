# Copilot Instructions for ctxseg

## Project Overview
- **ctxseg** is a modular Python project for segmentation and modeling, organized by domain-specific submodules.
- Main package: `ctxseg/` with subfolders for modeling, ops, sampling, segmentation, training, and utils.
- Data, experiments, and logs are managed in `run/`.

## Architecture & Key Components
- **Modeling**: `ctxseg/modeling/` contains model definitions (e.g., `convnext.py`, `ctxseg.py`, `diffusion.py`).
- **Segmentation**: `ctxseg/segmentation/` provides segmentation logic, metrics, and utilities.
- **Training**: `ctxseg/training/` includes training routines and preconditioning logic.
- **Ops**: `ctxseg/ops/` implements custom operations (e.g., `ndimage.py`).
- **Sampling**: `ctxseg/sampling/` contains samplers for data or model outputs.
- **Utils**: `ctxseg/utils/` offers plotting and general utilities.
- **Experimentation**: `run/` holds scripts, notebooks, logs, and results. Notebooks are used for prototyping and analysis.

## Developer Workflows
- **No explicit build system**: Code is run directly as scripts or imported as a package.
- **Testing**: Test scripts are in `run/` (e.g., `test_biopb.py`, `test_d.py`, `test_p.py`). Run with `python run/test_biopb.py`.
- **Notebooks**: Jupyter notebooks in `run/` for training, evaluation, and data distillation. Use `jupyter lab` or `jupyter notebook` to launch.
- **Logging**: Logs are written to files in `run/` (e.g., `*.log`).

## Project-Specific Patterns
- **Modular imports**: Use relative imports within `ctxseg/` (e.g., `from .utils import plotting`).
- **Experiment isolation**: Each experiment or evaluation is kept in its own subfolder or notebook under `run/`.
- **Pickle files**: Intermediate results and models are serialized as `.pickle` in `run/`.
- **Metrics and reports**: CSV and PKL files in `run/eval_*` and `run/edm_noctx/` for evaluation results.

## Integration Points
- **External dependencies**: Managed via `pyproject.toml`. Install with `pip install -e .` from the repo root.
- **No explicit API boundary**: Modules interact via direct imports; no REST or RPC.
- **Data flow**: Data and results move between scripts, notebooks, and serialized files in `run/`.

## Examples
- To run a test: `python run/test_biopb.py`
- To train a model: open and run `run/train.ipynb` in Jupyter
- To analyze results: inspect CSV/PKL files in `run/eval_cp/` or `run/edm_noctx/`

## Conventions
- Keep new experiments in separate subfolders or notebooks under `run/`
- Use relative imports for intra-package code
- Serialize intermediate results as `.pickle` in `run/`
- Log outputs to `run/*.log`

---
_Review and update these instructions as the project evolves. For questions, see `README.md` or ask maintainers._
