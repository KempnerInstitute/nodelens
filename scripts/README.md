# Production Scripts

This directory contains production-ready scripts for running alignment experiments.

## Scripts

### `run_experiment.py`
The main experiment runner for the alignment framework. This is the primary tool for running research experiments with full configuration support.

**Usage:**
```bash
# From repository root
python scripts/run_experiment.py --config configs/unified_config.yaml
```

## Difference from Examples

- **`examples/`**: Educational scripts for learning the framework
- **`scripts/`**: Production tools for research experiments

The examples demonstrate concepts with hardcoded parameters for clarity, while the scripts here are fully configurable tools designed for actual research work. 