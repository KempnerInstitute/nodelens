# Tests

This directory contains unit and integration tests for the alignment framework.

## Test Structure

### Unit Tests (`unit/`)
- `test_models.py` - Model architecture and loading tests
- `test_metrics.py` - Alignment metric computation tests
- `test_experiments.py` - Experiment configuration and execution tests
- `test_checkpoint.py` - Checkpointing and state management tests
- `metrics/` - Detailed tests for specific metric categories
  - `test_rayleigh_metrics.py` - Rayleigh quotient and related metrics
  - `test_information_metrics.py` - Information-theoretic metrics
  - `test_similarity_metrics.py` - Similarity and correlation metrics

### Integration Tests (`integration/`)
- `test_all_completed.py` - End-to-end workflow tests

## Running Tests

### Run All Tests
```bash
cd /path/to/alignment
python -m pytest tests/
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/

# Specific test file
python -m pytest tests/unit/test_models.py

# Specific test function
python -m pytest tests/unit/test_models.py::test_model_creation
```

### Test with Coverage
```bash
python -m pytest tests/ --cov=alignment --cov-report=html
```

## Test Configuration

Tests use `conftest.py` for shared fixtures and configuration. The test suite automatically:
- Sets up temporary directories for test outputs
- Configures logging for test runs
- Provides common test data and model fixtures
- Cleans up after test completion

## Requirements

Tests require the alignment package to be installed in development mode:
```bash
pip install -e .
```

Additional test dependencies are specified in `pyproject.toml`.