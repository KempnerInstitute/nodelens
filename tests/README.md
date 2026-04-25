# Tests

Unit and integration tests.

## Running Tests

```bash
pytest tests/
pytest tests/unit/ -v
pytest tests/unit/test_models.py
pytest tests/ --cov=nodelens
```

## Structure

```
tests/
├── unit/
|   ├── test_models.py
|   ├── test_metrics.py
|   ├── test_experiments.py
|   └── metrics/
└── integration/
```
