# Experiments Module

Structured experiment framework for alignment analysis.

## Available Experiments

### GeneralAlignmentExperiment

Vision and general model analysis with training, metrics, and pruning.

```python
from alignment.experiments import GeneralAlignmentExperiment

experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()
```

### LLMAlignmentExperiment

LLM analysis with SCAR metrics, supernode detection, and structured pruning.

```python
from alignment.experiments import LLMAlignmentExperiment

experiment = LLMAlignmentExperiment(config)
experiment.setup()
scores = experiment.compute_importance_scores()
```

## Running Experiments

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
python scripts/run_experiment.py --config configs/examples/llm_alignment.yaml
```

## Base Classes

All experiments inherit from `BaseExperiment` which provides:
- Configuration management
- Model wrapping
- Result tracking
- Device management
