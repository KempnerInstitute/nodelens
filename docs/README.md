# Documentation

## Guides

- [Usage Guide](usage.md) - Running experiments and configuration
- [API Reference](api_reference.md) - Core classes and functions
- [LLM Guide](llm_guide.md) - LLM-specific analysis and pruning

## Configuration

- [Template](../configs/template.yaml) - Complete parameter reference
- [Examples](../configs/examples/) - Example configurations

## Quick Start

```bash
# Run experiment
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# Generate analysis
python scripts/run_analysis.py --results-dir ./results --output-dir ./plots --quick
```
