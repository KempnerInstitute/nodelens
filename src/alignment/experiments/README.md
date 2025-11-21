# Experiments Module

Framework for running structured neural network alignment experiments.

## Available Experiments

### GeneralAlignmentExperiment
Complete pipeline for alignment analysis including training, metric computation, and pruning.

```python
from alignment.experiments import GeneralAlignmentExperiment

# From YAML configuration
experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()

# Programmatic configuration
from alignment.experiments import GeneralAlignmentConfig
config = GeneralAlignmentConfig(
    dataset_name="mnist",
    model_name="mlp",
    alignment_metrics=["rayleigh_quotient"],
    pruning_strategy="magnitude"
)
experiment = GeneralAlignmentExperiment(config)
results = experiment.run()
```

### LLMAlignmentExperiment
Specialized pipeline for **large language models (LLMs)** using Hugging Face checkpoints.

- Supports HF causal LMs via the `hf_causal_lm` registry entry
- Computes activation- and information-based metrics (e.g. `activation_l2_norm`, `activation_outlier_index`, `rayleigh_quotient`, `pairwise_redundancy_gaussian`)
- Implements **supernode-aware pruning** for LLaMA-style FFNs (gate/up/down projections)
- Optionally computes **SCAR-style** metrics:
  - `scar_activation_power` (E\[u_i²\])
  - `scar_taylor` (first-order saliency)
  - `scar_curvature` (Rayleigh-style curvature)
  - `scar_loss_proxy` (second-order loss proxy used for pruning)

Example (via config and unified runner):

```bash
python scripts/run_experiment.py --config configs/projects/llm_supernode.yaml
```

This will:

- Load `meta-llama/Llama-3.1-8B` (or another HF model),
- Compute alignment and SCAR metrics on a calibration corpus,
- Perform structured FFN pruning using `scar_loss_proxy` while protecting a supernode core,
- Optionally evaluate perplexity before and after pruning.

### Other Experiments

The repository also contains more specialized pruning and analysis experiments, including:

- **Layer-isolated pruning** and **progressive pruning** (see `alignment/pruning/experiments/`)
- **Parallel pruning experiments** for comparing multiple strategies/modes

See the example scripts in `examples/` for end-to-end workflows such as:

- `06_redundancy_aware_pruning.py` – redundancy/synergy-aware pruning on vision models
- `08_llama_ffn_pruning.py` – FFN per-neuron analysis and structured pruning in LLaMA-like models

## Base Classes

All experiments inherit from `BaseExperiment` which provides:
- Configuration management
- Model wrapping for metrics
- Result tracking and checkpointing
- Device management

## Creating Custom Experiments

```python
from alignment.experiments import BaseExperiment, ExperimentConfig

class MyExperiment(BaseExperiment):
    def run(self) -> Dict[str, Any]:
        # Your experiment logic
        metrics = self.compute_metrics(data)
        results = {"metrics": metrics}
        self.save_results(results)
        return results
```

## Configuration

Use YAML files for reproducible experiments:

```yaml
name: "my_experiment"
model_name: "resnet18"
dataset_name: "cifar10"
metrics: ["rayleigh_quotient", "spectral_gap"]
device: "cuda"
```

## Integration

Works seamlessly with the analysis module for automatic result processing and reporting. 