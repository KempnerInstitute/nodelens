# Alignment Metrics Framework - Refactored

A comprehensive, modular framework for computing and analyzing neural network alignment metrics. This refactored version provides a clean, extensible architecture optimized for multi-GPU HPC environments.

## 🎯 Overview

This framework provides tools for measuring how neural network representations align with their inputs and weights through various information-theoretic and geometric metrics. The refactored architecture emphasizes:

- **Modularity**: Clear separation of concerns with protocol-based interfaces
- **Performance**: Optimized for large-scale models with automatic memory management
- **Extensibility**: Easy to add new metrics, models, and experiments
- **Distributed**: Built-in support for multi-GPU training

## 📁 Architecture

```
src/alignment_refactor/
├── core/               # Core protocols, registry, and base classes
├── metrics/            # Alignment metrics organized by computational method
│   ├── rayleigh/      # Rayleigh quotient-based metrics
│   ├── information/   # Information-theoretic metrics
│   └── similarity/    # Similarity-based metrics
├── models/            # Model wrappers and activation tracking
├── data/              # Dataset wrappers and loaders
├── experiments/       # Experiment runners and configurations
├── analysis/          # Result aggregation and visualization
└── utils/             # Distributed computing, logging, checkpointing
```

## 🚀 Key Features

### 1. **Protocol-Based Design**
All components implement well-defined protocols, making it easy to extend:

```python
from alignment_refactor.core.protocols import AlignmentMetric

class MyMetric(AlignmentMetric):
    def compute(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Your implementation
        pass
```

### 2. **Registry System**
Automatic discovery and instantiation of components:

```python
from alignment_refactor.core.registry import register_metric, get_metric

@register_metric("my_metric")
class MyMetric(AlignmentMetric):
    pass

# Later...
metric = get_metric("my_metric")()
```

### 3. **Memory-Aware Computation**
Automatic CPU offloading for large operations:

```python
metric = RayleighQuotient(
    force_cpu_for_large_ops=True,
    cpu_threshold=1e7  # Offload if tensor > 10M elements
)
```

### 4. **Distributed Support**
Built-in distributed computing with automatic reduction:

```python
# Metrics automatically handle distributed reduction
scores = metric.compute_distributed(
    inputs=local_inputs,
    weights=weights,
    world_size=4,
    rank=rank
)
```

### 5. **Flexible Model Wrapping**
Track activations with automatic layer discovery:

```python
from alignment_refactor.models import ModelWrapper

# Automatically discovers trackable layers
wrapper = ModelWrapper(model)

# Or specify layers
wrapper = ModelWrapper(model, tracked_layers=['layer1', 'layer2'])

# Get activations
outputs, activations = wrapper.forward_with_activations(inputs)
```

## 📊 Implemented Metrics

### Rayleigh Quotient-Based (`metrics/rayleigh/`)
- **RayleighQuotient**: Standard and relative RQ computation
- **PatchWiseRayleighQuotient**: For convolutional layers
- **DeltaAlignment**: RQ on weight changes
- **NormalizedDeltaAlignment**: Scale-invariant version

### Information-Theoretic (`metrics/information/`)
- **MutualInformationGaussian**: MI with Gaussian approximation
- **MutualInformationBinning**: MI using histogram binning
- **ConditionalMutualInformation**: CMI implementation
- **SharedInformation** (PID): Redundant information between inputs
- **UniqueInformationX/Y** (PID): Unique information from each input
- **SynergisticInformation** (PID): Emergent information from both inputs
- **AverageRedundancy**: Redundancy between neurons
- **NodeRedundancy**: Input feature redundancy
- **LayerRedundancy**: Overall layer redundancy

### Similarity-Based (`metrics/similarity/`)
- **WeightCosineSimilarity**: Cosine similarity between weight vectors
- **ActivationCosineSimilarity**: Similarity between activation patterns
- **WeightActivationAlignment**: Alignment with activation PCs

## 🧪 Running Experiments

### Single Experiment
```python
from alignment_refactor.experiments import ProgressiveDropoutExperiment, ExperimentConfig

config = ExperimentConfig(
    name="dropout_analysis",
    model_name="resnet18",
    dataset_name="cifar10",
    metrics=["rayleigh_quotient", "mutual_information_gaussian", "pid_shared"],
    dropout_rates=[0.0, 0.2, 0.4, 0.6, 0.8]
)

experiment = ProgressiveDropoutExperiment(config)
results = experiment.run()
```

### Grid Search
```python
from alignment_refactor.experiments.runner import ExperimentRunner

runner = ExperimentRunner(base_config=config)
runner.add_grid_search(
    "progressive_dropout",
    param_grid={
        'dropout_structure': ['random', 'magnitude'],
        'batch_size': [64, 128, 256]
    }
)
all_results = runner.run_all()
```

## 📈 Analysis and Visualization

### Result Aggregation
```python
from alignment_refactor.analysis import ResultAggregator

aggregator = ResultAggregator()
aggregator.load_from_directory("./results")
df = aggregator.to_dataframe()

# Get statistics
stats = aggregator.compute_statistics("rayleigh_quotient", "layer1")
```

### Visualization
```python
from alignment_refactor.analysis import MetricVisualizer, LayerVisualizer

# Plot metric evolution
visualizer = MetricVisualizer()
fig = visualizer.plot_metric_evolution(
    steps, values,
    title="RQ Evolution",
    save_path="rq_evolution.png"
)

# Layer comparison
layer_viz = LayerVisualizer()
fig = layer_viz.plot_layer_comparison(
    layer_metrics,
    title="Layer-wise Alignment"
)
```

### Report Generation
```python
from alignment_refactor.analysis import HTMLReporter

reporter = HTMLReporter("Experiment Analysis")
reporter.add_dataframe("Results", df)
reporter.add_figure("evolution.png", "Metric Evolution")
reporter.generate("report.html")
```

## 🛠️ Utilities

### Distributed Training
```python
from alignment_refactor.utils import setup_distributed, is_main_process

# Setup
setup_distributed(backend="nccl")

# Use throughout code
if is_main_process():
    # Save checkpoints, log, etc.
    pass
```

### Checkpoint Management
```python
from alignment_refactor.utils import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    max_checkpoints=5,
    metric_name="val_loss",
    mode="min"
)

# Save
manager.save(model.state_dict(), step=1000, metrics={"val_loss": 0.5})

# Load best
checkpoint = manager.load_best()
```

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd alignment

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## 📚 Examples

See the `examples/` directory for complete examples:
- `example_usage.py`: Basic metric computation
- `experiment_example.py`: Running experiments
- `analysis_example.py`: Analyzing results

## 🤝 Contributing

Contributions are welcome! The modular architecture makes it easy to add:
- New metrics (inherit from `AlignmentMetric`)
- New models (inherit from `BaseModelWrapper`)
- New experiments (inherit from `BaseExperiment`)
- New datasets (inherit from `BaseDataset`)

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

This refactored version builds upon the original alignment metrics codebase, reorganizing it for better modularity, performance, and extensibility. 