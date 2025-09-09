# Alignment Analysis Framework

A comprehensive framework for analyzing neural network alignment, pruning, and information-theoretic properties.

## Features

- Alignment Analysis: Measure how neural representations align with data and task structure
- Pruning Experiments: Test various pruning strategies and their effects on model performance  
- Multi-Network Analysis: Train and analyze multiple networks in parallel
- 30+ Metrics: Rayleigh quotient, mutual information, spectral metrics, and more
- Extensible Design: Easy to add custom metrics and strategies

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd alignment

# Create conda environment
conda env create -f environment.yml
conda activate networkAlignmentAnalysis

# Install in development mode
pip install -e .
```

## Quick Start

### Using Configuration Files (Recommended)
```bash
# Run ResNet-18 experiment on CIFAR-10
python scripts/run_experiment.py --config configs/examples/resnet18_analysis.yaml --device cuda

# Run comprehensive analysis
python scripts/run_experiment.py --config configs/examples/resnet50_analysis.yaml --device cuda
```

### Using Python API
```python
from alignment.configs.config_loader import load_config
from alignment.experiments import GeneralAlignmentExperiment

# Load configuration
config = load_config('configs/examples/resnet18_analysis.yaml')

# Run experiment
experiment = GeneralAlignmentExperiment(config)
results = experiment.run()
```

## Supported Models

### Vision Models (via torchvision/timm)
- ResNet (18, 34, 50, 101, 152)
- VGG (11, 13, 16, 19)
- AlexNet
- EfficientNet (B0-B7)
- Vision Transformers (ViT, DeiT)
- MobileNet, DenseNet

### Custom Models
- Multi-layer Perceptrons (MLP)
- Convolutional Neural Networks (CNN)
- Custom architectures via model registry

## Datasets

- MNIST, Fashion-MNIST
- CIFAR-10, CIFAR-100
- ImageNet
- Custom datasets via dataset registry

## Documentation

Build documentation locally:
```bash
cd docs
make html
```

View at: `docs/build/html/index.html`

## Project Structure

```
alignment/
├── src/alignment/        # Main package
│   ├── core/            # Core functionality and registry
│   ├── models/          # Model architectures and loaders
│   ├── metrics/         # Alignment metrics (30+ implementations)
│   ├── pruning/         # Pruning strategies and experiments
│   ├── experiments/     # Experiment framework
│   ├── data/            # Dataset handling
│   ├── analysis/        # Result analysis and visualization
│   └── configs/         # Configuration management
├── configs/             # Configuration templates and examples
├── examples/            # Example scripts
├── scripts/             # Experiment runner scripts
├── tests/               # Unit and integration tests
└── docs/                # Documentation source
```

## Usage Examples

### Basic Alignment Analysis
```bash
# Simple MLP on MNIST
python scripts/run_experiment.py --config configs/examples/mnist_mlp_standard.yaml

# Vision model analysis
python scripts/run_experiment.py --config configs/examples/resnet18_analysis.yaml
```

### Custom Configuration
1. Copy a template: `cp configs/template_basic.yaml configs/my_experiment.yaml`
2. Edit the configuration file
3. Run: `python scripts/run_experiment.py --config configs/my_experiment.yaml`

## Results and Outputs

Experiments generate:
- Training logs and metrics
- Alignment analysis results
- Pruning performance comparisons
- Professional visualizations (PNG plots)
- Comprehensive experiment reports

Results are saved in timestamped directories: `results/experiment_name_YYYYMMDD_HHMMSS/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

See `docs/source/contributing.rst` for detailed guidelines.

## License

See LICENSE file for details.