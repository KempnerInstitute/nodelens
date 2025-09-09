# Configuration Files

This directory contains configuration templates and examples for alignment experiments.

## Templates

### Core Templates
- `template_comprehensive.yaml` - Complete reference with ALL available parameters
- `template_basic.yaml` - Basic configuration for simple experiments  
- `template_minimal.yaml` - Minimal configuration with essential parameters only

### Usage
Copy a template and modify for your needs:
```bash
cp configs/template_basic.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
python scripts/run_experiment.py --config configs/my_experiment.yaml
```

## Examples

The `examples/` subdirectory contains ready-to-run configurations for specific use cases:

### Vision Models
- `resnet18_analysis.yaml` - ResNet-18 on CIFAR-10 (lightweight, fast)
- `resnet50_analysis.yaml` - ResNet-50 on CIFAR-10 (standard benchmark)
- `alexnet_analysis.yaml` - AlexNet on CIFAR-10 (classic architecture)
- `vgg16_analysis.yaml` - VGG-16 on CIFAR-10 (deep convolutional)
- `efficientnet_b0_analysis.yaml` - EfficientNet-B0 (modern efficient)
- `vit_b16_analysis.yaml` - Vision Transformer (attention-based)

### Master Reference
- `vision_networks_master.yaml` - Complete example showing how to configure all supported models
- `master_config.yaml` - Comprehensive configuration with all options documented

### Simple Models
- `mnist_mlp_standard.yaml` - Basic MLP on MNIST for testing

## Quick Start

### For Vision Models
```bash
# Fast experiment with ResNet-18
python scripts/run_experiment.py --config configs/examples/resnet18_analysis.yaml --device cuda

# Comprehensive analysis with ResNet-50  
python scripts/run_experiment.py --config configs/examples/resnet50_analysis.yaml --device cuda
```

### For Custom Experiments
1. Start with a template: `cp configs/template_basic.yaml configs/my_config.yaml`
2. Modify the model, dataset, and experiment parameters
3. Run: `python scripts/run_experiment.py --config configs/my_config.yaml`

## Configuration Structure

All configurations support:
- **Models**: MLP, CNN, ResNet, VGG, EfficientNet, Vision Transformers
- **Datasets**: MNIST, CIFAR-10/100, ImageNet
- **Metrics**: 30+ alignment and similarity metrics
- **Pruning**: Magnitude, alignment-based, random, hybrid strategies
- **Visualization**: Professional plots and analysis reports