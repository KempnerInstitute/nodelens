# Usage Guide

## Running Experiments

The framework uses YAML configuration files to specify experiments. This approach allows reproducible experiments and easy parameter management.

### Basic Usage

```bash
conda activate alignment
cd /path/to/alignment
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```

### Example Configurations

The `configs/examples/` directory contains pre-configured experiments:

**MNIST Basic Analysis**
```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```
Trains an MLP on MNIST and computes alignment scores.

**ResNet Pruning**
```bash
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
```
Applies pruning to ResNet-18 on CIFAR-10 using alignment-based importance scores.

**LLaMA-3 Scoring**
```bash
python scripts/run_experiment.py --config configs/examples/llama3_scoring.yaml
```
Computes per-neuron importance scores for LLaMA model feed-forward layers.

**LLaMA-3 Pruning**
```bash
python scripts/run_experiment.py --config configs/examples/llama3_pruning.yaml
```
Prunes LLaMA model using information-theoretic importance scores.

**LLaMA-3 Supernode Pruning (SCAR-style)**
```bash
python scripts/run_experiment.py --config configs/projects/llm_supernode.yaml
```
Runs the `LLMAlignmentExperiment` on a Hugging Face LLaMA-3.1 model, computes activation, redundancy,
and SCAR-style supernode metrics (activation power, first-order saliency, curvature, loss proxy),
and performs structured FFN pruning using the `scar_loss_proxy` metric while protecting a supernode core.

**Vision Synergy / Redundancy Analysis (ResNet-18)**
```bash
python scripts/run_experiment.py --config configs/projects/vision_synergy.yaml
```
Analyzes alignment, Gaussian PID synergy, and pairwise redundancy in a pretrained ResNet-18 on CIFAR-10,
then performs redundancy- and synergy-aware pruning using a composite alignment score.

## Command-Line Overrides

Override configuration parameters from the command line:

```bash
python scripts/run_experiment.py \
  --config configs/examples/resnet_pruning.yaml \
  --device cuda:1 \
  --batch-size 64 \
  --target-sparsity 0.5
```

Common override options:
- `--device cuda:0` - Select GPU device
- `--batch-size 64` - Set batch size
- `--target-sparsity 0.7` - Set pruning target
- `--epochs 50` - Set training epochs
- `--output-dir ./results` - Set output directory

## Creating Custom Configurations

### From Template

Copy the template and modify for your needs:

```bash
cp configs/template.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with desired parameters
python scripts/run_experiment.py --config configs/my_experiment.yaml
```

### From Existing Example

Start with an example configuration:

```bash
cp configs/examples/resnet_pruning.yaml configs/my_resnet.yaml
# Modify specific parameters
python scripts/run_experiment.py --config configs/my_resnet.yaml
```

## Configuration Structure

All configuration files follow the same structure:

```yaml
```yaml
experiment:
  name: "my_experiment"
  device: "cuda"
  
model:
  name: "resnet18"
  pretrained: true
  
dataset:
  name: "cifar10"
  batch_size: 128

metrics:
  enabled: ['rayleigh_quotient']

pruning:
  enabled: false
```

See `configs/template.yaml` for all available parameters.

## Experiment Types

### Computing Metrics

Compute alignment and information-theoretic scores:

```yaml
metrics:
  enabled: ['rayleigh_quotient', 'pairwise_redundancy_gaussian', 'synergy_gaussian_mmi']
  
  rayleigh_quotient:
    relative: true
    regularization: 1.0e-6
  
  pairwise_redundancy_gaussian:
    mode: 'output_based'
    num_pairs: 10

training:
  enabled: false
pruning:
  enabled: false
```

### Training Networks

Train from scratch with optional metric tracking:

```yaml
training:
  enabled: true
  epochs: 100
  learning_rate: 0.001
  optimizer: 'adam'
  compute_metrics_during_training: false
```

### Pruning Networks

Apply pruning with specified strategy:

```yaml
pruning:
  enabled: true
  strategy: 'composite'
  target_sparsity: 0.7
  distribution: 'adaptive_sensitivity'
  scoring: 'rayleigh_quotient'
  structured: true
  
  fine_tune:
    enabled: true
    epochs: 20
    learning_rate: 0.0001
```

### Multi-Level Pruning

Test multiple sparsity levels:

```yaml
pruning:
  enabled: true
  sparsity_levels: [0.3, 0.5, 0.7, 0.9]
  strategy: 'magnitude'
```

## Output Structure

Results are saved to the specified output directory:

```
results/[experiment_name]/
├── config.yaml           # Configuration used
├── results.json          # Numerical results
├── scores/               # Per-layer importance scores
├── plots/                # Visualizations
└── checkpoints/          # Model checkpoints
```

## Workflow

1. Create or select configuration file
2. Activate environment: `conda activate alignment`
3. Run experiment: `python scripts/run_experiment.py --config [path]`
4. Results saved to output directory
5. Analyze results and visualizations
```

See `configs/template.yaml` for complete parameter reference.

---

## Output

Results saved to experiment output directory:

```
results/
└── [experiment_name]/
    ├── config.yaml (saved configuration)
    ├── results.json (numerical results)
    ├── scores/ (per-layer scores)
    ├── plots/ (visualizations)
    └── checkpoints/ (model checkpoints)
```

---

## Examples for Different Tasks

### Compute Metrics Only

```yaml
metrics:
  enabled: ['rayleigh_quotient', 'pairwise_redundancy_gaussian']
training:
  enabled: false
pruning:
  enabled: false
```

### Training with Metric Tracking

```yaml
training:
  enabled: true
  epochs: 50
  compute_metrics_during_training: true
  metric_frequency: 100
```

### Pruning Experiments

```yaml
pruning:
  enabled: true
  strategy: 'ultimate'  # or 'magnitude', 'composite', etc.
  target_sparsity: 0.7
  distribution: 'adaptive_sensitivity'
  scoring: 'composite'
```

### Multi-Sparsity Comparison

```yaml
pruning:
  enabled: true
  sparsity_levels: [0.3, 0.5, 0.7, 0.9]
  # Automatically tests all levels
```

---

## Workflow

1. Choose or create config file
2. Activate environment: `conda activate alignment`
3. Run: `python scripts/run_experiment.py --config [path]`
4. Results saved to `results/[experiment_name]/`
5. View plots in `results/[experiment_name]/plots/`

---

This single script handles ALL experiment types through YAML configuration.

