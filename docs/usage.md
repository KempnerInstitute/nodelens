# How to Run Experiments with YAML Configs

## Quick Start

### Step 1: Activate Environment

```bash
conda activate alignment
cd /path/to/alignment
```

### Step 2: Run Experiment

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```

---

## Available Example Configs

### 1. MNIST Basic Analysis

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```

Computes: RQ scores for simple MLP on MNIST

### 2. ResNet Pruning

```bash
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
```

Performs: Redundancy-aware pruning on ResNet-18 with CIFAR-10

### 3. LLaMA-3 Scoring

```bash
python scripts/run_experiment.py --config configs/examples/llama3_scoring.yaml
```

Computes: Per-neuron importance scores for LLaMA-3 FFN

### 4. LLaMA-3 Pruning

```bash
python scripts/run_experiment.py --config configs/examples/llama3_pruning.yaml
```

Performs: Redundancy-aware pruning of LLaMA-3 model

---

## Command-Line Overrides

Override any parameter:

```bash
python scripts/run_experiment.py \
  --config configs/examples/resnet_pruning.yaml \
  --device cuda:1 \
  --batch-size 64 \
  --target-sparsity 0.5
```

Common overrides:
- `--device cuda:0` - GPU selection
- `--batch-size 64` - Batch size
- `--target-sparsity 0.7` - Pruning amount
- `--epochs 50` - Training epochs
- `--output-dir ./my_results` - Output directory

---

## Creating Custom Configs

### Method 1: Copy and Modify Template

```bash
cp configs/template.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
python scripts/run_experiment.py --config configs/my_experiment.yaml
```

### Method 2: Copy Existing Example

```bash
cp configs/examples/resnet_pruning.yaml configs/my_resnet.yaml
# Modify my_resnet.yaml
python scripts/run_experiment.py --config configs/my_resnet.yaml
```

---

## Config Structure

All configs have the same structure:

```yaml
experiment:        # Experiment settings
  name: "..."
  seed: 42
  device: "cuda"

model:            # Model architecture
  name: "..."     # 'resnet18', 'mlp', 'meta-llama/...'
  pretrained: true

dataset:          # Data source
  name: "..."     # 'mnist', 'cifar10', 'wikitext'
  batch_size: 128

metrics:          # Metrics to compute
  enabled: ['rayleigh_quotient']

pruning:          # Pruning settings (optional)
  enabled: true
  strategy: '...'
  target_sparsity: 0.7

# ... other sections as needed
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

