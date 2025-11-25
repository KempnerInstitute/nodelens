# Configuration Files

## Structure

```
configs/
├── template.yaml              # Complete template with all options
├── examples/                  # Ready-to-use examples
│   ├── mnist_basic.yaml       # MNIST RQ analysis
│   ├── resnet_pruning.yaml    # ResNet pruning
│   ├── llama3_pruning.yaml    # Llama-3 pruning
│   ├── llm_alignment.yaml     # LLM supernode analysis
│   └── vision_comprehensive.yaml
└── projects/                  # Project configs
    ├── llm_supernode.yaml
    └── vision_synergy.yaml
```

## Usage

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
python scripts/run_experiment.py --config configs/examples/llama3_pruning.yaml
```

## Configuration Blocks

| Block | Purpose |
|-------|---------|
| `experiment` | Name, type (`alignment_analysis` or `llm_alignment`), seed, device |
| `model` | Architecture, pretrained, tracked_layers. For LLMs: model_id, torch_dtype |
| `dataset` | Dataset name, batch_size, data_path |
| `metrics` | `enabled`: list of metrics. `num_samples`: calibration samples. `composite_weights`: for composite scoring |
| `training` | `enabled`, epochs, learning_rate, optimizer |
| `supernode` | Detection settings: score_metric, core_fraction, protect_core |
| `pruning` | strategy, sparsity_levels, scoring, direction, structured |
| `llm` | LLM-specific: scar_metrics, evaluate_perplexity |
| `cnn` | CNN-specific: mode (unfold, patchwise) |
| `analysis` | save_scores, generate_plots, plots to enable |
| `visualization` | format, dpi |

## Metrics

Specify metrics to compute in `metrics.enabled`:

- `rayleigh_quotient` - Input-weight alignment
- `activation_l2_norm` - Activation magnitude
- `activation_outlier_index` - Outlier detection
- `pairwise_redundancy_gaussian` - Pairwise redundancy
- `synergy_gaussian_mmi` - Synergistic information
- `mutual_information_gaussian` - MI estimate

## Composite Scoring

Define weights in `metrics.composite_weights` for combining metrics:

```yaml
metrics:
  composite_weights:
    activation_l2_norm: 0.2
    rayleigh_quotient: 0.3
    pairwise_redundancy_gaussian: -0.2  # Negative penalizes redundancy
```

Used when `pruning.scoring: "composite"` or `supernode.score_metric: "composite"`.
