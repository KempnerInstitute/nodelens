# LLM Analysis Guide

NodeLens can analyze Hugging Face causal language models at the channel level.
The LLM workflow is designed for activation and gradient capture, FFN channel
metrics, ablation probes, and structured pruning.

## Quick Start

```bash
python scripts/run_experiment.py --config configs/examples/gpt2_fast_test.yaml
python scripts/run_experiment.py --config configs/prune_llm/llama3_8b_unified.yaml
```

The GPT-2 config is a small smoke test. The Llama, Mistral, Qwen, and OLMo
configs under `configs/prune_llm/` are larger workflows and may require model
access, GPU memory planning, and local cache setup.

## What The LLM Workflow Computes

- FFN activation statistics, including activation magnitude and outlier scores.
- Gradient-informed scores such as Taylor, curvature, and SCAR loss proxy.
- Supernode-style protected cores when a config asks for top-scoring channels.
- Halo and cross-layer diagnostics when enabled.
- Structured FFN channel pruning and perplexity evaluation.

## Example Config Structure

```yaml
experiment:
  name: "llama3_8b_analysis"
  type: "llm_alignment"
  device: "cuda"

model:
  name: "hf_causal_lm"
  model_id: "meta-llama/Llama-3.1-8B"
  dtype: "bfloat16"
  device_map: "auto"
  tracked_layers:
    - "model.model.layers.*.mlp.up_proj"
    - "model.model.layers.*.mlp.gate_proj"
    - "model.model.layers.*.mlp.down_proj"

dataset:
  name: "wikitext"
  subset: "wikitext-2-raw-v1"
  split: "train"
  batch_size: 1

calibration:
  num_samples: 128
  max_length: 2048
  batch_size: 4

metrics:
  scar:
    enabled: true
    num_samples: 64
    max_length: 512

supernode:
  enabled: true
  score_metric: "scar_loss_proxy"
  core_fraction: 0.01
  halo_fraction: 0.10
  protect_core: true

pruning:
  enabled: true
  ratios: [0.1, 0.3, 0.5]
  structured: true
  dependency_aware: true
  algorithms:
    - "magnitude"
    - "wanda"
    - "sparsegpt"
    - "scar_loss_proxy"
    - "supernode_protection_score"
```

## Metric Families

| Family | Examples |
|--------|----------|
| Activation | `activation_l2_norm`, `activation_variance`, `activation_outlier_index` |
| SCAR | `scar_activation_power`, `scar_taylor`, `scar_curvature`, `scar_loss_proxy` |
| Alignment | `rayleigh_quotient`, `delta_alignment` |
| Information | `mutual_information_gaussian`, `average_redundancy`, `pairwise_redundancy_gaussian` |
| Baselines | `magnitude`, `weight_magnitude`, `wanda`, `sparsegpt` |

## Structured FFN Pruning

For Llama-style FFNs, structured channel pruning masks the corresponding
intermediate channel across `gate_proj`, `up_proj`, and `down_proj`. This asks a
channel-level question: which full FFN units can be removed while preserving
model quality?

Unstructured weight pruning is a different setting. It can be useful as a
compression baseline, but it should be labeled separately from structured
channel pruning.

## Supernode And Halo Diagnostics

When `supernode.enabled` is true, NodeLens ranks channels by the configured
`score_metric` and marks the top `core_fraction` as a protected or analyzed
core. The same outputs can be used for ablation, pruning protection, or overlap
analysis with activation-defined outliers.

Halo diagnostics are optional. They measure local write-overlap and redundancy
around the high-scoring core and are useful when the question is whether
neighboring non-core channels behave differently from other channels.

## Memory Notes

- Use `batch_size: 1` for very large models.
- Use `device_map: "auto"` when model parallelism is available.
- Use `torch_dtype: "bfloat16"` or `"float16"` when supported.
- Reduce `calibration.num_samples`, `calibration.max_length`, or SCAR sample
  counts for smoke tests.

## Outputs

LLM runs usually write:

- per-layer metric arrays and score summaries
- pruning and ablation results
- perplexity or downstream-task evaluations
- plots, tables, and JSON summaries when enabled

Use the copied `experiment_config.yaml` in each output directory to audit the
exact settings for a run.
