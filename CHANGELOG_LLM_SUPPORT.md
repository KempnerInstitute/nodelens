# Changelog: LLM Support Addition

## Summary

Added comprehensive Large Language Model (LLM) support to the alignment framework, enabling pruning and analysis of transformer-based models like LLaMA, Mistral, and GPT-2.

## Date

October 2025

## Changes

### New Modules

#### `alignment/experiments/llm_experiments.py`
- **`LLMAlignmentExperiment`**: Main experiment class for LLM alignment analysis
  - Loads HuggingFace causal LMs
  - Computes importance scores using alignment metrics
  - Supports structured pruning of MLP and attention layers
  - Evaluates perplexity on text datasets
  - Fully integrated with existing config system

#### `alignment/data/datasets/text_datasets.py`
- **`TextDataset`**: Generic text dataset wrapper
- **`WikiTextDataset`**: WikiText dataset for language modeling
- **`C4Dataset`**: Streaming C4 dataset support
- **`load_text_dataset()`**: Unified interface for loading text datasets

### Modified Modules

#### `alignment/experiments/__init__.py`
- Added `LLMAlignmentExperiment` to exports
- Registered `llm_alignment` experiment type

#### `alignment/data/datasets/__init__.py`
- Added text dataset exports
- Integrated with existing dataset registry

#### `alignment/models/hub.py` (existing)
- Already supported `HFCausalLM` for loading LLMs
- No changes needed - works out of the box

#### `alignment/models/wrappers_transformer.py` (existing)
- Already supported transformer wrappers
- No changes needed - works with LLMs

## Features Added

### 1. LLM Model Loading
```python
from alignment.models import HFCausalLM

model = HFCausalLM(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype="bfloat16",
    device_map="auto"
)
```

### 2. Neuron Importance Computation
```python
from alignment.experiments import LLMAlignmentExperiment

experiment = LLMAlignmentExperiment(config)
experiment.setup()
scores = experiment.compute_importance_scores()
# Returns: {layer_name: {metric_name: tensor}}
```

### 3. Structured Pruning
```python
masks = experiment.apply_pruning(
    sparsity=0.2,
    metric='rayleigh_quotient',
    mode='low'  # or 'high' for ablation studies
)
```

### 4. Perplexity Evaluation
```python
perplexity = experiment.evaluate_perplexity(
    dataset='wikitext',
    split='test',
    num_samples=100
)
```

### 5. Wildcard Layer Selection
```yaml
wrapper:
  tracked_layers:
    - "model.layers.*.mlp"      # All MLP layers
    - "model.layers.[0-15].self_attn"  # First 16 attention layers
```

## Configuration Example

```yaml
experiment:
  name: "llama3_alignment_analysis"
  type: "llm_alignment"

model:
  name: "hf_causal_lm"
  model_id: "meta-llama/Meta-Llama-3-8B-Instruct"
  torch_dtype: "bfloat16"
  device_map: "auto"

wrapper:
  name: "transformer_wrapper"
  tracked_layers:
    - "model.layers.*.mlp"

alignment:
  metrics: ["rayleigh_quotient", "mutual_information_gaussian"]

pruning:
  enabled: true
  sparsity_levels: [0.1, 0.2, 0.3]
  alignment_metric: "rayleigh_quotient"

evaluation:
  compute_perplexity: true
  dataset: "wikitext"
  num_samples: 100
```

## API Compatibility

- **Backward compatible**: Existing experiments continue to work
- **Consistent API**: LLM experiments use same interface as vision experiments
- **Registry integration**: LLM experiment registered as `"llm_alignment"`

## Testing

Tested with:
- ✅ LLaMA 3 8B (Instruct)
- ✅ Mistral 7B
- ✅ GPT-2
- ✅ Multiple alignment metrics (RQ, MI, cosine similarity)
- ✅ Structured pruning of MLP layers
- ✅ Perplexity evaluation on WikiText

## Dependencies

New optional dependencies for LLM support:
```txt
transformers>=4.40.0
accelerate>=0.30.0
datasets>=2.19.0
```

These are optional - alignment still works without them for non-LLM experiments.

## Usage Examples

### Basic Importance Analysis
```python
from alignment.experiments import LLMAlignmentExperiment

config = {
    'model': {'name': 'hf_causal_lm', 'model_id': 'meta-llama/Meta-Llama-3-8B-Instruct'},
    'wrapper': {'tracked_layers': ['model.layers.*.mlp']},
    'alignment': {'metrics': ['rayleigh_quotient']},
}

experiment = LLMAlignmentExperiment(config)
experiment.setup()
scores = experiment.compute_importance_scores()
```

### Pruning + Evaluation
```python
config['pruning'] = {
    'enabled': True,
    'sparsity_levels': [0.2],
    'alignment_metric': 'rayleigh_quotient'
}
config['evaluation'] = {
    'compute_perplexity': True,
    'dataset': 'wikitext'
}

experiment = LLMAlignmentExperiment(config)
experiment.setup()
results = experiment.run()

print(f"Baseline: {results['evaluation']['baseline_perplexity']}")
print(f"Pruned: {results['pruning_results']['sparsity_0.2']['perplexity']}")
```

## Integration with PruneLLM

The alignment framework's LLM support is now used by the PruneLLM project:
- PruneLLM scripts are simple wrappers around `LLMAlignmentExperiment`
- All core functionality lives in alignment codebase
- Clean separation: alignment (general infrastructure) + PruneLLM (project-specific analysis)

See: `PruneLLM/alignment-based-pruning/README.md`

## Future Work

Potential enhancements:
- [ ] Support for encoder-decoder models (T5, BART)
- [ ] Attention head pruning (in addition to neuron pruning)
- [ ] Knowledge distillation integration
- [ ] Quantization-aware pruning
- [ ] Multi-GPU parallelism for large models

## Authors

Alignment Framework Team

## Notes

- All metrics work with LLMs (RQ, MI, PID, etc.)
- All pruning strategies work with LLMs
- Existing experiment runner (`scripts/run_experiment.py`) supports LLM configs
- No breaking changes to existing code

