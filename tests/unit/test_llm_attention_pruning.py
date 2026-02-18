"""
Unit tests for LLM attention/MLP structured pruning.

These tests exercise:
- Head-level attention pruning with shared Q/K/V/O masks.
- MLP pruning that keeps gate/up/down projections in sync.
"""

from typing import Dict

import pytest
import torch
import torch.nn as nn

# Skip entire module if transformers not installed
pytest.importorskip("transformers")

from alignment.experiments.base import ExperimentConfig
from alignment.experiments.llm_experiments import LLMAlignmentExperiment


class _TinySelfAttention(nn.Module):
    """Minimal LLaMA-style self-attention block for testing."""

    def __init__(self, embed_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Not a real attention implementation; sufficient for shape checks.
        self.q_proj(x)
        self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(v)


class _TinyMLP(nn.Module):
    """Minimal LLaMA-style MLP block for testing."""

    def __init__(self, hidden_dim: int = 8, intermediate_dim: int = 12):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gate_proj(x))
        h = self.down_proj(h)
        return h


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TinySelfAttention(embed_dim=16, num_heads=4)
        self.mlp = _TinyMLP(hidden_dim=16, intermediate_dim=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x)
        x = self.mlp(x)
        return x


class _TinyTransformer(nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_TinyBlock() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _TinyRoot(nn.Module):
    """
    Root module that mimics HF causal LM structure:
    - Top-level module has a single submodule called 'model'
    - Attention/MLP blocks live under 'model.layers.*'
    """

    def __init__(self, num_layers: int = 1):
        super().__init__()
        self.model = _TinyTransformer(num_layers=num_layers)

    @property
    def layers(self):
        # Convenience so tests can still access .layers[0].mlp
        return self.model.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _Wrapper:
    """
    Minimal wrapper exposing `_model` and `_tracked_layers` to satisfy
    LLMAlignmentExperiment expectations.
    """

    def __init__(self, model: nn.Module, tracked_layers):
        self._model = model
        self._tracked_layers = tracked_layers

    def forward_with_activations(self, inputs: Dict[str, torch.Tensor]):
        x = inputs["input_ids"].float()
        activations = {}
        for name, module in self._model.named_modules():
            if name in self._tracked_layers:

                def hook(mod, inp, out, key=name):
                    activations[f"{key}_output"] = out.detach()

                handle = module.register_forward_hook(hook)
                x = module(x) if hasattr(module, "forward") else x
                handle.remove()
        return x, activations


@pytest.fixture
def tiny_llm_experiment(monkeypatch):
    """Create an LLMAlignmentExperiment with a tiny transformer backend."""
    # Avoid initializing full metric stack for this tiny synthetic test.
    from alignment.experiments.base import BaseExperiment

    monkeypatch.setattr(BaseExperiment, "_initialize_components", lambda self: None)
    monkeypatch.setattr(BaseExperiment, "_setup_directories", lambda self: None)

    model = _TinyRoot(num_layers=1)
    tracked_layers = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
    ]

    config = ExperimentConfig(
        name="tiny_llm_test",
        experiment_type="llm_alignment",
        model_name="hf_causal_lm",
        model_config={"model_id": "dummy", "model_backend": "hf"},
        dataset_name="wikitext",
        batch_size=1,
        device="cpu",
        alignment_methods=["activation_l2_norm"],
        alignment_data_num_samples=1,
    )

    # Inject our tiny model directly so BaseExperiment skips HF loading.
    config.model = model

    exp = LLMAlignmentExperiment(config)
    exp.wrapped_model = _Wrapper(model, tracked_layers)
    exp.tokenizer = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 0})

    # Bypass dataset loading and importance computation: directly inject scores.
    scores = torch.linspace(0.0, 1.0, steps=16)
    exp.importance_scores = {
        "model.layers.0.self_attn.q_proj": {"activation_l2_norm": scores},
        "model.layers.0.mlp.gate_proj": {"activation_l2_norm": scores},
    }

    return exp


def test_attention_pruning_shared_masks(tiny_llm_experiment):
    """Attention pruning should apply the same neuron mask to Q/K/V and O projections."""
    exp = tiny_llm_experiment

    # Run pruning at a moderate sparsity
    masks = exp.apply_pruning(sparsity=0.5, metric="activation_l2_norm", mode="low")

    # All attention projections should share the same mask shape and content.
    q_mask = masks["model.layers.0.self_attn.q_proj"]
    k_mask = masks["model.layers.0.self_attn.k_proj"]
    v_mask = masks["model.layers.0.self_attn.v_proj"]
    o_mask = masks["model.layers.0.self_attn.o_proj"]

    assert q_mask.shape == k_mask.shape == v_mask.shape
    # O projection mask applies on input dim; must still have same length.
    assert o_mask.numel() == q_mask.numel()
    assert torch.equal(q_mask, k_mask)
    assert torch.equal(q_mask, v_mask)

    # Verify that pruning actually zeroed out the corresponding weights.
    attn = exp.wrapped_model._model.layers[0].self_attn

    # Q/K/V: mask on output rows
    for proj_name, proj_mask in [
        ("q_proj", q_mask),
        ("k_proj", k_mask),
        ("v_proj", v_mask),
    ]:
        proj = getattr(attn, proj_name)
        rows_zero = proj.weight.data.abs().sum(dim=1) == 0
        assert torch.all(rows_zero == (~proj_mask.bool()))

    # O projection: mask on input columns
    cols_zero = attn.o_proj.weight.data.abs().sum(dim=0) == 0
    assert torch.all(cols_zero == (~o_mask.bool()))

    # Heads should be pruned as whole units: within each head, all dims share the same mask value.
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    for h in range(num_heads):
        start = h * head_dim
        end = start + head_dim
        assert q_mask[start:end].unique().numel() == 1


def test_mlp_pruning_gate_up_down_consistency(tiny_llm_experiment):
    """MLP pruning should keep gate/up/down projections structurally consistent."""
    exp = tiny_llm_experiment

    masks = exp.apply_pruning(sparsity=0.5, metric="activation_l2_norm", mode="low")

    gate_mask = masks["model.layers.0.mlp.gate_proj"]
    # Check that MLP weights respect the gate mask (rows / columns).
    mlp = exp.wrapped_model._model.layers[0].mlp

    # gate_proj: mask on output dimension (rows)
    rows_zero = mlp.gate_proj.weight.data.abs().sum(dim=1) == 0
    assert torch.all(rows_zero == (~gate_mask.bool()))

    # down_proj: corresponding columns should be zero
    cols_zero = mlp.down_proj.weight.data.abs().sum(dim=0) == 0
    assert torch.all(cols_zero == (~gate_mask.bool()))


def test_attention_supernode_core_protection(tiny_llm_experiment):
    """
    Supernode protection should prevent core neurons from being pruned in attention.

    We mark index 0 as a core neuron via a supernode mask and enable protection;
    the resulting attention mask should keep the entire head containing index 0.
    """
    exp = tiny_llm_experiment
    attn = exp.wrapped_model._model.layers[0].self_attn

    layer_key = "model.layers.0.self_attn.q_proj"

    # Override scores with a simple ramp so behavior is deterministic.
    scores = torch.arange(attn.embed_dim, dtype=torch.float32)
    exp.importance_scores[layer_key]["activation_l2_norm"] = scores.clone()

    # Configure supernode protection and mark index 0 as core.
    core_mask = torch.zeros_like(scores, dtype=torch.bool)
    core_mask[0] = True
    exp.config.supernode_config = {
        "enabled": True,
        "protect_core": True,
        "score_metric": "activation_l2_norm",
        "core_fraction": 0.1,
        "min_core_neurons": 1,
    }
    exp.importance_scores[layer_key]["supernode_mask"] = core_mask

    neuron_mask, heads_kept, total_heads = exp._create_attention_neuron_mask(
        scores=scores.clone(),
        attn_module=attn,
        mode="low",
        sparsity=0.75,
        layer_key=layer_key,
        metric="activation_l2_norm",
    )

    assert neuron_mask is not None
    assert total_heads == attn.num_heads
    # Index 0 (and its entire head) should be kept.
    assert neuron_mask[0] == 1.0
    head_dim = attn.head_dim
    assert neuron_mask[0:head_dim].min().item() == 1.0
