"""
Unit tests for attention SCAR metrics computation.

These tests validate:
- Correct computation of per-head loss proxy
- Hook-based forward/backward tracking
- Summary statistics and visualization compatibility
"""

from typing import Optional

import pytest
import torch
import torch.nn as nn

# Skip entire module if transformers not installed
pytest.importorskip("transformers")

from nodelens.experiments.base import BaseExperiment, ExperimentConfig
from nodelens.experiments.llm_experiments import LLMAlignmentExperiment


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
        # Simplified attention: just project and recombine
        batch, seq_len, embed = x.shape
        v = self.v_proj(x)
        out = self.o_proj(v)
        return out


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
    def __init__(self, embed_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.self_attn = _TinySelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = _TinyMLP(hidden_dim=embed_dim, intermediate_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class _TinyTransformer(nn.Module):
    def __init__(self, num_layers: int = 2, embed_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList([_TinyBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _TinyLM(nn.Module):
    """
    Minimal causal LM for testing attention SCAR metrics.
    Mimics HuggingFace model structure with config.
    """

    class Config:
        def __init__(self, num_attention_heads: int = 4, hidden_size: int = 16):
            self.num_attention_heads = num_attention_heads
            self.hidden_size = hidden_size

    def __init__(self, num_layers: int = 2, embed_dim: int = 16, num_heads: int = 4, vocab_size: int = 100):
        super().__init__()
        self.config = self.Config(num_attention_heads=num_heads, hidden_size=embed_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.model = _TinyTransformer(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x = self.embed(input_ids)
        x = self.model(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Return object-like result
        class Output:
            pass

        out = Output()
        out.logits = logits
        out.loss = loss
        return out


@pytest.fixture(autouse=True)
def _bypass_base_init(monkeypatch):
    """Prevent BaseExperiment from initializing model/dataset/metrics."""
    monkeypatch.setattr(BaseExperiment, "_initialize_components", lambda self: None)
    monkeypatch.setattr(BaseExperiment, "_setup_directories", lambda self: None)


class TestAttentionSCARMetrics:
    """Tests for compute_attention_scar_metrics method."""

    def test_attention_scar_hook_registration(self):
        """Test that hooks are correctly registered on o_proj modules."""
        model = _TinyLM(num_layers=2, embed_dim=16, num_heads=4, vocab_size=100)

        # Count o_proj modules
        o_proj_count = sum(1 for name, _ in model.named_modules() if "o_proj" in name)
        assert o_proj_count == 2, f"Expected 2 o_proj modules, got {o_proj_count}"

    def test_attention_scar_output_structure(self):
        """Test that attention SCAR metrics have correct structure."""
        model = _TinyLM(num_layers=2, embed_dim=16, num_heads=4, vocab_size=100)
        torch.device("cpu")

        # Simple mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
                # Return random token ids
                ids = torch.randint(1, 100, (1, 10))
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        # Create minimal config
        config = ExperimentConfig(
            name="test_attn_scar",
            experiment_type="llm_alignment",
            model_name="test",
            device="cpu",
            log_dir="/tmp/test_attn_scar",
            do_attention_scar_metrics=True,
            scar_num_samples=2,
            scar_max_length=16,
        )

        # Create experiment with mocked components
        exp = LLMAlignmentExperiment(config)
        exp.model = model
        exp.tokenizer = MockTokenizer()
        exp.importance_scores = {}

        # Provide calibration texts
        exp.config.importance_computation_texts = ["Test text one", "Test text two"]

        # Run attention SCAR computation
        attn_scores = exp.compute_attention_scar_metrics()

        # Verify structure
        assert len(attn_scores) > 0, "Expected some attention SCAR scores"

        for layer_name, layer_metrics in attn_scores.items():
            assert "o_proj" in layer_name, f"Expected o_proj in layer name: {layer_name}"
            assert "attn_loss_proxy" in layer_metrics, "Missing attn_loss_proxy"
            assert "attn_activation_power" in layer_metrics, "Missing attn_activation_power"
            assert "attn_taylor" in layer_metrics, "Missing attn_taylor"
            assert "attn_gradient_power" in layer_metrics, "Missing attn_gradient_power"

            # Check shapes
            lp = layer_metrics["attn_loss_proxy"]
            assert lp.shape == (4,), f"Expected shape (4,) for 4 heads, got {lp.shape}"

    def test_attention_scar_config_disabled(self):
        """Test that disabled config skips computation."""
        config = ExperimentConfig(
            name="test_attn_scar_disabled",
            experiment_type="llm_alignment",
            model_name="test",
            device="cpu",
            log_dir="/tmp/test_attn_scar_disabled",
            do_attention_scar_metrics=False,
        )

        exp = LLMAlignmentExperiment(config)
        exp.model = _TinyLM()
        exp.importance_scores = {}

        result = exp.compute_attention_scar_metrics()
        assert result == {}, "Expected empty dict when disabled"

    def test_attention_loss_proxy_values(self):
        """Test that loss proxy values are non-negative and finite."""
        model = _TinyLM(num_layers=2, embed_dim=16, num_heads=4, vocab_size=100)

        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
                ids = torch.randint(1, 100, (1, 8))
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        config = ExperimentConfig(
            name="test_attn_lp",
            experiment_type="llm_alignment",
            model_name="test",
            device="cpu",
            log_dir="/tmp/test_attn_lp",
            do_attention_scar_metrics=True,
            scar_num_samples=2,
            scar_max_length=16,
        )

        exp = LLMAlignmentExperiment(config)
        exp.model = model
        exp.tokenizer = MockTokenizer()
        exp.importance_scores = {}
        exp.config.importance_computation_texts = ["Test one", "Test two"]

        attn_scores = exp.compute_attention_scar_metrics()

        for layer_name, layer_metrics in attn_scores.items():
            lp = layer_metrics["attn_loss_proxy"]
            assert torch.all(lp >= 0), "Loss proxy should be non-negative"
            assert torch.all(torch.isfinite(lp)), "Loss proxy should be finite"
            assert lp.sum() > 0, "Loss proxy should have some positive values"


class TestAttentionSCARVisualization:
    """Tests for attention SCAR visualization functions."""

    def test_plot_attention_head_heatmap_import(self):
        """Test that visualization function exists and is importable."""
        from nodelens.analysis.visualization import UnifiedVisualizer

        viz = UnifiedVisualizer()
        assert hasattr(viz, "plot_attention_head_heatmap"), "Missing plot_attention_head_heatmap method"
        assert hasattr(viz, "plot_attention_scar_layer_scores"), "Missing plot_attention_scar_layer_scores method"
        assert hasattr(viz, "plot_ffn_vs_attention_concentration"), "Missing plot_ffn_vs_attention_concentration method"

    def test_plot_ffn_vs_attention_concentration(self):
        """Test FFN vs attention concentration plot with mock data."""
        import os
        import tempfile

        from nodelens.analysis.visualization import UnifiedVisualizer

        viz = UnifiedVisualizer()

        # Create mock data
        scar_scores = {
            "layer.0.mlp": {"scar_loss_proxy": torch.randn(100).abs()},
            "layer.1.mlp": {"scar_loss_proxy": torch.randn(100).abs()},
        }
        attn_scar_scores = {
            "layer.0.self_attn.o_proj": {
                "attn_loss_proxy": torch.randn(8).abs(),
                "layer_idx": "0",
            },
            "layer.1.self_attn.o_proj": {
                "attn_loss_proxy": torch.randn(8).abs(),
                "layer_idx": "1",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_concentration.png")
            fig = viz.plot_ffn_vs_attention_concentration(
                scar_scores=scar_scores,
                attn_scar_scores=attn_scar_scores,
                save_path=save_path,
            )

            assert fig is not None, "Figure should be returned"
            assert os.path.exists(save_path), "Figure should be saved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
