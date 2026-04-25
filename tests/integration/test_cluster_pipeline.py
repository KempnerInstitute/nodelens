"""
Integration test: full cluster analysis pipeline.

End-to-end: tiny CNN + synthetic data -> compute metrics -> cluster -> halo -> CAP prune
-> verify valid masks and model still runs.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from nodelens.analysis.cascade_analysis import CascadeAnalysis
from nodelens.analysis.clustering.cross_layer_halo import CrossLayerHaloAnalysis
from nodelens.analysis.clustering.metric_clustering import MetricSpaceClustering
from nodelens.pruning.strategies.cluster_aware import ClusterAwarePruning, ClusterAwarePruningConfig

# ---------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------


class _PipelineCNN(nn.Module):
    """3-layer CNN for pipeline testing."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestClusterPipeline:
    def test_full_pipeline(self):
        """Run metrics -> cluster -> halo -> prune -> verify model still works."""
        torch.manual_seed(42)
        np.random.seed(42)

        # 1. Create model and synthetic data
        model = _PipelineCNN(n_classes=5)
        model.eval()
        n_samples = 50
        images = torch.randn(n_samples, 3, 8, 8)
        labels = torch.randint(0, 5, (n_samples,))

        # 2. Collect activations per layer
        activations = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, inp, out):
                activations[name] = out.detach()

            return hook_fn

        for name, mod in [("conv1", model.conv1), ("conv2", model.conv2), ("conv3", model.conv3)]:
            hooks.append(mod.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            logits = model(images)

        for h in hooks:
            h.remove()

        # 3. Compute per-channel metrics for conv2 (middle layer)
        acts = activations["conv2"]  # [B, 32, H, W]
        acts_gap = acts.mean(dim=(2, 3)).numpy()  # [B, 32]

        # RQ proxy: activation variance / weight norm
        var = np.var(acts_gap, axis=0)
        w = model.conv2.weight.detach().numpy()
        w_flat = w.reshape(w.shape[0], -1)
        w_norm_sq = np.sum(w_flat**2, axis=1)
        rq = var / (w_norm_sq + 1e-10)

        # Redundancy: mean pairwise Gaussian MI
        corr = np.corrcoef(acts_gap.T)
        corr = np.clip(corr, -0.999, 0.999)
        mi = -0.5 * np.log(1 - corr**2)
        np.fill_diagonal(mi, 0)
        red = np.mean(mi, axis=1)

        # Synergy: simple proxy (variance of logit margin per channel)
        logits_np = logits.numpy()
        correct_logits = logits_np[np.arange(n_samples), labels.numpy()]
        margins = correct_logits - logits_np.mean(axis=1)
        syn = np.zeros(32)
        for i in range(32):
            r = np.corrcoef(acts_gap[:, i], margins)[0, 1]
            syn[i] = abs(r)

        # 4. Cluster channels
        clusterer = MetricSpaceClustering(n_clusters=4, seed=42)
        cluster_result = clusterer.fit(rq, red, syn, name="conv2")

        assert cluster_result.n_clusters == 4
        assert len(cluster_result.labels) == 32
        assert set(cluster_result.type_mapping.values()) == {
            "critical",
            "redundant",
            "synergistic",
            "background",
        }

        # 5. Halo analysis: conv2 -> conv3
        halo_analyzer = CrossLayerHaloAnalysis(percentile=80)
        w_next = model.conv3.weight.detach().numpy()
        # For conv: [out, in, k, k] -> sum kernel -> [out, in]
        w_next_2d = np.abs(w_next).sum(axis=(2, 3))

        activations["conv3"].mean(dim=(2, 3)).numpy()
        influence = halo_analyzer.compute_influence(w_next_2d, acts_gap)

        critical_idx = np.where(cluster_result.labels == [k for k, v in cluster_result.type_mapping.items() if v == "critical"][0])[0]
        halo_idx, rel_infl = halo_analyzer.find_halo(influence, critical_idx)
        assert rel_infl.shape[0] == 32  # n_out for conv3

        # 6. CAP prune at 50%
        metrics = {"rq": rq, "redundancy": red, "synergy": syn}
        clusters = {
            "labels": cluster_result.labels,
            "centroids": cluster_result.centroids,
            "type_mapping": cluster_result.type_mapping,
            "type_counts": cluster_result.type_counts,
        }

        cap = ClusterAwarePruning(
            config=ClusterAwarePruningConfig(
                amount=0.5,
                protect_critical_frac=0.3,
                target_redundant=True,
            ),
            precomputed_metrics=metrics,
            precomputed_clusters=clusters,
        )

        scores = cap.compute_importance_scores(
            module=model.conv2,
            layer_name="conv2",
        )
        assert scores.shape == (32,)
        assert torch.all(torch.isfinite(scores))

        n_prune = 16
        pruned = cap.select_channels_to_prune(scores, n_prune, layer_name="conv2")
        assert len(pruned) == n_prune
        assert all(0 <= i < 32 for i in pruned)

        # 7. Apply pruning mask and verify model still runs
        mask = torch.ones(32)
        for i in pruned:
            mask[i] = 0.0

        # Zero out pruned channels
        with torch.no_grad():
            model.conv2.weight.data *= mask.view(-1, 1, 1, 1)
            if model.conv2.bias is not None:
                model.conv2.bias.data *= mask

        # Model should still produce valid output
        with torch.no_grad():
            out = model(images[:5])
        assert out.shape == (5, 5)
        assert torch.all(torch.isfinite(out))

    def test_cascade_by_cluster_after_clustering(self):
        """Cascade analysis should work with cluster labels from MetricSpaceClustering."""
        torch.manual_seed(42)
        model = _PipelineCNN(n_classes=5)
        model.eval()

        # Synthetic loader
        images = torch.randn(30, 3, 8, 8)
        labels = torch.randint(0, 5, (30,))
        loader = [(images[:15], labels[:15]), (images[15:], labels[15:])]

        # Quick cluster (using random metrics for speed)
        rng = np.random.default_rng(42)
        rq = rng.uniform(0.1, 10.0, 32)
        red = rng.uniform(0, 1, 32)
        syn = rng.uniform(0, 1, 32)

        clusterer = MetricSpaceClustering(n_clusters=4, seed=42)
        result = clusterer.fit(rq, red, syn, name="conv2")

        # Run cascade
        ca = CascadeAnalysis(model, loader, device="cpu")
        cascade_results = ca.by_cluster(
            "conv2",
            result.labels,
            result.type_mapping,
            n_rm=3,
        )
        assert len(cascade_results) > 0
        for ctype, cr in cascade_results.items():
            assert cr.n_removed <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
