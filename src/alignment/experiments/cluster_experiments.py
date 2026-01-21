"""
Cluster-based analysis experiments for neural networks.

This module provides a general experiment runner for:
1. Computing per-channel metrics (RQ, Redundancy, Synergy with continuous target)
2. Clustering channels/neurons into functional types (Critical, Redundant, Synergistic, Background)
3. Cross-layer halo analysis (downstream dependencies)
4. Cascade/damage prediction experiments
5. Cluster-aware pruning with baseline comparisons

Compatible with any neural network architecture:
- Vision: ResNet, VGG, MobileNet, etc.
- LLMs: Can be adapted for FFN analysis
- Any model with convolutional or linear layers
"""

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..analysis.clustering import MetricSpaceClustering, CrossLayerHaloAnalysis
from ..analysis.cascade_analysis import CascadeAnalysis, DamagePrediction
from ..pruning.pipeline import PruningPipelineOptions, run_pruning_pipeline

logger = logging.getLogger(__name__)


class _CovAccumulator:
    """
    Streaming Gaussian-statistics accumulator for a layer.

    Maintains sufficient statistics to compute:
    - per-channel variance
    - channel-channel covariance/correlation
    - covariance between scalar target T and channels
    """

    def __init__(self, n_channels: int):
        self.n = 0
        self.sum_y = np.zeros(n_channels, dtype=np.float64)
        self.sum_abs_y = np.zeros(n_channels, dtype=np.float64)
        self.sum_yy = np.zeros((n_channels, n_channels), dtype=np.float64)
        self.sum_t = 0.0
        self.sum_tt = 0.0
        self.sum_ty = np.zeros(n_channels, dtype=np.float64)

    def update(self, y: np.ndarray, t: np.ndarray) -> None:
        """
        Args:
            y: [N, C] channel samples (float)
            t: [N] target samples (float)
        """
        if y.size == 0:
            return
        y = np.asarray(y, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        if y.ndim != 2:
            raise ValueError(f"Expected y as [N,C], got shape {y.shape}")
        if t.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched sample count: y has {y.shape[0]}, t has {t.shape[0]}")

        self.n += int(y.shape[0])
        self.sum_y += y.sum(axis=0)
        # For activation-magnitude baselines (mean |activation| per channel)
        self.sum_abs_y += np.abs(y).sum(axis=0)
        self.sum_yy += y.T @ y
        self.sum_t += float(t.sum())
        self.sum_tt += float((t * t).sum())
        self.sum_ty += (t[:, None] * y).sum(axis=0)

    def finalize(self) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            var_t: scalar
            var_y: [C]
            cov_yy: [C, C]
            cov_ty: [C]
        """
        if self.n < 2:
            c = self.sum_y.shape[0]
            return 0.0, np.zeros(c), np.zeros((c, c)), np.zeros(c)

        n = float(self.n)
        mean_y = self.sum_y / n
        mean_t = self.sum_t / n

        # Unbiased covariance estimates (divide by n-1)
        cov_yy = (self.sum_yy - n * np.outer(mean_y, mean_y)) / (n - 1.0)
        var_y = np.clip(np.diag(cov_yy), 1e-12, None)

        var_t = float((self.sum_tt - n * mean_t * mean_t) / (n - 1.0))
        var_t = max(var_t, 1e-12)

        cov_ty = (self.sum_ty - n * mean_t * mean_y) / (n - 1.0)
        return var_t, var_y, cov_yy, cov_ty


@dataclass
class ClusterAnalysisConfig:
    """Configuration for cluster-based analysis experiments."""
    model_name: str = "resnet18"
    dataset_name: str = "cifar10"
    n_calibration: int = 5000
    n_clusters: int = 4
    # Where to read the channel signal Y_i for within-layer statistics:
    # - "pre_bn": hook Conv2d outputs (pre-BN, pre-ReLU). (Backward compatible default.)
    # - "post_bn": hook the matching BatchNorm outputs when available (post-BN, pre-ReLU).
    #   For RQ we fold BN scaling into the denominator so the metric stays comparable.
    activation_point: str = "pre_bn"
    # How to form channel samples from Conv outputs Y[B,C,H,W]
    # - "flatten_spatial": treat spatial positions as samples (subsample per image)
    # - "gap": global-average-pool per image (one sample per image)
    activation_samples: str = "flatten_spatial"
    spatial_samples_per_image: int = 16  # used when activation_samples="flatten_spatial"
    synergy_target: str = "logit_margin"  # logit_margin, correct_logit
    # Synergy settings:
    # - synergy_candidate_pool: number of candidate partners per channel (chosen by redundancy)
    # - synergy_pairs: top-m partners to average (Eq. per_channel_syn)
    synergy_candidate_pool: int = 50
    synergy_pairs: int = 10
    halo_percentile: float = 90.0
    use_activation_weight: bool = True  # Use activation-weighted influence for halos
    cascade_n_remove: int = 5
    damage_sample_frac: float = 0.2
    # Pruning experiment settings
    pruning_ratios: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    pruning_methods: List[str] = field(default_factory=lambda: [
        'random', 'magnitude', 'taylor', 'network_slimming', 'composite', 'cluster_aware'
    ])
    fine_tune_after_pruning: bool = False  # Whether to fine-tune after pruning
    fine_tune_epochs: int = 10
    fine_tune_lr: float = 0.0001
    fine_tune_max_batches: Optional[int] = None
    fine_tune_weight_decay: float = 0.0
    # Output
    output_dir: str = "results/cluster_analysis"
    device: str = "cuda"
    seed: int = 42
    # Multi-seed support for robust statistics
    seeds: Optional[List[int]] = None  # If provided, run experiment with each seed
    # Ablation settings
    run_metric_ablation: bool = False  # Run clustering with metric subsets
    metric_ablations: List[str] = field(default_factory=lambda: ["all", "rq_red", "rq_syn", "red_syn"])
    # Permutation baseline settings
    run_permutation_baseline: bool = False  # Run halo permutation tests
    n_permutations: int = 100


# Backward compatibility alias
VisionExperimentConfig = ClusterAnalysisConfig


class ClusterAnalysisExperiment:
    """
    General experiment class for cluster-based neural network analysis.
    
    Works with any architecture that has Conv2d or Linear layers.
    
    Example:
        >>> config = ClusterAnalysisConfig(model_name="resnet18")
        >>> exp = ClusterAnalysisExperiment(config, model, train_loader, test_loader)
        >>> results = exp.run()
    """
    
    def __init__(
        self,
        config: ClusterAnalysisConfig,
        model: "nn.Module",
        train_loader: "DataLoader",
        test_loader: "DataLoader",
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = config.device
        
        # Results storage
        self.layer_metrics = {}
        self.cluster_results = {}
        self.halo_results = {}
        self.halo_flow_results = {}
        self.permutation_results = {}  # Permutation baseline results
        self.ablation_results = {}     # Metric ablation results
        self.cascade_results = {}
        self.pruning_results = {}
        self.pruning_cluster_distributions = {}
        # Cache for expensive pruning scores (e.g., gradient-based Taylor)
        self._pruning_score_cache: Dict[str, Dict[str, "torch.Tensor"]] = {}
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get analyzable layers
        self.layers = self._get_conv_layers()
        logger.info(f"Found {len(self.layers)} convolutional layers")
    
    def _get_conv_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get all Conv2d layers for analysis."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels >= 4:
                layers.append((name, module))
        return layers
    
    def compute_metrics(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute per-channel metrics for all layers.
        
        Returns:
            Dict mapping layer_name to dict of metric arrays
        """
        logger.info("Computing per-channel metrics (streaming)...")
        self.model.eval()

        # Per-layer accumulators (filled lazily once we see a batch for the layer)
        accs: Dict[str, _CovAccumulator] = {}

        # Temporary per-batch activations captured by hooks
        batch_acts: Dict[str, "torch.Tensor"] = {}

        def hook_fn(name: str):
            def fn(_m, _inp, out):
                # Store only for this batch; processed after logits are computed
                batch_acts[name] = out.detach()
            return fn

        # Register hooks.
        # By default we hook conv outputs (pre-BN); optionally hook matching BN outputs (post-BN)
        # while still storing under the conv's name so downstream code stays consistent.
        modules = dict(self.model.named_modules())
        activation_point = str(getattr(self.config, "activation_point", "pre_bn")).lower()

        def _bn_for_conv_name(conv_name: str):
            # Best-effort mapping using common naming conventions (ResNet/VGG).
            cand = [
                conv_name.replace("conv", "bn"),
                conv_name.replace(".conv", ".bn"),
                conv_name + "_bn",
            ]
            if "downsample.0" in conv_name:
                cand.append(conv_name.replace("downsample.0", "downsample.1"))
            for n in cand:
                m = modules.get(n)
                if m is not None and m.__class__.__name__.lower().startswith("batchnorm"):
                    return n, m
            return None, None

        handles = []
        for name, layer in self.layers:
            hook_mod = layer
            if activation_point in {"post_bn", "postbn", "bn"}:
                _bn_name, bn = _bn_for_conv_name(name)
                if bn is not None:
                    hook_mod = bn
            handles.append(hook_mod.register_forward_hook(hook_fn(name)))

        activation_mode = str(getattr(self.config, "activation_samples", "flatten_spatial")).lower()
        samples_per_img = int(getattr(self.config, "spatial_samples_per_image", 16))
        samples_per_img = max(1, samples_per_img)

        rng = np.random.default_rng(int(getattr(self.config, "seed", 42)))

        n_seen = 0
        with torch.no_grad():
            for x, y in self.train_loader:
                if n_seen >= self.config.n_calibration:
                    break

                # Trim last batch to hit n_calibration exactly
                remaining = int(self.config.n_calibration) - int(n_seen)
                if remaining <= 0:
                    break
                if x.size(0) > remaining:
                    x = x[:remaining]
                    y = y[:remaining]

                x = x.to(self.device)
                y = y.to(self.device)

                batch_acts.clear()
                logits = self.model(x)

                # Continuous target T (logit margin)
                bsz = logits.size(0)
                correct_logits = logits[torch.arange(bsz, device=logits.device), y]
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[torch.arange(bsz, device=logits.device), y] = False
                max_incorrect = logits.masked_fill(~mask, float("-inf")).max(dim=1)[0]
                T_img = (correct_logits - max_incorrect).detach().cpu().numpy()  # [B]

                # Update each layer accumulator using the captured activations
                for name, layer in self.layers:
                    out = batch_acts.get(name)
                    if out is None:
                        continue
                    if out.ndim != 4:
                        continue

                    out_cpu = out.detach().cpu()  # [B, C, H, W]
                    b, c, h, w = out_cpu.shape

                    if activation_mode in {"gap", "global", "global_avg", "global_average"}:
                        y_s = out_cpu.mean(dim=(2, 3)).numpy()  # [B, C]
                        t_s = T_img
                    else:
                        # Spatially-flattened samples, subsampled per image
                        hw = int(h * w)
                        p = min(samples_per_img, hw)
                        # [B, HW, C] as numpy for fast per-image patch subsampling
                        y_hw_np = out_cpu.permute(0, 2, 3, 1).reshape(b, hw, c).numpy()
                        if p < hw:
                            idx = rng.integers(0, hw, size=(b, p), endpoint=False)
                            row = np.arange(b)[:, None]
                            y_s = y_hw_np[row, idx, :].reshape(b * p, c)
                            t_s = np.repeat(T_img, p)
                        else:
                            y_s = y_hw_np.reshape(b * hw, c)
                            t_s = np.repeat(T_img, hw)

                    if name not in accs:
                        accs[name] = _CovAccumulator(n_channels=c)
                    accs[name].update(y_s, t_s)

                n_seen += int(x.size(0))

        # Remove hooks
        for h in handles:
            h.remove()

        # Compute metrics per layer from accumulated Gaussian stats
        for name, layer in self.layers:
            acc = accs.get(name)
            if acc is None:
                continue

            var_t, var_y, cov_yy, cov_ty = acc.finalize()
            n_channels = int(var_y.shape[0])

            metrics: Dict[str, np.ndarray] = {}

            # 0) Activation magnitude baselines (computed from the same calibration samples)
            # Mean absolute activation per channel (requested baseline)
            if acc.n > 0:
                metrics["activation_mean"] = (acc.sum_abs_y / float(acc.n))[:n_channels].astype(np.float64)
                # RMS activation (close cousin of activation L2 norm; scale doesn't affect ranking)
                y2 = np.clip(np.diag(acc.sum_yy) / float(acc.n), 0.0, None)
                metrics["activation_rms"] = np.sqrt(y2)[:n_channels].astype(np.float64)

            # 1) Rayleigh Quotient proxy: Var(Y_i) / ||w_i||^2
            weight = layer.weight.data.cpu()  # [C_out, C_in, k, k]
            weight_flat = weight.view(weight.size(0), -1)  # [C_out, ...]
            weight_norm = weight_flat.norm(dim=1).numpy().astype(np.float64) ** 2
            # If we used post-BN activations as Y, fold the BN scale into the denominator so
            # RQ remains comparable to the pre-BN definition (since Var(BN(y)) scales by gamma^2/rv).
            if activation_point in {"post_bn", "postbn", "bn"}:
                _bn_name, bn = _bn_for_conv_name(name)
                if bn is not None and hasattr(bn, "weight") and hasattr(bn, "running_var"):
                    try:
                        gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
                        rv = bn.running_var.detach().cpu().numpy().astype(np.float64)
                        eps = float(getattr(bn, "eps", 1e-5))
                        scale_sq = (gamma[:n_channels] ** 2) / (rv[:n_channels] + eps)
                        denom = (weight_norm[:n_channels] * scale_sq) + 1e-10
                        rq = var_y / denom
                    except Exception:
                        rq = var_y / (weight_norm[:n_channels] + 1e-10)
                else:
                    rq = var_y / (weight_norm[:n_channels] + 1e-10)
            else:
                rq = var_y / (weight_norm[:n_channels] + 1e-10)
            metrics["rq"] = rq.astype(np.float64)

            # 2) Redundancy via Gaussian MI from correlations
            denom = np.sqrt(np.outer(var_y, var_y)) + 1e-12
            corr = cov_yy / denom
            corr = np.clip(corr, -0.999, 0.999)
            mi_matrix = -0.5 * np.log(1.0 - corr ** 2)
            np.fill_diagonal(mi_matrix, 0.0)
            metrics["redundancy"] = mi_matrix.mean(axis=1).astype(np.float64)

            # 3) Synergy with scalar target under Gaussian approximation (MMI)
            # MI(T;Y_i) depends only on corr(T,Y_i)
            corr_ty = cov_ty / (np.sqrt(var_t * var_y) + 1e-12)
            corr_ty = np.clip(corr_ty, -0.999, 0.999)
            mi_t = np.maximum(0.0, -0.5 * np.log(1.0 - corr_ty ** 2))

            candidate_pool = int(getattr(self.config, "synergy_candidate_pool", 50))
            top_m = int(getattr(self.config, "synergy_pairs", 10))
            candidate_pool = max(2, min(candidate_pool, n_channels))
            top_m = max(1, min(top_m, candidate_pool - 1))

            synergy = np.zeros(n_channels, dtype=np.float64)

            # Precompute partner ordering by redundancy (MI) per channel
            # Use the MI matrix row i, excluding i.
            for i in range(n_channels):
                order = np.argsort(-mi_matrix[i])
                order = order[order != i]
                cand = order[:candidate_pool]
                if cand.size == 0:
                    continue

                mi_i = float(mi_t[i])
                syn_vals: List[float] = []
                for j in cand:
                    j = int(j)
                    mi_j = float(mi_t[j])
                    cov_i_j = float(cov_yy[i, j])
                    mi_joint = self._gaussian_mi_joint_from_stats(
                        var_t=var_t,
                        var_i=float(var_y[i]),
                        var_j=float(var_y[j]),
                        cov_t_i=float(cov_ty[i]),
                        cov_t_j=float(cov_ty[j]),
                        cov_i_j=cov_i_j,
                    )
                    s = mi_joint - mi_i - mi_j + min(mi_i, mi_j)
                    syn_vals.append(float(s))

                if syn_vals:
                    syn_vals.sort(reverse=True)
                    synergy[i] = float(np.mean(syn_vals[:top_m]))

            metrics["synergy"] = synergy

            self.layer_metrics[name] = metrics
            logger.info(
                "  %s: %d channels (mode=%s, n_samples=%d)",
                name,
                n_channels,
                activation_mode,
                acc.n,
            )

        return self.layer_metrics
    
    def _gaussian_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Gaussian MI between two variables."""
        rho = np.corrcoef(x, y)[0, 1]
        rho = np.clip(rho, -0.999, 0.999)
        return max(0, -0.5 * np.log(1 - rho ** 2))
    
    def _gaussian_mi_joint(self, t: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float:
        """Compute Gaussian MI I(T; [Y1, Y2])."""
        joint = np.column_stack([t, y1, y2])
        cov = np.cov(joint.T) + 1e-8 * np.eye(3)
        var_t = cov[0, 0]
        cov_y = cov[1:, 1:]
        det_all = np.linalg.det(cov)
        det_y = np.linalg.det(cov_y)
        if det_all <= 0 or det_y <= 0 or var_t <= 0:
            return 0.
        return max(0, 0.5 * np.log(var_t * det_y / det_all))

    def _gaussian_mi_joint_from_stats(
        self,
        *,
        var_t: float,
        var_i: float,
        var_j: float,
        cov_t_i: float,
        cov_t_j: float,
        cov_i_j: float,
    ) -> float:
        """Gaussian MI I(T; [Y_i, Y_j]) from covariance statistics (no raw samples)."""
        # 3x3 covariance matrix for (T, Y_i, Y_j)
        cov = np.array(
            [
                [var_t, cov_t_i, cov_t_j],
                [cov_t_i, var_i, cov_i_j],
                [cov_t_j, cov_i_j, var_j],
            ],
            dtype=np.float64,
        )
        cov += 1e-10 * np.eye(3)
        cov_y = np.array([[var_i, cov_i_j], [cov_i_j, var_j]], dtype=np.float64)
        cov_y += 1e-10 * np.eye(2)

        det_all = float(np.linalg.det(cov))
        det_y = float(np.linalg.det(cov_y))
        if det_all <= 0.0 or det_y <= 0.0 or var_t <= 0.0:
            return 0.0
        return max(0.0, 0.5 * float(np.log(var_t * det_y / det_all)))
    
    def run_clustering(self, run_ablation: Optional[bool] = None) -> Dict[str, Any]:
        """
        Cluster channels in each layer.
        
        Args:
            run_ablation: If True, also run ablation study with metric subsets.
                         Uses config.run_metric_ablation if not specified.
        
        Returns:
            Dict with cluster results (and ablation results if enabled)
        """
        logger.info("Clustering channels...")
        
        run_ablation = run_ablation if run_ablation is not None else getattr(
            self.config, 'run_metric_ablation', False
        )
        
        clusterer = MetricSpaceClustering(
            n_clusters=self.config.n_clusters,
            seed=self.config.seed,
        )
        
        ablation_results = {}
        
        for name, metrics in self.layer_metrics.items():
            result = clusterer.fit(
                metrics["rq"],
                metrics["redundancy"],
                metrics["synergy"],
                name,
            )
            self.cluster_results[name] = {
                "labels": result.labels,
                "centroids": result.centroids,
                "silhouette": result.silhouette,
                "type_mapping": result.type_mapping,
                "type_counts": result.type_counts,
                "layer_name": name,
                "ablation_mode": "all",
            }
            logger.info(f"  {name}: silhouette={result.silhouette:.3f}, types={result.type_counts}")
            
            # Run ablation study if enabled
            if run_ablation:
                ablations = getattr(self.config, 'metric_ablations', 
                                   ["all", "rq_red", "rq_syn", "red_syn"])
                abl_results = clusterer.run_ablation_study(
                    metrics["rq"],
                    metrics["redundancy"],
                    metrics["synergy"],
                    name,
                    ablations=ablations,
                )
                ablation_results[name] = {
                    ablation: {
                        "silhouette": res.silhouette,
                        "ari_vs_full": res.ari_vs_full,
                        "ami_vs_full": res.ami_vs_full,
                        "type_counts": res.cluster_result.type_counts,
                    }
                    for ablation, res in abl_results.items()
                }
                logger.info(f"    Ablation: {[f'{k}: sil={v.silhouette:.3f}' for k,v in abl_results.items()]}")
        
        if run_ablation:
            self.cluster_results["_ablation"] = ablation_results
        
        return self.cluster_results
    
    def run_halo_analysis(
        self,
        run_permutation: Optional[bool] = None,
        n_permutations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze cross-layer halos with activation-weighted influence.
        
        Uses effective influence: ||W||_1 * std(Y) to account for
        batch normalization scaling effects.
        
        Args:
            run_permutation: If True, run permutation test to establish null baseline.
                            Uses config.run_permutation_baseline if not specified.
            n_permutations: Number of permutations for baseline (default: config.n_permutations)
        
        Returns:
            Dict with halo results per transition
        """
        logger.info("Analyzing cross-layer halos...")
        
        # Get permutation settings
        run_permutation = run_permutation if run_permutation is not None else getattr(
            self.config, 'run_permutation_baseline', False
        )
        n_permutations = n_permutations if n_permutations is not None else getattr(
            self.config, 'n_permutations', 100
        )
        
        # Initialize permutation results storage if needed
        if not hasattr(self, 'permutation_results'):
            self.permutation_results = {}
        
        halo_analyzer = CrossLayerHaloAnalysis(
            percentile=self.config.halo_percentile,
            use_activation_weight=getattr(self.config, 'use_activation_weight', True),
        )
        
        layer_names = list(self.cluster_results.keys())
        # Filter out special keys like "_ablation"
        layer_names = [n for n in layer_names if not n.startswith("_")]
        modules = dict(self.model.named_modules())
        
        # Choose halo transitions along *direct weight-connected* edges by matching channel dimensions.
        # This avoids spurious transitions in residual blocks (e.g., conv2 -> downsample conv),
        # while still supporting skip-branch convs as valid sources into the next block.
        for i, src_name in enumerate(layer_names[:-1]):
            src_layer = modules.get(src_name)
            if src_layer is None or not hasattr(src_layer, "weight"):
                continue

            src_out = int(src_layer.weight.shape[0])

            tgt_name = None
            for j in range(i + 1, len(layer_names)):
                cand_name = layer_names[j]
                cand_layer = modules.get(cand_name)
                if cand_layer is None or not hasattr(cand_layer, "weight"):
                    continue
                w = cand_layer.weight
                if w is None or w.ndim < 2:
                    continue
                cand_in = int(w.shape[1])
                if cand_in == src_out:
                    tgt_name = cand_name
                    break

            if tgt_name is None:
                continue
            
            src_result = self.cluster_results[src_name]
            tgt_result = self.cluster_results.get(tgt_name, {})
            src_metrics = self.layer_metrics.get(src_name, {})
            tgt_metrics = self.layer_metrics.get(tgt_name, {})
            
            if not tgt_metrics:
                continue
            
            # Get weight matrix between layers
            tgt_layer = modules[tgt_name]
            tgt_weight = tgt_layer.weight.data.cpu().numpy()
            n_out, n_in = tgt_weight.shape[0], tgt_weight.shape[1]
            
            # Base influence: L1 norm over kernel dimensions
            influence = np.abs(tgt_weight.reshape(n_out, n_in, -1)).sum(axis=2)
            
            # Apply activation weighting (effective influence = weight * std)
            # This accounts for BN scaling: channels with large gamma/sqrt(var)
            # have larger effective signal even if outgoing weights are small
            # Activation-weighted influence proxy.
            # We approximate sigma_i as the (post-BN when present) channel std:
            #   sigma_conv = sqrt(RQ_i * ||w_i||^2)  (since RQ_i = Var(Y_i)/||w_i||^2)
            #   sigma_postBN ≈ sigma_conv * |gamma| / sqrt(running_var + eps)
            if "rq" in src_metrics:
                w_src = src_layer.weight.data.cpu().numpy().astype(np.float64)
                w_norm_sq = np.sum(w_src.reshape(w_src.shape[0], -1) ** 2, axis=1)
                rq = np.asarray(src_metrics["rq"], dtype=np.float64).reshape(-1)
                sigma = np.sqrt(np.clip(rq[: len(w_norm_sq)] * w_norm_sq[: len(rq)], 0.0, None))

                bn = self._find_bn_for_conv(self.model, src_name)
                if bn is not None and hasattr(bn, "weight") and hasattr(bn, "running_var"):
                    gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
                    rv = bn.running_var.detach().cpu().numpy().astype(np.float64)
                    eps = float(getattr(bn, "eps", 1e-5))
                    scale = np.abs(gamma) / np.sqrt(rv + eps)
                    m = min(len(sigma), len(scale))
                    sigma[:m] = sigma[:m] * scale[:m]

                n_in_actual = min(n_in, len(sigma))
                influence[:, :n_in_actual] = influence[:, :n_in_actual] * sigma[:n_in_actual]
            
            halo_data = {}
            for cid, ctype in src_result["type_mapping"].items():
                cluster_idx = np.where(src_result["labels"] == cid)[0]
                if len(cluster_idx) == 0 or cluster_idx.max() >= n_in:
                    continue
                
                halo_idx, rel_infl = halo_analyzer.find_halo(influence, cluster_idx)
                if len(halo_idx) == 0:
                    continue
                
                halo_data[ctype] = {
                    "halo_size": len(halo_idx),
                    "halo_red": float(np.mean(tgt_metrics["redundancy"][halo_idx])),
                    "halo_syn": float(np.mean(tgt_metrics["synergy"][halo_idx])),
                    "cluster_type": ctype,
                }
            
            self.halo_results[f"{src_name}->{tgt_name}"] = halo_data
            logger.info(f"  {src_name}->{tgt_name}: {len(halo_data)} cluster halos analyzed")

            # Also compute cluster-to-cluster flow matrix (for influence heatmaps)
            try:
                src_labels = np.asarray(src_result.get("labels", np.array([], dtype=int))).astype(int)
                tgt_labels = np.asarray(tgt_result.get("labels", np.array([], dtype=int))).astype(int)
                if src_labels.size > 0 and tgt_labels.size > 0:
                    # Trim labels to match influence matrix dimensions if needed
                    src_labels = src_labels[: min(len(src_labels), n_in)]
                    tgt_labels = tgt_labels[: min(len(tgt_labels), n_out)]

                    flow = halo_analyzer.compute_cluster_to_cluster_flow(
                        influence,
                        source_labels=src_labels,
                        target_labels=tgt_labels,
                        source_types=src_result.get("type_mapping", {}),
                        target_types=tgt_result.get("type_mapping", {}),
                    )
                    self.halo_flow_results[f"{src_name}->{tgt_name}"] = flow
            except Exception as exc:
                logger.debug("Could not compute halo flow matrix for %s->%s: %s", src_name, tgt_name, exc)
            
            # Run permutation baseline if enabled
            if run_permutation:
                try:
                    src_labels = np.asarray(src_result.get("labels", np.array([], dtype=int))).astype(int)
                    src_labels = src_labels[: min(len(src_labels), n_in)]
                    
                    perm_results = halo_analyzer.permutation_baseline(
                        influence=influence,
                        labels=src_labels,
                        type_mapping=src_result["type_mapping"],
                        redundancy=tgt_metrics["redundancy"],
                        synergy=tgt_metrics["synergy"],
                        n_permutations=n_permutations,
                        seed=self.config.seed,
                    )
                    self.permutation_results[f"{src_name}->{tgt_name}"] = perm_results
                    
                    # Log significant results
                    for ctype, pres in perm_results.items():
                        if pres.get('p_syn', 1.0) < 0.05 or pres.get('p_red', 1.0) < 0.05:
                            logger.info(
                                f"    Permutation test {ctype}: z_syn={pres['z_syn']:.2f} "
                                f"(p={pres['p_syn']:.3f}), z_red={pres['z_red']:.2f} (p={pres['p_red']:.3f})"
                            )
                except Exception as exc:
                    logger.debug("Permutation baseline failed for %s->%s: %s", src_name, tgt_name, exc)
        
        return self.halo_results
    
    def run_cascade_test(self) -> Dict[str, Any]:
        """Run cascade damage test by cluster type."""
        logger.info("Running cascade tests...")
        
        cascade = CascadeAnalysis(self.model, self.test_loader, self.device)
        cascade.baseline()
        
        for name, cluster_data in self.cluster_results.items():
            results = cascade.by_cluster(
                name,
                cluster_data["labels"],
                cluster_data["type_mapping"],
                n_rm=self.config.cascade_n_remove,
            )
            self.cascade_results[name] = {
                ctype: {
                    "accuracy_drop": r.accuracy_drop,
                    "loss_increase": r.loss_increase,
                    "n_removed": r.n_removed,
                }
                for ctype, r in results.items()
            }
            logger.info(f"  {name}: {len(results)} cluster types tested")
        
        return self.cascade_results
    
    def run_pruning_experiments(
        self,
        ratios: Optional[List[float]] = None,
        methods: Optional[List[str]] = None,
        fine_tune_epochs: int = 0,
        fine_tune_lr: float = 0.0001,
        fine_tune_max_batches: Optional[int] = None,
        fine_tune_weight_decay: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run pruning experiments comparing different methods.
        
        Args:
            ratios: Sparsity ratios to test (default: [0.3, 0.5, 0.7])
            methods: Pruning methods to compare (default: all)
            fine_tune_epochs: Number of fine-tuning epochs after pruning
            fine_tune_lr: Learning rate for fine-tuning (unused when fine_tune_epochs=0)
            
        Returns:
            Dict mapping (method, ratio) to accuracy results
        """
        import copy
        
        ratios = ratios or getattr(self.config, "pruning_ratios", None) \
                       or getattr(self.config, "pruning_amounts", None) \
                       or [0.1, 0.3, 0.5, 0.7]
        
        default_methods = [
            "random", "magnitude",
            "network_slimming",
            "rq_low", "rq_high",
            "redundancy_low", "redundancy_high",
            "synergy_low", "synergy_high",
            "composite", "composite_pos_red",
            "rq_minus_red", "rq_plus_red",
            "magnitude_plus_rq", "magnitude_minus_red", "magnitude_plus_red",
        ]
        methods = methods or getattr(self.config, "pruning_methods", None) \
                          or getattr(self.config, "pruning_algorithms", None) \
                          or getattr(self.config, "pruning_strategies", None) \
                          or default_methods
        
        pipeline_options = PruningPipelineOptions(
            distribution=getattr(self.config, "pruning_distribution", "uniform"),
            dependency_aware=bool(getattr(self.config, "dependency_aware_pruning", False)),
            min_amount=getattr(self.config, "pruning_min_per_layer", 0.0),
            max_amount=getattr(self.config, "pruning_max_per_layer", 0.95),
        )
        
        baseline_acc = self._evaluate_accuracy()
        logger.info(f"Baseline accuracy: {baseline_acc:.2%}")
        
        if baseline_acc < 0.7:
            logger.warning("Baseline accuracy is low; pruning comparisons may be noisy.")
        
        results = {"baseline": baseline_acc, "methods": {}}
        
        for method in methods:
            logger.info(f"Running pruning method: {method}")
            method_results = {}
            results["methods"][method] = method_results
            
            for ratio in ratios:
                logger.info(f"  Target sparsity: {ratio:.0%}")
                model_copy = copy.deepcopy(self.model)
                layer_modules = self._get_layer_module_map(model_copy)
                selection_mode = self._selection_mode_for_method(method)
                
                try:
                    if method.startswith("cluster_aware"):
                        pipeline_result = self._run_cluster_aware_pruning(
                            model_copy,
                            layer_modules=layer_modules,
                            ratio=ratio,
                            method=method,
                        )
                    else:
                        layer_scores = self._compute_layer_scores_for_method(method, model_copy)
                        if not layer_scores:
                            raise ValueError("No layer scores available for method")

                        pipeline_result = run_pruning_pipeline(
                            model_copy,
                            layer_scores,
                            layer_modules=layer_modules,
                            target_sparsity=ratio,
                            selection_mode=selection_mode,
                            options=pipeline_options,
                        )
                    
                    self._zero_batchnorm_from_masks(model_copy, pipeline_result.get("masks", {}))
                    
                    acc_before = self._evaluate_accuracy(model_copy)
                    acc_after = acc_before
                    if fine_tune_epochs > 0:
                        model_copy = self._fine_tune(
                            model_copy,
                            epochs=fine_tune_epochs,
                            lr=fine_tune_lr,
                            max_batches=fine_tune_max_batches,
                            weight_decay=fine_tune_weight_decay,
                            masks=pipeline_result.get("masks", {}) if isinstance(pipeline_result, dict) else None,
                        )
                        acc_after = self._evaluate_accuracy(model_copy)
                    
                    method_results[ratio] = {
                        "accuracy_before_ft": acc_before,
                        "accuracy_after_ft": acc_after,
                        "accuracy_drop": baseline_acc - acc_before,
                        "accuracy_recovery": acc_after - acc_before if fine_tune_epochs > 0 else 0.0,
                        "selection_mode": selection_mode,
                        "mask_stats": pipeline_result.get("stats", {}),
                    }
                    
                    logger.info("    Result: %.2f%% (drop %.2f%%)", acc_after * 100, (baseline_acc - acc_after) * 100)
                except Exception as exc:
                    import traceback
                    logger.warning("    Pruning failed for %s @ %.0f%%: %s", method, ratio * 100, exc)
                    logger.warning("    Traceback:\n%s", traceback.format_exc())
                    method_results[ratio] = {"error": str(exc)}
                finally:
                    del model_copy
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        self.pruning_results = results
        with open(self.output_dir / "pruning_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    def _get_layer_module_map(self, model: nn.Module) -> Dict[str, nn.Module]:
        modules = dict(model.named_modules())
        return {name: modules.get(name) for name, _ in self.layers if name in modules}

    def _selection_mode_for_method(self, method: str) -> str:
        if method == "random":
            return "random"
        high_methods = {"rq_high", "redundancy_high", "synergy_high", "magnitude_high", "activation_l2_norm_high"}
        if method in high_methods:
            return "high"
        return "low"

    def _compute_taylor_channel_scores(self, model: nn.Module) -> Dict[str, "torch.Tensor"]:
        """
        Compute per-output-channel Taylor saliency scores for each analyzed conv layer.

        Uses weight-based first-order Taylor approximation:
          score_i = sum_d | w_i[d] * grad_w_i[d] |

        Computed over a small calibration subset from self.train_loader.
        """
        if not HAS_TORCH:
            return {}

        # Keep this small by default; configurable via config if present.
        max_samples = int(getattr(self.config, "taylor_samples", 1024))
        max_samples = max(1, max_samples)

        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        model.zero_grad(set_to_none=True)

        n_seen = 0
        for x, y in self.train_loader:
            if n_seen >= max_samples:
                break

            remaining = max_samples - n_seen
            if x.size(0) > remaining:
                x = x[:remaining]
                y = y[:remaining]

            x = x.to(self.device)
            y = y.to(self.device)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            n_seen += int(x.size(0))

        modules = dict(model.named_modules())
        out: Dict[str, "torch.Tensor"] = {}
        for name, _layer in self.layers:
            m = modules.get(name)
            if m is None or not hasattr(m, "weight") or m.weight is None:
                continue
            if m.weight.grad is None:
                continue
            g = m.weight.grad.detach()
            w = m.weight.detach()
            # Reduce to [C_out]
            score = (g * w).abs().view(w.shape[0], -1).sum(dim=1)
            out[name] = score.detach().cpu()

        model.zero_grad(set_to_none=True)
        return out

    def _compute_geometric_median_channel_scores(self, model: nn.Module) -> Dict[str, "torch.Tensor"]:
        """
        Geometric-median (FPGM-style) per-channel importance for Conv layers.

        For each conv layer, treat each output channel filter as a vector and compute
        the geometric median m (Weiszfeld). Channels closest to m are considered
        more redundant; we prune LOW distances.
        """
        if not HAS_TORCH:
            return {}

        # Weiszfeld settings (keep small; this is run once and cached)
        iters = int(getattr(self.config, "geometric_median_iters", 10))
        iters = max(1, min(iters, 50))
        eps = float(getattr(self.config, "geometric_median_eps", 1e-8))
        eps = max(eps, 1e-12)

        modules = dict(model.named_modules())
        out: Dict[str, "torch.Tensor"] = {}
        for name, _layer in self.layers:
            m = modules.get(name)
            if m is None or not hasattr(m, "weight") or m.weight is None:
                continue

            w = m.weight.detach().float().cpu()
            if w.ndim < 2:
                continue
            x = w.view(w.shape[0], -1)  # [C_out, D]
            if x.numel() == 0:
                continue

            # Initialize at the mean
            med = x.mean(dim=0)
            for _ in range(iters):
                d = torch.norm(x - med, p=2, dim=1).clamp_min(eps)  # [C_out]
                inv = 1.0 / d
                med = (inv[:, None] * x).sum(dim=0) / inv.sum()

            # Importance = distance to median (prune low)
            dist = torch.norm(x - med, p=2, dim=1)
            out[name] = dist.detach().cpu()

        return out

    def _compute_hrank_channel_scores(self, model: nn.Module) -> Dict[str, "torch.Tensor"]:
        """
        HRank-style baseline: per-channel average feature-map rank.

        We approximate the rank of each channel's feature map by:
          - adaptive average pooling to (p x p)
          - computing matrix rank via singular values on the pooled map
          - averaging across a small calibration subset

        Channels with LOW average rank are pruned.
        """
        if not HAS_TORCH:
            return {}

        import torch.nn.functional as F

        max_images = int(getattr(self.config, "hrank_images", 256))
        max_images = max(1, max_images)
        pool = int(getattr(self.config, "hrank_pool", 8))
        pool = max(2, min(pool, 32))
        sv_eps = float(getattr(self.config, "hrank_sv_eps", 1e-3))
        sv_eps = max(sv_eps, 1e-6)

        model = model.to(self.device)
        model.eval()

        modules = dict(model.named_modules())

        rank_sum: Dict[str, "torch.Tensor"] = {}
        rank_count: Dict[str, int] = {}

        def _svdvals(x: "torch.Tensor") -> "torch.Tensor":
            # Batched singular values. Fall back gracefully across torch versions.
            try:
                return torch.linalg.svdvals(x)
            except Exception:
                try:
                    # torch.linalg.svd returns U,S,Vh
                    return torch.linalg.svd(x, full_matrices=False).S
                except Exception:
                    # Old fallback
                    return torch.svd(x).S

        def hook_fn(layer_name: str):
            def fn(_m, _inp, out):
                # out: [B,C,H,W]
                try:
                    if out is None or out.ndim != 4:
                        return
                    out_f = out.float()
                    b, c, _h, _w = out_f.shape
                    pooled = F.adaptive_avg_pool2d(out_f, (pool, pool))  # [B,C,p,p]
                    mats = pooled.reshape(b * c, pool, pool)  # [B*C,p,p]

                    sv = _svdvals(mats)  # [B*C,p]
                    thr = sv.max(dim=1).values * sv_eps + 1e-12  # [B*C]
                    r = (sv > thr[:, None]).sum(dim=1).float()  # [B*C]
                    r = r.view(b, c).sum(dim=0).detach().cpu().double()  # [C]

                    if layer_name not in rank_sum:
                        rank_sum[layer_name] = torch.zeros(c, dtype=torch.float64)
                        rank_count[layer_name] = 0
                    rank_sum[layer_name] += r
                    rank_count[layer_name] += int(b)
                except Exception as exc:
                    logger.debug("HRank hook failed for %s (%s)", layer_name, exc)
            return fn

        handles = []
        for name, _layer in self.layers:
            m = modules.get(name)
            if isinstance(m, nn.Conv2d):
                handles.append(m.register_forward_hook(hook_fn(name)))

        n_seen = 0
        with torch.no_grad():
            for x, _y in self.train_loader:
                if n_seen >= max_images:
                    break

                remaining = max_images - n_seen
                if x.size(0) > remaining:
                    x = x[:remaining]

                x = x.to(self.device)

                _ = model(x)

                bsz = int(x.size(0))
                n_seen += bsz

        for h in handles:
            h.remove()

        out_scores: Dict[str, "torch.Tensor"] = {}
        for lname, s in rank_sum.items():
            cnt = int(rank_count.get(lname, 0))
            if cnt <= 0:
                continue
            out_scores[lname] = (s / float(cnt)).float().cpu()

        return out_scores

    def _compute_layer_scores_for_method(self, method: str, model: nn.Module) -> Dict[str, torch.Tensor]:
        layer_scores: Dict[str, torch.Tensor] = {}
        modules = self._get_layer_module_map(model)
        metric_map = {
            "rq_low": "rq",
            "rq_high": "rq",
            "redundancy_low": "redundancy",
            "redundancy_high": "redundancy",
            "synergy_low": "synergy",
            "synergy_high": "synergy",
        }
        for name, layer in modules.items():
            if layer is None or not hasattr(layer, "weight"):
                continue
            weight = layer.weight
            device = weight.device
            metrics = self.layer_metrics.get(name, {})
            n_channels = weight.shape[0]

            if method == "random":
                layer_scores[name] = torch.rand(n_channels, device=device)
            elif method in {"activation_mean", "activation_rms"}:
                values = metrics.get(method)
                if values is None:
                    continue
                layer_scores[name] = torch.as_tensor(values, dtype=torch.float32, device=device)
            elif method in {"magnitude", "magnitude_high", "activation_l2_norm", "activation_l2_norm_high"}:
                w_flat = weight.view(n_channels, -1)
                mags = torch.norm(w_flat, p=2, dim=1)
                layer_scores[name] = mags
            elif method in {"network_slimming", "bn_scale"}:
                # Network Slimming baseline: prune small BN gamma (|gamma|)
                bn = self._find_bn_for_conv(model, name)
                if bn is not None and hasattr(bn, "weight") and bn.weight is not None:
                    gamma = bn.weight.detach().abs()
                    if gamma.numel() != n_channels:
                        logger.warning(
                            "BN gamma size mismatch for %s: gamma=%d, channels=%d; falling back to weight magnitude",
                            name,
                            int(gamma.numel()),
                            int(n_channels),
                        )
                        w_flat = weight.view(n_channels, -1)
                        layer_scores[name] = torch.norm(w_flat, p=2, dim=1)
                    else:
                        layer_scores[name] = gamma.to(device=device, dtype=torch.float32)
                else:
                    # Fallback for layers without BN (rare in these vision backbones)
                    w_flat = weight.view(n_channels, -1)
                    layer_scores[name] = torch.norm(w_flat, p=2, dim=1)
            elif method == "taylor":
                # Gradient-based baseline. Compute once per experiment and cache on CPU.
                if "taylor" not in self._pruning_score_cache:
                    try:
                        self._pruning_score_cache["taylor"] = self._compute_taylor_channel_scores(model)
                    except Exception as exc:
                        logger.warning("Taylor score computation failed (%s); falling back to magnitude", exc)
                        self._pruning_score_cache["taylor"] = {}
                cpu_scores = self._pruning_score_cache.get("taylor", {}).get(name)
                if cpu_scores is None or cpu_scores.numel() != n_channels:
                    # Fallback: weight magnitude if we couldn't compute gradients or mismatch
                    w_flat = weight.view(n_channels, -1)
                    layer_scores[name] = torch.norm(w_flat, p=2, dim=1)
                else:
                    layer_scores[name] = cpu_scores.to(device=device, dtype=torch.float32)
            elif method in {"geometric_median", "fpgm"}:
                cache_key = "geometric_median"
                if cache_key not in self._pruning_score_cache:
                    try:
                        self._pruning_score_cache[cache_key] = self._compute_geometric_median_channel_scores(model)
                    except Exception as exc:
                        logger.warning("Geometric median score computation failed (%s); falling back to magnitude", exc)
                        self._pruning_score_cache[cache_key] = {}
                cpu_scores = self._pruning_score_cache.get(cache_key, {}).get(name)
                if cpu_scores is None or cpu_scores.numel() != n_channels:
                    w_flat = weight.view(n_channels, -1)
                    layer_scores[name] = torch.norm(w_flat, p=2, dim=1)
                else:
                    layer_scores[name] = cpu_scores.to(device=device, dtype=torch.float32)
            elif method == "hrank":
                cache_key = "hrank"
                if cache_key not in self._pruning_score_cache:
                    try:
                        self._pruning_score_cache[cache_key] = self._compute_hrank_channel_scores(model)
                    except Exception as exc:
                        logger.warning("HRank score computation failed (%s); falling back to magnitude", exc)
                        self._pruning_score_cache[cache_key] = {}
                cpu_scores = self._pruning_score_cache.get(cache_key, {}).get(name)
                if cpu_scores is None or cpu_scores.numel() != n_channels:
                    w_flat = weight.view(n_channels, -1)
                    layer_scores[name] = torch.norm(w_flat, p=2, dim=1)
                else:
                    layer_scores[name] = cpu_scores.to(device=device, dtype=torch.float32)
            elif method in metric_map:
                values = metrics.get(metric_map[method])
                if values is None:
                    continue
                layer_scores[name] = torch.as_tensor(values, dtype=torch.float32, device=device)
            elif method in {
                "composite",
                "composite_pos_red",
                "rq_minus_red",
                "rq_plus_red",
                "magnitude_plus_rq",
                "magnitude_minus_red",
                "magnitude_plus_red",
            }:
                comp = self._compute_composite_metric(method, metrics, layer)
                if comp is not None:
                    layer_scores[name] = comp.to(device)
            else:
                logger.warning("Unknown pruning method '%s'; skipping layer scores", method)
                return {}
        return layer_scores

    def _compute_composite_metric(self, method: str, metrics: Dict[str, np.ndarray], layer: nn.Module) -> Optional[torch.Tensor]:
        rq = np.log(np.clip(metrics.get("rq", np.ones(layer.weight.shape[0])), 1e-10, None))
        redundancy = metrics.get("redundancy", np.zeros_like(rq))
        synergy = metrics.get("synergy", np.zeros_like(rq))

        def normalize(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            min_v = arr.min()
            max_v = arr.max()
            if max_v - min_v < 1e-8:
                return np.zeros_like(arr)
            return (arr - min_v) / (max_v - min_v)

        rq_norm = normalize(rq)
        red_norm = normalize(redundancy)
        syn_norm = normalize(synergy)

        if method == "composite":
            scores = rq_norm + 0.5 * syn_norm - 0.3 * red_norm
        elif method == "composite_pos_red":
            scores = rq_norm + 0.5 * syn_norm + 0.3 * red_norm
        elif method == "rq_minus_red":
            scores = rq_norm - 0.5 * red_norm
        elif method == "rq_plus_red":
            scores = rq_norm + 0.5 * red_norm
        elif method == "magnitude_plus_rq":
            w = layer.weight.detach().view(layer.weight.shape[0], -1)
            mag = normalize(w.norm(p=2, dim=1).cpu().numpy())
            scores = mag + 0.5 * rq_norm
        elif method == "magnitude_minus_red":
            w = layer.weight.detach().view(layer.weight.shape[0], -1)
            mag = normalize(w.norm(p=2, dim=1).cpu().numpy())
            scores = mag - 0.3 * red_norm
        elif method == "magnitude_plus_red":
            w = layer.weight.detach().view(layer.weight.shape[0], -1)
            mag = normalize(w.norm(p=2, dim=1).cpu().numpy())
            scores = mag + 0.3 * red_norm
        else:
            return None

        return torch.as_tensor(scores, dtype=torch.float32)

    def _compute_halo_syn_proxy(
        self,
        *,
        layer_name: str,
        layer: nn.Module,
        next_layer: Optional[nn.Module],
        next_layer_name: Optional[str],
        halo_percentile: float,
        use_activation_weight: bool,
    ) -> np.ndarray:
        """
        Compute per-channel HaloSyn proxy without needing raw activations.

        Uses effective influence:
          influence[j,i] = ||W_{j,i}||_1 * sigma_i

        Where sigma_i is approximated from cached RQ and weight norms:
          Var(Y_i) = RQ_i * ||w_i||^2  => sigma_i = sqrt(Var(Y_i))
        """
        metrics = self.layer_metrics.get(layer_name, {})
        rq = np.asarray(metrics.get("rq", np.array([])), dtype=np.float64).reshape(-1)
        if rq.size == 0 or next_layer is None or next_layer_name is None or not hasattr(next_layer, "weight"):
            return np.zeros(int(layer.weight.shape[0]), dtype=np.float64)

        # sigma proxy from rq and weight norms (and BN scaling when present)
        w = layer.weight.detach().view(layer.weight.shape[0], -1).cpu().numpy().astype(np.float64)
        w_norm_sq = np.sum(w * w, axis=1)
        sigma = np.sqrt(np.clip(rq[: len(w_norm_sq)] * w_norm_sq[: len(rq)], 0.0, None))

        bn = self._find_bn_for_conv(self.model, layer_name)
        if bn is not None and hasattr(bn, "weight") and hasattr(bn, "running_var"):
            gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
            rv = bn.running_var.detach().cpu().numpy().astype(np.float64)
            eps = float(getattr(bn, "eps", 1e-5))
            scale = np.abs(gamma) / np.sqrt(rv + eps)
            m = min(len(sigma), len(scale))
            sigma[:m] = sigma[:m] * scale[:m]

        w_next = next_layer.weight.detach().cpu().numpy().astype(np.float64)
        if w_next.ndim == 4:
            influence = np.abs(w_next).sum(axis=(2, 3))  # [out, in]
        elif w_next.ndim == 3:
            influence = np.abs(w_next).sum(axis=2)  # [out, in]
        else:
            influence = np.abs(w_next)

        # Apply activation weighting via sigma_i (effective influence)
        if use_activation_weight:
            n_in = min(influence.shape[1], sigma.shape[0])
            influence[:, :n_in] = influence[:, :n_in] * sigma[:n_in][None, :]

        next_metrics = self.layer_metrics.get(next_layer_name, {}) if next_layer_name else {}
        next_syn = np.asarray(next_metrics.get("synergy", np.array([])), dtype=np.float64).reshape(-1)
        if next_syn.size == 0:
            next_syn = np.zeros(influence.shape[0], dtype=np.float64)
        else:
            next_syn = next_syn[: influence.shape[0]]

        halo_syn = np.zeros(int(layer.weight.shape[0]), dtype=np.float64)
        total_infl = influence.sum(axis=1) + 1e-10
        pct = float(halo_percentile)
        for i in range(min(halo_syn.shape[0], influence.shape[1])):
            rel_infl = influence[:, i] / total_infl
            thresh = np.percentile(rel_infl, pct)
            mask = rel_infl >= thresh
            if mask.sum() > 0:
                halo_syn[i] = float(np.mean(next_syn[mask]))
        return halo_syn

    def _run_cluster_aware_pruning(
        self,
        model: nn.Module,
        *,
        layer_modules: Dict[str, nn.Module],
        ratio: float,
        method: str,
    ) -> Dict[str, Any]:
        """
        Apply cluster-aware pruning using the paper strategy (halo score + constraints).

        Returns a pipeline-like dict with:
          - masks: {layer_name: [C] mask}
          - stats: {layer_name: mask stats}
        Also stores a pruned-by-cluster summary under self.pruning_cluster_distributions.
        """
        from ..pruning.strategies.cluster_aware import ClusterAwarePruning, ClusterAwarePruningConfig
        from ..services.mask_ops import MaskOperations

        # Base config
        cfg = ClusterAwarePruningConfig(amount=float(ratio), structured=True)

        # Variants for ablations / controls
        if method == "cluster_aware_no_halo":
            cfg.lambda_halo = 0.0
        elif method == "cluster_aware_no_constraints":
            cfg.protect_critical_frac = 1.0
            cfg.target_redundant = False
            cfg.synergy_pair_constraint = False
        elif method == "cluster_aware_protect_redundant":
            # Inverted priority (rough proxy): do not preferentially prune redundant/background
            cfg.target_redundant = False
        elif method == "cluster_aware_annealed":
            # Anneal constraints + mix in a strong low-sparsity baseline (Taylor) so we
            # behave like Taylor/Magnitude at low sparsity and like Cluster-aware at high sparsity.
            #
            # anneal_w(r)=0 below start, 1 above end.
            start = float(getattr(self.config, "cluster_aware_anneal_start", 0.70))
            end = float(getattr(self.config, "cluster_aware_anneal_end", 0.90))
            if end <= start:
                end = start + 1e-6
            if ratio <= start:
                w_anneal = 0.0
            elif ratio >= end:
                w_anneal = 1.0
            else:
                w_anneal = float((ratio - start) / (end - start))

            # Constraints: off at low sparsity, on at high sparsity
            base_lambda = float(cfg.lambda_halo)
            base_protect = float(cfg.protect_critical_frac)
            cfg.lambda_halo = base_lambda * w_anneal
            cfg.protect_critical_frac = 1.0 - w_anneal * (1.0 - base_protect)
            cfg.target_redundant = bool(w_anneal >= 0.5)
            cfg.synergy_pair_constraint = bool(w_anneal >= 0.5)

        # Allow paper scripts / SLURM jobs to sweep score weights via config overrides
        cfg.alpha = float(getattr(self.config, "cluster_aware_alpha", cfg.alpha))
        cfg.beta = float(getattr(self.config, "cluster_aware_beta", cfg.beta))
        cfg.gamma = float(getattr(self.config, "cluster_aware_gamma", cfg.gamma))
        cfg.lambda_halo = float(getattr(self.config, "cluster_aware_lambda_halo", cfg.lambda_halo))
        cfg.protect_critical_frac = float(getattr(self.config, "cluster_aware_protect_critical_frac", cfg.protect_critical_frac))

        # Keep halo settings consistent with experiment config unless overridden
        cfg.halo_percentile = float(getattr(self.config, "halo_percentile", cfg.halo_percentile))
        cfg.use_activation_weight = bool(getattr(self.config, "use_activation_weight", cfg.use_activation_weight))
        cfg.n_clusters = int(getattr(self.config, "n_clusters", cfg.n_clusters))

        masks: Dict[str, torch.Tensor] = {}
        stats: Dict[str, Any] = {}

        # Aggregate pruning distribution by cluster type
        by_type_pruned: Dict[str, int] = {}
        by_type_total: Dict[str, int] = {}

        layer_names = [nm for nm, _ in self.layers]
        module_map = dict(model.named_modules())

        # ------------------------------------------------------------------
        # Respect the same pruning distribution knobs as the baseline pipeline.
        #
        # - pruning_distribution controls how much to prune per layer (uniform,
        #   global_threshold, size_proportional, importance_weighted, ...)
        # - pruning_{min,max}_per_layer bound the per-layer amounts
        #
        # For score-dependent strategies (global_threshold / importance_weighted),
        # we compute the per-layer cluster-aware scores first, then allocate per
        # layer amounts from those scores.
        # ------------------------------------------------------------------
        distribution = getattr(self.config, "pruning_distribution", "uniform")
        min_amount = float(getattr(self.config, "pruning_min_per_layer", 0.0))
        max_amount = float(getattr(self.config, "pruning_max_per_layer", 0.95))

        # First pass: compute per-layer cluster-aware scores (no pruning yet)
        layer_scores: Dict[str, torch.Tensor] = {}
        layer_pruners: Dict[str, "ClusterAwarePruning"] = {}
        layer_num_channels: Dict[str, int] = {}

        for idx, layer_name in enumerate(layer_names):
            layer = module_map.get(layer_name)
            if layer is None or not hasattr(layer, "weight") or layer.weight is None:
                continue

            n_channels = int(layer.weight.shape[0])
            layer_num_channels[layer_name] = n_channels

            # Pick the next *weight-connected* layer by matching channel dimensions (same logic as halo analysis).
            src_out = int(layer.weight.shape[0])
            next_layer_name = None
            for j in range(idx + 1, len(layer_names)):
                cand_name = layer_names[j]
                cand_layer = module_map.get(cand_name)
                if cand_layer is None or not hasattr(cand_layer, "weight"):
                    continue
                w = cand_layer.weight
                if w is None or w.ndim < 2:
                    continue
                if int(w.shape[1]) == src_out:
                    next_layer_name = cand_name
                    break
            next_layer = module_map.get(next_layer_name) if next_layer_name else None

            # Cached metrics + clusters from the original (unpruned) analysis
            pre_metrics = self.layer_metrics.get(layer_name, {})
            pre_clusters = self.cluster_results.get(layer_name, {})

            labels = np.asarray(pre_clusters.get("labels", np.zeros(n_channels, dtype=int))).astype(int)
            type_mapping = pre_clusters.get("type_mapping", {})

            # HaloSyn proxy (uses sigma from RQ and next-layer synergy)
            halo_syn = self._compute_halo_syn_proxy(
                layer_name=layer_name,
                layer=layer,
                next_layer=next_layer,
                next_layer_name=next_layer_name,
                halo_percentile=cfg.halo_percentile,
                use_activation_weight=cfg.use_activation_weight,
            )

            pruner = ClusterAwarePruning(
                cfg,
                precomputed_metrics=pre_metrics,
                precomputed_clusters={"labels": labels, "type_mapping": type_mapping},
                precomputed_halos={"halo_syn": halo_syn},
            )

            scores = pruner.compute_importance_scores(
                layer,
                outputs=None,  # halo syn is precomputed
                next_layer_weights=next_layer.weight if next_layer is not None else None,
                next_layer_metrics=self.layer_metrics.get(next_layer_name, {}) if next_layer_name else None,
                layer_name=layer_name,
            )

            # Optional annealed mixing: blend cluster-aware score with Taylor at low sparsity.
            if method == "cluster_aware_annealed":
                # Ensure Taylor cache exists (computed once; reused across ratios/methods).
                if "taylor" not in self._pruning_score_cache:
                    try:
                        self._pruning_score_cache["taylor"] = self._compute_taylor_channel_scores(self.model)
                    except Exception:
                        self._pruning_score_cache["taylor"] = {}

                t_cpu = (self._pruning_score_cache.get("taylor", {}) or {}).get(layer_name)
                if t_cpu is None or (hasattr(t_cpu, "numel") and int(t_cpu.numel()) != int(n_channels)):
                    # Fallback to weight magnitude if Taylor is unavailable/mismatched
                    w_flat = layer.weight.detach().view(n_channels, -1)
                    t = w_flat.norm(p=2, dim=1).detach().cpu()
                else:
                    t = t_cpu.detach().cpu()

                # Normalize both to [0,1] per-layer for stable mixing
                def _minmax(x: "torch.Tensor") -> "torch.Tensor":
                    x = x.float()
                    if x.numel() == 0:
                        return x
                    mn = float(x.min().item())
                    mx = float(x.max().item())
                    if mx - mn < 1e-12:
                        return torch.zeros_like(x)
                    return (x - mn) / (mx - mn)

                s_ca = _minmax(scores.detach().cpu())
                s_t = _minmax(t)

                start = float(getattr(self.config, "cluster_aware_anneal_start", 0.70))
                end = float(getattr(self.config, "cluster_aware_anneal_end", 0.90))
                if end <= start:
                    end = start + 1e-6
                if ratio <= start:
                    w_anneal = 0.0
                elif ratio >= end:
                    w_anneal = 1.0
                else:
                    w_anneal = float((ratio - start) / (end - start))

                mixed = (1.0 - w_anneal) * s_t + w_anneal * s_ca
                scores = mixed.to(device=scores.device)

            layer_scores[layer_name] = scores.detach()
            layer_pruners[layer_name] = pruner

        # Compute per-layer amounts using the shared distribution manager.
        try:
            from ..pruning.distribution import PruningDistributionManager

            manager = PruningDistributionManager(
                strategy=str(distribution),
                target_sparsity=float(ratio),
                min_amount=float(min_amount),
                max_amount=float(max_amount),
            )
            # Only include layers we actually scored
            scored_names = [nm for nm in layer_names if nm in layer_scores]
            per_layer_amounts = manager.compute_distribution(model, scored_names, layer_scores=layer_scores)
        except Exception as exc:
            logger.warning(
                "Cluster-aware pruning: failed to compute distribution '%s' (%s); falling back to uniform",
                distribution,
                exc,
            )
            clipped = max(min_amount, min(max_amount, float(ratio)))
            per_layer_amounts = {nm: clipped for nm in layer_scores.keys()}

        # Second pass: apply pruning using per-layer allocated amounts
        for layer_name in layer_names:
            layer = module_map.get(layer_name)
            if layer is None or not hasattr(layer, "weight") or layer.weight is None:
                continue
            if layer_name not in layer_scores or layer_name not in layer_pruners:
                continue

            n_channels = int(layer_num_channels.get(layer_name, layer.weight.shape[0]))
            amount = float(per_layer_amounts.get(layer_name, float(ratio)))
            n_prune = int(n_channels * amount)
            if n_prune <= 0:
                masks[layer_name] = torch.ones(n_channels, dtype=torch.bool, device=layer.weight.device)
                stats[layer_name] = MaskOperations.get_mask_statistics(masks[layer_name])
                continue

            # Cached clusters from the original (unpruned) analysis (for by-type summaries)
            pre_clusters = self.cluster_results.get(layer_name, {})

            labels = np.asarray(pre_clusters.get("labels", np.zeros(n_channels, dtype=int))).astype(int)
            type_mapping = pre_clusters.get("type_mapping", {})

            pruner = layer_pruners[layer_name]
            scores = layer_scores[layer_name].to(device=layer.weight.device)
            prune_idx = pruner.select_channels_to_prune(scores, n_prune, layer_name=layer_name)

            mask = torch.ones(n_channels, dtype=torch.bool, device=layer.weight.device)
            if prune_idx:
                mask[torch.as_tensor(prune_idx, device=layer.weight.device)] = False

                with torch.no_grad():
                    layer.weight.data[~mask] = 0.0
                    if getattr(layer, "bias", None) is not None and layer.bias.data.numel() == n_channels:
                        layer.bias.data[~mask] = 0.0

            masks[layer_name] = mask
            stats[layer_name] = MaskOperations.get_mask_statistics(mask)

            # Update by-type counts for diagnostics/figures
            # Trim labels if necessary
            labels = labels[: min(len(labels), n_channels)]
            for cid, ctype in type_mapping.items():
                cid_int = int(cid)
                idxs = np.where(labels == cid_int)[0]
                by_type_total[ctype] = by_type_total.get(ctype, 0) + int(len(idxs))
                if len(idxs) > 0:
                    pruned = int((~mask.detach().cpu().numpy().astype(bool))[idxs].sum())
                    by_type_pruned[ctype] = by_type_pruned.get(ctype, 0) + pruned

        # Store summary for paper figures
        self.pruning_cluster_distributions.setdefault(method, {})
        self.pruning_cluster_distributions[method][float(ratio)] = {
            "pruned": by_type_pruned,
            "total": by_type_total,
        }

        return {"masks": masks, "stats": stats}

    def _zero_batchnorm_from_masks(self, model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
        for layer_name, mask in masks.items():
            bn_layer = self._find_bn_for_conv(model, layer_name)
            if bn_layer is None or not hasattr(bn_layer, "weight"):
                continue
            mask_bool = mask.to(bn_layer.weight.device).bool()
            if mask_bool.numel() != bn_layer.weight.data.numel():
                continue
            with torch.no_grad():
                bn_layer.weight.data[~mask_bool] = 0.0
                if getattr(bn_layer, "bias", None) is not None:
                    bn_layer.bias.data[~mask_bool] = 0.0
                if hasattr(bn_layer, "running_mean"):
                    bn_layer.running_mean.data[~mask_bool] = 0.0
                if hasattr(bn_layer, "running_var"):
                    bn_layer.running_var.data[~mask_bool] = 1.0
    
    def _apply_pruning(self, model: nn.Module, method: str, ratio: float) -> nn.Module:
        """
        Apply a specific pruning method.
        
        Supported methods:
        
        BASELINE:
        - 'random': Random channel selection
        - 'magnitude': Prune lowest activation magnitude (standard baseline)
        - 'taylor': Prune by gradient-based importance
        
        SINGLE METRICS (prune LOW values = assume low is unimportant):
        - 'rq_low': Prune channels with lowest Rayleigh Quotient
        - 'redundancy_low': Prune channels with lowest redundancy (MI)
        - 'synergy_low': Prune channels with lowest synergy
        
        SINGLE METRICS (prune HIGH values = assume high is unimportant):
        - 'rq_high': Prune channels with highest RQ
        - 'redundancy_high': Prune channels with highest redundancy
        - 'synergy_high': Prune channels with highest synergy
        - 'magnitude_high': Prune highest magnitude channels
        
        COMPOSITE COMBINATIONS:
        - 'composite': Original formula: score = RQ + syn - red (prune low)
        - 'composite_pos_red': Flipped: score = RQ + syn + red (prune low)
        - 'rq_minus_red': score = RQ - redundancy (prune low)
        - 'rq_plus_red': score = RQ + redundancy (prune low)
        - 'magnitude_plus_rq': score = magnitude + RQ (prune low)
        
        CLUSTER-AWARE:
        - 'cluster_aware': Cluster-constrained pruning (targets redundant cluster)
        - 'cluster_aware_protect_redundant': Inverted (protects redundant, targets critical)
        """
        model = model.to(self.device)
        pruner = None  # Will use metric-based pruning for most methods
        
        if method == 'random':
            from ..pruning.strategies import RandomPruning
            from ..pruning.base import PruningConfig
            pruner = RandomPruning(PruningConfig(amount=ratio, structured=True))
            
        elif method == 'magnitude':
            from ..pruning.strategies import MagnitudePruning
            from ..pruning.base import PruningConfig
            pruner = MagnitudePruning(PruningConfig(amount=ratio, structured=True))
            
        elif method == 'taylor':
            # Taylor pruning needs gradients from a backward pass. In this
            # analysis-only flow we do not run backward, so running Taylor here
            # would be misleading. Fail fast so results clearly mark it unusable.
            raise ValueError("Taylor pruning requires gradients; not available in analysis-only mode.")
            
        elif method == 'composite':
            from ..pruning.strategies import CompositePruning, ClusterAwarePruningConfig
            config = ClusterAwarePruningConfig(amount=ratio)
            pruner = CompositePruning(config)
            
        elif method == 'cluster_aware':
            from ..pruning.strategies import ClusterAwarePruning, ClusterAwarePruningConfig
            config = ClusterAwarePruningConfig(amount=ratio)
            pruner = ClusterAwarePruning(
                config,
                precomputed_metrics=None,
                precomputed_clusters=None,
            )
            
        elif method == 'cluster_aware_protect_redundant':
            # Inverted cluster-aware: protect redundant, target critical
            from ..pruning.strategies import ClusterAwarePruning, ClusterAwarePruningConfig
            config = ClusterAwarePruningConfig(
                amount=ratio,
                target_redundant=False,  # Don't target redundant
                protect_critical_frac=1.0,  # Don't protect critical
            )
            pruner = ClusterAwarePruning(config)
        
        # Apply to each conv layer in the COPIED model (not self.model!)
        # Get layer references from the passed model, not self.layers
        model_modules = dict(model.named_modules())
        
        for name, orig_layer in self.layers:
            # Get the corresponding layer from model_copy, not self.model
            if name not in model_modules:
                logger.debug(f"    {name}: not found in model copy, skipping")
                continue
            layer = model_modules[name]
            
            if not hasattr(layer, 'weight'):
                continue
            
            n_channels = layer.weight.shape[0]
            n_prune = int(n_channels * ratio)
            if n_prune == 0:
                continue
            
            # Get cached metrics for this layer (from original model analysis)
            metrics = self.layer_metrics.get(name, {})
            clusters = self.cluster_results.get(name, {})
            
            # Debug: log if metrics are missing
            if not metrics:
                logger.warning(f"    {name}: NO METRICS CACHED! Using defaults (will select channels 0,1,2...)")
            elif 'rq' not in metrics:
                logger.warning(f"    {name}: 'rq' not in metrics. Keys: {list(metrics.keys())}")
            
            try:
                # ================================================================
                # METRIC-BASED PRUNING (single metrics and combinations)
                # ================================================================
                if method.startswith('rq_') or method.startswith('redundancy_') or \
                   method.startswith('synergy_') or method.startswith('magnitude_') or \
                   method.startswith('composite') or method.startswith('rq_'):
                    
                    # Get metric arrays
                    rq = np.array(metrics.get('rq', np.ones(n_channels)))
                    redundancy = np.array(metrics.get('redundancy', np.zeros(n_channels)))
                    synergy = np.array(metrics.get('synergy', np.zeros(n_channels)))
                    
                    # Compute magnitude from activations if available
                    acts = metrics.get('_activations', None)
                    if acts is not None:
                        magnitude = np.mean(np.abs(acts), axis=0)
                    else:
                        # Use weight L2 norm as proxy
                        w = layer.weight.data.cpu().numpy()
                        magnitude = np.sqrt(np.sum(w.reshape(w.shape[0], -1)**2, axis=1))
                    
                    # Normalize metrics to [0, 1] for stable combination
                    def normalize(x):
                        x_min, x_max = x.min(), x.max()
                        if x_max > x_min:
                            return (x - x_min) / (x_max - x_min)
                        return np.zeros_like(x)
                    
                    rq_norm = normalize(np.log(np.clip(rq, 1e-10, None)))
                    red_norm = normalize(redundancy)
                    syn_norm = normalize(synergy)
                    mag_norm = normalize(magnitude)
                    
                    # Compute scores based on method
                    # SINGLE METRICS - prune LOW
                    if method == 'rq_low':
                        scores = rq_norm  # Low RQ → prune
                    elif method == 'redundancy_low':
                        scores = red_norm  # Low redundancy → prune
                    elif method == 'synergy_low':
                        scores = syn_norm  # Low synergy → prune
                    
                    # SINGLE METRICS - prune HIGH
                    elif method == 'rq_high':
                        scores = -rq_norm  # High RQ → prune (invert)
                    elif method == 'redundancy_high':
                        scores = -red_norm  # High redundancy → prune
                    elif method == 'synergy_high':
                        scores = -syn_norm  # High synergy → prune
                    elif method == 'magnitude_high':
                        scores = -mag_norm  # High magnitude → prune
                    
                    # COMPOSITE COMBINATIONS
                    elif method == 'composite':
                        # Original: High RQ + High Syn - High Red = important
                        # Prune LOW scores
                        scores = rq_norm + 0.5 * syn_norm - 0.3 * red_norm
                    elif method == 'composite_pos_red':
                        # Flipped: High RQ + High Syn + High Red = important
                        scores = rq_norm + 0.5 * syn_norm + 0.3 * red_norm
                    elif method == 'rq_minus_red':
                        scores = rq_norm - 0.5 * red_norm
                    elif method == 'rq_plus_red':
                        scores = rq_norm + 0.5 * red_norm
                    elif method == 'magnitude_plus_rq':
                        scores = mag_norm + 0.5 * rq_norm
                    elif method == 'magnitude_minus_red':
                        scores = mag_norm - 0.3 * red_norm
                    elif method == 'magnitude_plus_red':
                        scores = mag_norm + 0.3 * red_norm
                    else:
                        raise ValueError(f"Unknown metric-based method: {method}")
                    
                    # Select lowest scores to prune
                    scores_tensor = torch.from_numpy(scores).float()
                    prune_idx = torch.argsort(scores_tensor)[:n_prune].tolist()
                    
                    # Debug: check if all channels have same score (bug indicator)
                    score_range = scores.max() - scores.min()
                    if score_range < 1e-8:
                        logger.warning(f"    {name}: ALL SCORES ARE IDENTICAL ({scores[0]:.6f})! Selecting channels {prune_idx[:5]}...")
                
                # ================================================================
                # CLUSTER-AWARE PRUNING
                # ================================================================
                elif method in ['cluster_aware', 'cluster_aware_protect_redundant']:
                    pruner.precomputed_metrics = metrics
                    pruner.precomputed_clusters = clusters
                    pruner._metrics_cache[name] = metrics
                    pruner._cluster_cache[name] = clusters
                    
                    scores = pruner.compute_importance_scores(layer, layer_name=name)
                    prune_idx = pruner.select_channels_to_prune(scores, n_prune, name)
                
                # ================================================================
                # COMPOSITE PRUNING (using pruner class)
                # ================================================================
                elif method == 'composite_class':
                    pruner.precomputed_metrics = metrics
                    pruner.precomputed_clusters = clusters
                    pruner._metrics_cache[name] = metrics
                    pruner._cluster_cache[name] = clusters
                    
                    scores = pruner.compute_importance_scores(layer, layer_name=name)
                    prune_idx = torch.argsort(scores)[:n_prune].tolist()
                
                # ================================================================
                # STANDARD PRUNERS (random, magnitude, taylor)
                # ================================================================
                elif pruner is not None:
                    # Get weight-level importance scores
                    weight_scores = pruner.compute_importance_scores(layer, layer_name=name)
                    
                    # Convert to channel-level scores by averaging over non-channel dims
                    # weight_scores shape: [C_out, C_in, k, k] for Conv2d
                    if len(weight_scores.shape) == 4:
                        # Average over input channels and kernel dims
                        channel_scores = weight_scores.abs().mean(dim=(1, 2, 3))
                    elif len(weight_scores.shape) == 2:
                        # Linear: [out, in] -> average over input dim
                        channel_scores = weight_scores.abs().mean(dim=1)
                    else:
                        # Fallback: flatten and take first n_channels
                        channel_scores = weight_scores.view(n_channels, -1).abs().mean(dim=1)
                    
                    prune_idx = torch.argsort(channel_scores)[:n_prune].tolist()
                    logger.debug(f"    {name}: pruner {method}, channel scores range [{channel_scores.min():.4f}, {channel_scores.max():.4f}]")
                
                # ================================================================
                # TAYLOR PRUNING (gradient-weight product)
                # ================================================================
                elif method == 'taylor':
                    # Taylor needs gradients - compute them on the fly
                    # Use weight L2 norm as importance (magnitude-based fallback)
                    # since we don't have gradients readily available
                    w = layer.weight.data
                    if len(w.shape) == 4:
                        # Conv: L2 norm per output channel
                        channel_scores = w.pow(2).sum(dim=(1, 2, 3)).sqrt()
                    else:
                        channel_scores = w.pow(2).sum(dim=1).sqrt()
                    
                    prune_idx = torch.argsort(channel_scores)[:n_prune].tolist()
                    logger.debug(f"    {name}: taylor (magnitude fallback), scores range [{channel_scores.min():.4f}, {channel_scores.max():.4f}]")
                
                else:
                    raise ValueError(f"Unknown pruning method: {method}")
                
                # Zero out pruned channels in conv layer
                with torch.no_grad():
                    layer.weight.data[prune_idx] = 0
                    if layer.bias is not None:
                        layer.bias.data[prune_idx] = 0
                
                # Also zero corresponding BatchNorm parameters
                bn_layer = self._find_bn_for_conv(model, name)
                if bn_layer is not None:
                    with torch.no_grad():
                        bn_layer.weight.data[prune_idx] = 0
                        bn_layer.bias.data[prune_idx] = 0
                        bn_layer.running_mean.data[prune_idx] = 0
                        bn_layer.running_var.data[prune_idx] = 1  # Avoid div by zero
                
                # Verify pruning: count zeroed channels
                n_zeroed = (layer.weight.data.view(n_channels, -1).abs().sum(dim=1) == 0).sum().item()
                logger.debug(f"    {name}: pruned {n_prune} channels, verified {n_zeroed} are zeroed")
                        
            except Exception as e:
                logger.debug(f"Pruning {name} with {method} failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        return model
    
    def _find_bn_for_conv(self, model: nn.Module, conv_name: str) -> Optional[nn.Module]:
        """
        Find the BatchNorm layer that corresponds to a conv layer.
        
        In standard architectures (ResNet, VGG-BN), BN follows conv with naming like:
        - conv1 -> bn1
        - layer1.0.conv1 -> layer1.0.bn1
        """
        modules = dict(model.named_modules())
        
        # Try common naming patterns
        patterns = [
            conv_name.replace('conv', 'bn'),  # conv1 -> bn1
            conv_name.replace('.conv', '.bn'),  # layer1.0.conv1 -> layer1.0.bn1
            conv_name + '_bn',  # some architectures
        ]
        
        for pattern in patterns:
            if pattern in modules:
                bn = modules[pattern]
                if isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    return bn
        
        # For downsample layers: layer2.0.downsample.0 -> layer2.0.downsample.1
        if 'downsample.0' in conv_name:
            bn_name = conv_name.replace('downsample.0', 'downsample.1')
            if bn_name in modules:
                bn = modules[bn_name]
                if isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    return bn

        # Generic Sequential convention: Conv at index i, BN at index i+1
        # Covers VGG16-BN (features.0 -> features.1) and MobileNetV2 (....0 -> ....1).
        parts = conv_name.split(".")
        if parts and parts[-1].isdigit():
            try:
                i = int(parts[-1])
                cand = ".".join(parts[:-1] + [str(i + 1)])
                bn = modules.get(cand)
                if isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    return bn
            except Exception:
                pass
        
        return None
    
    def _fine_tune(
        self,
        model: nn.Module,
        epochs: int,
        lr: float,
        max_batches: Optional[int] = None,
        weight_decay: float = 0.0,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> nn.Module:
        """Fine-tune a pruned model.

        Important: when fine-tuning after structured pruning, we must keep pruned
        channels pruned. We do this by re-applying channel masks after each
        optimizer step (and keeping the corresponding BatchNorm params zeroed).
        """
        import torch.optim as optim
        
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay or 0.0))
        criterion = nn.CrossEntropyLoss()

        module_map: Dict[str, nn.Module] = dict(model.named_modules())
        masks_dev: Dict[str, torch.Tensor] = {}
        bn_map: Dict[str, nn.Module] = {}

        if masks:
            for layer_name, mask in masks.items():
                m = module_map.get(layer_name)
                if m is None or not hasattr(m, "weight") or getattr(m, "weight", None) is None:
                    continue
                try:
                    mb = mask.to(m.weight.device).bool()
                    if mb.numel() != int(m.weight.shape[0]):
                        continue
                    masks_dev[layer_name] = mb
                    bn = self._find_bn_for_conv(model, layer_name)
                    if bn is not None:
                        bn_map[layer_name] = bn
                except Exception:
                    continue

        def _reapply_masks() -> None:
            if not masks_dev:
                return
            with torch.no_grad():
                for layer_name, mb in masks_dev.items():
                    m = module_map.get(layer_name)
                    if m is None or not hasattr(m, "weight") or getattr(m, "weight", None) is None:
                        continue
                    if mb.numel() != int(m.weight.shape[0]):
                        continue

                    # Zero pruned output channels
                    m.weight.data[~mb] = 0.0
                    if getattr(m, "bias", None) is not None and m.bias.data.numel() == mb.numel():
                        m.bias.data[~mb] = 0.0

                    # Keep matched BatchNorm channels zeroed too (when present)
                    bn = bn_map.get(layer_name)
                    if bn is None or not hasattr(bn, "weight") or getattr(bn, "weight", None) is None:
                        continue
                    if mb.numel() != bn.weight.data.numel():
                        continue
                    bn.weight.data[~mb] = 0.0
                    if getattr(bn, "bias", None) is not None:
                        bn.bias.data[~mb] = 0.0
                    if hasattr(bn, "running_mean"):
                        bn.running_mean.data[~mb] = 0.0
                    if hasattr(bn, "running_var"):
                        bn.running_var.data[~mb] = 1.0
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                _reapply_masks()
                
                total_loss += loss.item()
                n_batches += 1
                if max_batches is not None and n_batches >= int(max_batches):
                    break
            
            if epoch == 0 or (epoch + 1) % 5 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                logger.debug(f"    FT epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        model.eval()
        return model
    
    def _evaluate_accuracy(self, model: Optional[nn.Module] = None) -> float:
        """Evaluate model accuracy on test set."""
        model = model or self.model
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def run_full_analysis(self, include_pruning: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            include_pruning: Whether to run pruning comparison experiments
        """
        logger.info(f"Starting full analysis for {self.config.model_name}")
        
        # 1. Compute metrics
        self.compute_metrics()
        
        # 2. Clustering
        self.run_clustering()
        
        # 3. Halo analysis
        self.run_halo_analysis()
        
        # 4. Cascade test
        self.run_cascade_test()
        
        # 5. Pruning experiments (optional)
        if include_pruning and getattr(self.config, 'pruning_ratios', None):
            # Check if fine-tuning is enabled
            fine_tune_enabled = getattr(self.config, 'fine_tune_after_pruning', True)
            fine_tune_epochs = getattr(self.config, 'fine_tune_epochs', 10) if fine_tune_enabled else 0
            fine_tune_lr = getattr(self.config, 'fine_tune_lr', 0.0001)
            fine_tune_max_batches = getattr(self.config, "fine_tune_max_batches", None)
            fine_tune_weight_decay = float(getattr(self.config, "fine_tune_weight_decay", 0.0) or 0.0)
            
            logger.info(f"Fine-tuning after pruning: {'enabled' if fine_tune_epochs > 0 else 'disabled'}")
            
            self.run_pruning_experiments(
                ratios=self.config.pruning_ratios,
                methods=getattr(self.config, "pruning_methods", None),
                fine_tune_epochs=fine_tune_epochs,
                fine_tune_lr=fine_tune_lr,
                fine_tune_max_batches=fine_tune_max_batches,
                fine_tune_weight_decay=fine_tune_weight_decay,
            )
        
        # Save results (including centroids for visualization)
        results = {
            "config": {
                "model_name": self.config.model_name,
                "dataset_name": self.config.dataset_name,
                "n_clusters": self.config.n_clusters,
                "activation_samples": getattr(self.config, "activation_samples", "flatten_spatial"),
                "spatial_samples_per_image": getattr(self.config, "spatial_samples_per_image", 16),
                "seed": getattr(self.config, "seed", 42),
            },
            "layer_metrics": self.layer_metrics,
            "cluster_results": {
                k: {
                    "labels": v["labels"].tolist() if hasattr(v.get("labels", None), "tolist") else v.get("labels", []),
                    "type_counts": v["type_counts"],
                    "silhouette": v["silhouette"],
                    "centroids": v["centroids"].tolist() if hasattr(v["centroids"], 'tolist') else v["centroids"],
                    "type_mapping": {str(kk): vv for kk, vv in v["type_mapping"].items()},
                }
                for k, v in self.cluster_results.items() if not k.startswith("_")
            },
            "halo_results": self.halo_results,
            "halo_flow_results": self.halo_flow_results,
            "permutation_results": getattr(self, 'permutation_results', {}),
            "ablation_results": self.cluster_results.get("_ablation", {}),
            "cascade_results": self.cascade_results,
            "pruning_results": getattr(self, 'pruning_results', {}),
            "pruning_cluster_distributions": getattr(self, "pruning_cluster_distributions", {}),
        }
        
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.output_dir}")
        return results
    
    def run(self) -> Dict[str, Any]:
        """
        Standard run method for compatibility with run_experiment.py.
        
        This is the main entry point when running via:
            python scripts/run_experiment.py --config configs/vision/resnet18_cifar10_full.yaml
        """
        # Run full analysis
        results = self.run_full_analysis()
        
        # Generate figures
        self.generate_figures()
        
        return results
    
    def generate_figures(self) -> None:
        """Generate all visualization figures using centralized visualization module."""
        # Import visualization functions from the unified module
        from ..analysis.visualization.cluster_plots import (
            plot_metric_scatter,
            plot_cluster_evolution,
            plot_metric_scatter_3d,
            plot_influence_matrix,
            plot_cascade_test,
            plot_halo_properties,
            plot_pruning_comparison,
            plot_pruning_by_cluster_type,
            plot_centroid_evolution,
            plot_centroid_depth_profiles,
            plot_metric_distributions_for_layer,
            plot_layer_metric_summary,
            plot_layer_metric_trends,
            plot_metric_statistics_table,
        )
        from ..analysis.visualization.metric_plots import (
            plot_metric_histogram,
            plot_metric_violin,
            plot_metric_correlation_heatmap,
            plot_top_neurons_bar,
        )
        from ..analysis.visualization.pruning_plots import (
            plot_pruning_recovery_chart,
            plot_pruning_accuracy_loss_grid,
            plot_pruning_bar_comparison,
            plot_pruning_heatmap,
            plot_pruning_ranking,
        )
        
        # Determine figures directory - check both new "figures" and old "plots" subdirectories
        if (self.output_dir / "figures").exists():
            fig_dir = self.output_dir / "figures"
        elif (self.output_dir / "plots").exists():
            fig_dir = self.output_dir / "plots"
        else:
            fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True, parents=True)

        # Helper: keep backward-compatible root-level copies for paper scripts
        # while also writing into organized subfolders.
        try:
            import shutil

            def _copy_legacy(src: "Path", dst: "Path") -> None:
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

        except Exception:
            shutil = None  # type: ignore

            def _copy_legacy(_src: "Path", _dst: "Path") -> None:
                return
        
        # Create organized subdirectories
        distributions_dir = fig_dir / "01_distributions"
        distributions_dir.mkdir(exist_ok=True)
        
        summary_dir = fig_dir / "02_summary"
        summary_dir.mkdir(exist_ok=True)
        
        clustering_dir = fig_dir / "03_clustering"
        clustering_dir.mkdir(exist_ok=True)
        
        cascade_dir = fig_dir / "04_cascade"
        cascade_dir.mkdir(exist_ok=True)
        
        halo_dir = fig_dir / "05_halo"
        halo_dir.mkdir(exist_ok=True)
        
        pruning_dir = fig_dir / "06_pruning"
        pruning_dir.mkdir(exist_ok=True)
        
        # ==================================================================
        # 1. Metric Distributions (Histograms) - NEW
        # ==================================================================
        logger.info("Generating metric distribution plots...")
        for name, metrics in self.layer_metrics.items():
            safe_name = name.replace('.', '_')
            
            # Combined histogram for all metrics in this layer
            plot_metric_distributions_for_layer(
                metrics=metrics,
                layer_name=name,
                save_dir=distributions_dir,
            )
            
            # Individual histograms with percentile highlighting
            for metric_name in ['rq', 'redundancy', 'synergy']:
                if metric_name in metrics:
                    plot_metric_histogram(
                        values=metrics[metric_name],
                        metric_name=metric_name,
                        layer_name=name,
                        highlight_percentile=95,
                        log_scale=(metric_name == 'rq'),
                        save_path=distributions_dir / f"{metric_name}_{safe_name}.png",
                    )
        
        # ==================================================================
        # 2. Layer-wise Violin/Boxplots for each metric
        # ==================================================================
        logger.info("Generating layer-wise metric plots...")
        for metric_name in ['rq', 'redundancy', 'synergy']:
            layer_data = {
                name: metrics.get(metric_name, np.array([]))
                for name, metrics in self.layer_metrics.items()
                if metric_name in metrics
            }
            if layer_data:
                plot_metric_violin(
                    layer_metrics=layer_data,
                    metric_name=metric_name,
                    save_path=summary_dir / f"{metric_name}_violin_all_layers.png",
                )
        
        # ==================================================================
        # 3. Metric Correlation Heatmap per layer
        # ==================================================================
        for name, metrics in self.layer_metrics.items():
            if len(metrics) >= 2:
                safe_name = name.replace('.', '_')
                plot_metric_correlation_heatmap(
                    metrics=metrics,
                    layer_name=name,
                    save_path=distributions_dir / f"correlation_{safe_name}.png",
                )
        
        # ==================================================================
        # 4. Layer Metric Summary (overview of all layers and metrics)
        # ==================================================================
        if self.layer_metrics:
            # Original heatmap-style summary
            _p = summary_dir / "layer_metric_summary.png"
            plot_layer_metric_summary(
                layer_metrics=self.layer_metrics,
                save_path=_p,
            )
            _copy_legacy(_p, fig_dir / "layer_metric_summary.png")
            
            # NEW: Smoother trend plots with confidence intervals
            _p = summary_dir / "layer_metric_trends.png"
            plot_layer_metric_trends(
                layer_metrics=self.layer_metrics,
                metrics_to_plot=['rq', 'redundancy', 'synergy'],
                smooth_window=3,  # Moving average over 3 layers
                show_ci=True,
                ci_percentile=95,
                save_path=_p,
            )
            _copy_legacy(_p, fig_dir / "layer_metric_trends.png")
            
            # NEW: Statistics table for paper/report
            _p = summary_dir / "metric_statistics_table.png"
            plot_metric_statistics_table(
                layer_metrics=self.layer_metrics,
                save_path=_p,
            )
            _copy_legacy(_p, fig_dir / "metric_statistics_table.png")
        
        # ==================================================================
        # 5. Cluster scatter for each layer
        # ==================================================================
        logger.info("Generating cluster scatter plots...")
        for name, metrics in self.layer_metrics.items():
            cluster = self.cluster_results.get(name, {})
            if not cluster:
                continue
            plot_metric_scatter(
                metrics["rq"],
                metrics["redundancy"],
                metrics["synergy"],
                cluster["labels"],
                cluster["type_mapping"],
                name,
                clustering_dir / f"cluster_scatter_{name.replace('.', '_')}.png",
            )
            _copy_legacy(
                clustering_dir / f"cluster_scatter_{name.replace('.', '_')}.png",
                fig_dir / f"cluster_scatter_{name.replace('.', '_')}.png",
            )

        # Representative 3D scatter for the paper (best-effort)
        try:
            if self.cluster_results and self.layer_metrics:
                rep_layer = None
                for candidate in self.cluster_results.keys():
                    if "layer3" in candidate or "layer4" in candidate:
                        rep_layer = candidate
                        break
                if rep_layer is None:
                    rep_layer = list(self.cluster_results.keys())[len(self.cluster_results) // 2]

                rep_cluster = self.cluster_results.get(rep_layer, {})
                rep_metrics = self.layer_metrics.get(rep_layer, {})
                if rep_cluster and rep_metrics:
                    _p = clustering_dir / "cluster_3d_scatter.png"
                    plot_metric_scatter_3d(
                        rq=rep_metrics.get("rq", np.array([])),
                        redundancy=rep_metrics.get("redundancy", np.array([])),
                        synergy=rep_metrics.get("synergy", np.array([])),
                        labels=rep_cluster.get("labels", np.array([])),
                        type_mapping=rep_cluster.get("type_mapping", {}),
                        layer_name=rep_layer,
                        save_path=_p,
                    )
                    _copy_legacy(_p, fig_dir / "cluster_3d_scatter.png")
        except Exception as exc:
            logger.debug("Could not generate representative 3D cluster scatter: %s", exc)
        
        # ==================================================================
        # 6. Cluster evolution across depth
        # ==================================================================
        layer_results = [
            {"layer_name": k, "type_counts": v["type_counts"]}
            for k, v in self.cluster_results.items()
        ]
        _p = clustering_dir / "cluster_evolution.png"
        plot_cluster_evolution(layer_results, _p)
        _copy_legacy(_p, fig_dir / "cluster_evolution.png")
        
        # ==================================================================
        # 7. Cascade test results
        # ==================================================================
        logger.info("Generating cascade test plots...")
        for name, cascade in self.cascade_results.items():
            if cascade:
                from ..analysis.cascade_analysis import CascadeResult
                results = {
                    ct: CascadeResult(name, ct, d["n_removed"], d["accuracy_drop"], d["loss_increase"])
                    for ct, d in cascade.items()
                }
                _p = cascade_dir / f"cascade_{name.replace('.', '_')}.png"
                plot_cascade_test(results, _p)
                # Paper scripts glob fig_dir/"cascade_*.png" (non-recursive)
                _copy_legacy(_p, fig_dir / f"cascade_{name.replace('.', '_')}.png")
        
        # ==================================================================
        # 8. Halo properties
        # ==================================================================
        if self.halo_results:
            halo_summary = []
            for transition, clusters in self.halo_results.items():
                for ctype, data in clusters.items():
                    halo_summary.append({
                        "cluster_type": ctype,
                        "halo_red": data.get("halo_red", 0),
                        "halo_syn": data.get("halo_syn", 0),
                    })
            if halo_summary:
                from collections import defaultdict
                by_type = defaultdict(lambda: {"halo_red": [], "halo_syn": []})
                for h in halo_summary:
                    by_type[h["cluster_type"]]["halo_red"].append(h["halo_red"])
                    by_type[h["cluster_type"]]["halo_syn"].append(h["halo_syn"])
                
                avg_halo = [
                    {
                        "cluster_type": ct,
                        "halo_red": np.mean(v["halo_red"]),
                        "halo_syn": np.mean(v["halo_syn"]),
                    }
                    for ct, v in by_type.items()
                ]
                # Save into the organized halo subfolder, but also keep a
                # root-level copy for backward compatibility (paper scripts expect it).
                halo_props_path = halo_dir / "halo_properties.png"
                plot_halo_properties(avg_halo, halo_props_path)
                try:
                    _copy_legacy(halo_props_path, fig_dir / "halo_properties.png")
                except Exception:
                    pass

        # Representative cluster-to-cluster influence matrix for the paper (best-effort)
        try:
            if self.halo_flow_results:
                rep_transition = None
                for t in self.halo_flow_results.keys():
                    if "layer3" in t or "layer4" in t:
                        rep_transition = t
                        break
                if rep_transition is None:
                    rep_transition = list(self.halo_flow_results.keys())[0]
                flow = self.halo_flow_results.get(rep_transition, {})
                if flow:
                    halo_infl_path = halo_dir / "halo_influence_matrix.png"
                    plot_influence_matrix(
                        flow=flow,
                        layer_name=rep_transition,
                        save_path=halo_infl_path,
                    )
                    try:
                        _copy_legacy(halo_infl_path, fig_dir / "halo_influence_matrix.png")
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("Could not generate influence matrix plot: %s", exc)
        
        # ==================================================================
        # 9. Pruning comparison (using unified interface)
        # ==================================================================
        logger.info("Generating pruning plots...")
        if hasattr(self, 'pruning_results') and self.pruning_results:
            baseline = self.pruning_results.get('baseline', 0.9)
            methods = self.pruning_results.get('methods', {})
            
            if methods:
                # Main pruning comparison (line plot) - shows accuracy vs sparsity
                plot_pruning_comparison(
                    methods, baseline,
                    pruning_dir / "01_accuracy_vs_sparsity.png"
                )
        
                # Accuracy recovery chart
                plot_pruning_recovery_chart(
                    results=methods,
                    baseline_value=baseline,
                    metric='accuracy',
                    title='Accuracy Recovery After Pruning',
                    save_path=pruning_dir / "02_accuracy_recovery.png",
                )
                
                # ============================================================
                # Bar charts for method comparison at specific sparsities
                # ============================================================
                # Bar chart at 30% sparsity (conservative)
                plot_pruning_bar_comparison(
                    results=methods,
                    baseline_value=baseline,
                    target_sparsity=0.3,
                    metric='accuracy',
                    show_before_ft=True,
                    title='Pruning Methods at 30% Sparsity',
                    save_path=pruning_dir / "03_bar_30pct_sparsity.png",
                )
                
                # Bar chart at 50% sparsity (standard comparison point)
                plot_pruning_bar_comparison(
                    results=methods,
                    baseline_value=baseline,
                    target_sparsity=0.5,
                    metric='accuracy',
                    show_before_ft=True,
                    title='Pruning Methods at 50% Sparsity',
                    save_path=pruning_dir / "04_bar_50pct_sparsity.png",
                )
                
                # Bar chart at 70% sparsity (aggressive)
                plot_pruning_bar_comparison(
                    results=methods,
                    baseline_value=baseline,
                    target_sparsity=0.7,
                    metric='accuracy',
                    show_before_ft=True,
                    title='Pruning Methods at 70% Sparsity',
                    save_path=pruning_dir / "05_bar_70pct_sparsity.png",
                )
                
                # Heatmap of all methods x all sparsities
                plot_pruning_heatmap(
                    results=methods,
                    metric='accuracy',
                    title='Pruning Performance Heatmap (Accuracy %)',
                    save_path=pruning_dir / "06_heatmap_all_methods.png",
                )
                
                # Ranking plot (methods ranked by average performance)
                plot_pruning_ranking(
                    results=methods,
                    metric='accuracy',
                    title='Pruning Method Ranking (by Average Accuracy)',
                    save_path=pruning_dir / "07_method_ranking.png",
                )
                
                # Accuracy + Loss grid (if loss data available)
                has_loss = any(
                    'loss' in d or 'test_loss' in d
                    for ratio_data in methods.values()
                    for d in ratio_data.values()
                    if isinstance(d, dict)
                )
                if has_loss:
                    plot_pruning_accuracy_loss_grid(
                        results=methods,
                        baseline_acc=baseline,
                        title='Pruning: Accuracy and Loss',
                        save_path=pruning_dir / "08_accuracy_loss_grid.png",
                    )

                # Paper figure: which channels get pruned by cluster-aware?
                try:
                    dist = getattr(self, "pruning_cluster_distributions", {}).get("cluster_aware", {})
                    rep_ratio = 0.5 if 0.5 in dist else (sorted(dist.keys())[0] if dist else None)
                    if rep_ratio is not None:
                        summary = dist.get(rep_ratio, {})
                        pruned = summary.get("pruned", {})
                        total = summary.get("total", {})
                        if pruned and total:
                            _p = pruning_dir / "pruning_by_cluster.png"
                            plot_pruning_by_cluster_type(
                                pruned=pruned,
                                total=total,
                                save_path=_p,
                                title=f"Cluster-aware pruning (sparsity={rep_ratio:.0%})",
                            )
                            _copy_legacy(_p, fig_dir / "pruning_by_cluster.png")
                except Exception as exc:
                    logger.debug("Could not generate pruning-by-cluster plot: %s", exc)
        
        # ==================================================================
        # 10. Centroid evolution across depth
        # ==================================================================
        if self.cluster_results:
            layer_names = list(self.cluster_results.keys())
            layer_centroids = []
            for depth, name in enumerate(layer_names):
                cluster_data = self.cluster_results[name]
                if "centroids" in cluster_data:
                    layer_centroids.append({
                        "layer_name": name,
                        "depth": depth,
                        "centroids": cluster_data["centroids"].tolist() if hasattr(cluster_data["centroids"], 'tolist') else cluster_data["centroids"],
                        "type_mapping": cluster_data["type_mapping"],
                    })
            
            if layer_centroids:
                plot_centroid_evolution(layer_centroids, clustering_dir / "centroid_evolution_2d.png")
                plot_centroid_depth_profiles(layer_centroids, clustering_dir / "centroid_depth_profiles.png")
        
        # ==================================================================
        # 11. Top neurons by each metric (for first and last layers)
        # ==================================================================
        if self.layer_metrics:
            layer_names = list(self.layer_metrics.keys())
            key_layers = [layer_names[0], layer_names[-1]] if len(layer_names) > 1 else layer_names
            
            for name in key_layers:
                metrics = self.layer_metrics[name]
                safe_name = name.replace('.', '_')
                for metric_name in ['rq', 'redundancy', 'synergy']:
                    if metric_name in metrics:
                        plot_top_neurons_bar(
                            values=metrics[metric_name],
                            metric_name=metric_name,
                            layer_name=name,
                            top_k=15,
                            save_path=distributions_dir / f"top_neurons_{metric_name}_{safe_name}.png",
                        )
        
        logger.info(f"All figures saved to {fig_dir}")


# Backward compatibility aliases
VisionExperiment = ClusterAnalysisExperiment


def aggregate_multi_seed_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple seed runs into mean ± std statistics.
    
    This is the key function for robust statistical reporting. It computes
    mean and standard deviation across seeds for all numeric metrics.
    
    Args:
        results_list: List of result dictionaries from run_full_analysis(),
                     one per seed.
    
    Returns:
        Aggregated results with 'mean', 'std', 'seeds', and 'n_seeds' fields
        for all numeric values.
    
    Example:
        >>> seeds = [42, 123, 456]
        >>> all_results = []
        >>> for seed in seeds:
        ...     config.seed = seed
        ...     exp = ClusterAnalysisExperiment(config, model, train_loader, test_loader)
        ...     all_results.append(exp.run_full_analysis())
        >>> aggregated = aggregate_multi_seed_results(all_results)
        >>> print(aggregated['pruning_results']['methods']['cluster_aware'][0.5])
        # {'accuracy_mean': 0.923, 'accuracy_std': 0.004, 'n_seeds': 3}
    """
    if not results_list:
        return {}
    
    if len(results_list) == 1:
        # Single seed - just return with metadata
        result = results_list[0].copy()
        result["_aggregation"] = {"n_seeds": 1, "seeds": [result.get("config", {}).get("seed", 42)]}
        return result
    
    seeds = [r.get("config", {}).get("seed", i) for i, r in enumerate(results_list)]
    
    def _aggregate_numeric(values: List[Any]) -> Dict[str, Any]:
        """Aggregate a list of values into mean/std."""
        numeric = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
        if not numeric:
            return {"value": values[0] if values else None, "n_seeds": len(values)}
        arr = np.array(numeric, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_seeds": len(numeric),
        }
    
    def _aggregate_dict(dicts: List[Dict]) -> Dict:
        """Recursively aggregate dictionaries."""
        if not dicts or not all(isinstance(d, dict) for d in dicts):
            return {}
        
        all_keys = set()
        for d in dicts:
            all_keys.update(d.keys())
        
        result = {}
        for key in all_keys:
            values = [d.get(key) for d in dicts if key in d]
            
            if not values:
                continue
            
            # Check type of first non-None value
            first = next((v for v in values if v is not None), None)
            
            if first is None:
                result[key] = None
            elif isinstance(first, dict):
                result[key] = _aggregate_dict([v for v in values if isinstance(v, dict)])
            elif isinstance(first, (int, float)) and not isinstance(first, bool):
                result[key] = _aggregate_numeric(values)
            elif isinstance(first, list) and all(isinstance(x, (int, float)) for x in first):
                # List of numbers - aggregate element-wise
                try:
                    arr = np.array([v for v in values if isinstance(v, list)], dtype=np.float64)
                    result[key] = {
                        "mean": np.mean(arr, axis=0).tolist(),
                        "std": np.std(arr, axis=0).tolist(),
                        "n_seeds": len(arr),
                    }
                except Exception:
                    result[key] = values[0]
            else:
                # Non-numeric - just take first value
                result[key] = first
        
        return result
    
    # Aggregate main result sections
    aggregated = {
        "config": results_list[0].get("config", {}),
        "_aggregation": {
            "n_seeds": len(results_list),
            "seeds": seeds,
        },
    }
    
    # Sections to aggregate
    for section in ["pruning_results", "cascade_results", "halo_results", "permutation_results"]:
        section_data = [r.get(section, {}) for r in results_list]
        if any(section_data):
            aggregated[section] = _aggregate_dict(section_data)
    
    # For cluster results, aggregate silhouette scores
    cluster_sections = [r.get("cluster_results", {}) for r in results_list]
    if any(cluster_sections):
        aggregated["cluster_results"] = {}
        all_layers = set()
        for cs in cluster_sections:
            all_layers.update(cs.keys())
        
        for layer in all_layers:
            layer_data = [cs.get(layer, {}) for cs in cluster_sections if layer in cs]
            if layer_data:
                sil_values = [d.get("silhouette", 0.0) for d in layer_data]
                aggregated["cluster_results"][layer] = {
                    "silhouette": _aggregate_numeric(sil_values),
                    "type_counts": layer_data[0].get("type_counts", {}),  # Take first
                    "type_mapping": layer_data[0].get("type_mapping", {}),
                }
    
    # Copy ablation results (typically don't vary much across seeds)
    if "ablation_results" in results_list[0]:
        aggregated["ablation_results"] = results_list[0]["ablation_results"]
    
    return aggregated


def run_multi_seed_experiment(
    config: ClusterAnalysisConfig,
    model_fn,
    train_loader,
    test_loader,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run the full experiment across multiple seeds and aggregate results.
    
    Args:
        config: Base configuration (seed field will be overwritten per run)
        model_fn: Callable that returns a fresh model instance for each seed
        train_loader: Training data loader
        test_loader: Test data loader
        seeds: List of random seeds (default: [42, 123, 456, 789, 1000])
    
    Returns:
        Aggregated results with mean ± std across seeds
    
    Example:
        >>> def make_model():
        ...     return torchvision.models.resnet18(pretrained=True)
        >>> config = ClusterAnalysisConfig(model_name="resnet18")
        >>> results = run_multi_seed_experiment(
        ...     config, make_model, train_loader, test_loader,
        ...     seeds=[42, 123, 456]
        ... )
    """
    import copy
    
    seeds = seeds or getattr(config, 'seeds', None) or [42, 123, 456, 789, 1000]
    
    all_results = []
    
    for i, seed in enumerate(seeds):
        logger.info(f"=== Running seed {seed} ({i+1}/{len(seeds)}) ===")
        
        # Create fresh config and model for this seed
        seed_config = copy.deepcopy(config)
        seed_config.seed = seed
        seed_config.output_dir = str(Path(config.output_dir) / f"seed_{seed}")
        
        # Set random seeds
        if HAS_TORCH:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # Create fresh model
        model = model_fn()
        
        # Run experiment
        exp = ClusterAnalysisExperiment(seed_config, model, train_loader, test_loader)
        results = exp.run_full_analysis(
            include_pruning=getattr(config, 'pruning_ratios', None) is not None
        )
        all_results.append(results)
        
        # Clean up
        del model, exp
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Aggregate results
    aggregated = aggregate_multi_seed_results(all_results)
    
    # Save aggregated results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    logger.info(f"Aggregated results from {len(seeds)} seeds saved to {output_dir}")
    
    return aggregated
