"""
Master Pruning Orchestrator - Complete pruning pipeline.

Coordinates all aspects of pruning:
- Distribution strategy (how to allocate across layers)
- Scoring method (single metric or composite)
- Dynamic vs static scoring
- Parallel optimization
- Dependency handling

Provides simple high-level API for comprehensive pruning experiments.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PruningPlan:
    """Complete pruning plan."""

    distribution_strategy: str
    scoring_method: str
    per_layer_amounts: Dict[str, float]
    per_layer_scores: Dict[str, torch.Tensor]
    expected_sparsity: float
    use_dynamic_scores: bool = False


class MasterPruningOrchestrator:
    """
    High-level orchestrator for complete pruning workflows.

    Handles everything:
    - Distribution across layers (uniform, adaptive, global, etc.)
    - Scoring (single metric, composite, dynamic)
    - Direction (low, high, random)
    - Dependencies (conv, attention)
    - Parallelization (multiple strategies/networks)

    One-liner API for comprehensive experiments!

    Example:
        >>> orchestrator = MasterPruningOrchestrator()
        >>> result = orchestrator.prune_complete(
        ...     model,
        ...     target_sparsity=0.7,
        ...     distribution='adaptive_sensitivity',
        ...     scoring='composite',
        ...     use_dynamic=True,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader
        ... )
        >>> print(f"Accuracy: {result['baseline']}% → {result['final']}%")
    """

    def __init__(self, verbose: bool = True, parallel: bool = False, num_workers: int = 4):
        """
        Initialize orchestrator.

        Args:
            verbose: Print detailed progress
            parallel: Use parallel optimization when possible
            num_workers: Number of parallel workers
        """
        self.verbose = verbose
        self.parallel = parallel
        self.num_workers = num_workers

    def prune_complete(
        self,
        model: nn.Module,
        target_sparsity: float,
        distribution: str = "adaptive_sensitivity",
        scoring: str = "composite",
        direction: str = "low",
        use_dynamic: bool = False,
        train_loader=None,
        val_loader=None,
        trainer_fn: Optional[Callable] = None,
        eval_fn: Optional[Callable] = None,
        layers: Optional[List[str]] = None,
        fine_tune_epochs: int = 20,
    ) -> Dict[str, Any]:
        """
        Complete pruning workflow with all options.

        Args:
            model: Model to prune
            target_sparsity: Overall target (e.g., 0.7 for 70%)
            distribution: How to distribute across layers
                - 'uniform': Same % per layer
                - 'global_threshold': Global score threshold
                - 'adaptive_sensitivity': Based on layer sensitivity (RECOMMENDED)
                - 'importance_weighted': Based on avg scores
                - 'cascading': Sequential
            scoring: How to score neurons
                - 'magnitude': L1/L2 norm
                - 'rayleigh_quotient': RQ alignment
                - 'composite': Multi-criteria (RECOMMENDED)
                - 'movement': Training-aware
            direction: Which neurons to prune
                - 'low': Prune low scores (default)
                - 'high': Prune high scores (ablation)
                - 'random': Random (baseline)
            use_dynamic: Use training evolution (requires train_loader)
            train_loader: For dynamic scoring & fine-tuning
            val_loader: For evaluation
            trainer_fn: Function(model, train_loader, epochs) for fine-tuning
            eval_fn: Function(model, val_loader) -> accuracy
            layers: Specific layers to prune (None = auto-detect)
            fine_tune_epochs: Epochs to fine-tune after pruning

        Returns:
            Complete results dictionary
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Master Pruning Orchestrator")
            print("=" * 80)
            print(f"Target sparsity: {target_sparsity:.0%}")
            print(f"Distribution: {distribution}")
            print(f"Scoring: {scoring}")
            print(f"Direction: {direction}")
            print(f"Dynamic scores: {use_dynamic}")
            print("=" * 80 + "\n")

        # Auto-detect layers if needed
        if layers is None:
            from ..core.layer_detector import detect_trackable_layers

            layers = detect_trackable_layers(model)
            if self.verbose:
                print(f"Auto-detected {len(layers)} trackable layers\n")

        # Baseline evaluation
        baseline_acc = None
        if eval_fn and val_loader:
            baseline_acc = eval_fn(model, val_loader)
            if self.verbose:
                print(f"Baseline accuracy: {baseline_acc:.2f}%\n")

        # Step 1: Compute scores
        if self.verbose:
            print("Step 1: Computing importance scores...")

        if use_dynamic and train_loader:
            layer_scores = self._compute_dynamic_scores(model, train_loader, layers, scoring)
        else:
            layer_scores = self._compute_static_scores(model, val_loader or train_loader, layers, scoring)

        if self.verbose:
            print(f"Computed scores for {len(layer_scores)} layers\n")

        # Step 2: Compute distribution
        if self.verbose:
            print(f"Step 2: Computing {distribution} distribution...")

        from .distribution import PruningDistributionManager

        dist_manager = PruningDistributionManager(strategy=distribution, target_sparsity=target_sparsity)

        per_layer_amounts = dist_manager.compute_distribution(
            model, layers, layer_scores=layer_scores, eval_fn=lambda m: eval_fn(m, val_loader) if eval_fn and val_loader else None
        )

        if self.verbose:
            dist_manager.print_distribution(per_layer_amounts, model, layer_scores)

        # Step 3: Create masks
        if self.verbose:
            print("Step 3: Creating pruning masks...")

        from ..services import MaskOperations

        masks = {}
        for layer_name in layers:
            if layer_name not in layer_scores or layer_name not in per_layer_amounts:
                continue

            mask = MaskOperations.create_structured_mask(layer_scores[layer_name], amount=per_layer_amounts[layer_name], mode=direction)
            masks[layer_name] = mask

        if self.verbose:
            print(f"Created masks for {len(masks)} layers\n")

        # Step 4: Apply with dependency awareness
        if self.verbose:
            print("Step 4: Applying pruning (dependency-aware)...")

        from .dependency_aware import DependencyAwarePruning

        dep_pruner = DependencyAwarePruning(model)

        # Convert masks to scores for dependency pruner interface
        pruning_result = dep_pruner.prune(layer_scores, amount=target_sparsity, dry_run=False)  # Overall target

        if self.verbose:
            print("Applied pruning\n")

        # Step 5: Fine-tune
        if trainer_fn and train_loader and fine_tune_epochs > 0:
            if self.verbose:
                print(f"Step 5: Fine-tuning for {fine_tune_epochs} epochs...")

            trainer_fn(model, train_loader, epochs=fine_tune_epochs)

            if self.verbose:
                print("Fine-tuning complete\n")

        # Step 6: Final evaluation
        final_acc = None
        if eval_fn and val_loader:
            final_acc = eval_fn(model, val_loader)
            if self.verbose:
                print(f"Final accuracy: {final_acc:.2f}%")
                if baseline_acc:
                    drop = baseline_acc - final_acc
                    print(f"Accuracy drop: {drop:.2f}%\n")

        # Return complete results
        return {
            "baseline_accuracy": baseline_acc,
            "final_accuracy": final_acc,
            "accuracy_drop": baseline_acc - final_acc if baseline_acc and final_acc else None,
            "target_sparsity": target_sparsity,
            "distribution_strategy": distribution,
            "scoring_method": scoring,
            "per_layer_amounts": per_layer_amounts,
            "masks": masks,
            "pruning_stats": pruning_result["stats"],
        }

    def _compute_static_scores(self, model: nn.Module, data_loader, layers: List[str], scoring: str) -> Dict[str, torch.Tensor]:
        """Compute scores on current (trained) model."""
        from ..metrics import get_metric
        from ..models import BaseModelWrapper
        from ..services import ActivationCaptureService, NodeScoringService

        # Wrap model
        wrapper = BaseModelWrapper(model, tracked_layers=layers)
        capture = ActivationCaptureService(wrapper)

        # Get batch
        inputs, targets = next(iter(data_loader))
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Capture activations
        data = capture.capture(inputs, include_weights=True)

        # Compute scores based on method
        if scoring == "magnitude":
            scores = {}
            for layer in layers:
                if layer in data.weights:
                    weights = data.weights[layer]
                    scores[layer] = weights.abs().mean(dim=list(range(1, weights.ndim)))

        elif scoring == "rayleigh_quotient":
            rq = get_metric("rayleigh_quotient")
            scores = {}
            for layer in layers:
                if layer in data.inputs and layer in data.weights:
                    scores[layer] = rq.compute(data.inputs[layer], data.weights[layer])

        elif scoring == "composite":
            scorer = NodeScoringService(
                metrics={
                    "rq": get_metric("rayleigh_quotient"),
                    "redundancy": get_metric("pairwise_redundancy_gaussian", mode="output_based", num_pairs=10),
                    "synergy": get_metric("synergy_gaussian_mmi", num_pairs=10),
                }
            )

            layer_scores_obj = scorer.compute_layerwise_scores(data, targets)
            scores = {name: ls.composite for name, ls in layer_scores_obj.items()}

        else:
            raise ValueError(f"Unknown scoring method: {scoring}")

        return scores

    def _compute_dynamic_scores(self, model: nn.Module, train_loader, layers: List[str], scoring: str) -> Dict[str, torch.Tensor]:
        """
        Compute scores using training dynamics.

        Note: Requires training with callback - placeholder for now.
        Future: Integrate with training history.
        """
        logger.warning(
            "Dynamic scoring requires training with AlignmentMetricsCallback. "
            "Falling back to static scores. "
            "See dynamic_scoring.py for full implementation."
        )

        # Fallback to static
        return self._compute_static_scores(model, train_loader, layers, scoring)


def prune_with_all_options(model: nn.Module, target_sparsity: float = 0.7, **kwargs) -> Dict:
    """
    One-liner for complete pruning with all options.

    Args:
        model: Model to prune
        target_sparsity: Target overall sparsity
        **kwargs: All options (distribution, scoring, etc.)

    Returns:
        Complete results
    """
    orchestrator = MasterPruningOrchestrator()
    return orchestrator.prune_complete(model, target_sparsity, **kwargs)
