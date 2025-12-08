#!/usr/bin/env python3
"""
Unified Alignment Experiment Runner

Run alignment experiments with configuration files.

Usage:
    python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
    python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml --device cuda:0
    python scripts/run_experiment.py --analysis-only --experiment-dir results/my_experiment_20240101
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add the project root and src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "src"))

# Configure tqdm globally to avoid ANSI escape codes in log files
# This is especially important when running under SLURM where output is redirected to files
# The [A escape codes you see in logs are cursor movement codes from tqdm progress bars

# Set environment variable for libraries that respect it (e.g., transformers)
# This tells tqdm to use simpler formatting
os.environ.setdefault('TQDM_DISABLE', '0')  # Keep tqdm enabled but configure it

try:
    from tqdm import tqdm
    import tqdm as tqdm_module
    
    # Check if we're in a terminal (TTY) - if not, we're likely logging to a file
    is_tty = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
    
    # Also check if we're running under SLURM (common case where logs go to files)
    is_slurm = 'SLURM_JOB_ID' in os.environ
    
    if not is_tty or is_slurm:
        # When not in terminal or under SLURM, configure tqdm to avoid ANSI escape codes
        # This prevents escape codes like [A from appearing in log files
        original_tqdm = tqdm_module.tqdm
        
        def patched_tqdm(*args, **kwargs):
            # Force ASCII mode and simpler formatting when output might go to a file
            kwargs.setdefault('ascii', True)  # Use ASCII instead of Unicode blocks (prevents █▋ characters)
            kwargs.setdefault('ncols', 100)   # Fixed width
            kwargs.setdefault('file', sys.stderr)  # Always use stderr
            # Disable dynamic resizing which can cause issues
            kwargs.setdefault('dynamic_ncols', False)
            # Minimize escape codes
            kwargs.setdefault('leave', False)  # Don't leave progress bar after completion
            return original_tqdm(*args, **kwargs)
        
        tqdm_module.tqdm = patched_tqdm
except ImportError:
    pass  # tqdm not available, skip configuration

from alignment.experiments.general_alignment import GeneralAlignmentExperiment
from alignment.pruning.experiments.cascading_layer import CascadingLayerPruningExperiment
from alignment.pruning.experiments.layer_wise import LayerIsolatedPruningExperiment
from alignment.experiments.llm_experiments import LLMAlignmentExperiment
from alignment.experiments.cluster_experiments import (
    ClusterAnalysisExperiment,
    ClusterAnalysisConfig,
    VisionExperiment,  # backward compat
    VisionExperimentConfig,  # backward compat
)

logger = logging.getLogger(__name__)


def _create_cluster_experiment(config):
    """Create ClusterAnalysisExperiment from unified config."""
    import torch
    import torchvision
    import torchvision.transforms as transforms
    
    # Helper to safely get nested config values
    def _get_nested(obj, key, default):
        """Get nested config value, handling both dict and object attributes."""
        if hasattr(obj, key):
            val = getattr(obj, key)
            if isinstance(val, dict):
                return val
            return default
        return default
    
    # Extract nested configs with proper defaults
    model_cfg = _get_nested(config, "model", {})
    dataset_cfg = _get_nested(config, "dataset", {})
    metrics_cfg = _get_nested(config, "metrics", {})
    clustering_cfg = _get_nested(config, "clustering", {})
    halo_cfg = _get_nested(config, "halo_analysis", {})
    
    # Build ClusterAnalysisConfig from the loaded config
    cluster_config = ClusterAnalysisConfig(
        model_name=getattr(config, "model_name", model_cfg.get("name", "resnet18") if isinstance(model_cfg, dict) else "resnet18"),
        dataset_name=getattr(config, "dataset_name", dataset_cfg.get("name", "cifar10") if isinstance(dataset_cfg, dict) else "cifar10"),
        n_calibration=getattr(config, "n_calibration", metrics_cfg.get("n_calibration_samples", 5000) if isinstance(metrics_cfg, dict) else 5000),
        n_clusters=getattr(config, "n_clusters", clustering_cfg.get("n_clusters", 4) if isinstance(clustering_cfg, dict) else 4),
        synergy_target=getattr(config, "synergy_target", metrics_cfg.get("synergy_target", "logit_margin") if isinstance(metrics_cfg, dict) else "logit_margin"),
        synergy_pairs=getattr(config, "synergy_pairs", metrics_cfg.get("synergy_num_pairs", 10) if isinstance(metrics_cfg, dict) else 10),
        halo_percentile=getattr(config, "halo_percentile", halo_cfg.get("percentile", 90.0) if isinstance(halo_cfg, dict) else 90.0),
        output_dir=getattr(config, "experiment_dir", "results/cluster_analysis"),
        device=getattr(config, "device", "cuda"),
        seed=getattr(config, "seed", 42),
    )
    
    # Load model
    model_name = cluster_config.model_name.lower()
    num_classes = 10 if "cifar" in cluster_config.dataset_name.lower() else 1000
    
    if "resnet18" in model_name:
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "resnet50" in model_name:
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "vgg16" in model_name:
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif "mobilenet" in model_name:
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load dataset
    dataset_name = cluster_config.dataset_name.lower()
    if "cifar10" in dataset_name:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif "cifar100" in dataset_name:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    batch_size = getattr(config, "batch_size", 128)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4)
    
    # Fine-tune the model on target dataset before experiments
    # This is necessary because we replaced the classifier head with random weights
    model = _finetune_model_for_dataset(
        model, train_loader, test_loader, 
        device=cluster_config.device,
        epochs=getattr(config, "pretrain_epochs", 20),
        lr=getattr(config, "pretrain_lr", 0.001),
    )
    
    return ClusterAnalysisExperiment(cluster_config, model, train_loader, test_loader)


def _finetune_model_for_dataset(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    epochs: int = 20,
    lr: float = 0.001,
) -> torch.nn.Module:
    """
    Fine-tune a pretrained model on the target dataset.
    
    This is necessary when using ImageNet pretrained models on CIFAR-10/100
    because the classifier head is replaced with random weights.
    
    Args:
        model: Model with replaced classifier head
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to train on
        epochs: Number of fine-tuning epochs
        lr: Learning rate
        
    Returns:
        Fine-tuned model
    """
    import torch.optim as optim
    
    model = model.to(device)
    
    # Check initial accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    initial_acc = correct / total
    
    # If already trained (>50% accuracy), skip fine-tuning
    if initial_acc > 0.5:
        logger.info(f"Model already trained (accuracy: {initial_acc:.2%}), skipping fine-tuning")
        return model
    
    logger.info(f"Fine-tuning model on target dataset (initial accuracy: {initial_acc:.2%})...")
    
    # Use different learning rates for pretrained vs new layers
    # Freeze early layers, fine-tune later layers + new classifier
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)
    
    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': lr * 0.1},  # Lower LR for pretrained
        {'params': new_params, 'lr': lr},  # Higher LR for new classifier
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={train_loss/len(train_loader):.4f}, acc={acc:.2%}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    logger.info(f"Fine-tuning complete. Best accuracy: {best_acc:.2%}")
    
    return model


def run_post_analysis(config, results_file: Path, output_dir: Path):
    """Run post-experiment analysis using AnalysisRunner."""
    post_analysis_config = getattr(config, "post_analysis", {})
    if not post_analysis_config:
        return
    
    logger.info("Running post-experiment analysis...")
    
    try:
        from alignment.analysis import AnalysisRunner, AnalysisConfig
        
        # Build analysis config from post_analysis block
        analysis_config = AnalysisConfig(
            results_file=str(results_file),
            output_dir=str(output_dir / "analysis"),
            style=post_analysis_config.get("style", "seaborn-v0_8-paper"),
            format=post_analysis_config.get("format", config.plot_format),
            dpi=post_analysis_config.get("dpi", config.plot_dpi),
            analyses=post_analysis_config.get("analyses", ["all"]),
            histograms=post_analysis_config.get("histograms", {}),
            scatter_plots=post_analysis_config.get("scatter_plots", {}),
            heatmaps=post_analysis_config.get("heatmaps", {}),
            pruning_curves=post_analysis_config.get("pruning_curves", {}),
            layer_distributions=post_analysis_config.get("layer_distributions", {}),
            scar_analysis=post_analysis_config.get("scar_analysis", {}),
        )
        
        runner = AnalysisRunner(analysis_config)
        outputs = runner.run()
        
        total_files = sum(len(v) for v in outputs.values())
        logger.info(f"Post-analysis complete: generated {total_files} files in {output_dir / 'analysis'}")
        
    except Exception as e:
        logger.error(f"Post-analysis failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Alignment Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Load existing experiment and regenerate analysis/plots",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        help="Path to existing experiment directory (with --analysis-only)",
    )

    args, unknown = parser.parse_known_args()

    # Parse overrides
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.seed:
        overrides["seed"] = args.seed

    # Load config
    from alignment.configs.config_loader import load_config as proper_load_config
    config = proper_load_config(args.config)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    is_analysis_only = bool(args.analysis_only)

    if is_analysis_only:
        if not args.experiment_dir:
            raise ValueError("--analysis-only requires --experiment-dir")
        output_dir = Path(args.experiment_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {output_dir}")

        config.experiment_dir = str(output_dir)
        config.checkpoint_dir = str(output_dir / "checkpoints")
        config.log_dir = str(output_dir / "logs")
        plots_dir = output_dir / "plots"
        config.plots_dir = str(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        config_save_path = output_dir / "experiment_config.yaml"
        timestamp = None
    else:
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = getattr(config, "name", "experiment")

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f"results/{experiment_name}_{timestamp}")

        output_dir.mkdir(parents=True, exist_ok=True)

        config_save_path = output_dir / "experiment_config.yaml"
        config.save(config_save_path)

        config.checkpoint_dir = str(output_dir / "checkpoints")
        config.log_dir = str(output_dir / "logs")
        config.experiment_dir = str(output_dir)

        plots_dir = output_dir / "plots"
        config.plots_dir = str(plots_dir)

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Print experiment info
    print(f"\n{'='*60}")
    print("Alignment Experiment" + (" (Analysis Only)" if is_analysis_only else ""))
    print(f"{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"Experiment directory: {output_dir}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")

    # Determine experiment type
    experiment_type = getattr(config, "experiment_type", "alignment_analysis")
    logger.info(f"Running {experiment_type} experiment")

    if experiment_type in {"llm_alignment", "llm_supernode", "llm"}:
        experiment = LLMAlignmentExperiment(config)
    elif experiment_type in {"alignment_analysis", "vision_synergy", "general_alignment"}:
        experiment = GeneralAlignmentExperiment(config)
    elif experiment_type in {"cluster_analysis", "vision_cluster_analysis", "metric_cluster_analysis"}:
        # Cluster-based analysis experiment (works for any architecture)
        experiment = _create_cluster_experiment(config)
    elif experiment_type == "layer_isolated_pruning":
        experiment = LayerIsolatedPruningExperiment(config)
    elif experiment_type == "cascading_layer_pruning":
        experiment = CascadingLayerPruningExperiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Analysis-only mode
    if is_analysis_only:
        if isinstance(experiment, GeneralAlignmentExperiment):
            result_files = sorted(output_dir.glob("results_*.json"))
            if not result_files:
                raise FileNotFoundError(f"No results_*.json found in {output_dir}")
            results_path = result_files[-1]
            with results_path.open("r") as f:
                results = json.load(f)

            experiment.train_results = results.get("train_results", {})
            experiment.test_results = results.get("test_results", {})
            experiment.dropout_results = results.get("dropout_results", {})
            experiment.pruning_results = results.get("pruning_results", {})
            experiment.eigenfeature_results = results.get("eigenfeature_results", {})

            if getattr(config, "generate_plots", True):
                experiment._generate_visualizations()
                logger.info("Regenerated visualizations from existing results")
            
            # Run post-analysis if configured
            run_post_analysis(config, results_path, output_dir)
        else:
            logger.warning(f"Analysis-only mode not supported for {experiment_type}")

        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"{'='*60}\n")
        return

    # Full experiment run
    results = experiment.run()

    # Save results
    results_file = output_dir / f"results_{timestamp}.json"

    def convert_to_serializable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    serializable_results = convert_to_serializable(results)
    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Run post-analysis if configured
    run_post_analysis(config, results_file, output_dir)

    # Print completion
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")

    if "test_results" in results:
        print(f"Final accuracy: {results['test_results'].get('final_accuracy', 'N/A'):.2f}%")

    print(f"\nResults saved in: {output_dir}")
    print(f"  - Configuration: {config_save_path}")
    print(f"  - Results: {results_file}")
    print(f"  - Plots: {plots_dir}/")
    
    # Check for analysis output
    analysis_dir = output_dir / "analysis"
    if analysis_dir.exists():
        analysis_files = list(analysis_dir.rglob("*"))
        print(f"  - Analysis ({len([f for f in analysis_files if f.is_file()])} files): {analysis_dir}/")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
