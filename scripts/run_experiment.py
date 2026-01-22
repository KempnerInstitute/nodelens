#!/usr/bin/env python3
"""
Unified Alignment Experiment Runner

Run alignment experiments with configuration files.

Usage:
    python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
    python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml --device cuda:0
    python scripts/run_experiment.py --analysis-only --experiment-dir results/my_experiment_20240101

Job Directory Structure:
    When base_output_dir is specified in config, experiments create unique job directories:
    
    {base_output_dir}/
        {experiment_name}_{timestamp}_{job_id}/
            results/          # JSON results files
            logs/             # experiment.log
            checkpoints/      # Model checkpoints
            figures/          # All visualizations
            analysis/         # Post-experiment analysis
            experiment_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml

try:
    from alignment.experiments.general_alignment import GeneralAlignmentExperiment
    from alignment.experiments.llm_experiments import LLMAlignmentExperiment
    from alignment.experiments.cluster_experiments import (
        ClusterAnalysisExperiment,
        ClusterAnalysisConfig,
        VisionExperiment,  # backward compat
        VisionExperimentConfig,  # backward compat
    )
except ImportError:
    # Repo-local runs (without installing the package): add project root + src/ to sys.path.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(current_dir)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, "src"))

    from alignment.experiments.general_alignment import GeneralAlignmentExperiment
    from alignment.experiments.llm_experiments import LLMAlignmentExperiment
    from alignment.experiments.cluster_experiments import (
        ClusterAnalysisExperiment,
        ClusterAnalysisConfig,
        VisionExperiment,  # backward compat
        VisionExperimentConfig,  # backward compat
    )

# Configure tqdm to avoid ANSI escape codes in log files (common under SLURM).
try:
    from tqdm import tqdm
    import tqdm as tqdm_module

    # Check if we're in a terminal (TTY) - if not, we're likely logging to a file.
    is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    # Also check if we're running under SLURM (common case where logs go to files).
    is_slurm = "SLURM_JOB_ID" in os.environ

    if not is_tty or is_slurm:
        original_tqdm = tqdm_module.tqdm

        def patched_tqdm(*args, **kwargs):
            # Force ASCII mode and simpler formatting when output might go to a file.
            kwargs.setdefault("ascii", True)  # prevent Unicode blocks in logs
            kwargs.setdefault("ncols", 100)  # fixed width
            kwargs.setdefault("file", sys.stderr)  # always use stderr
            kwargs.setdefault("dynamic_ncols", False)  # avoid resizing escape codes
            kwargs.setdefault("leave", False)  # don't leave progress bars in logs
            return original_tqdm(*args, **kwargs)

        tqdm_module.tqdm = patched_tqdm
except ImportError:
    pass  # tqdm not available, skip configuration

# Set environment variable for libraries that respect it (e.g., transformers).
os.environ.setdefault("TQDM_DISABLE", "0")

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
    pruning_cfg = _get_nested(config, "pruning", {})
    
    # Get pruning settings
    pruning_ratios = getattr(config, "pruning_amounts", None) or \
                     (pruning_cfg.get("ratios") if isinstance(pruning_cfg, dict) else None) or \
                     [0.1, 0.3, 0.5, 0.7]

    # Pruning distribution / constraints (used by ClusterAnalysisExperiment pruning pipeline)
    pruning_distribution = (
        getattr(config, "pruning_distribution", None)
        or (pruning_cfg.get("distribution") if isinstance(pruning_cfg, dict) else None)
        or "uniform"
    )
    dependency_aware_pruning = bool(
        getattr(config, "dependency_aware_pruning", None)
        if hasattr(config, "dependency_aware_pruning")
        else (pruning_cfg.get("dependency_aware", False) if isinstance(pruning_cfg, dict) else False)
    )
    pruning_min_per_layer = float(
        getattr(config, "pruning_min_per_layer", None)
        if hasattr(config, "pruning_min_per_layer")
        else (pruning_cfg.get("min_per_layer", 0.0) if isinstance(pruning_cfg, dict) else 0.0)
    )
    pruning_max_per_layer = float(
        getattr(config, "pruning_max_per_layer", None)
        if hasattr(config, "pruning_max_per_layer")
        else (pruning_cfg.get("max_per_layer", 0.95) if isinstance(pruning_cfg, dict) else 0.95)
    )

    # Optional pruning layer filters (e.g., MobileNetV2 pointwise-only pruning)
    pruning_pointwise_only = bool(
        getattr(config, "pruning_pointwise_only", False)
        if hasattr(config, "pruning_pointwise_only")
        else (pruning_cfg.get("pointwise_only", False) if isinstance(pruning_cfg, dict) else False)
    )
    pruning_skip_depthwise = bool(
        getattr(config, "pruning_skip_depthwise", False)
        if hasattr(config, "pruning_skip_depthwise")
        else (pruning_cfg.get("skip_depthwise", False) if isinstance(pruning_cfg, dict) else False)
    )
    
    # Get fine-tuning settings
    fine_tune_cfg = pruning_cfg.get("fine_tune", {}) if isinstance(pruning_cfg, dict) else {}
    fine_tune_enabled = getattr(config, "fine_tune_after_pruning", 
                                fine_tune_cfg.get("enabled", False) if isinstance(fine_tune_cfg, dict) else False)
    fine_tune_epochs = getattr(config, "fine_tune_epochs",
                              fine_tune_cfg.get("epochs", 10) if isinstance(fine_tune_cfg, dict) else 10)
    fine_tune_lr = getattr(config, "fine_tune_learning_rate",
                          fine_tune_cfg.get("learning_rate", 0.0001) if isinstance(fine_tune_cfg, dict) else 0.0001)
    fine_tune_max_batches = getattr(
        config,
        "fine_tune_max_batches",
        fine_tune_cfg.get("max_batches", None) if isinstance(fine_tune_cfg, dict) else None,
    )
    fine_tune_weight_decay = float(
        getattr(
            config,
            "fine_tune_weight_decay",
            fine_tune_cfg.get("weight_decay", 0.0) if isinstance(fine_tune_cfg, dict) else 0.0,
        ) or 0.0
    )
    
    # Get pruning algorithms/methods
    pruning_methods = getattr(config, "pruning_strategies", None) or \
                     (pruning_cfg.get("methods") if isinstance(pruning_cfg, dict) else None) or \
                     (pruning_cfg.get("algorithms") if isinstance(pruning_cfg, dict) else None) or \
                     ['random', 'magnitude', 'taylor', 'composite', 'cluster_aware']
    
    # Build ClusterAnalysisConfig from the loaded config
    cluster_config = ClusterAnalysisConfig(
        model_name=getattr(config, "model_name", model_cfg.get("name", "resnet18") if isinstance(model_cfg, dict) else "resnet18"),
        dataset_name=getattr(config, "dataset_name", dataset_cfg.get("name", "cifar10") if isinstance(dataset_cfg, dict) else "cifar10"),
        n_calibration=getattr(config, "n_calibration", metrics_cfg.get("n_calibration_samples", 5000) if isinstance(metrics_cfg, dict) else 5000),
        n_clusters=getattr(config, "n_clusters", clustering_cfg.get("n_clusters", 4) if isinstance(clustering_cfg, dict) else 4),
        activation_point=str(
            getattr(
                config,
                "activation_point",
                metrics_cfg.get("activation_point", "pre_bn") if isinstance(metrics_cfg, dict) else "pre_bn",
            )
        ),
        activation_samples=getattr(
            config,
            "activation_samples",
            metrics_cfg.get("activation_samples", "flatten_spatial") if isinstance(metrics_cfg, dict) else "flatten_spatial",
        ),
        spatial_samples_per_image=int(
            getattr(
                config,
                "spatial_samples_per_image",
                metrics_cfg.get("spatial_samples_per_image", 16) if isinstance(metrics_cfg, dict) else 16,
            )
        ),
        synergy_target=getattr(config, "synergy_target", metrics_cfg.get("synergy_target", "logit_margin") if isinstance(metrics_cfg, dict) else "logit_margin"),
        synergy_candidate_pool=int(
            getattr(
                config,
                "synergy_candidate_pool",
                metrics_cfg.get("synergy_candidate_pool", 50) if isinstance(metrics_cfg, dict) else 50,
            )
        ),
        synergy_pairs=getattr(config, "synergy_pairs", metrics_cfg.get("synergy_num_pairs", 10) if isinstance(metrics_cfg, dict) else 10),
        halo_percentile=getattr(config, "halo_percentile", halo_cfg.get("percentile", 90.0) if isinstance(halo_cfg, dict) else 90.0),
        pruning_ratios=pruning_ratios,
        pruning_methods=pruning_methods,
        fine_tune_after_pruning=fine_tune_enabled,
        fine_tune_epochs=fine_tune_epochs,
        fine_tune_lr=fine_tune_lr,
        fine_tune_max_batches=fine_tune_max_batches,
        fine_tune_weight_decay=fine_tune_weight_decay,
        output_dir=getattr(config, "experiment_dir", "results/cluster_analysis"),
        device=getattr(config, "device", "cuda"),
        seed=getattr(config, "seed", 42),
    )

    # Propagate pruning distribution knobs into ClusterAnalysisConfig so all pruning
    # methods (including cluster-aware) use the same allocation regime.
    setattr(cluster_config, "pruning_distribution", str(pruning_distribution))
    setattr(cluster_config, "dependency_aware_pruning", bool(dependency_aware_pruning))
    setattr(cluster_config, "pruning_min_per_layer", float(pruning_min_per_layer))
    setattr(cluster_config, "pruning_max_per_layer", float(pruning_max_per_layer))
    setattr(cluster_config, "pruning_pointwise_only", bool(pruning_pointwise_only))
    setattr(cluster_config, "pruning_skip_depthwise", bool(pruning_skip_depthwise))

    # Optional: allow sweeping cluster-aware score weights via nested pruning config:
    #   pruning.cluster_aware.{alpha,beta,gamma,lambda_halo,protect_critical_frac}
    if isinstance(pruning_cfg, dict):
        ca = pruning_cfg.get("cluster_aware", {})
        if isinstance(ca, dict):
            if "alpha" in ca:
                setattr(cluster_config, "cluster_aware_alpha", float(ca["alpha"]))
            if "beta" in ca:
                setattr(cluster_config, "cluster_aware_beta", float(ca["beta"]))
            if "gamma" in ca:
                setattr(cluster_config, "cluster_aware_gamma", float(ca["gamma"]))
            if "lambda_halo" in ca:
                setattr(cluster_config, "cluster_aware_lambda_halo", float(ca["lambda_halo"]))
            if "protect_critical_frac" in ca:
                setattr(cluster_config, "cluster_aware_protect_critical_frac", float(ca["protect_critical_frac"]))
            # Annealed variant schedule (when using cluster_aware_annealed)
            if "anneal_start" in ca:
                setattr(cluster_config, "cluster_aware_anneal_start", float(ca["anneal_start"]))
            if "anneal_end" in ca:
                setattr(cluster_config, "cluster_aware_anneal_end", float(ca["anneal_end"]))

    # Also support the flat ExperimentConfig fields used by our config loader
    # (and mapped from unified-style dotted CLI overrides).
    for attr in (
        "cluster_aware_alpha",
        "cluster_aware_beta",
        "cluster_aware_gamma",
        "cluster_aware_lambda_halo",
        "cluster_aware_protect_critical_frac",
        "cluster_aware_anneal_start",
        "cluster_aware_anneal_end",
    ):
        if hasattr(config, attr):
            setattr(cluster_config, attr, float(getattr(config, attr)))
    
    # Load model
    model_name = cluster_config.model_name.lower()
    dataset_name = cluster_config.dataset_name.lower()
    # Prefer explicit num_classes from config.model.num_classes when present
    num_classes = (
        int(model_cfg.get("num_classes")) if isinstance(model_cfg, dict) and model_cfg.get("num_classes") is not None
        else (10 if "cifar10" in dataset_name else 100 if "cifar100" in dataset_name else 100 if "imagenet100" in dataset_name else 1000)
    )
    
    # Check for pre-trained checkpoint
    model_cfg = _get_nested(config, "model", {})
    checkpoint_path = model_cfg.get("checkpoint", None) if isinstance(model_cfg, dict) else None
    checkpoint_path = checkpoint_path or getattr(config, "model_checkpoint", None)
    
    pretrained = bool(model_cfg.get("pretrained", True)) if isinstance(model_cfg, dict) else True
    weights_name = model_cfg.get("weights", None) if isinstance(model_cfg, dict) else None
    weights_arg = weights_name if pretrained else None

    if "resnet18" in model_name:
        model = torchvision.models.resnet18(weights=weights_arg or 'IMAGENET1K_V1')
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "resnet50" in model_name:
        model = torchvision.models.resnet50(weights=weights_arg or 'IMAGENET1K_V1')
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "vgg16" in model_name:
        model = torchvision.models.vgg16_bn(weights=weights_arg or 'IMAGENET1K_V1')
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif "mobilenet" in model_name:
        model = torchvision.models.mobilenet_v2(weights=weights_arg or 'IMAGENET1K_V1')
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif "alexnet" in model_name:
        model = torchvision.models.alexnet(weights=weights_arg or 'IMAGENET1K_V1')
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint if available, otherwise model needs to be trained
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        needs_training = False
    else:
        logger.warning(f"No checkpoint found - model needs to be trained on {cluster_config.dataset_name}")
        needs_training = True
    
    # Load dataset
    if "cifar10" in dataset_name:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        root = (
            (dataset_cfg.get("root") if isinstance(dataset_cfg, dict) else None)
            or getattr(config, "data_path", None)
            or "./data"
        )
        # Use standard CIFAR augmentation when training so baseline accuracies match common reporting.
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    elif "cifar100" in dataset_name:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        root = (
            (dataset_cfg.get("root") if isinstance(dataset_cfg, dict) else None)
            or getattr(config, "data_path", None)
            or "./data"
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)
    elif "imagenet100" in dataset_name:
        # Expected folder structure: {root}/train/* and {root}/val/* (ImageFolder)
        root = dataset_cfg.get("root", "./data/imagenet100") if isinstance(dataset_cfg, dict) else "./data/imagenet100"
        train_dir = Path(root) / "train"
        val_dir = Path(root) / "val"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"ImageNet-100 not found. Expected ImageFolder dirs at: {train_dir} and {val_dir}"
            )

        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        image_size = int(dataset_cfg.get("image_size", 224)) if isinstance(dataset_cfg, dict) else 224
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        train_dataset = torchvision.datasets.ImageFolder(root=str(train_dir), transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=str(val_dir), transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    batch_size = int(getattr(config, "batch_size", 128))
    num_workers = int(getattr(config, "num_workers", 4))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    
    # Track architecture tweaks so we can reproduce the exact model when loading checkpoints later.
    resnet_cifar_stem_tweaked = False
    mobilenet_cifar_stride1 = False
    
    # CIFAR-specific stem tweak: using the ImageNet stem (7x7,stride2 + maxpool)
    # degrades CIFAR accuracy. Use the standard CIFAR stem and (when pretrained)
    # seed weights by center-cropping the 7x7 conv filter.
    if ("cifar" in dataset_name) and ("resnet" in model_name):
        if hasattr(model, "conv1") and hasattr(model, "maxpool"):
            old_conv = model.conv1
            new_conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            try:
                if pretrained and hasattr(old_conv, "weight") and old_conv.weight.shape[-1] == 7:
                    with torch.no_grad():
                        new_conv.weight.copy_(old_conv.weight[:, :, 2:5, 2:5])
            except Exception:
                pass
            model.conv1 = new_conv
            model.maxpool = torch.nn.Identity()
            resnet_cifar_stem_tweaked = True

    # MobileNetV2 CIFAR stem tweak: the ImageNet stride-2 stem collapses spatial resolution too early
    # on 32x32 inputs and can lead to unstable/weak CIFAR fine-tuning. Use stride=1 for the first conv.
    if ("cifar" in dataset_name) and ("mobilenet" in model_name):
        try:
            conv0 = model.features[0][0]  # ConvBNReLU: [conv, bn, relu]
            if isinstance(conv0, torch.nn.Conv2d):
                conv0.stride = (1, 1)
                mobilenet_cifar_stride1 = True
        except Exception:
            pass

    # Train/fine-tune the model on target dataset before experiments.
    # If you want a pure "no-training" analysis, provide an explicit checkpoint and set do_train=false.
    do_train = bool(getattr(config, "do_train", True))
    if needs_training and not do_train:
        raise RuntimeError(
            f"Model checkpoint not found and training is disabled (do_train=false). "
            f"Provide model.checkpoint/model_checkpoint or enable training in the config."
        )
    if needs_training and do_train:
        model = _finetune_model_for_dataset(
            model,
            train_loader,
            test_loader,
            device=cluster_config.device,
            epochs=int(getattr(config, "training_epochs", 30)),
            lr=float(getattr(config, "learning_rate", 1e-3)),
            optimizer_name=str(getattr(config, "optimizer", "adam")),
            weight_decay=float(getattr(config, "weight_decay", 0.0) or 0.0),
            momentum=float(getattr(config, "momentum", 0.9) or 0.9),
            scheduler=getattr(config, "scheduler", None),
            scheduler_config=getattr(config, "scheduler_config", {}) or {},
        )
    
    # Save the trained model checkpoint
    output_dir = Path(cluster_config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    trained_checkpoint = checkpoint_dir / "trained_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'dataset_name': dataset_name,
        'num_classes': num_classes,
        # Architecture metadata for reproducibility when loading from paper scripts
        'cifar_resnet_stem_tweaked': resnet_cifar_stem_tweaked,
        'cifar_mobilenet_stride1': mobilenet_cifar_stride1,
    }, trained_checkpoint)
    logger.info(f"Saved trained model checkpoint to {trained_checkpoint}")
    
    return ClusterAnalysisExperiment(cluster_config, model, train_loader, test_loader)


def _finetune_model_for_dataset(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    epochs: int = 20,
    lr: float = 0.001,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    scheduler: Optional[str] = "cosine",
    scheduler_config: Optional[dict] = None,
    max_batches: Optional[int] = None,
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
    
    # If already well-trained, skip fine-tuning (useful when an explicit checkpoint is provided).
    if initial_acc > 0.90:
        logger.info(f"Model already well-trained (accuracy: {initial_acc:.2%}), skipping fine-tuning")
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
    
    opt_name = (optimizer_name or "adam").lower()
    if opt_name in {"sgd", "momentum", "sgd_momentum"}:
        optimizer = optim.SGD(
            [
                {"params": pretrained_params, "lr": lr * 0.1},
                {"params": new_params, "lr": lr},
            ],
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            nesterov=True,
        )
    elif opt_name in {"adamw"}:
        optimizer = optim.AdamW(
            [
                {"params": pretrained_params, "lr": lr * 0.1},
                {"params": new_params, "lr": lr},
            ],
            weight_decay=float(weight_decay),
        )
    else:
        optimizer = optim.Adam(
            [
                {"params": pretrained_params, "lr": lr * 0.1},
                {"params": new_params, "lr": lr},
            ],
            weight_decay=float(weight_decay),
        )

    # Scheduler (optional)
    sch_name = (str(scheduler).lower() if scheduler is not None else "none")
    scheduler_config = scheduler_config or {}
    if sch_name in {"none", "null", ""}:
        lr_scheduler = None
    elif sch_name in {"step", "steplr"}:
        step_size = int(scheduler_config.get("step_size", max(1, epochs // 3)))
        gamma = float(scheduler_config.get("gamma", 0.1))
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        # Default: cosine
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for bi, (x, y) in enumerate(train_loader):
            if max_batches is not None and bi >= int(max_batches):
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
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


def _regenerate_llm_visualizations(experiment, results: dict, output_dir: Path):
    """
    Regenerate visualizations for LLM experiments from saved results.
    
    Args:
        experiment: LLMAlignmentExperiment instance
        results: Loaded results dictionary from JSON
        output_dir: Output directory for plots
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    from alignment.analysis.visualization import UnifiedVisualizer
    
    # Determine plots directory
    if (output_dir / "figures").exists():
        plots_dir = output_dir / "figures"
    elif (output_dir / "plots").exists():
        plots_dir = output_dir / "plots"
    else:
        plots_dir = output_dir / "figures"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    viz = UnifiedVisualizer()
    
    # Extract data from results
    importance_scores = results.get("importance_scores", {})
    pruning_results = results.get("pruning_results", {})
    scar_results = results.get("scar_metrics", results.get("scar_scores", {}))
    supernode_results = results.get("supernode_analysis", {})
    halo_results = results.get("halo_analysis", {})
    
    logger.info(f"Regenerating plots to: {plots_dir}")
    logger.info(f"  - Found importance_scores: {bool(importance_scores)}")
    logger.info(f"  - Found pruning_results: {bool(pruning_results)}")
    logger.info(f"  - Found scar_results: {bool(scar_results)}")
    logger.info(f"  - Found supernode_results: {bool(supernode_results)}")
    logger.info(f"  - Found halo_results: {bool(halo_results)}")
    
    # Convert list values back to numpy arrays for plotting
    def to_numpy(data):
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, dict):
            return {k: to_numpy(v) for k, v in data.items()}
        return data
    
    importance_scores = to_numpy(importance_scores)
    scar_results = to_numpy(scar_results)
    
    # 1. SCAR metrics plots
    if scar_results:
        scar_plots_dir = plots_dir / "scar"
        scar_plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Layer-wise SCAR distributions
            for metric_name in ["scar_loss_proxy", "scar_activation_power", "scar_curvature", "scar_taylor"]:
                # Check if this metric exists in any layer
                has_metric = any(metric_name in layer_data for layer_data in scar_results.values() if isinstance(layer_data, dict))
                if has_metric:
                    try:
                        fig = viz.plot_scar_layer_scores(
                            scar_results,
                            metric_name=metric_name,
                            plot_type="violin",
                            save_path=scar_plots_dir / f"{metric_name}_layers.png",
                        )
                        plt.close(fig)
                        logger.info(f"  Generated: {metric_name}_layers.png")
                    except Exception as e:
                        logger.debug(f"Could not generate {metric_name} plot: {e}")
        except Exception as e:
            logger.warning(f"Error generating SCAR plots: {e}")
    
    # 2. Pruning comparison plots
    if pruning_results:
        pruning_plots_dir = plots_dir / "pruning"
        pruning_plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract pruning data for visualization
            all_pruning_data = []
            for method_name, method_results in pruning_results.items():
                if isinstance(method_results, dict):
                    for sparsity, data in method_results.items():
                        if isinstance(data, dict):
                            entry = {
                                "method": method_name,
                                "sparsity": float(sparsity) if isinstance(sparsity, str) else sparsity,
                                "perplexity": data.get("perplexity", data.get("test_perplexity")),
                                "accuracy": data.get("accuracy"),
                            }
                            if entry["perplexity"] is not None or entry["accuracy"] is not None:
                                all_pruning_data.append(entry)
            
            if all_pruning_data:
                # Generate pruning comparison plots
                fig = viz.plot_pruning_comparison_curves(
                    all_pruning_data,
                    metric="perplexity",
                    title="Pruning Methods Comparison (Perplexity)",
                    save_path=pruning_plots_dir / "pruning_comparison_perplexity.png",
                )
                if fig:
                    plt.close(fig)
                    logger.info("  Generated: pruning_comparison_perplexity.png")
        except Exception as e:
            logger.warning(f"Error generating pruning plots: {e}")
    
    # 3. Redundancy plots (if halo analysis results exist)
    if halo_results:
        redundancy_plots_dir = plots_dir / "redundancy"
        redundancy_plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for layer_name, layer_data in halo_results.items():
                if isinstance(layer_data, dict):
                    high_vals = layer_data.get("high_connected_redundancy", [])
                    low_vals = layer_data.get("low_connected_redundancy", [])
                    
                    if high_vals and low_vals:
                        high_vals = np.array(high_vals) if isinstance(high_vals, list) else high_vals
                        low_vals = np.array(low_vals) if isinstance(low_vals, list) else low_vals
                        
                        figs = viz.plot_redundancy_comparison(
                            high_vals, low_vals, layer_name,
                            save_dir=redundancy_plots_dir,
                        )
                        for fig in figs:
                            plt.close(fig)
                        logger.info(f"  Generated redundancy plots for: {layer_name}")
        except Exception as e:
            logger.warning(f"Error generating redundancy plots: {e}")
    
    # 4. Importance score histograms
    if importance_scores:
        importance_plots_dir = plots_dir / "importance"
        importance_plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for layer_name, layer_metrics in importance_scores.items():
                if isinstance(layer_metrics, dict):
                    for metric_name, values in layer_metrics.items():
                        if values is not None and len(values) > 0:
                            try:
                                values_np = np.array(values) if isinstance(values, list) else values
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.hist(values_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
                                ax.set_xlabel(metric_name)
                                ax.set_ylabel("Count")
                                ax.set_title(f"{metric_name} Distribution\n{layer_name}")
                                
                                safe_layer = layer_name.replace(".", "_").replace("/", "_")
                                save_path = importance_plots_dir / f"{metric_name}_{safe_layer}.png"
                                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                                plt.close(fig)
                            except Exception as e:
                                logger.debug(f"Could not plot {metric_name} for {layer_name}: {e}")
        except Exception as e:
            logger.warning(f"Error generating importance plots: {e}")
    
    logger.info(f"Visualization regeneration complete. Plots saved to: {plots_dir}")


def _create_job_directory(config, args, timestamp: str) -> Path:
    """
    Create job directory using the new unified structure.
    
    Priority for base_output_dir:
    1. --output-dir CLI argument (full path)
    2. config.base_output_dir (new field)
    3. config.experiment.output_dir or config.output.dir
    4. Default: ./results
    
    Returns:
        Path to the created job directory.
    """
    from alignment.infrastructure.storage import create_job_directory, get_slurm_job_id
    
    experiment_name = getattr(config, "name", "experiment")
    
    # If --output-dir is given as full path, use it directly
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    # Check for base_output_dir in config (new unified approach)
    base_output_dir = getattr(config, "base_output_dir", None)
    
    # If base_output_dir is set, use new job directory structure
    if base_output_dir:
        job_id = get_slurm_job_id()  # Will be None for non-SLURM runs
        output_dir = create_job_directory(
            base_output_dir=base_output_dir,
            experiment_name=experiment_name,
            timestamp=timestamp,
            job_id=job_id,
            create_subdirs=True,
        )
        return output_dir
    
    # Fallback to old behavior: results/{name}_{timestamp}
    output_dir = Path(f"results/{experiment_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for consistency
    for subdir in ["results", "logs", "checkpoints", "figures", "analysis"]:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Alignment Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--output-dir", type=str, help="Override output directory (full path)")
    parser.add_argument("--base-output-dir", type=str, help="Override base output directory (creates job subdir)")
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

    # Load config (support key=value overrides passed after args)
    # Example:
    #   python scripts/run_experiment.py --config ... name="llama3_8b_paper_main" supernode.protect_core=false
    from alignment.configs.config_loader import load_config_with_overrides as proper_load_config
    cli_overrides = [x for x in (unknown or []) if isinstance(x, str) and "=" in x]
    config = proper_load_config(args.config, overrides=overrides or None, cli_args=cli_overrides or None)
    
    # Override base_output_dir if provided via CLI
    if args.base_output_dir:
        config.base_output_dir = args.base_output_dir

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
        
        # Support both old 'plots' and new 'figures' directory names
        if (output_dir / "figures").exists():
            plots_dir = output_dir / "figures"
        else:
            plots_dir = output_dir / "plots"
        config.plots_dir = str(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        config_save_path = output_dir / "experiment_config.yaml"
        timestamp = None
    else:
        # Create job directory with new structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _create_job_directory(config, args, timestamp)

        config_save_path = output_dir / "experiment_config.yaml"
        config.save(config_save_path)

        # Set paths - use new subdirectory structure
        config.checkpoint_dir = str(output_dir / "checkpoints")
        config.log_dir = str(output_dir / "logs")
        config.experiment_dir = str(output_dir)

        # Use 'figures' subdirectory for new structure, 'plots' for compatibility
        if (output_dir / "figures").exists():
            plots_dir = output_dir / "figures"
        else:
            plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        config.plots_dir = str(plots_dir)

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging - use logs subdirectory if it exists (new structure)
    logs_subdir = output_dir / "logs"
    if logs_subdir.exists():
        log_file = logs_subdir / "experiment.log"
    else:
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
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}. "
                        f"Supported types: llm_alignment, llm_supernode, llm, "
                        f"alignment_analysis, vision_synergy, general_alignment, "
                        f"cluster_analysis, vision_cluster_analysis, metric_cluster_analysis")

    # Analysis-only mode
    if is_analysis_only:
        # Find results file - check both root and results subdirectory
        results_subdir = output_dir / "results"
        if results_subdir.exists():
            result_files = sorted(results_subdir.glob("results_*.json"))
        else:
            result_files = sorted(output_dir.glob("results_*.json"))
        
        if not result_files:
            raise FileNotFoundError(f"No results_*.json found in {output_dir} or {results_subdir}")
        results_path = result_files[-1]
        
        with results_path.open("r") as f:
            results = json.load(f)

        if isinstance(experiment, GeneralAlignmentExperiment):
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
        
        elif isinstance(experiment, LLMAlignmentExperiment):
            # For LLM experiments, regenerate visualizations from saved results
            logger.info("Regenerating LLM experiment visualizations from saved results...")
            
            # Setup experiment (loads model/tokenizer for any model-dependent plots)
            try:
                experiment.setup()
            except Exception as e:
                logger.warning(f"Could not setup model (may not be needed for plots): {e}")
            
            # Regenerate visualizations using the unified visualizer
            if getattr(config, "generate_plots", True):
                _regenerate_llm_visualizations(experiment, results, output_dir)
                logger.info("Regenerated LLM visualizations from existing results")
            
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

    # Save results - use results subdirectory if it exists (new structure)
    results_subdir = output_dir / "results"
    if results_subdir.exists():
        results_file = results_subdir / f"results_{timestamp}.json"
    else:
        results_file = output_dir / f"results_{timestamp}.json"

    def convert_to_serializable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        elif hasattr(obj, "item"):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
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
