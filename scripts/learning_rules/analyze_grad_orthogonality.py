#!/usr/bin/env python3
"""Analyze the cosine between the analytic input-capture gradient and the analytic
target-relevance gradient for trained checkpoints.

This is the falsification test for the diagnosis that backpropagation already
implements residualized credit assignment implicitly via gradient orthogonality
(Safaai et al. 2026, two-axis paper, Proposition: Gaussian residualization of
the two gradients).

For a Conv2d layer with patch-flattened input ``X`` (shape ``[N_patch, F]``,
``F = C_in * kH * kW``) and per-output-channel flattened weight ``w_i``:

    sigma_w_i = Sigma_X @ w_i              # input-capture direction
    b_i       = w_i @ c                    # raw task-cov of channel i
    D_i       = w_i @ sigma_w_i            # channel variance
    g_T_i     = c - (b_i / D_i) * sigma_w_i  # target-relevance direction
    cos_i     = (sigma_w_i . g_T_i) / (||sigma_w_i|| ||g_T_i||)

``c = cov_patch(X, T_broadcast)`` with ``T`` the per-image task margin
broadcast to every spatial patch of that image.

The script loads each run's saved ``trained_model.pth``, reconstructs the model
from ``experiment_config.yaml``, runs a calibration forward pass on the
training dataset, and writes one row per ``(run_dir, layer_name, channel)``
plus a per-layer summary row. The result is the cos-trajectory data for the
"BP already does it" plot.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class LayerStats:
    layer_name: str
    n_channels: int
    n_patches: int
    cos_mean: float
    cos_std: float
    abs_cos_mean: float
    cos_p25: float
    cos_p50: float
    cos_p75: float
    sigma_norm_mean: float
    g_t_norm_mean: float


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def build_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """Reconstruct a model matching the run's experiment_config.yaml.

    Mirrors the small subset of run_experiment.py needed to instantiate the
    model architecture (no checkpoint loading here).
    """
    requested_model_name = str(config.get("model_name", "")).lower()
    model_cfg = config.get("model_config") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    num_classes = int(config.get("num_classes", model_cfg.get("num_classes", 100)))

    if requested_model_name in {"torchvision_model", "torchvision"}:
        resolved_model_name = str(model_cfg.get("model_name", "")).lower()
    else:
        resolved_model_name = requested_model_name

    is_custom_cifar_resnet = resolved_model_name in {
        "cifar_resnet18",
        "resnet18_cifar",
        "cifar_resnet18_width",
    }
    if is_custom_cifar_resnet:
        from nodelens.models.architectures.cifar_resnet import cifar_resnet18

        return cifar_resnet18(
            num_classes=num_classes,
            width_multiplier=float(model_cfg.get("width_multiplier", 1.0)),
            base_width=int(model_cfg.get("base_width", 64)),
        )

    if resolved_model_name in {"cifar_vgg16", "vgg16_cifar"}:
        from nodelens.models.architectures.cifar_vgg import cifar_vgg16

        return cifar_vgg16(
            num_classes=num_classes,
            width_multiplier=float(model_cfg.get("width_multiplier", 1.0)),
        )

    import torchvision

    aliases = {"vgg16": "vgg16_bn", "mobilenetv2": "mobilenet_v2", "mobilenet": "mobilenet_v2"}
    tv_func_name = aliases.get(resolved_model_name, resolved_model_name)
    if not hasattr(torchvision.models, tv_func_name):
        raise ValueError(f"Cannot reconstruct model {resolved_model_name} from this script.")
    tv_func = getattr(torchvision.models, tv_func_name)
    model = tv_func(weights=None)
    if hasattr(model, "fc") and num_classes != model.fc.out_features:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential) and num_classes != model.classifier[-1].out_features:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    dataset_name = str(config.get("dataset_name", "")).lower()
    if dataset_name:
        from nodelens.models.hub import adapt_model_for_dataset

        adapt_model_for_dataset(model, resolved_model_name, dataset_name, pretrained=False)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")


def make_cifar100_loader(root: str, batch_size: int, num_samples: int, num_workers: int) -> torch.utils.data.DataLoader:
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))])
    dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=transform)
    if num_samples and num_samples < len(dataset):
        gen = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(dataset), generator=gen)[:num_samples]
        dataset = torch.utils.data.Subset(dataset, idx.tolist())
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def task_margin_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, targets.long().view(-1, 1)).squeeze(1)
    masked = logits.detach().clone()
    masked.scatter_(1, targets.long().view(-1, 1), float("-inf"))
    return (correct - masked.max(dim=1).values).detach()


@dataclass
class _PatchAccum:
    sum_x: torch.Tensor  # [F]
    sum_xt: torch.Tensor  # [F]
    sum_xx: torch.Tensor  # [F, F]
    sum_t: torch.Tensor  # scalar
    sum_t2: torch.Tensor  # scalar
    n: int


def _accumulate_layer(layer: nn.Conv2d, accum: Optional[_PatchAccum], x: torch.Tensor, t: torch.Tensor, max_patches: int) -> _PatchAccum:
    """Accumulate patch sums for one minibatch into ``accum``.

    ``x`` is the layer input ``[B, C_in, H, W]``; ``t`` is the per-image task
    margin ``[B]``. Patches are extracted via ``F.unfold`` matching the layer's
    kernel/padding/stride; per-image ``t`` is broadcast to all patches of that
    image. Patches are subsampled per-batch to keep ``F^2`` accumulators bounded.
    """
    patches = F.unfold(
        x,
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=layer.padding,
        stride=layer.stride,
    )
    # patches: [B, F, L] where F = C_in * kH * kW, L = H_out * W_out
    B, F_dim, L = patches.shape
    patches = patches.permute(0, 2, 1).reshape(B * L, F_dim).float()
    t_repeat = t.float().repeat_interleave(L)
    if patches.shape[0] > max_patches:
        gen = torch.Generator(device=patches.device).manual_seed(0)
        idx = torch.randperm(patches.shape[0], generator=gen, device=patches.device)[:max_patches]
        patches = patches[idx]
        t_repeat = t_repeat[idx]
    if accum is None:
        accum = _PatchAccum(
            sum_x=torch.zeros(F_dim, device=patches.device, dtype=torch.float64),
            sum_xt=torch.zeros(F_dim, device=patches.device, dtype=torch.float64),
            sum_xx=torch.zeros(F_dim, F_dim, device=patches.device, dtype=torch.float64),
            sum_t=torch.zeros((), device=patches.device, dtype=torch.float64),
            sum_t2=torch.zeros((), device=patches.device, dtype=torch.float64),
            n=0,
        )
    p64 = patches.double()
    t64 = t_repeat.double()
    accum.sum_x += p64.sum(dim=0)
    accum.sum_xt += (p64 * t64[:, None]).sum(dim=0)
    accum.sum_xx += p64.T @ p64
    accum.sum_t += t64.sum()
    accum.sum_t2 += (t64 * t64).sum()
    accum.n += int(p64.shape[0])
    return accum


def _finalize_accum(accum: _PatchAccum, ridge: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert raw patch sums into ``Sigma_X`` (centered, ridged) and ``c``."""
    n = max(accum.n, 1)
    mean_x = accum.sum_x / n
    mean_t = accum.sum_t / n
    cov = accum.sum_xx / n - torch.outer(mean_x, mean_x)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    cov = cov + ridge * eye
    c = accum.sum_xt / n - mean_x * mean_t
    return cov, c


def _layer_cosines(weight: torch.Tensor, sigma: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-output-channel cosines and norms.

    ``weight`` is ``[C_out, F]`` (already flattened over kernel).
    Returns ``(cos, sigma_norm, g_t_norm)`` each shape ``[C_out]``.
    """
    eps = 1e-12
    sigma_w = weight @ sigma  # [C_out, F]
    b = weight @ c  # [C_out]
    D = (weight * sigma_w).sum(dim=1).clamp_min(eps)  # [C_out]
    g_t = c[None, :] - (b / D)[:, None] * sigma_w  # [C_out, F]
    sigma_norm = sigma_w.norm(dim=1).clamp_min(eps)
    g_t_norm = g_t.norm(dim=1).clamp_min(eps)
    cos = (sigma_w * g_t).sum(dim=1) / (sigma_norm * g_t_norm)
    return cos.clamp(min=-1.0, max=1.0), sigma_norm, g_t_norm


def analyze_run(
    run_dir: Path,
    *,
    data_root: str,
    device: str,
    calibration_size: int,
    batch_size: int,
    max_patches_per_layer: int,
    max_layer_features: int,
    ridge: float,
) -> Tuple[List[LayerStats], List[Dict[str, Any]]]:
    config_path = run_dir / "experiment_config.yaml"
    ckpt_path = run_dir / "checkpoints" / "trained_model.pth"
    if not config_path.exists() or not ckpt_path.exists():
        return [], []

    config = load_yaml(config_path)
    model = build_model_from_config(config)
    load_checkpoint(model, ckpt_path)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    conv_layers: List[Tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and not (module.groups == module.in_channels and module.out_channels == module.in_channels):
            f_dim = int(module.in_channels * module.kernel_size[0] * module.kernel_size[1])
            if f_dim <= max_layer_features:
                conv_layers.append((name, module))

    if not conv_layers:
        return [], []

    accums: Dict[str, Optional[_PatchAccum]] = {name: None for name, _ in conv_layers}
    handles: List[torch.utils.hooks.RemovableHandle] = []
    layer_inputs: Dict[str, torch.Tensor] = {}

    def make_hook(layer_name: str):
        def hook(_module, inputs, _output):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            if torch.is_tensor(x):
                layer_inputs[layer_name] = x.detach()

        return hook

    for name, module in conv_layers:
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        loader = make_cifar100_loader(data_root, batch_size, calibration_size, num_workers=0)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                margin = task_margin_from_logits(logits, y)
                for name, module in conv_layers:
                    if name not in layer_inputs:
                        continue
                    accums[name] = _accumulate_layer(module, accums[name], layer_inputs[name], margin, max_patches_per_layer)
                layer_inputs.clear()
    finally:
        for h in handles:
            h.remove()

    summary_rows: List[LayerStats] = []
    channel_rows: List[Dict[str, Any]] = []
    for name, module in conv_layers:
        accum = accums.get(name)
        if accum is None or accum.n == 0:
            continue
        sigma, c = _finalize_accum(accum, ridge)
        weight = module.weight.detach().to(sigma.device).to(sigma.dtype).reshape(module.out_channels, -1)
        cos, sigma_norm, g_t_norm = _layer_cosines(weight, sigma, c)
        cos_cpu = cos.detach().float().cpu()
        sigma_norm_cpu = sigma_norm.detach().float().cpu()
        g_t_norm_cpu = g_t_norm.detach().float().cpu()

        summary_rows.append(
            LayerStats(
                layer_name=name,
                n_channels=int(cos_cpu.numel()),
                n_patches=int(accum.n),
                cos_mean=float(cos_cpu.mean()),
                cos_std=float(cos_cpu.std(unbiased=False)),
                abs_cos_mean=float(cos_cpu.abs().mean()),
                cos_p25=float(cos_cpu.quantile(0.25)),
                cos_p50=float(cos_cpu.quantile(0.5)),
                cos_p75=float(cos_cpu.quantile(0.75)),
                sigma_norm_mean=float(sigma_norm_cpu.mean()),
                g_t_norm_mean=float(g_t_norm_cpu.mean()),
            )
        )
        for i in range(int(cos_cpu.numel())):
            channel_rows.append(
                {
                    "run_dir": str(run_dir),
                    "layer_name": name,
                    "channel": i,
                    "cos": float(cos_cpu[i]),
                    "sigma_norm": float(sigma_norm_cpu[i]),
                    "g_t_norm": float(g_t_norm_cpu[i]),
                }
            )

    return summary_rows, channel_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_runs(results_root: Path, pattern: str) -> List[Path]:
    if results_root.is_file():
        return [results_root.parent]
    matches = sorted(p.parent for p in results_root.glob(pattern))
    return [p for p in matches if (p / "checkpoints" / "trained_model.pth").exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, default=None, help="Single run directory to analyze.")
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/learning_rules/resnet18_cifar100",
        help="Sweep root if --run-dir is not given.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/experiment_config.yaml",
        help="Glob pattern (relative to results-root) for run discovery.",
    )
    parser.add_argument("--data-root", type=str, default="./data", help="CIFAR-100 root.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-patches-per-layer", type=int, default=8192)
    parser.add_argument("--max-layer-features", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="projects/replaceability_learning_rules/paper_artifacts/tables",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="grad_orthogonality",
        help="Filename prefix for the two output CSVs.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Cap on runs analyzed (debug).")
    args = parser.parse_args()

    if args.run_dir:
        runs = [Path(args.run_dir)]
    else:
        runs = discover_runs(Path(args.results_root), args.pattern)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    if not runs:
        print("No runs found.")
        return 1

    print(f"analyzing {len(runs)} runs on {args.device}")
    all_summary: List[Dict[str, Any]] = []
    all_channel: List[Dict[str, Any]] = []
    for run_dir in runs:
        print(f"  - {run_dir.name}")
        try:
            summary, channels = analyze_run(
                run_dir,
                data_root=args.data_root,
                device=args.device,
                calibration_size=args.calibration_size,
                batch_size=args.batch_size,
                max_patches_per_layer=args.max_patches_per_layer,
                max_layer_features=args.max_layer_features,
                ridge=args.ridge,
            )
        except Exception as exc:  # keep going across runs
            print(f"    [error] {exc}")
            continue
        for s in summary:
            row = {"run_dir": str(run_dir), **s.__dict__}
            all_summary.append(row)
        all_channel.extend(channels)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / f"{args.out_prefix}_summary.csv", all_summary)
    write_csv(out_dir / f"{args.out_prefix}_channels.csv", all_channel)
    print(f"wrote {out_dir/(args.out_prefix + '_summary.csv')} ({len(all_summary)} rows)")
    print(f"wrote {out_dir/(args.out_prefix + '_channels.csv')} ({len(all_channel)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
