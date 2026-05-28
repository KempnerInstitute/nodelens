"""Training-loop helpers for replaceability-aware learning rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .regularizers import (
    anti_decoupling_penalty,
    compact_hull_penalty,
    cross_layer_weights,
    peer_reconstructability_penalty,
    synergy_pair_penalty,
    task_aware_redundancy_loss,
    variance_floor_loss,
)
from .statistics import (
    average_squared_peer_correlation,
    capacity_masses,
    channel_correlation,
    compact_hull_from_correlation,
    peer_reconstructability_from_correlation,
)


def task_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return correct-class logit minus best incorrect-class logit."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1:
        targets = targets.reshape(-1)
    correct = logits.gather(1, targets.long().view(-1, 1)).squeeze(1)
    masked = logits.detach().clone()
    masked.scatter_(1, targets.long().view(-1, 1), float("-inf"))
    return correct.detach() - masked.max(dim=1).values.detach()


def pooled_channel_activations(activations: torch.Tensor) -> torch.Tensor:
    """Return per-example channel activations as ``[batch, channels]``."""
    if activations.ndim == 2:
        return activations
    if activations.ndim < 2:
        raise ValueError("activations must have at least 2 dimensions")
    reduce_dims = tuple(range(2, activations.ndim))
    return activations.mean(dim=reduce_dims)


def task_relevance_from_logits(activations: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Estimate per-channel task relevance by squared correlation with margin."""
    pooled = pooled_channel_activations(activations).float().detach()
    margin = task_margin(logits, targets).float().detach()
    if pooled.shape[0] != margin.numel():
        raise ValueError("activation batch size must match logits/targets batch size")
    if pooled.shape[0] < 2:
        return pooled.new_zeros(pooled.shape[1])

    pooled = pooled - pooled.mean(dim=0, keepdim=True)
    margin = margin - margin.mean()
    cov = (pooled * margin[:, None]).sum(dim=0)
    denom = torch.sqrt(pooled.square().sum(dim=0) * margin.square().sum() + eps)
    corr = cov / denom
    return corr.square().clamp(min=0.0, max=1.0)


def sigmoid_task_gate(task_relevance: torch.Tensor, *, temperature: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """Convert task relevance into a median-centered stop-gradient gate."""
    if task_relevance.numel() == 0:
        return task_relevance
    center = task_relevance.detach().median()
    temp = max(float(temperature), eps)
    return torch.sigmoid((task_relevance.detach() - center) / temp)


@dataclass
class LearningRuleConfig:
    """Configuration for training-time replaceability regularization."""

    method: str = "none"
    weight: float = 0.0
    schedule: str = "warmup"
    warmup_epochs: int = 0
    ramp_epochs: int = 0
    trigger_metric: str = "rho_cap"
    trigger_threshold: Optional[float] = None
    trigger_direction: str = "below"
    trigger_min_epoch: int = 0
    layer_filter: str = "conv2d"
    max_layers: Optional[int] = None
    skip_depthwise: bool = True
    pointwise_only: bool = False
    task_gate_temperature: float = 0.05
    task_gate_source: str = "task"
    rtc_ridge: float = 1e-3
    peer_proxy: str = "avg_corr2"
    variance_weight: float = 0.0
    variance_floor: float = 0.0
    cross_layer_alloc: str = "uniform"
    cross_layer_alpha: float = 1.0
    hull_max_size: int = 10
    hull_eps: float = 0.05
    grad_projection_strength: float = 0.0
    grad_projection_ema: float = 0.95
    grad_projection_ridge: float = 1e-3
    grad_projection_update_period: int = 1
    grad_projection_max_patches: int = 4096
    synergy_sample_pairs: int = 256
    anti_decouple_target_rho: float = 0.3

    @property
    def enabled(self) -> bool:
        loss_enabled = self.method.lower() not in {"", "none", "off", "bp"} and float(self.weight) != 0.0
        return loss_enabled or self.grad_projection_enabled

    @property
    def metric_triggered(self) -> bool:
        return str(self.schedule).lower() in {"metric", "metric_triggered", "triggered"}

    @property
    def grad_projection_enabled(self) -> bool:
        return float(self.grad_projection_strength) > 0.0


class ActivationCollector:
    """Collect layer activations through forward hooks."""

    def __init__(self, model: nn.Module, module_names: Iterable[str]):
        module_map = dict(model.named_modules())
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        for name in module_names:
            module = module_map.get(name)
            if module is None:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, tuple):
                output = output[0]
            if torch.is_tensor(output):
                self.activations[name] = output

        return hook

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class InputActivationCollector:
    """Collect Conv2d inputs through forward_pre_hooks.

    Used by the gradient-projection rule: it needs the input ``X`` to each
    convolutional layer so it can build the per-layer ``Sigma_X = X^T X / N``
    from unfolded patches and project the weight gradient onto the residual
    of ``Sigma_X w_i``.
    """

    def __init__(self, model: nn.Module, module_names: Iterable[str]):
        module_map = dict(model.named_modules())
        self.inputs: Dict[str, torch.Tensor] = {}
        self.modules: Dict[str, nn.Module] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        for name in module_names:
            module = module_map.get(name)
            if module is None or not isinstance(module, nn.Conv2d):
                continue
            self.modules[name] = module
            self._handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            if torch.is_tensor(x):
                self.inputs[name] = x.detach()

        return hook

    def clear(self) -> None:
        self.inputs.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def select_regularized_layers(model: nn.Module, config: LearningRuleConfig) -> List[str]:
    """Select modules to regularize from a model."""
    selected: List[str] = []
    filter_name = str(config.layer_filter or "conv2d").lower()
    for name, module in model.named_modules():
        if not name:
            continue
        if filter_name in {"conv", "conv2d"} and not isinstance(module, nn.Conv2d):
            continue
        if filter_name in {"linear", "fc"} and not isinstance(module, nn.Linear):
            continue
        if isinstance(module, nn.Conv2d):
            is_depthwise = module.groups == module.in_channels and module.out_channels == module.in_channels
            is_pointwise = tuple(module.kernel_size) == (1, 1)
            if config.skip_depthwise and is_depthwise:
                continue
            if config.pointwise_only and not is_pointwise:
                continue
        selected.append(name)
        if config.max_layers is not None and len(selected) >= int(config.max_layers):
            break
    return selected


def scheduled_regularizer_weight(config: LearningRuleConfig, epoch: int, *, trigger_epoch: Optional[int] = None) -> float:
    """Return the regularizer weight for an epoch."""
    base = float(config.weight)
    if config.metric_triggered:
        if trigger_epoch is None:
            return 0.0
        warmup = max(0, int(trigger_epoch))
    else:
        warmup = max(0, int(config.warmup_epochs))
    ramp = max(0, int(config.ramp_epochs))
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return base
    progress = min(1.0, float(epoch - warmup + 1) / float(ramp))
    return base * progress


def _peer_reconstructability_proxy(correlation: torch.Tensor, config: LearningRuleConfig) -> torch.Tensor:
    if str(config.peer_proxy).lower() in {"ridge", "full", "full_ridge"}:
        return peer_reconstructability_from_correlation(correlation)
    return average_squared_peer_correlation(correlation)


def residualized_task_relevance_from_logits(
    activations: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float = 1e-3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate task relevance after linear peer residualization.

    This is a minibatch proxy for residualized task credit (RTC). It computes
    per-example pooled channel activations, removes the linear prediction of
    each channel from its same-layer peers via a regularized precision matrix,
    and returns squared correlation between the residual channel and task
    margin. The result is detached and intended as a gate, not as a direct MI
    estimator with gradients through the peer regression.
    """
    pooled = pooled_channel_activations(activations).float().detach()
    margin = task_margin(logits, targets).float().detach()
    if pooled.shape[0] != margin.numel():
        raise ValueError("activation batch size must match logits/targets batch size")
    if pooled.shape[0] < 2:
        return pooled.new_zeros(pooled.shape[1])

    centered = pooled - pooled.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.square().sum(dim=0, keepdim=True) / max(centered.shape[0] - 1, 1) + eps)
    z = centered / std
    n_channels = z.shape[1]
    if n_channels <= 1:
        residual = z
    else:
        corr = z.T @ z / max(z.shape[0] - 1, 1)
        eye = torch.eye(n_channels, device=z.device, dtype=z.dtype)
        precision = torch.linalg.solve(corr + float(ridge) * eye, eye)
        residual = z @ precision
        residual = residual / torch.diagonal(precision).clamp_min(eps)[None, :]

    margin = margin - margin.mean()
    residual = residual - residual.mean(dim=0, keepdim=True)
    cov = (residual * margin[:, None]).sum(dim=0)
    denom = torch.sqrt(residual.square().sum(dim=0) * margin.square().sum() + eps)
    corr = cov / denom
    return corr.square().clamp(min=0.0, max=1.0)


def _vector_correlation(x: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    if x.numel() != y.numel() or x.numel() < 2:
        return x.new_tensor(0.0)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().sum() * y.square().sum() + eps)
    if float(denom.detach().cpu()) <= eps:
        return x.new_tensor(0.0)
    return ((x * y).sum() / denom).clamp(min=-1.0, max=1.0)


def _capacity_diagnostics(
    task_relevance: torch.Tensor,
    task_gate: torch.Tensor,
    peer_reconstructability: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Dict[str, float]:
    task = task_relevance.detach().float()
    gate = task_gate.detach().float()
    q = peer_reconstructability.detach().float().clamp(min=0.0, max=1.0)
    duplicate, non_replaceable = capacity_masses(task, q)
    task_mass = task.sum()
    return {
        "duplicate_task_mass": float(duplicate.cpu()),
        "non_replaceable_task_mass": float(non_replaceable.cpu()),
        "duplicate_task_fraction": float((duplicate / (task_mass + eps)).cpu()),
        "non_replaceable_task_fraction": float((non_replaceable / (task_mass + eps)).cpu()),
        "task_relevance_mean": float(task.mean().cpu()),
        "task_gate_mean": float(gate.mean().cpu()),
        "peer_reconstructability_mean": float(q.mean().cpu()),
        "rho_cap": float(_vector_correlation(task, q).cpu()),
    }


_HULL_METHODS = {"bp_hull", "hull"}
_TARD_METHODS = {"bp_tard", "tard", "bp_rtc_tard", "rtc_tard", "bp_rtc_gate", "rtc_gate"}
_RTP_METHODS = {"bp_rtp", "rtp"}
_SYNERGY_METHODS = {"bp_synergy", "synergy"}
_ANTIDECOUPLE_METHODS = {"bp_antidecouple", "antidecouple", "bp_anticoupling"}
_GATED_METHODS = _TARD_METHODS | _RTP_METHODS | _HULL_METHODS
_GATE_FREE_METHODS = _SYNERGY_METHODS | _ANTIDECOUPLE_METHODS
_RTC_METHODS = {"bp_rtc_tard", "rtc_tard", "bp_rtc_gate", "rtc_gate"}


def replaceability_regularization_loss(
    activations: Dict[str, torch.Tensor],
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: LearningRuleConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute a replaceability-aware regularization loss over collected activations."""
    method = str(config.method).lower()
    if method in {"", "none", "off", "bp"} or not activations:
        zero = logits.sum() * 0.0
        return zero, {"regularizer_raw": 0.0, "regularized_layers": 0.0}

    variance_total = logits.sum() * 0.0
    layer_losses: List[torch.Tensor] = []
    layer_capacity_stats: List[Dict[str, float]] = []
    layer_diagnostics: List[Dict[str, float]] = []

    for acts in activations.values():
        if not torch.is_tensor(acts) or acts.ndim < 2:
            continue
        gate = None
        relevance = None
        gate_relevance = None
        corr = None
        q = None
        hull_size = None
        hull_score = None

        if method in _GATED_METHODS:
            relevance = task_relevance_from_logits(acts, logits, targets)
            gate_relevance = relevance
            if str(config.task_gate_source).lower() in {"rtc", "residual", "residualized"} or method in _RTC_METHODS:
                gate_relevance = residualized_task_relevance_from_logits(acts, logits, targets, ridge=float(config.rtc_ridge))
            gate = sigmoid_task_gate(gate_relevance, temperature=float(config.task_gate_temperature))

        if method in {"decov", "bp_decov", "covariance"}:
            layer_loss = task_aware_redundancy_loss(acts, task_gate=None)
        elif method in _TARD_METHODS:
            corr = channel_correlation(acts)
            layer_loss = task_aware_redundancy_loss(acts, task_gate=gate, correlation=corr)
        elif method in _RTP_METHODS:
            corr = channel_correlation(acts)
            q = _peer_reconstructability_proxy(corr, config)
            layer_loss = peer_reconstructability_penalty(gate, q)
        elif method in _HULL_METHODS:
            corr = channel_correlation(acts)
            hull_size, full_r2, hull_score = compact_hull_from_correlation(
                corr,
                max_size=int(config.hull_max_size),
                eps=float(config.hull_eps),
                ridge=float(config.rtc_ridge),
            )
            q = full_r2
            layer_loss = compact_hull_penalty(gate, hull_score)
        elif method in _SYNERGY_METHODS:
            pooled = pooled_channel_activations(acts).float()
            margin = task_margin(logits, targets)
            layer_loss = synergy_pair_penalty(
                pooled,
                margin,
                sample_pairs=int(config.synergy_sample_pairs),
            )
        elif method in _ANTIDECOUPLE_METHODS:
            pooled = pooled_channel_activations(acts).float()
            margin = task_margin(logits, targets)
            layer_loss = anti_decoupling_penalty(
                pooled,
                margin,
                target_rho=float(config.anti_decouple_target_rho),
            )
        else:
            raise ValueError(f"Unknown learning rule method: {config.method}")

        layer_losses.append(layer_loss)

        capacity_payload: Dict[str, float] = {}
        if relevance is not None and gate is not None:
            if corr is None:
                corr = channel_correlation(acts)
            if q is None:
                q = _peer_reconstructability_proxy(corr, config)
            capacity_payload = _capacity_diagnostics(relevance, gate, q)
            capacity_payload["gate_relevance_mean"] = float(gate_relevance.detach().float().mean().cpu())
            if gate_relevance is not relevance:
                capacity_payload["rtc_relevance_mean"] = float(gate_relevance.detach().float().mean().cpu())
            if hull_size is not None and hull_score is not None:
                capacity_payload["hull_size_mean"] = float(hull_size.float().mean().cpu())
                capacity_payload["hull_score_mean"] = float(hull_score.detach().float().mean().cpu())
        elif method in _SYNERGY_METHODS:
            capacity_payload = {"synergy_raw_mean": float(-layer_loss.detach().cpu().item())}
        elif method in _ANTIDECOUPLE_METHODS:
            # Recompute rho_l for logging
            pooled_l = pooled_channel_activations(acts).float().detach()
            margin_l = task_margin(logits, targets).detach()
            p = pooled_l - pooled_l.mean(dim=0, keepdim=True)
            p_var = (p.square().sum(dim=0) / max(p.shape[0] - 1, 1)).clamp_min(1e-8)
            i_x = torch.log1p(p_var)
            p_norm = p.norm(dim=0, keepdim=True).clamp_min(1e-8)
            p_unit = p / p_norm
            t = margin_l.float() - margin_l.float().mean()
            t_unit = t / t.norm().clamp_min(1e-8)
            rho_iT = (p_unit * t_unit[:, None]).sum(dim=0)
            i_t = rho_iT.square()
            ix_c = i_x - i_x.mean()
            it_c = i_t - i_t.mean()
            denom = (ix_c.norm() * it_c.norm()).clamp_min(1e-8)
            rho_l = float(((ix_c * it_c).sum() / denom).cpu())
            capacity_payload = {
                "antidecouple_rho_l": rho_l,
                "antidecouple_target_rho": float(config.anti_decouple_target_rho),
                "antidecouple_gap": rho_l - float(config.anti_decouple_target_rho),
            }
        layer_capacity_stats.append(capacity_payload)
        layer_diagnostics.append(capacity_payload)

        if float(config.variance_weight) > 0.0 and float(config.variance_floor) > 0.0:
            variance_total = variance_total + variance_floor_loss(acts, min_std=float(config.variance_floor))

    layer_count = len(layer_losses)
    if layer_count == 0:
        zero = logits.sum() * 0.0
        return zero, {"regularizer_raw": 0.0, "regularized_layers": 0.0}

    weights = cross_layer_weights(
        layer_capacity_stats,
        mode=str(config.cross_layer_alloc),
        alpha=float(config.cross_layer_alpha),
    )
    weighted = layer_losses[0].new_zeros(())
    for w, layer_loss in zip(weights, layer_losses):
        weighted = weighted + float(w) * layer_loss
    total = weighted / float(layer_count)

    if float(config.variance_weight) > 0.0 and float(config.variance_floor) > 0.0:
        total = total + float(config.variance_weight) * variance_total / float(layer_count)

    stats = {
        "regularizer_raw": float(total.detach().cpu().item()),
        "regularized_layers": float(layer_count),
    }
    diagnostic_layers = sum(1 for d in layer_diagnostics if d)
    if diagnostic_layers:
        diagnostic_totals: Dict[str, float] = {}
        for d in layer_diagnostics:
            for key, value in d.items():
                diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value)
        for key, value in diagnostic_totals.items():
            stats[key] = float(value / float(diagnostic_layers))
        stats["diagnostic_layers"] = float(diagnostic_layers)
    cross_alloc_mode = str(config.cross_layer_alloc).lower()
    if cross_alloc_mode not in {"", "uniform", "none", "off"} and weights:
        stats["cross_layer_alloc_mode"] = cross_alloc_mode
        stats["cross_layer_weight_mean"] = float(sum(weights) / len(weights))
        stats["cross_layer_weight_max"] = float(max(weights))
        stats["cross_layer_weight_min"] = float(min(weights))
    return total, stats


class _SigmaXEMA:
    """Per-layer EMA of the input second-moment matrix ``Sigma_X``.

    For each registered Conv2d we maintain an EMA of ``X^T X / N`` over
    unfolded input patches. The EMA is updated lazily (on demand) when the
    gradient-projection rule needs a fresh covariance.
    """

    def __init__(self, decay: float = 0.95, max_patches: int = 4096) -> None:
        self.decay = float(decay)
        self.max_patches = int(max_patches)
        self.state: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}

    def update(self, name: str, sigma: torch.Tensor) -> torch.Tensor:
        prev = self.state.get(name)
        if prev is None or prev.shape != sigma.shape:
            self.state[name] = sigma
        else:
            self.state[name] = self.decay * prev + (1.0 - self.decay) * sigma
        self.counts[name] = self.counts.get(name, 0) + 1
        return self.state[name]


def _layer_sigma_x(module: nn.Conv2d, x: torch.Tensor, *, max_patches: int, eps: float = 1e-8) -> torch.Tensor:
    """Compute the channel-input second-moment matrix ``X^T X / N`` for a Conv2d.

    Inputs ``x`` are unfolded with the layer's kernel/padding/stride/dilation,
    flattened to ``[N_patch, F]`` with ``F = C_in * kH * kW``, and optionally
    subsampled to ``max_patches`` for cost control.
    """
    import torch.nn.functional as F

    patches = F.unfold(
        x.detach(),
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    B, F_dim, L = patches.shape
    patches = patches.permute(0, 2, 1).reshape(B * L, F_dim).float()
    if patches.shape[0] > max_patches:
        idx = torch.randint(0, patches.shape[0], (max_patches,), device=patches.device)
        patches = patches[idx]
    n = max(patches.shape[0], 1)
    return patches.T @ patches / float(n)


def project_signal_power_gradients(
    inputs: Dict[str, torch.Tensor],
    modules: Dict[str, nn.Module],
    sigma_state: _SigmaXEMA,
    config: LearningRuleConfig,
    *,
    step: int = 0,
) -> Dict[str, float]:
    """Project each Conv2d's weight gradient orthogonally to ``Sigma_X w_i``.

    For each registered Conv2d layer with input ``X`` cached this minibatch:

    1. Update the EMA of ``Sigma_X = X^T X / N`` (every ``update_period`` steps).
    2. For each output channel ``i`` with weight ``w_i`` (flattened over the
       kernel), compute ``sigma_w_i = (Sigma_X + ridge*I) w_i``.
    3. Replace the channel's weight gradient ``g_i`` with
       ``g_i - alpha * <g_i, sigma_w_i> / ||sigma_w_i||^2 * sigma_w_i``.

    ``alpha = grad_projection_strength``. ``alpha == 1`` removes the full
    signal-power component; ``alpha == 0`` is a no-op. The intent is to
    suppress the part of the loss gradient that points along the
    input-capture direction predicted by the two-axis paper, leaving the
    residualized-target direction.

    Returns per-call summary stats (mean cosine before projection, mean
    norm-shrink ratio, layer count). Modifies ``module.weight.grad`` in place.
    """
    alpha = float(config.grad_projection_strength)
    if alpha <= 0.0:
        return {"grad_projection_layers": 0.0}

    ridge = float(config.grad_projection_ridge)
    update_period = max(1, int(config.grad_projection_update_period))
    eps = 1e-8

    total_cos = 0.0
    total_shrink = 0.0
    total_layers = 0

    for name, module in modules.items():
        if not isinstance(module, nn.Conv2d):
            continue
        weight = module.weight
        if weight.grad is None:
            continue
        x = inputs.get(name)
        if x is None:
            cached = sigma_state.state.get(name)
            if cached is None:
                continue
            sigma = cached
        else:
            if int(step) % update_period == 0 or name not in sigma_state.state:
                fresh = _layer_sigma_x(module, x, max_patches=int(config.grad_projection_max_patches))
                sigma = sigma_state.update(name, fresh)
            else:
                sigma = sigma_state.state.get(name)
                if sigma is None:
                    fresh = _layer_sigma_x(module, x, max_patches=int(config.grad_projection_max_patches))
                    sigma = sigma_state.update(name, fresh)

        c_out = weight.shape[0]
        f_dim = weight[0].numel()
        if sigma.shape[0] != f_dim:
            continue

        w_flat = weight.detach().reshape(c_out, f_dim).to(sigma.dtype)
        g_flat = weight.grad.detach().reshape(c_out, f_dim).to(sigma.dtype)
        ridge_eye = ridge * torch.eye(f_dim, device=sigma.device, dtype=sigma.dtype)
        sigma_w = w_flat @ (sigma + ridge_eye)  # [C_out, F]
        sigma_w_sqnorm = sigma_w.square().sum(dim=1).clamp_min(eps)  # [C_out]
        proj_coeff = (g_flat * sigma_w).sum(dim=1) / sigma_w_sqnorm  # [C_out]

        g_norm = g_flat.norm(dim=1).clamp_min(eps)
        sw_norm = sigma_w.norm(dim=1).clamp_min(eps)
        cos_g_sw = ((g_flat * sigma_w).sum(dim=1) / (g_norm * sw_norm)).clamp(min=-1.0, max=1.0)
        total_cos += float(cos_g_sw.abs().mean().cpu())

        new_g = g_flat - alpha * proj_coeff.unsqueeze(1) * sigma_w
        shrink = (new_g.norm(dim=1).clamp_min(eps) / g_norm).clamp_max(1.0)
        total_shrink += float(shrink.mean().cpu())

        new_grad = new_g.to(weight.grad.dtype).reshape_as(weight.grad)
        weight.grad.copy_(new_grad)
        total_layers += 1

    if total_layers == 0:
        return {"grad_projection_layers": 0.0}
    return {
        "grad_projection_layers": float(total_layers),
        "grad_projection_abs_cos_mean": float(total_cos / total_layers),
        "grad_projection_norm_shrink_mean": float(total_shrink / total_layers),
        "grad_projection_strength": float(alpha),
    }
