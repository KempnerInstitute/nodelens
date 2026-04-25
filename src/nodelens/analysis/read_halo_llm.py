"""
Optional cross-layer "read-halo" analysis for transformer FFNs.

This module is intentionally self-contained and **not** used by default pruning.

High-level idea
---------------
Given a layer ℓ, suppose we have identified a set of hidden-dimension positions S (in the residual stream)
that are strongly influenced by supernodes in layer ℓ (e.g., high mass in the aggregated supernode write pattern).

In the *next* layer ℓ+1, each intermediate FFN channel j reads from the residual stream through two linear maps:
  gate_proj[j, :] and up_proj[j, :].

We define a "read connectivity" score for channel j:

  ReadConn_j = (|W_gate[j,S]| + |W_up[j,S]|) / (|W_gate[j,:]| + |W_up[j,:]|).

The read-halo is the top-η fraction of channels by ReadConn.

We then test whether read-halo channels are redundant to each other by measuring within-set redundancy
on their *actual* intermediate activations u (the input to down_proj).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ReadHaloConfig:
    enabled: bool = False
    read_halo_fraction: float = 0.10
    num_texts: int = 4
    max_length: int = 256
    random_seed: int = 0
    # Optional: "bus dependence" probe.
    # If enabled, we run a paired forward pass (baseline vs bus-ablation) and measure
    # mean |Δu_j| per next-layer FFN channel j, then relate it to ReadConn_j.
    compute_dependence: bool = False
    # For speed: limit to a subset of channels when plotting (computation still runs on all channels).
    dependence_max_points: int = 20000


def _resolve_module(module_dict: Dict[str, nn.Module], name: str) -> Optional[nn.Module]:
    """Best-effort resolve module by name with common prefix/suffix variants."""
    if name in module_dict:
        return module_dict[name]
    # Strip a single leading "model." (e.g., LlamaModel uses "layers.*" vs LlamaForCausalLM uses "model.layers.*").
    if name.startswith("model."):
        alt0 = name[len("model.") :]
        if alt0 in module_dict:
            return module_dict[alt0]
    if name.startswith("model.layers."):
        alt = "model.model." + name
        if alt in module_dict:
            return module_dict[alt]
    for k, v in module_dict.items():
        if k.endswith(name):
            return v
    return None


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation (robust, no SciPy dependency)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    # Rank transform
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
    rho = float((ra @ rb) / denom)
    return rho if np.isfinite(rho) else 0.0


def _mean_abs_corr(x: torch.Tensor) -> float:
    """Mean absolute off-diagonal correlation for a [T, K] activation matrix."""
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    k = int(x.shape[1])
    if k <= 1:
        return 0.0
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    x = x / std
    corr = (x.T @ x) / max(1, int(x.shape[0]) - 1)
    corr = torch.clamp(corr, -1.0, 1.0)
    mask = ~torch.eye(k, dtype=torch.bool)
    return float(corr[mask].abs().mean().item())


@torch.no_grad()
def compute_next_layer_read_halo(
    *,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    source_layer_name: str,
    next_layer_idx: int,
    follower_indices: np.ndarray,
    calibration_texts: List[str],
    cfg: ReadHaloConfig,
    plots_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compute read-halo redundancy for a single (layer ℓ -> layer ℓ+1) transition.

    follower_indices: hidden-dimension indices (d) that are strongly influenced by supernodes in layer ℓ.
    """
    rh = float(cfg.read_halo_fraction)
    rh = max(0.0, min(1.0, rh))
    if rh <= 0.0:
        return {"error": "read_halo_fraction <= 0"}
    if follower_indices is None or len(follower_indices) == 0:
        return {"error": "empty follower_indices"}
    if not calibration_texts:
        return {"error": "no calibration texts"}

    module_dict = dict(model.named_modules())
    gate_name = f"model.layers.{next_layer_idx}.mlp.gate_proj"
    up_name = f"model.layers.{next_layer_idx}.mlp.up_proj"
    down_name = f"model.layers.{next_layer_idx}.mlp.down_proj"

    gate_mod = _resolve_module(module_dict, gate_name)
    up_mod = _resolve_module(module_dict, up_name)
    down_mod = _resolve_module(module_dict, down_name)

    if gate_mod is None or up_mod is None or down_mod is None:
        return {
            "error": "could not resolve next-layer modules",
            "gate_found": gate_mod is not None,
            "up_found": up_mod is not None,
            "down_found": down_mod is not None,
        }
    if not (hasattr(gate_mod, "weight") and hasattr(up_mod, "weight")):
        return {"error": "gate/up missing weights"}

    # gate/up weights: [intermediate_dim, hidden_dim]
    Wg = gate_mod.weight.detach().float().cpu().abs()
    Wu = up_mod.weight.detach().float().cpu().abs()
    if Wg.ndim != 2 or Wu.ndim != 2 or Wg.shape != Wu.shape:
        return {"error": f"unexpected gate/up shapes gate={tuple(Wg.shape)} up={tuple(Wu.shape)}"}

    intermediate_dim, hidden_dim = int(Wg.shape[0]), int(Wg.shape[1])
    S = np.asarray(follower_indices, dtype=np.int64)
    S = S[(S >= 0) & (S < hidden_dim)]
    if S.size == 0:
        return {"error": "follower_indices out of bounds"}

    eps = 1e-8
    num = Wg[:, S].sum(dim=1) + Wu[:, S].sum(dim=1)
    den = Wg.sum(dim=1) + Wu.sum(dim=1) + eps
    read_conn = (num / den).clamp(0.0, 1.0)  # [intermediate_dim]

    num_halo = max(1, int(rh * intermediate_dim))
    halo_vals, halo_idx = torch.topk(read_conn, k=num_halo, largest=True)
    halo_idx_np = halo_idx.numpy()

    # Random baseline
    g = torch.Generator()
    g.manual_seed(int(cfg.random_seed) + int(next_layer_idx))
    rand_idx = torch.randperm(intermediate_dim, generator=g)[:num_halo].cpu().numpy()

    # Capture next-layer u (input to down_proj)
    halo_acts: List[torch.Tensor] = []
    rand_acts: List[torch.Tensor] = []

    def hook(_m: nn.Module, inputs: Tuple[torch.Tensor, ...], _out: torch.Tensor):
        if not inputs or inputs[0] is None:
            return
        u = inputs[0].detach().float()
        if u.ndim == 3:
            u = u.reshape(-1, u.shape[-1])
        halo_acts.append(u[:, halo_idx_np].cpu())
        rand_acts.append(u[:, rand_idx].cpu())

    h = down_mod.register_forward_hook(hook)
    try:
        model.eval()
        for text in calibration_texts[: max(1, int(cfg.num_texts))]:
            toks = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(cfg.max_length),
                padding=False,
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            try:
                model(**toks)
            except Exception:
                # Best-effort: ignore problematic prompts / OOM-safe failures.
                pass
    finally:
        h.remove()

    if not halo_acts or not rand_acts:
        return {"error": "no activations captured"}

    all_halo = torch.cat(halo_acts, dim=0)
    all_rand = torch.cat(rand_acts, dim=0)

    halo_red = _mean_abs_corr(all_halo)
    rand_red = _mean_abs_corr(all_rand)
    effect = halo_red - rand_red

    # ---------------------------------------------------------------------
    # Optional: dependence on the bus support S (activation-level, causal-ish)
    # ---------------------------------------------------------------------
    dependence: Optional[Dict[str, Any]] = None
    if bool(getattr(cfg, "compute_dependence", False)):
        # Best-effort: ablate bus dims at the *input* to gate/up (hidden stream of this block).
        # Prefer post_attention_layernorm output; fallback to gate/up pre-hooks.
        ln_name = f"model.layers.{next_layer_idx}.post_attention_layernorm"
        ln_mod = _resolve_module(module_dict, ln_name)

        S_gpu = torch.as_tensor(S, dtype=torch.long, device=device)

        def _ablate_hidden(x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, hidden_dim] (or [T, hidden_dim])
            if x is None:
                return x
            if x.ndim >= 2:
                # Clone to avoid in-place surprises on shared tensors.
                y = x.clone()
                y.index_fill_(-1, S_gpu, 0.0)
                return y
            return x

        @torch.no_grad()
        def _capture_u_for_text(text: str, *, ablate: bool) -> Optional[torch.Tensor]:
            u_holder: Dict[str, torch.Tensor] = {}

            def down_in_hook(_m: nn.Module, inputs: Tuple[torch.Tensor, ...], _out: torch.Tensor):
                if not inputs or inputs[0] is None:
                    return
                u = inputs[0].detach().float()
                if u.ndim == 3:
                    u = u.reshape(-1, u.shape[-1])
                u_holder["u"] = u.cpu()

            # Bus ablation hook
            ln_handle = None
            gate_handle = None
            up_handle = None

            if ablate:
                if ln_mod is not None:

                    def ln_hook(_m: nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                        return _ablate_hidden(out)

                    ln_handle = ln_mod.register_forward_hook(ln_hook)
                else:
                    # Fallback: ablate inputs to both gate and up (may clone twice per forward).
                    def pre_hook(_m: nn.Module, inputs: Tuple[torch.Tensor, ...]):
                        if not inputs or inputs[0] is None:
                            return inputs
                        x = inputs[0]
                        return (_ablate_hidden(x),) + tuple(inputs[1:])

                    gate_handle = gate_mod.register_forward_pre_hook(pre_hook)
                    up_handle = up_mod.register_forward_pre_hook(pre_hook)

            down_handle = down_mod.register_forward_hook(down_in_hook)
            try:
                toks = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(cfg.max_length),
                    padding=False,
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                try:
                    model(**toks)
                except Exception:
                    pass
            finally:
                down_handle.remove()
                if ln_handle is not None:
                    ln_handle.remove()
                if gate_handle is not None:
                    gate_handle.remove()
                if up_handle is not None:
                    up_handle.remove()

            return u_holder.get("u")

        # Stream-accumulate mean |Δu| per channel
        sum_abs_delta = torch.zeros(intermediate_dim, dtype=torch.float32)
        total_tokens = 0

        for text in calibration_texts[: max(1, int(cfg.num_texts))]:
            u0 = _capture_u_for_text(text, ablate=False)
            u1 = _capture_u_for_text(text, ablate=True)
            if u0 is None or u1 is None:
                continue
            if u0.shape != u1.shape:
                continue
            # u: [T, m]
            t = int(u0.shape[0])
            if t <= 0:
                continue
            sum_abs_delta += (u1 - u0).abs().sum(dim=0)
            total_tokens += t

        if total_tokens > 0:
            mean_abs_delta = (sum_abs_delta / float(total_tokens)).numpy()
            read_conn_np = read_conn.numpy()
            rho = _spearman(read_conn_np, mean_abs_delta)

            # Group summaries: read-halo vs random
            num_halo = int(num_halo)
            read_halo_idx = halo_idx_np
            rand_idx2 = rand_idx  # from earlier random baseline (size-matched)
            halo_mean = float(np.mean(mean_abs_delta[read_halo_idx])) if read_halo_idx.size else 0.0
            rand_mean = float(np.mean(mean_abs_delta[rand_idx2])) if rand_idx2.size else 0.0

            dependence = {
                "description": "Bus dependence of next-layer FFN channels under input-subspace ablation (mean |Δu|)",
                "support_size": int(S.size),
                "num_texts": int(min(len(calibration_texts), max(1, int(cfg.num_texts)))),
                "spearman_readconn_vs_mean_abs_delta_u": float(rho),
                "mean_abs_delta_u": {
                    "read_halo": halo_mean,
                    "random": rand_mean,
                    "difference": float(halo_mean - rand_mean),
                },
                "delta_u_summary": {
                    "mean": float(np.mean(mean_abs_delta)),
                    "std": float(np.std(mean_abs_delta)),
                    "min": float(np.min(mean_abs_delta)),
                    "max": float(np.max(mean_abs_delta)),
                },
            }

            # Optional plots
            if plots_dir is not None:
                try:
                    import matplotlib.pyplot as plt

                    out_dir = Path(plots_dir) / "read_halo"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    suffix = source_layer_name.replace(".", "_")

                    # Downsample for scatter readability
                    max_pts = int(getattr(cfg, "dependence_max_points", 20000) or 20000)
                    n = int(read_conn_np.size)
                    if n > max_pts:
                        g = np.random.default_rng(int(cfg.random_seed) + int(next_layer_idx))
                        idx_plot = g.choice(n, size=max_pts, replace=False)
                    else:
                        idx_plot = np.arange(n)

                    plt.figure(figsize=(5.2, 3.4))
                    plt.scatter(
                        read_conn_np[idx_plot],
                        mean_abs_delta[idx_plot],
                        s=6,
                        alpha=0.25,
                        color="#2c3e50",
                        edgecolors="none",
                    )
                    plt.xlabel("ReadConn")
                    plt.ylabel(r"Mean $|\Delta u_j|$ under bus ablation")
                    plt.title(f"ReadConn predicts bus dependence (layer {next_layer_idx})\nSpearman ρ={rho:+.3f}")
                    plt.grid(True, alpha=0.2)
                    plt.tight_layout()
                    plt.savefig(out_dir / f"readhalo_dependence_scatter_{suffix}.png", dpi=220)
                    plt.close()

                    plt.figure(figsize=(4.2, 3.0))
                    plt.bar([0, 1], [halo_mean, rand_mean], color=["#f39c12", "#95a5a6"])
                    plt.xticks([0, 1], ["Read-halo", "Random"])
                    plt.ylabel(r"Mean $|\Delta u_j|$")
                    plt.title(f"Bus dependence gap\nΔ={halo_mean - rand_mean:+.4f}")
                    plt.tight_layout()
                    plt.savefig(out_dir / f"readhalo_dependence_bar_{suffix}.png", dpi=220)
                    plt.close()
                except Exception:
                    pass

    # Optional plots
    if plots_dir is not None:
        try:
            import matplotlib.pyplot as plt

            out_dir = Path(plots_dir) / "read_halo"
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = source_layer_name.replace(".", "_")

            plt.figure(figsize=(5, 3))
            plt.hist(read_conn.numpy(), bins=80, alpha=0.85, color="#4c72b0")
            thr = float(halo_vals[-1].item()) if halo_vals.numel() > 0 else float("nan")
            plt.axvline(thr, color="#c0392b", linestyle="--", linewidth=1, label=f"threshold={thr:.3f}")
            plt.xlabel("ReadConn")
            plt.ylabel("Count")
            plt.title(f"ReadConn distribution (layer {next_layer_idx})")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(out_dir / f"readconn_hist_{suffix}.png", dpi=200)
            plt.close()

            plt.figure(figsize=(4, 3))
            plt.bar([0, 1], [halo_red, rand_red], color=["#f39c12", "#95a5a6"])
            plt.xticks([0, 1], ["Read-halo", "Random"])
            plt.ylabel("Mean |corr| within set (u)")
            plt.title(f"Within-set redundancy (layer {next_layer_idx})\nΔ={effect:+.4f}")
            plt.tight_layout()
            plt.savefig(out_dir / f"readhalo_redundancy_{suffix}.png", dpi=200)
            plt.close()
        except Exception:
            pass

    return {
        "description": "Read-halo in next layer (channels reading from supernode-influenced hidden dims)",
        "source_layer": source_layer_name,
        "target_layer_idx": int(next_layer_idx),
        "hidden_dim_support_size": int(S.size),
        "intermediate_dim": int(intermediate_dim),
        "num_read_halo": int(num_halo),
        "readconn": {
            "mean": float(read_conn.mean().item()),
            "std": float(read_conn.std().item()),
            "min": float(read_conn.min().item()),
            "max": float(read_conn.max().item()),
            "threshold": float(halo_vals[-1].item()) if halo_vals.numel() > 0 else float("nan"),
        },
        "redundancy_u": {
            "read_halo_mean_abs_corr": float(halo_red),
            "random_mean_abs_corr": float(rand_red),
            "difference": float(effect),
        },
        "dependence_u": dependence,
    }
