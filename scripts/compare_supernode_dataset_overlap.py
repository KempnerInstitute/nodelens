#!/usr/bin/env python3
"""
Compare supernode Jaccard overlap across multiple text datasets.

This script computes activation-based supernode sets for each target layer and dataset,
then reports pairwise Jaccard overlap matrices across datasets.

Example:
    python scripts/compare_supernode_dataset_overlap.py \
        --config configs/examples/llama3_dataset_supernode_overlap.yaml
"""

import argparse
from datetime import datetime
import fnmatch
from itertools import combinations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise RuntimeError("transformers is required for this script") from exc

try:
    from alignment.dataops.datasets.text_datasets import WikiTextDataset, load_text_dataset
except ImportError:
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    from alignment.dataops.datasets.text_datasets import WikiTextDataset, load_text_dataset


LOGGER = logging.getLogger("dataset_supernode_overlap")


@dataclass
class DatasetSpec:
    name: str
    split: str
    max_samples: int
    max_length: int
    kwargs: Dict[str, Any]


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if not dtype_name:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype_name).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[key]


def _find_layers(model: torch.nn.Module, patterns: Sequence[str]) -> Dict[str, torch.nn.Module]:
    matched: Dict[str, torch.nn.Module] = {}
    for layer_name, layer_module in model.named_modules():
        if not hasattr(layer_module, "weight"):
            continue
        if any(fnmatch.fnmatch(layer_name, p) for p in patterns):
            matched[layer_name] = layer_module
    if not matched:
        raise ValueError(f"No layers matched patterns: {patterns}")
    return matched


def _extract_texts(tokenizer: Any, spec: DatasetSpec) -> List[str]:
    dataset_kwargs = dict(spec.kwargs)

    # `load_text_dataset` uses `dataset_name` to select dataset type (wikitext/c4/code/arxiv).
    # For WikiText, users may also pass a WikiText subset as `kwargs.dataset_name`.
    # Handle that explicitly to avoid Python's duplicate keyword error.
    wikitext_subset = None
    if str(spec.name).lower() == "wikitext" and "dataset_name" in dataset_kwargs:
        wikitext_subset = str(dataset_kwargs.pop("dataset_name"))

    dataset_name_l = str(spec.name).lower()

    if dataset_name_l == "wikitext" and wikitext_subset is not None:
        ds = WikiTextDataset(
            tokenizer=tokenizer,
            split=spec.split,
            max_length=spec.max_length,
            dataset_name=wikitext_subset,
        )
    else:
        if "dataset_name" in dataset_kwargs:
            raise ValueError(
                "Use `name` to choose dataset type and reserve `kwargs.dataset_name` for wikitext only. "
                f"Received kwargs.dataset_name for dataset '{spec.name}'."
            )

        # Newer `datasets` versions reject script-based datasets like `scientific_papers`.
        # Keep arxiv support by falling back to a non-script dataset when needed.
        if dataset_name_l == "arxiv":
            try:
                ds = load_text_dataset(
                    dataset_name=spec.name,
                    tokenizer=tokenizer,
                    split=spec.split,
                    max_length=spec.max_length,
                    max_samples=spec.max_samples,
                    **dataset_kwargs,
                )
            except RuntimeError as e:
                msg = str(e)
                if "Dataset scripts are no longer supported" not in msg:
                    raise

                from datasets import load_dataset

                hf_dataset = str(dataset_kwargs.get("hf_dataset", "ccdv/arxiv-summarization"))
                text_field = str(dataset_kwargs.get("text_field", "article"))
                LOGGER.warning(
                    "Primary arxiv loader unavailable (%s). Falling back to %s[%s].",
                    msg,
                    hf_dataset,
                    text_field,
                )
                raw_ds = load_dataset(hf_dataset, split=spec.split)
                texts_fallback: List[str] = []
                for item in raw_ds:
                    t = item.get(text_field) or item.get("text") or item.get("article") or item.get("abstract")
                    if not t or len(str(t).strip()) == 0:
                        continue
                    texts_fallback.append(str(t))
                    if spec.max_samples and len(texts_fallback) >= int(spec.max_samples):
                        break

                class _SimpleTextContainer:
                    def __init__(self, texts: List[str]):
                        self.texts = texts

                ds = _SimpleTextContainer(texts_fallback)
        else:
            ds = load_text_dataset(
                dataset_name=spec.name,
                tokenizer=tokenizer,
                split=spec.split,
                max_length=spec.max_length,
                max_samples=spec.max_samples,
                **dataset_kwargs,
            )

    texts = list(getattr(ds, "texts", []) or [])
    texts = [t for t in texts if isinstance(t, str) and t.strip()]

    if spec.max_samples > 0:
        texts = texts[: spec.max_samples]

    if not texts:
        raise ValueError(
            f"Dataset '{spec.name}' did not provide raw texts. "
            "Please use a dataset option supported by load_text_dataset with textual outputs."
        )

    LOGGER.info("Loaded %d texts for dataset=%s split=%s", len(texts), spec.name, spec.split)
    return texts


def _chunked(seq: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def _compute_layer_activation_l2(
    model: torch.nn.Module,
    tokenizer: Any,
    layer_map: Mapping[str, torch.nn.Module],
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    accum_sq: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {name: 0 for name in layer_map.keys()}

    def make_hook(layer_name: str):
        def hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x = x.detach().float()
            if x.ndim == 3:
                x2 = x.reshape(-1, x.shape[-1])
            elif x.ndim == 2:
                x2 = x
            else:
                return

            sq_sum = (x2 * x2).sum(dim=0).cpu()
            if layer_name not in accum_sq:
                accum_sq[layer_name] = sq_sum
            else:
                accum_sq[layer_name] += sq_sum
            counts[layer_name] += int(x2.shape[0])

        return hook

    hooks = [m.register_forward_hook(make_hook(n)) for n, m in layer_map.items()]

    model.eval()
    with torch.no_grad():
        for batch_texts in _chunked(list(texts), batch_size):
            encoded = tokenizer(
                list(batch_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            _ = model(**encoded)

    for h in hooks:
        h.remove()

    scores: Dict[str, torch.Tensor] = {}
    for layer_name in layer_map.keys():
        if layer_name not in accum_sq or counts[layer_name] == 0:
            raise RuntimeError(f"No activations captured for layer: {layer_name}")
        mean_sq = accum_sq[layer_name] / float(counts[layer_name])
        scores[layer_name] = torch.sqrt(mean_sq)
    return scores


def _compute_scar_metric_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    layer_map: Mapping[str, torch.nn.Module],
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute SCAR metrics for down_proj layers using forward+backward passes."""
    scar_state: Dict[str, Dict[str, Any]] = {}
    hooks: List[Any] = []

    for layer_name, module in layer_map.items():
        scar_state[layer_name] = {
            "u_sqr_sum": None,
            "R_sum": None,
            "T_sum": None,
            "loss_proxy_sum": None,
            "count": 0,
        }

        def make_hooks(name: str):
            def fwd_hook(mod: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                if not inputs:
                    return
                u = inputs[0]
                if u is None:
                    return

                u_flat = u.detach()
                if u_flat.ndim > 2:
                    u_flat = u_flat.reshape(-1, u_flat.shape[-1])

                state = scar_state[name]
                m = int(u_flat.shape[-1])
                if state["u_sqr_sum"] is None:
                    state["u_sqr_sum"] = torch.zeros(m, device=u_flat.device, dtype=torch.float32)
                    state["R_sum"] = torch.zeros_like(state["u_sqr_sum"])
                    state["T_sum"] = torch.zeros_like(state["u_sqr_sum"])
                    state["loss_proxy_sum"] = torch.zeros_like(state["u_sqr_sum"])

                u_f = u_flat.float()
                state["u_sqr_sum"] += (u_f * u_f).sum(dim=0)
                state["count"] += int(u_flat.shape[0])
                mod._scar_last_u = u.detach()

            def bwd_hook(mod: torch.nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
                if not grad_input or grad_input[0] is None:
                    return
                if not hasattr(mod, "_scar_last_u"):
                    return

                state = scar_state[name]
                g_u = grad_input[0]
                u = mod._scar_last_u
                delattr(mod, "_scar_last_u")

                u_flat = u.reshape(-1, u.shape[-1]) if u.ndim > 2 else u.reshape(-1, u.shape[-1])
                g_u_flat = g_u.reshape(-1, g_u.shape[-1]) if g_u.ndim > 2 else g_u.reshape(-1, g_u.shape[-1])
                if u_flat.shape != g_u_flat.shape:
                    return

                s_flat = g_u_flat.float()
                u_flat_f = u_flat.float()

                state["R_sum"] += (s_flat * s_flat).sum(dim=0)
                state["T_sum"] += torch.abs(s_flat * u_flat_f).sum(dim=0)
                q = u_flat_f * s_flat
                state["loss_proxy_sum"] += (q * q).sum(dim=0)

            return fwd_hook, bwd_hook

        fwd_hook, bwd_hook = make_hooks(layer_name)
        hooks.append(module.register_forward_hook(fwd_hook))
        hooks.append(module.register_full_backward_hook(bwd_hook))

    model.eval()
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i : i + batch_size])
            if not batch_texts:
                continue

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            labels = inputs["input_ids"].clone()
            pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            inputs["labels"] = labels

            model.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            loss.backward()
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name, state in scar_state.items():
        count = int(state["count"])
        if count <= 0 or state["u_sqr_sum"] is None:
            continue

        u2_mean = state["u_sqr_sum"] / float(count)
        r_vals = state["R_sum"] / float(count)
        t_vals = state["T_sum"] / float(count)
        loss_proxy_joint = 0.5 * (state["loss_proxy_sum"] / float(count))
        loss_proxy_factored = 0.5 * u2_mean * r_vals

        out[layer_name] = {
            "scar_activation_power": u2_mean.cpu(),
            "scar_taylor": t_vals.cpu(),
            "scar_curvature": r_vals.cpu(),
            "scar_loss_proxy": loss_proxy_joint.cpu(),
            "scar_loss_proxy_factored": loss_proxy_factored.cpu(),
        }

    return out


def _compute_rq_for_layer(
    model: torch.nn.Module,
    tokenizer: Any,
    layer_name: str,
    layer_module: torch.nn.Module,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Compute RQ proxy per neuron for a layer using input covariance and layer weights."""
    if not hasattr(layer_module, "weight"):
        return None

    weight = layer_module.weight.data.float().to(device)

    activations: List[torch.Tensor] = []

    def hook_fn(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if not inputs:
            return
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        if not torch.is_tensor(inp):
            return
        activations.append(inp.detach().float())

    hook_handle = layer_module.register_forward_hook(hook_fn)
    try:
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = list(texts[i : i + batch_size])
                if not batch_texts:
                    continue
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)

        if not activations:
            return None

        all_acts = torch.cat([a.view(-1, a.shape[-1]) for a in activations], dim=0).to(device)
        if all_acts.shape[0] < 2:
            return None

        mean = all_acts.mean(dim=0, keepdim=True)
        centered = all_acts - mean
        cov = (centered.T @ centered) / (all_acts.shape[0] - 1)

        out_dim, in_dim = weight.shape
        input_dim = all_acts.shape[1]

        if "down_proj" in layer_name and in_dim == input_dim:
            var_j = torch.diag(cov)
            col_norm_sq = (weight**2).sum(dim=0)
            rq_per_intermediate = var_j * col_norm_sq
            return rq_per_intermediate.cpu()

        if in_dim == input_dim:
            w_cov = weight @ cov
            w_cov_w = torch.sum(w_cov * weight, dim=1)
            w_w = torch.sum(weight**2, dim=1)
            rq = w_cov_w / (w_w + 1e-10)
            return rq.cpu()

        return torch.var(all_acts, dim=0).cpu()
    finally:
        hook_handle.remove()


def _compute_layer_scores_by_metric(
    model: torch.nn.Module,
    tokenizer: Any,
    layer_map: Mapping[str, torch.nn.Module],
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    supernode_metric: str,
) -> Dict[str, torch.Tensor]:
    metric = str(supernode_metric).strip().lower()

    activation_aliases = {"activation_l2", "activation_l2_norm"}
    if metric in activation_aliases:
        return _compute_layer_activation_l2(model, tokenizer, layer_map, texts, batch_size, max_length, device)

    scar_metrics = {
        "scar_activation_power",
        "scar_taylor",
        "scar_curvature",
        "scar_loss_proxy",
        "scar_loss_proxy_factored",
    }
    if metric in scar_metrics:
        scar_all = _compute_scar_metric_scores(model, tokenizer, layer_map, texts, batch_size, max_length, device)
        out: Dict[str, torch.Tensor] = {}
        for ln in layer_map.keys():
            layer_scores = scar_all.get(ln, {})
            if metric in layer_scores:
                out[ln] = layer_scores[metric]
        return out

    rq_aliases = {"rayleigh_quotient", "rq"}
    mi_aliases = {"gaussian_mi_analytic", "mutual_information", "mi"}
    if metric in rq_aliases or metric in mi_aliases:
        out: Dict[str, torch.Tensor] = {}
        for ln, mod in layer_map.items():
            rq_scores = _compute_rq_for_layer(model, tokenizer, ln, mod, texts, batch_size, max_length, device)
            if rq_scores is None:
                continue
            if metric in mi_aliases:
                noise_var = 0.1
                snr = rq_scores / (noise_var + 1e-10)
                out[ln] = 0.5 * torch.log1p(snr.clamp(min=0))
            else:
                out[ln] = rq_scores
        return out

    raise ValueError(
        "Unsupported supernode_metric. Supported: activation_l2_norm, activation_l2, "
        "scar_activation_power, scar_taylor, scar_curvature, scar_loss_proxy, scar_loss_proxy_factored, "
        "rayleigh_quotient, rq, gaussian_mi_analytic, mutual_information, mi"
    )


def _topk_indices(scores: torch.Tensor, fraction: float) -> Set[int]:
    n = int(scores.numel())
    k = max(1, int(round(fraction * n)))
    k = min(k, n)
    topk = torch.topk(scores.flatten(), k=k, largest=True).indices.cpu().numpy().tolist()
    return set(int(i) for i in topk)


def _jaccard(a: Set[int], b: Set[int]) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return float(len(a & b) / union)


def _jaccard_many(sets: Sequence[Set[int]]) -> float:
    if not sets:
        return 0.0
    inter = set(sets[0])
    union = set(sets[0])
    for s in sets[1:]:
        inter &= s
        union |= s
    if not union:
        return 0.0
    return float(len(inter) / len(union))


def _save_heatmap(matrix: np.ndarray, labels: List[str], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3, 1.2 * len(labels) + 2.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < 0.55 else "black")

    fig.colorbar(im, ax=ax, label="Jaccard")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _save_fraction_sweep_plot(
    fractions: List[float],
    series_map: Mapping[str, Sequence[float]],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for label, values in series_map.items():
        ax.plot(fractions, list(values), marker="o", linewidth=1.6, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Supernode Fraction")
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    if len(series_map) <= 12:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _build_fraction_sweep(configured_values: Optional[Sequence[Any]]) -> Optional[List[float]]:
    if configured_values is None:
        return None

    values = [float(v) for v in configured_values]
    cleaned = sorted({float(v) for v in values if float(v) > 0.0 and float(v) <= 1.0})
    if not cleaned:
        raise ValueError("fraction_sweep must contain values in (0, 1]")
    return cleaned


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Compare supernode overlap across datasets")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = _read_yaml(Path(args.config))

    model_cfg = cfg.get("model", {})
    analysis_cfg = cfg.get("analysis", {})
    output_cfg = cfg.get("output", {})

    model_id = model_cfg.get("model_id")
    if not model_id:
        raise ValueError("config.model.model_id is required")

    device_str = str(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    torch_dtype = _resolve_dtype(model_cfg.get("torch_dtype"))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    layer_patterns = list(model_cfg.get("tracked_layers", ["model.layers.*.mlp.down_proj"]))

    batch_size = int(analysis_cfg.get("batch_size", 4))
    max_length = int(analysis_cfg.get("max_length", 512))
    supernode_fraction = float(analysis_cfg.get("supernode_fraction", 0.01))
    supernode_metric = str(analysis_cfg.get("supernode_metric", "activation_l2")).strip().lower()
    fraction_sweep = _build_fraction_sweep(analysis_cfg.get("fraction_sweep"))
    jaccard_enabled = bool(analysis_cfg.get("jaccard_heatmap", analysis_cfg.get("jaccard_enabled", True)))
    jaccard_sweep_enabled = bool(analysis_cfg.get("jaccard_sweep", fraction_sweep is not None))
    dataset_specs_raw = list(cfg.get("datasets", []))

    if not dataset_specs_raw:
        raise ValueError("config.datasets must be a non-empty list")

    dataset_specs = []
    for item in dataset_specs_raw:
        dataset_specs.append(
            DatasetSpec(
                name=str(item["name"]),
                split=str(item.get("split", "test")),
                max_samples=int(item.get("max_samples", 128)),
                max_length=int(item.get("max_length", max_length)),
                kwargs=dict(item.get("kwargs", {})),
            )
        )

    base_output_dir = Path(output_cfg.get("output_dir", "./results/dataset_supernode_overlap"))
    unique_subdir = bool(output_cfg.get("unique_subdir", True))

    if unique_subdir:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = base_output_dir / f"run_{run_stamp}"
        suffix = 1
        while candidate.exists():
            candidate = base_output_dir / f"run_{run_stamp}_{suffix}"
            suffix += 1
        output_dir = candidate
    else:
        output_dir = base_output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model: %s", model_id)
    model_load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        model_load_kwargs["dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
    model.to(device)

    layer_map = _find_layers(model, layer_patterns)
    LOGGER.info("Matched %d layers", len(layer_map))

    dataset_scores: Dict[str, Dict[str, torch.Tensor]] = {}
    dataset_supernodes: Dict[str, Dict[str, Set[int]]] = {}

    for spec in dataset_specs:
        texts = _extract_texts(tokenizer, spec)
        scores = _compute_layer_scores_by_metric(
            model=model,
            tokenizer=tokenizer,
            layer_map=layer_map,
            texts=texts,
            batch_size=batch_size,
            max_length=spec.max_length,
            device=device,
            supernode_metric=supernode_metric,
        )
        dataset_scores[spec.name] = scores

        layer_to_supernodes: Dict[str, Set[int]] = {}
        for layer_name, score_vec in scores.items():
            layer_to_supernodes[layer_name] = _topk_indices(score_vec, supernode_fraction)
        dataset_supernodes[spec.name] = layer_to_supernodes

    dataset_names = [d.name for d in dataset_specs]
    per_layer_jaccard: Dict[str, List[List[float]]] = {}
    mean_matrix: Optional[np.ndarray] = None
    pct_tag = f"top{(supernode_fraction * 100):.1f}p".replace(".", "p")
    metric_tag = supernode_metric.replace(" ", "_")
    title_context = f"metric={supernode_metric}, fraction={supernode_fraction:g}"

    if jaccard_enabled:
        for layer_name in layer_map.keys():
            matrix = np.zeros((len(dataset_names), len(dataset_names)), dtype=float)
            for i, di in enumerate(dataset_names):
                for j, dj in enumerate(dataset_names):
                    matrix[i, j] = _jaccard(dataset_supernodes[di][layer_name], dataset_supernodes[dj][layer_name])
            per_layer_jaccard[layer_name] = matrix.tolist()

            layer_slug = layer_name.replace(".", "_")
            _save_heatmap(
                matrix=matrix,
                labels=dataset_names,
                title=f"Dataset Supernode Jaccard: {layer_name} ({title_context})",
                save_path=output_dir / f"jaccard_{metric_tag}_{pct_tag}_{layer_slug}.png",
            )

        # Mean matrix across layers.
        stacked = np.array([np.array(m) for m in per_layer_jaccard.values()])
        mean_matrix = stacked.mean(axis=0)
        _save_heatmap(
            matrix=mean_matrix,
            labels=dataset_names,
            title=f"Dataset Supernode Jaccard (Mean Across Layers) ({title_context})",
            save_path=output_dir / f"jaccard_{metric_tag}_{pct_tag}_mean_across_layers.png",
        )

    # Sweep over supernode fractions and plot overlap trends.
    common_layers = [ln for ln in layer_map.keys() if all(ln in dataset_scores[d] for d in dataset_names)]
    pairwise_names = list(combinations(dataset_names, 2))
    evaluated_fractions: List[float] = []
    mean_pairwise_vs_fraction: List[float] = []
    all_dataset_overlap_vs_fraction: List[float] = []
    pairwise_vs_fraction: Dict[str, List[float]] = {f"{a} vs {b}": [] for a, b in pairwise_names}

    for frac in ((fraction_sweep or []) if jaccard_sweep_enabled else []):
        topk_sets: Dict[str, Dict[str, Set[int]]] = {}
        for dname in dataset_names:
            topk_sets[dname] = {ln: _topk_indices(dataset_scores[dname][ln], frac) for ln in common_layers}

        per_layer_matrices: List[np.ndarray] = []
        for layer_name in common_layers:
            matrix = np.zeros((len(dataset_names), len(dataset_names)), dtype=float)
            for i, di in enumerate(dataset_names):
                for j, dj in enumerate(dataset_names):
                    s_i = topk_sets[di][layer_name]
                    s_j = topk_sets[dj][layer_name]
                    matrix[i, j] = _jaccard(s_i, s_j)
            per_layer_matrices.append(matrix)

        if not per_layer_matrices:
            continue

        evaluated_fractions.append(frac)
        mean_matrix_frac = np.mean(np.stack(per_layer_matrices, axis=0), axis=0)
        if len(dataset_names) > 1:
            tri = np.triu_indices(len(dataset_names), k=1)
            mean_pairwise_vs_fraction.append(float(np.mean(mean_matrix_frac[tri])))
        else:
            mean_pairwise_vs_fraction.append(1.0)

        layer_all_overlap_vals = [_jaccard_many([topk_sets[d][ln] for d in dataset_names]) for ln in common_layers]
        all_dataset_overlap_vs_fraction.append(float(np.mean(layer_all_overlap_vals)) if layer_all_overlap_vals else 0.0)

        for a, b in pairwise_names:
            ai = dataset_names.index(a)
            bi = dataset_names.index(b)
            pairwise_vs_fraction[f"{a} vs {b}"].append(float(mean_matrix_frac[ai, bi]))

    sweep_tag = f"{metric_tag}_sweep"
    if evaluated_fractions:
        mean_series: Dict[str, List[float]] = {"Mean pairwise (across layers)": mean_pairwise_vs_fraction}
        if len(dataset_names) >= 3:
            mean_series[f"All-{len(dataset_names)} overlap (across layers)"] = all_dataset_overlap_vs_fraction

        _save_fraction_sweep_plot(
            fractions=evaluated_fractions,
            series_map=mean_series,
            title=f"Mean Jaccard vs Supernode Fraction (metric={supernode_metric})",
            save_path=output_dir / f"jaccard_{sweep_tag}_mean_pairwise_vs_fraction.png",
        )

        if pairwise_vs_fraction:
            _save_fraction_sweep_plot(
                fractions=evaluated_fractions,
                series_map=pairwise_vs_fraction,
                title=f"Pairwise Jaccard vs Supernode Fraction (metric={supernode_metric})",
                save_path=output_dir / f"jaccard_{sweep_tag}_pairwise_vs_fraction.png",
            )

    result = {
        "model_id": model_id,
        "device": device_str,
        "tracked_layers": list(layer_map.keys()),
        "supernode_metric": supernode_metric,
        "supernode_fraction": supernode_fraction,
        "jaccard_enabled": jaccard_enabled,
        "jaccard_sweep_enabled": jaccard_sweep_enabled,
        "fraction_sweep": evaluated_fractions if evaluated_fractions else (fraction_sweep or []),
        "output_dir": str(output_dir),
        "datasets": dataset_names,
        "per_layer_jaccard": per_layer_jaccard,
        "mean_jaccard": mean_matrix.tolist() if mean_matrix is not None else [],
        "fraction_sweep_mean_pairwise_jaccard": mean_pairwise_vs_fraction,
        "fraction_sweep_all_dataset_jaccard": all_dataset_overlap_vs_fraction,
        "fraction_sweep_pairwise_jaccard": pairwise_vs_fraction,
    }

    out_json = output_dir / "dataset_supernode_overlap.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    LOGGER.info("Saved results to %s", out_json)
    LOGGER.info("Saved heatmaps to %s", output_dir)


if __name__ == "__main__":
    main()
