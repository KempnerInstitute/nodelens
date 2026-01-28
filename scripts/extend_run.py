#!/usr/bin/env python3
"""
Extend an existing experiment run directory with additional work.

Supported tasks:
- analysis_only: regenerate plots/analysis (delegates to scripts/run_experiment.py when supported)
- figures: regenerate cluster-analysis figures from saved results
- pruning: (re)run / extend cluster-analysis pruning sweeps from a saved checkpoint + results
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json_like(path: Path) -> Dict[str, Any]:
    txt = path.read_text()
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        try:
            import yaml  # type: ignore

            obj = yaml.safe_load(txt)
            return obj if isinstance(obj, dict) else {}
        except Exception as exc:
            raise RuntimeError(f"Failed to parse {path} as JSON/YAML: {exc}") from exc


def _latest_results_json(run_dir: Path) -> Optional[Path]:
    cands: List[Path] = []
    if (run_dir / "results").exists():
        cands.extend(sorted((run_dir / "results").glob("results_*.json")))
    cands.extend(sorted(run_dir.glob("results_*.json")))
    if (run_dir / "results.json").exists():
        cands.append(run_dir / "results.json")
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _detect_experiment_type(cfg: Dict[str, Any], res: Optional[Dict[str, Any]]) -> str:
    t = cfg.get("experiment_type", None)
    if isinstance(t, str) and t:
        return t
    if res and isinstance(res.get("metadata", None), dict):
        mt = res["metadata"].get("experiment_type", None)
        if isinstance(mt, str) and mt:
            return mt
    return "alignment_analysis"


def _resolve_checkpoint(run_dir: Path, cfg: Dict[str, Any], user_ckpt: Optional[str]) -> Path:
    if user_ckpt:
        p = Path(user_ckpt)
        if p.exists():
            return p
    p = run_dir / "checkpoints" / "trained_model.pth"
    if p.exists():
        return p
    mc = cfg.get("model_checkpoint", None)
    if isinstance(mc, str) and mc and Path(mc).exists():
        return Path(mc)
    ck_dir = run_dir / "checkpoints"
    if ck_dir.exists():
        cands = [q for q in ck_dir.glob("*.pth") if q.is_file()]
        cands.sort(key=lambda q: q.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
    raise FileNotFoundError(f"Checkpoint not found under {ck_dir} (or via --checkpoint)")


def _delegate_analysis_only(repo_root: Path, run_dir: Path, cfg_path: Path, *, device: Optional[str]) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_experiment.py"),
        "--config",
        str(cfg_path),
        "--analysis-only",
        "--experiment-dir",
        str(run_dir),
    ]
    if device:
        cmd.extend(["--device", str(device)])
    subprocess.check_call(cmd)


def _build_cluster_experiment(repo_root: Path, cfg: Dict[str, Any]):
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    from alignment.experiments.base import ExperimentConfig

    allowed = {f.name for f in fields(ExperimentConfig)}
    config = ExperimentConfig(**{k: v for k, v in cfg.items() if k in allowed})

    import importlib.util

    spec = importlib.util.spec_from_file_location("run_experiment", str(repo_root / "scripts" / "run_experiment.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    exp = mod._create_cluster_experiment(config)
    return exp, config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str)
    ap.add_argument("--tasks", default="analysis_only", type=str, help="analysis_only,figures,pruning")
    ap.add_argument("--device", default=None, type=str)
    ap.add_argument("--checkpoint", default=None, type=str)
    ap.add_argument("--methods", default=None, type=str)
    ap.add_argument("--ratios", default=None, type=str)
    ap.add_argument("--fine-tune-epochs", default=None, type=int)
    ap.add_argument("--fine-tune-lr", default=None, type=float)
    ap.add_argument("--fine-tune-max-batches", default=None, type=int)
    ap.add_argument("--fine-tune-weight-decay", default=None, type=float)
    ap.add_argument("--taylor-samples", default=None, type=int)
    ap.add_argument("--taylor-act-samples", default=None, type=int)
    ap.add_argument("--taylor-act-batch-size", default=None, type=int)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = run_dir / "experiment_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing experiment_config.yaml in {run_dir}")

    cfg = _load_json_like(cfg_path)
    res_path = _latest_results_json(run_dir)
    res = _load_json_like(res_path) if res_path else None
    exp_type = _detect_experiment_type(cfg, res)

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    if not tasks:
        tasks = ["analysis_only"]

    if "analysis_only" in tasks and exp_type not in {"cluster_analysis", "vision_cluster_analysis", "metric_cluster_analysis"}:
        _delegate_analysis_only(repo_root, run_dir, cfg_path, device=args.device)

    # Cluster-analysis tasks
    cluster_tasks = {"analysis_only", "figures", "pruning"}
    if not any(t in cluster_tasks for t in tasks):
        return
    if exp_type not in {"cluster_analysis", "vision_cluster_analysis", "metric_cluster_analysis"}:
        # Nothing else we can do here for non-cluster runs beyond delegated analysis_only.
        return

    cfg = dict(cfg)
    cfg["do_train"] = False
    cfg["experiment_dir"] = str(run_dir)
    cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
    cfg["log_dir"] = str(run_dir / "logs")
    if args.device is not None:
        cfg["device"] = str(args.device)
    cfg["model_checkpoint"] = str(_resolve_checkpoint(run_dir, cfg, args.checkpoint))
    if args.taylor_samples is not None:
        cfg["taylor_samples"] = int(args.taylor_samples)
    if args.taylor_act_samples is not None:
        cfg["taylor_act_samples"] = int(args.taylor_act_samples)
    if args.taylor_act_batch_size is not None:
        cfg["taylor_act_batch_size"] = int(args.taylor_act_batch_size)

    exp, config = _build_cluster_experiment(repo_root, cfg)

    if res and isinstance(res, dict):
        exp.layer_metrics = res.get("layer_metrics", {}) or {}
        exp.cluster_results = res.get("cluster_results", {}) or {}
        exp.halo_results = res.get("halo_results", {}) or {}
        exp.halo_flow_results = res.get("halo_flow_results", {}) or {}
        exp.cascade_results = res.get("cascade_results", {}) or {}
        exp.pruning_results = res.get("pruning_results", {}) or {}

    if "analysis_only" in tasks or "figures" in tasks:
        exp.generate_figures()

    if "pruning" in tasks:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()] if isinstance(args.methods, str) else None
        ratios = [float(x) for x in args.ratios.split(",") if x.strip()] if isinstance(args.ratios, str) else None

        pr_path = run_dir / "pruning_results.json"
        if args.backup and pr_path.exists():
            try:
                (run_dir / "pruning_results.json.bak").write_bytes(pr_path.read_bytes())
            except Exception:
                pass

        ft_epochs = int(args.fine_tune_epochs) if args.fine_tune_epochs is not None else (
            int(config.fine_tune_epochs) if bool(config.fine_tune_after_pruning) else 0
        )
        ft_lr = float(args.fine_tune_lr) if args.fine_tune_lr is not None else (
            float(config.fine_tune_learning_rate)
            if config.fine_tune_learning_rate is not None
            else float(config.learning_rate) * 0.1
        )
        ft_mb = args.fine_tune_max_batches if args.fine_tune_max_batches is not None else config.fine_tune_max_batches
        ft_wd = float(args.fine_tune_weight_decay) if args.fine_tune_weight_decay is not None else float(
            config.fine_tune_weight_decay or 0.0
        )

        exp.run_pruning_experiments(
            ratios=ratios,
            methods=methods,
            fine_tune_epochs=ft_epochs,
            fine_tune_lr=ft_lr,
            fine_tune_max_batches=ft_mb,
            fine_tune_weight_decay=ft_wd,
            resume=not bool(args.no_resume),
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()

