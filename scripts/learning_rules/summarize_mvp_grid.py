#!/usr/bin/env python3
"""Summarize learning-rule MVP runs into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "experiment_config.yaml"
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        data = None
    return data if isinstance(data, dict) else {}


def newest_results_json(run_dir: Path) -> dict[str, Any]:
    candidates = sorted((run_dir / "results").glob("results_*.json"))
    if not candidates:
        candidates = sorted(run_dir.glob("results_*.json"))
    if not candidates and (run_dir / "results.json").exists():
        candidates = [run_dir / "results.json"]
    return load_json(candidates[-1]) if candidates else {}


def run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("**/experiment_config.yaml"))


def method_from_config(config: dict[str, Any]) -> str:
    method = config.get("learning_rule_method")
    if method in (None, "", "none"):
        return "bp"
    return str(method)


def auc(xs: list[float], ys: list[float]) -> float | None:
    clean = sorted((float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None)
    if len(clean) < 2:
        return None
    x_arr = np.asarray([x for x, _ in clean], dtype=float)
    y_arr = np.asarray([y for _, y in clean], dtype=float)
    span = float(x_arr[-1] - x_arr[0])
    if span <= 0:
        return None
    return float(np.trapz(y_arr, x_arr) / span)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    pruning_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs(root):
        config = load_config(run_dir)
        history = load_json(run_dir / "results" / "training_history.json")
        results = newest_results_json(run_dir)
        pruning = load_json(run_dir / "pruning_results.json") or results.get("pruning_results", {})
        method = method_from_config(config)
        seed = config.get("seed")
        epochs = history.get("epochs", []) if isinstance(history, dict) else []
        last_epoch = epochs[-1] if epochs else {}
        actual_epochs = len(epochs)
        planned_epochs = config.get("training_epochs")
        training_complete = isinstance(planned_epochs, int) and actual_epochs >= planned_epochs and bool(history) and bool(results)

        run_rows.append(
            {
                "run_dir": str(run_dir),
                "name": config.get("name", run_dir.name),
                "method": method,
                "seed": seed,
                "model": config.get("model_name"),
                "width_multiplier": (
                    (config.get("model_config") or {}).get("width_multiplier") if isinstance(config.get("model_config"), dict) else None
                ),
                "dataset": config.get("dataset_name"),
                "training_epochs": config.get("training_epochs"),
                "actual_training_epochs": actual_epochs,
                "training_complete": training_complete,
                "learning_rule_lambda": config.get("learning_rule_lambda", 0.0),
                "learning_rule_schedule": config.get("learning_rule_schedule", "warmup"),
                "learning_rule_task_gate_source": config.get("learning_rule_task_gate_source", "task"),
                "learning_rule_peer_proxy": config.get("learning_rule_peer_proxy", "avg_corr2"),
                "learning_rule_cross_layer_alloc": config.get("learning_rule_cross_layer_alloc", "uniform"),
                "learning_rule_cross_layer_alpha": config.get("learning_rule_cross_layer_alpha", 1.0),
                "learning_rule_hull_max_size": config.get("learning_rule_hull_max_size"),
                "learning_rule_hull_eps": config.get("learning_rule_hull_eps"),
                "warmup_epochs": config.get("learning_rule_warmup_epochs", 0),
                "ramp_epochs": config.get("learning_rule_ramp_epochs", 0),
                "metric_trigger_epoch": history.get("learning_rule_metric_trigger_epoch") if isinstance(history, dict) else None,
                "initial_accuracy": history.get("initial_accuracy") if isinstance(history, dict) else None,
                "best_accuracy": history.get("best_accuracy") if isinstance(history, dict) else None,
                "final_train_accuracy": last_epoch.get("accuracy"),
                "final_ce_loss": last_epoch.get("train_ce_loss"),
                "final_total_loss": last_epoch.get("train_total_loss"),
                "final_regularizer_raw": last_epoch.get("regularizer_raw"),
                "final_regularizer_weight": last_epoch.get("regularizer_weight"),
                "final_regularizer_weighted": last_epoch.get("regularizer_weighted"),
                "final_duplicate_task_mass": last_epoch.get("duplicate_task_mass"),
                "final_non_replaceable_task_mass": last_epoch.get("non_replaceable_task_mass"),
                "final_duplicate_task_fraction": last_epoch.get("duplicate_task_fraction"),
                "final_non_replaceable_task_fraction": last_epoch.get("non_replaceable_task_fraction"),
                "final_rho_cap": last_epoch.get("rho_cap"),
                "final_peer_reconstructability_mean": last_epoch.get("peer_reconstructability_mean"),
                "final_task_relevance_mean": last_epoch.get("task_relevance_mean"),
                "final_gate_relevance_mean": last_epoch.get("gate_relevance_mean"),
                "final_rtc_relevance_mean": last_epoch.get("rtc_relevance_mean"),
                "final_task_gate_mean": last_epoch.get("task_gate_mean"),
                "final_hull_size_mean": last_epoch.get("hull_size_mean"),
                "final_hull_score_mean": last_epoch.get("hull_score_mean"),
                "final_cross_layer_weight_mean": last_epoch.get("cross_layer_weight_mean"),
                "final_cross_layer_weight_max": last_epoch.get("cross_layer_weight_max"),
                "final_cross_layer_weight_min": last_epoch.get("cross_layer_weight_min"),
                "final_grad_projection_abs_cos_mean": last_epoch.get("grad_projection_abs_cos_mean"),
                "final_grad_projection_norm_shrink_mean": last_epoch.get("grad_projection_norm_shrink_mean"),
                "learning_rule_grad_projection_strength": config.get("learning_rule_grad_projection_strength"),
                "final_synergy_raw_mean": last_epoch.get("synergy_raw_mean"),
                "final_antidecouple_rho_l": last_epoch.get("antidecouple_rho_l"),
                "final_antidecouple_gap": last_epoch.get("antidecouple_gap"),
                "learning_rule_anti_decouple_target_rho": config.get("learning_rule_anti_decouple_target_rho"),
                "regularized_layers": len(history.get("learning_rule_layers", [])) if isinstance(history, dict) else 0,
                "has_results": bool(results),
                "has_pruning": bool(isinstance(pruning, dict) and pruning.get("methods")),
                "pruning_baseline_accuracy": pruning.get("baseline") if isinstance(pruning, dict) else None,
            }
        )

        if not isinstance(pruning, dict):
            continue
        methods = pruning.get("methods", {})
        if not isinstance(methods, dict):
            continue
        for pruning_method, ratios in methods.items():
            if not isinstance(ratios, dict):
                continue
            xs: list[float] = []
            ys: list[float] = []
            for ratio_key, values in ratios.items():
                if not isinstance(values, dict):
                    continue
                target = values.get("target_sparsity")
                if target is None:
                    try:
                        target = float(ratio_key)
                    except Exception:
                        target = None
                achieved = values.get("achieved_sparsity_global")
                accuracy = values.get("accuracy_after_ft", values.get("accuracy_before_ft"))
                if accuracy is not None:
                    xs.append(float(achieved if achieved is not None else target))
                    ys.append(float(accuracy))
                pruning_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "method": method,
                        "seed": seed,
                        "pruning_method": pruning_method,
                        "target_sparsity": target,
                        "achieved_sparsity_global": achieved,
                        "accuracy_before_ft": values.get("accuracy_before_ft"),
                        "accuracy_after_ft": values.get("accuracy_after_ft"),
                        "accuracy_drop": values.get("accuracy_drop"),
                        "accuracy_recovery": values.get("accuracy_recovery"),
                        "error": values.get("error"),
                    }
                )
            auc_rows.append(
                {
                    "run_dir": str(run_dir),
                    "method": method,
                    "seed": seed,
                    "pruning_method": pruning_method,
                    "accuracy_auc": auc(xs, ys),
                    "n_points": len(xs),
                }
            )

    return run_rows, pruning_rows, auc_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/learning_rules/resnet18_cifar100")
    parser.add_argument("--out-dir", default="projects/replaceability_learning_rules/paper_artifacts/tables")
    args = parser.parse_args()

    run_rows, pruning_rows, auc_rows = summarize(Path(args.results_root))
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "run_summary.csv", run_rows)
    write_csv(out_dir / "pruning_curve.csv", pruning_rows)
    write_csv(out_dir / "pruning_auc.csv", auc_rows)

    print(f"runs: {len(run_rows)}")
    print(f"pruning rows: {len(pruning_rows)}")
    print(f"auc rows: {len(auc_rows)}")
    print(f"wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
