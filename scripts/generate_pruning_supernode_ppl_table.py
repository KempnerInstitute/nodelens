#!/usr/bin/env python3
"""Generate a LaTeX table (method vs supernodes pruned vs PPL) from pruning results JSON.

Examples:
  python scripts/generate_pruning_supernode_ppl_table.py \
      --results-json results/my_run/results/results_20260309_152848.json

  python scripts/generate_pruning_supernode_ppl_table.py \
      --experiment-dir results/my_run
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def method_label(metric: str) -> str:
    labels = {
        "weight_magnitude": "Magnitude (channel)",
        "activation_l2_norm": "Act-L2 (channel)",
        "wanda": "Wanda (channel)",
        "sparsegpt": "SparseGPT (channel)",
        "owl": "OWL (channel)",
        "llm_pruner": "LLM-Pruner (channel)",
        "flap": "FLAP (channel)",
        "ria": "RIA (channel)",
        "slimllm": "SlimLLM (channel)",
        "scar_loss_proxy": "SCAR-LP",
        "supernode_protection_score": "SCAR-Prot",
        "supernode_connectivity_score": "SCAR-Conn",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def find_latest_results_json(experiment_dir: Path) -> Path:
    experiment_dir = experiment_dir.expanduser().resolve()
    results_subdir = experiment_dir / "results"
    candidates = sorted(results_subdir.glob("results_*.json")) if results_subdir.exists() else []
    if not candidates:
        candidates = sorted(experiment_dir.glob("results_*.json"))
    if not candidates:
        # Allow passing a parent folder that contains one or more run directories.
        candidates = sorted(experiment_dir.rglob("results/results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No results_*.json found in {experiment_dir} or {results_subdir}")
    return candidates[-1]


def build_table(results: Dict[str, Any]) -> str:
    pruning_results = results.get("pruning_results", {})
    if not isinstance(pruning_results, dict) or not pruning_results:
        raise ValueError("No pruning_results found in JSON")

    row_pattern = re.compile(r"^(?P<metric>.+?)_(?P<mode>low|high|random)_sparsity_(?P<sparsity>.+)$")
    rows: List[
        Tuple[
            str,
            str,
            float,
            Optional[int],
            Optional[int],
            Optional[float],
            Optional[int],
            Optional[int],
            Optional[float],
            Optional[int],
            Optional[int],
            Optional[float],
            Optional[float],
        ]
    ] = []

    for key, entry in pruning_results.items():
        if not isinstance(entry, dict):
            continue

        m = row_pattern.match(str(key))
        if not m:
            continue

        metric = m.group("metric")
        mode = m.group("mode")
        sparsity = float(m.group("sparsity"))

        ppl_val = entry.get("perplexity")
        ppl = float(ppl_val) if isinstance(ppl_val, (int, float)) else None

        super_diag = entry.get("supernode_pruning", {}) or {}
        total_val = super_diag.get("supernodes_total")
        pruned_val = super_diag.get("supernodes_pruned")
        frac_val = super_diag.get("supernodes_pruned_frac")

        total = int(total_val) if isinstance(total_val, (int, float)) else None
        pruned = int(pruned_val) if isinstance(pruned_val, (int, float)) else None
        frac = float(frac_val) if isinstance(frac_val, (int, float)) else None

        nodes_total_val = super_diag.get("nodes_total")
        nodes_pruned_val = super_diag.get("nodes_pruned")
        nodes_frac_val = super_diag.get("nodes_pruned_frac")

        nodes_total = int(nodes_total_val) if isinstance(nodes_total_val, (int, float)) else None
        nodes_pruned = int(nodes_pruned_val) if isinstance(nodes_pruned_val, (int, float)) else None
        nodes_frac = float(nodes_frac_val) if isinstance(nodes_frac_val, (int, float)) else None

        both_total_val = super_diag.get("supernodes_both_scar_lp_activation_l2_total")
        both_pruned_val = super_diag.get("supernodes_both_scar_lp_activation_l2_pruned")
        both_frac_val = super_diag.get("supernodes_both_scar_lp_activation_l2_pruned_frac")

        both_total = int(both_total_val) if isinstance(both_total_val, (int, float)) else None
        both_pruned = int(both_pruned_val) if isinstance(both_pruned_val, (int, float)) else None
        both_frac = float(both_frac_val) if isinstance(both_frac_val, (int, float)) else None

        rows.append(
            (
                method_label(metric),
                mode,
                sparsity,
                nodes_pruned,
                nodes_total,
                nodes_frac,
                pruned,
                total,
                frac,
                both_pruned,
                both_total,
                both_frac,
                ppl,
            )
        )

    if not rows:
        raise ValueError("No pruning rows parsed from pruning_results")

    valid_fracs = [r[8] for r in rows if r[8] is not None]
    min_frac = min(valid_fracs) if valid_fracs else None

    rows.sort(key=lambda r: (r[2], r[0], r[1]))

    lines: List[str] = []
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Nodes pruned (\%)$\downarrow$ & Supernodes pruned (\%)$\downarrow$ & Pruned in SCAR-LP$\cap$Act-L2$\downarrow$ & PPL$\downarrow$ \\")
    lines.append(r"\midrule")

    for method, mode, _sparsity, nodes_pruned, nodes_total, nodes_frac, pruned, total, frac, both_pruned, both_total, both_frac, ppl in rows:
        method_txt = method if mode == "low" else f"{method} [{mode}]"

        if nodes_pruned is None or nodes_total is None or nodes_total <= 0:
            nodes_txt = "N/A"
        elif nodes_frac is None:
            nodes_txt = f"{nodes_pruned}/{nodes_total}"
        else:
            nodes_txt = f"{100.0 * nodes_frac:.1f} ({nodes_pruned}/{nodes_total})"

        if frac is None or pruned is None or total is None or total <= 0:
            super_txt = "N/A"
        else:
            val = f"{100.0 * frac:.1f} ({pruned}/{total})"
            super_txt = f"\\textbf{{{val}}}" if (min_frac is not None and abs(frac - min_frac) < 1e-12) else val

        if both_pruned is None or both_total is None or both_total <= 0:
            both_txt = "N/A"
        elif both_frac is None:
            both_txt = f"{both_pruned}/{both_total}"
        else:
            both_txt = f"{100.0 * both_frac:.1f} ({both_pruned}/{both_total})"

        ppl_txt = "N/A" if ppl is None else f"{ppl:.1f}"
        lines.append(f"{method_txt} & {nodes_txt} & {super_txt} & {both_txt} & {ppl_txt} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX supernode-pruning vs PPL table from pruning results JSON")
    parser.add_argument("--results-json", type=str, default=None, help="Path to results_*.json")
    parser.add_argument("--experiment-dir", type=str, default=None, help="Experiment run directory containing results/")
    parser.add_argument("--output", type=str, default=None, help="Output .tex path (default: next to results JSON)")
    args = parser.parse_args()

    if not args.results_json and not args.experiment_dir:
        raise ValueError("Provide either --results-json or --experiment-dir")

    if args.results_json:
        results_path = Path(args.results_json)
    else:
        results_path = find_latest_results_json(Path(args.experiment_dir))

    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    latex = build_table(results)

    if args.output:
        out_path = Path(args.output)
    else:
        stem = results_path.stem.replace("results_", "")
        out_path = results_path.parent / f"pruning_supernode_ppl_table_{stem}.tex"

    out_path.write_text(latex, encoding="utf-8")
    print(f"Saved table to: {out_path}")


if __name__ == "__main__":
    main()
