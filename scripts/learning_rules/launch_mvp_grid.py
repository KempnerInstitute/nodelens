#!/usr/bin/env python3
"""Launch or print the ResNet-18/CIFAR-100 learning-rules MVP grid."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

CONFIGS = [
    "configs/learning_rules/resnet18_cifar100_bp_baseline.yaml",
    "configs/learning_rules/resnet18_cifar100_bp_decov.yaml",
    "configs/learning_rules/resnet18_cifar100_bp_tard.yaml",
    "configs/learning_rules/resnet18_cifar100_bp_rtp.yaml",
]

SEEDS = [42, 123, 456, 789, 1011]


def value_token(value: float | int | str) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def build_command(
    config: str,
    seed: int,
    *,
    python: str,
    device: str | None,
    allow_dirty: bool,
    learning_rule_lambda: float | None = None,
    training_epochs: int | None = None,
    extra_overrides: list[str] | None = None,
    name_suffix: str | None = None,
) -> list[str]:
    cmd = [python, "scripts/run_experiment.py", "--config", config, "--seed", str(seed)]
    if device:
        cmd.extend(["--device", device])
    if allow_dirty:
        cmd.append("--allow-dirty")
    overrides = list(extra_overrides or [])
    if learning_rule_lambda is not None:
        overrides.append(f"learning_rule.lambda={learning_rule_lambda:g}")
    if training_epochs is not None:
        overrides.append(f"training_epochs={int(training_epochs)}")
    if name_suffix:
        overrides.append(f"name={Path(config).stem}_{name_suffix}")
    cmd.extend(overrides)
    return cmd


def build_sbatch_command(
    *,
    shell_command: str,
    job_name: str,
    log_path: Path,
    partition: str,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time: str,
    account: str | None,
) -> list[str]:
    cmd = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        "--job-name",
        job_name,
        "--output",
        str(log_path),
        "--partition",
        partition,
        "--gres",
        gres,
        "--cpus-per-task",
        str(cpus_per_task),
        "--mem",
        mem,
        "--time",
        time,
    ]
    if account:
        cmd.extend(["--account", account])
    cmd.extend(["--wrap", shell_command])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python executable for run_experiment.py")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda:0")
    parser.add_argument("--allow-dirty", action="store_true", help="Forward --allow-dirty to run_experiment.py")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--configs", nargs="*", default=CONFIGS, help="Config files to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS, help="Seeds to run")
    parser.add_argument("--learning-rule-lambda", type=float, default=None, help="Single learning-rule lambda override")
    parser.add_argument(
        "--learning-rule-lambdas",
        nargs="*",
        type=float,
        default=None,
        help="Launch one run per learning-rule lambda override",
    )
    parser.add_argument("--training-epochs", type=int, default=None, help="Override training_epochs for pilot runs")
    parser.add_argument(
        "--width-multipliers",
        nargs="*",
        type=float,
        default=None,
        help="Launch CIFAR-ResNet-18 width-constrained runs for each multiplier.",
    )
    parser.add_argument("--width-model-name", default="cifar_resnet18", help="Model name used with --width-multipliers")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra key=value override forwarded to run_experiment.py; can be repeated",
    )
    parser.add_argument("--slurm", action="store_true", help="Submit each run as an sbatch job")
    parser.add_argument("--partition", default="kempner_dev", help="SLURM partition")
    parser.add_argument("--account", default=None, help="Optional SLURM account")
    parser.add_argument("--gres", default="gpu:1", help="SLURM gres request")
    parser.add_argument("--cpus-per-task", type=int, default=8, help="SLURM CPUs per task")
    parser.add_argument("--mem", default="64G", help="SLURM memory request")
    parser.add_argument("--time", default="2-00:00:00", help="SLURM walltime")
    parser.add_argument("--log-dir", default="logs/learning_rules/slurm", help="Directory for SLURM logs")
    parser.add_argument("--job-name-prefix", default="lr-resnet18", help="Prefix for SLURM job names")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    log_dir = repo_root / args.log_dir
    if args.slurm:
        log_dir.mkdir(parents=True, exist_ok=True)

    lambda_values: list[float | None]
    if args.learning_rule_lambdas is not None:
        lambda_values = list(args.learning_rule_lambdas)
    elif args.learning_rule_lambda is not None:
        lambda_values = [args.learning_rule_lambda]
    else:
        lambda_values = [None]
    width_values: list[float | None] = list(args.width_multipliers) if args.width_multipliers is not None else [None]

    for config in args.configs:
        for learning_rule_lambda in lambda_values:
            for width_multiplier in width_values:
                suffix_parts = []
                extra_overrides = list(args.override)
                if width_multiplier is not None:
                    width_token = value_token(f"{width_multiplier:g}")
                    suffix_parts.append(f"w{width_token}")
                    extra_overrides.extend(
                        [
                            f"model.name={args.width_model_name}",
                            f"model.width_multiplier={width_multiplier:g}",
                        ]
                    )
                if learning_rule_lambda is not None:
                    suffix_parts.append(f"lam{value_token(f'{learning_rule_lambda:g}')}")
                if args.training_epochs is not None:
                    suffix_parts.append(f"e{int(args.training_epochs)}")
                name_suffix = "_".join(suffix_parts) if suffix_parts else None
                for seed in args.seeds:
                    cmd = build_command(
                        config,
                        seed,
                        python=args.python,
                        device=args.device,
                        allow_dirty=args.allow_dirty,
                        learning_rule_lambda=learning_rule_lambda,
                        training_epochs=args.training_epochs,
                        extra_overrides=extra_overrides,
                        name_suffix=name_suffix,
                    )
                    if args.slurm:
                        stem = Path(config).stem
                        suffix = f"-{name_suffix}" if name_suffix else ""
                        job_name = f"{args.job_name_prefix}-{stem}{suffix}-s{seed}"
                        log_path = log_dir / f"{job_name}-%j.out"
                        shell_command = f"cd {shlex.quote(str(repo_root))} && {shlex.join(cmd)}"
                        sbatch_cmd = build_sbatch_command(
                            shell_command=shell_command,
                            job_name=job_name,
                            log_path=log_path,
                            partition=args.partition,
                            gres=args.gres,
                            cpus_per_task=args.cpus_per_task,
                            mem=args.mem,
                            time=args.time,
                            account=args.account,
                        )
                        print(shlex.join(sbatch_cmd), flush=True)
                        if not args.dry_run:
                            completed = subprocess.run(sbatch_cmd, cwd=repo_root, check=True, text=True, capture_output=True)
                            print(f"submitted {job_name}: {completed.stdout.strip()}", flush=True)
                    else:
                        print(shlex.join(cmd), flush=True)
                        if not args.dry_run:
                            subprocess.run(cmd, cwd=repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
