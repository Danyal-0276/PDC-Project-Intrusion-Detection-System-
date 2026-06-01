#!/usr/bin/env python3
"""
One-command NIDS experiment runner.

Runs main.py (+ figure generation) with split and dataset size chosen via flags.
No need to edit config files or remember long environment variable lists.

Examples:
    # Both experiments (random -> output/, file -> output2/)
    python scripts/run_experiment.py --split both --size full --sample 0.3

    # File-based only, all models
    python scripts/run_experiment.py --split file --size full --sample 0.3

    # Quick test on weak VM
    python scripts/run_experiment.py --split random --size small --sample 0.1 --fast

    # From project root (short form):
    ./run.sh both full 0.3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"
FIGURES_PY = PROJECT_ROOT / "scripts" / "generate_figures.py"

SPLIT_OUTPUT = {
    "random": "output",
    "file": "output2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PySpark NIDS pipeline with one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset sizes (place matching CSV files in Data/):
  small   2 files  (~750k rows)   quick tests
  medium  5 files  (~2.2M rows)   balanced
  full    8 files  (~2.8M rows)   all CIC-IDS2017 files

Split modes:
  random  80/20 row split — high accuracy, same-day traffic in train & test
  file    weekday train / Friday PM test — realistic unseen-attack evaluation
  both    runs random then file -> output/ and output2/
        """,
    )
    parser.add_argument(
        "--split",
        choices=["random", "file", "both"],
        default="both",
        help="Train/test split strategy (default: both)",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "full"],
        default="full",
        dest="data_size",
        help="Which CSV subset to load (default: full)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=0.3,
        help="Fraction of rows after split, 0.05-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "multiclass"],
        default="binary",
        help="Classification mode (default: binary)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: Naive Bayes + Random Forest only (skip GBT/LR)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation after each run",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override output folder (only when --split is random or file, not both)",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Train only these models (comma-separated, e.g. gbt or logistic_regression)",
    )
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if not (0.05 <= args.sample <= 1.0):
        raise SystemExit("--sample must be between 0.05 and 1.0")
    if args.split != "both" and args.output_dir and args.output_dir in SPLIT_OUTPUT.values():
        pass  # ok
    if args.split == "both" and args.output_dir:
        raise SystemExit("--output-dir cannot be used with --split both (uses output/ and output2/)")


def _java_env() -> dict[str, str]:
    env = os.environ.copy()
    if "JAVA_HOME" not in env:
        for candidate in (
            "/usr/lib/jvm/java-11-openjdk-amd64",
            "/usr/lib/jvm/java-17-openjdk-amd64",
        ):
            if Path(candidate).exists():
                env["JAVA_HOME"] = candidate
                env["PATH"] = f"{candidate}/bin:{env.get('PATH', '')}"
                print(f"Using JAVA_HOME={candidate}")
                break
    return env


def _build_env(
    split: str,
    output_dir: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    env = _java_env()
    env["NIDS_OUTPUT_DIR"] = output_dir
    env["NIDS_DATA_SIZE"] = args.data_size
    env["NIDS_SAMPLE_FRACTION"] = str(args.sample)
    env["NIDS_SPLIT"] = split
    env["NIDS_MODE"] = args.mode
    env["NIDS_FAST"] = "1" if args.fast else "0"
    env["NIDS_CLASS_WEIGHTS"] = "0" if args.no_class_weights else "1"
    if args.models:
        env["NIDS_MODELS"] = args.models
    return env


def _run_step(label: str, cmd: list[str], env: dict[str, str]) -> int:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    return result.returncode


def _run_one_experiment(split: str, output_dir: str, args: argparse.Namespace) -> int:
    env = _build_env(split, output_dir, args)

    print("\n" + "#" * 70)
    print(f"EXPERIMENT: split={split}  size={args.data_size}  sample={args.sample}")
    print(f"  output -> {PROJECT_ROOT / output_dir}")
    print("#" * 70)

    code = _run_step("Running pipeline (main.py)...", [sys.executable, str(MAIN_PY)], env)
    if code != 0:
        return code

    if args.no_figures:
        return 0

    return _run_step(
        "Generating figures...",
        [sys.executable, str(FIGURES_PY)],
        env,
    )


def main() -> int:
    args = _parse_args()
    _validate(args)

    if not MAIN_PY.exists():
        raise SystemExit(f"main.py not found at {MAIN_PY}")

    if args.split == "both":
        plans = [("random", SPLIT_OUTPUT["random"]), ("file", SPLIT_OUTPUT["file"])]
    elif args.split == "random":
        out = args.output_dir or SPLIT_OUTPUT["random"]
        plans = [("random", out)]
    else:
        out = args.output_dir or SPLIT_OUTPUT["file"]
        plans = [("file", out)]

    for split, output_dir in plans:
        code = _run_one_experiment(split, output_dir, args)
        if code != 0:
            print(f"\nExperiment failed (split={split}, exit={code}).")
            return code

    print("\n" + "=" * 70)
    print("All experiments completed successfully.")
    for split, output_dir in plans:
        report = PROJECT_ROOT / output_dir / "research_paper_report.txt"
        if not report.exists():
            report = PROJECT_ROOT / output_dir / "report.txt"
        print(f"  [{split:6s}] {report}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
