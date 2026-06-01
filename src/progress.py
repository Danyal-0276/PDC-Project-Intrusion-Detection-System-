"""
Dataset statistics and training progress helpers.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config
from src.models import get_models_to_train

ENSEMBLE_NAMES = ("ensemble_hard_vote", "ensemble_soft_vote")


def total_model_steps() -> int:
    return len(get_models_to_train()) + len(ENSEMBLE_NAMES)


def print_dataset_summary(df: DataFrame, title: str = "Dataset summary") -> int:
    """
    Print total rows, per-file counts (if available), and label distribution.

    Returns:
        Total row count.
    """
    total = df.count()
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"Total records: {total:,}")
    print(f"Feature columns (excl. label): {len([c for c in df.columns if c != config.LABEL_COLUMN and c != 'source_file'])}")

    if "source_file" in df.columns:
        print("\n--- Rows per CSV file ---")
        per_file = (
            df.groupBy("source_file")
            .count()
            .orderBy("source_file")
            .collect()
        )
        for row in per_file:
            name = row["source_file"]
            count = row["count"]
            pct = (count / total * 100) if total else 0
            print(f"  {name:55s} {count:>12,}  ({pct:5.2f}%)")

    if config.LABEL_COLUMN in df.columns:
        print("\n--- Rows per class (label) ---")
        per_label = (
            df.groupBy(config.LABEL_COLUMN)
            .count()
            .orderBy(F.desc("count"))
            .collect()
        )
        for row in per_label:
            label = row[config.LABEL_COLUMN]
            count = row["count"]
            pct = (count / total * 100) if total else 0
            print(f"  {str(label):25s} {count:>12,}  ({pct:5.2f}%)")

    print("=" * 60)
    return total


def print_split_summary(train_df: DataFrame, test_df: DataFrame) -> None:
    """Print train vs test record counts and percentages."""
    train_n = train_df.count()
    test_n = test_df.count()
    total = train_n + test_n
    print(f"\n--- Train / test split ({config.SPLIT_MODE}) ---")
    print(f"  Training: {train_n:>12,}  ({train_n / total * 100 if total else 0:5.1f}%)")
    print(f"  Test:     {test_n:>12,}  ({test_n / total * 100 if total else 0:5.1f}%)")
    print(f"  Total:    {total:>12,}")


def print_model_progress(step: int, name: str, phase: str = "start") -> None:
    """
    Print which model step is running (e.g. Model 2/6).

    Args:
        step: 1-based step index.
        name: Model name.
        phase: 'start' or 'done'.
    """
    total = total_model_steps()
    if phase == "start":
        print(f"\n>>> Model {step}/{total}: {name} — training started...")
    else:
        print(f">>> Model {step}/{total}: {name} — done.")
