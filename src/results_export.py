"""
Export experiment metrics to JSON and CSV for paper tables.
"""

import json
from datetime import datetime

import config


def export_experiment_results(results: dict) -> None:
    """
    Save full experiment results (all models + per-class) to JSON and CSV files.

    Args:
        results: Dictionary returned by detector.train_and_evaluate().
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated": datetime.now().isoformat(),
        "dataset": config.DATASET,
        "classification_mode": config.CLASSIFICATION_MODE,
        "split_mode": config.SPLIT_MODE,
        "train_files": config.get_train_file_names(),
        "test_files": config.get_test_file_names(),
        "best_single": results.get("best_single"),
        "best_overall": results.get("best_overall"),
        "models": results.get("models", {}),
        "comparison": results.get("comparison", []),
        "per_class": results.get("per_class", {}),
    }

    json_path = config.RESULTS_DIR / "experiment_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    print(f"Results JSON saved to: {json_path}")

    _export_per_class_csv(results.get("per_class", {}))


def _json_default(obj):
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _export_per_class_csv(per_class: dict) -> None:
    """Write per-class metrics for each model to a single CSV."""
    path = config.PER_CLASS_CSV
    lines = ["model,class_label,precision,recall,f1,support\n"]

    for model_name, rows in per_class.items():
        for row in rows:
            lines.append(
                f"{model_name},{row['label']},{row['precision']:.6f},"
                f"{row['recall']:.6f},{row['f1']:.6f},{row['support']}\n"
            )

    path.write_text("".join(lines), encoding="utf-8")
    print(f"Per-class metrics CSV saved to: {path}")
