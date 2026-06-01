"""
Metric computation for multiclass intrusion detection models.
"""

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config


def _distinct_class_indices(
    predictions: DataFrame,
    label_col: str,
    prediction_col: str,
) -> list[float]:
    """Return sorted class indices seen in labels or predictions."""
    indices: set[float] = set()
    for row in predictions.select(label_col).distinct().collect():
        indices.add(float(row[label_col]))
    for row in predictions.select(prediction_col).distinct().collect():
        indices.add(float(row[prediction_col]))
    return sorted(indices)


def compute_per_class_metrics(
    predictions: DataFrame,
    prep_model: Pipeline,
    label_col: str = "label_idx",
    prediction_col: str = "prediction",
) -> list[dict]:
    """
    Compute precision, recall, F1, and support for each class label.

    Returns:
        List of dicts with human-readable label names and metrics.
    """
    label_indexer = None
    for stage in prep_model.stages:
        if isinstance(stage, StringIndexer) and stage.getInputCol() == config.LABEL_COLUMN:
            label_indexer = stage
            break

    rdd = predictions.select(prediction_col, label_col).rdd.map(
        lambda row: (float(row[prediction_col]), float(row[label_col]))
    )
    if rdd.isEmpty():
        return []

    mlib_metrics = MulticlassMetrics(rdd)
    label_indices = _distinct_class_indices(predictions, label_col, prediction_col)
    rows = []

    for idx in label_indices:
        idx_f = float(idx)
        precision = mlib_metrics.precision(idx_f)
        recall = mlib_metrics.recall(idx_f)
        f1 = mlib_metrics.fMeasure(idx_f)
        support = int(predictions.filter(F.col(label_col) == idx).count())

        if label_indexer and int(idx) < len(label_indexer.labels):
            label_name = label_indexer.labels[int(idx)]
        else:
            label_name = str(int(idx))

        rows.append(
            {
                "label": label_name,
                "label_idx": int(idx),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )

    return rows


def compute_metrics(
    predictions: DataFrame,
    prep_model: Pipeline | None = None,
    label_col: str = "label_idx",
    prediction_col: str = "prediction",
) -> dict:
    """
    Compute accuracy, weighted metrics, macro-F1, and optional per-class breakdown.

    Args:
        predictions: DataFrame with label and prediction columns.
        prep_model: Fitted preprocessing pipeline (for per-class label names).
        label_col: Ground-truth indexed label column.
        prediction_col: Predicted label column.

    Returns:
        Dictionary of metric name -> float, plus 'per_class' list if prep_model given.
    """
    evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
    )

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1_weighted = evaluator.setMetricName("f1").evaluate(predictions)

    rdd = predictions.select(prediction_col, label_col).rdd.map(
        lambda row: (float(row[prediction_col]), float(row[label_col]))
    )
    if rdd.isEmpty():
        macro_f1 = 0.0
    else:
        mlib_metrics = MulticlassMetrics(rdd)
        labels = _distinct_class_indices(predictions, label_col, prediction_col)
        if labels:
            macro_f1 = sum(mlib_metrics.fMeasure(float(l)) for l in labels) / len(labels)
        else:
            macro_f1 = 0.0

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_weighted": f1_weighted,
        "f1_macro": macro_f1,
    }

    if prep_model is not None:
        result["per_class"] = compute_per_class_metrics(
            predictions, prep_model, label_col, prediction_col
        )

    return result


def print_per_class_table(model_name: str, per_class: list[dict]) -> None:
    """Print per-class metrics table to stdout."""
    print(f"\n--- Per-class metrics: {model_name} ---")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    for row in per_class:
        print(
            f"{row['label']:<25} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} "
            f"{row['support']:>10}"
        )
