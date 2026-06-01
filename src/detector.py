"""
Train multiple classifiers, ensemble voting, and compare results.
"""

import time

from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config
from src.data_loader import split_train_test
from src.ensemble import build_hard_ensemble, build_soft_ensemble
from src.evaluation import compute_metrics, print_per_class_table
from src.models import SINGLE_MODELS, build_classifier, get_models_to_train
from src.results_export import export_experiment_results
from src.preprocessor import sanitize_ml_features
from src.progress import print_model_progress, print_split_summary

ENSEMBLE_HARD = "ensemble_hard_vote"
ENSEMBLE_SOFT = "ensemble_soft_vote"


def _add_class_weights(df: DataFrame) -> DataFrame:
    """Inverse-frequency sample weights so rare attack classes are not ignored."""
    stats = df.groupBy("label_idx").count().collect()
    if not stats:
        return df.withColumn("weight", F.lit(1.0))

    total = sum(int(row["count"]) for row in stats)
    n_classes = len(stats)
    weight_expr = None
    for row in stats:
        idx = row["label_idx"]
        count = int(row["count"])
        w = total / (n_classes * count)
        clause = F.when(F.col("label_idx") == idx, F.lit(float(w)))
        weight_expr = clause if weight_expr is None else weight_expr.when(
            F.col("label_idx") == idx, F.lit(float(w))
        )
    return df.withColumn("weight", weight_expr.otherwise(F.lit(1.0)))


def _get_label_indexer(prep_model: Pipeline) -> StringIndexer | None:
    for stage in prep_model.stages:
        if isinstance(stage, StringIndexer) and stage.getInputCol() == config.LABEL_COLUMN:
            return stage
    return None


def _print_confusion_matrix(predictions: DataFrame, prep_model: Pipeline, title: str) -> None:
    """Print readable confusion matrix for a model's predictions."""
    label_indexer = _get_label_indexer(prep_model)
    print(f"\n--- Confusion matrix: {title} ---")

    if label_indexer is None:
        predictions.groupBy("label_idx", "prediction").count().orderBy(
            "label_idx", "prediction"
        ).show(30, truncate=False)
        return

    readable = IndexToString(
        inputCol="label_idx",
        outputCol="actual_label",
        labels=label_indexer.labels,
    ).transform(predictions)
    readable = IndexToString(
        inputCol="prediction",
        outputCol="predicted_label",
        labels=label_indexer.labels,
    ).transform(readable)
    readable.groupBy("actual_label", "predicted_label").count().orderBy(
        F.desc("count")
    ).show(30, truncate=False)


def _print_metrics_row(name: str, metrics: dict) -> None:
    print(
        f"  {name:28s}  "
        f"Acc={metrics['accuracy']:.4f}  "
        f"Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}  "
        f"F1w={metrics['f1_weighted']:.4f}  "
        f"F1macro={metrics['f1_macro']:.4f}"
    )


def _run_ensembles(
    joined,
    pred_columns,
    prob_columns,
    prep_model,
    all_metrics,
    n_models: int,
) -> None:
    """Hard/soft voting when two or more classifiers were trained."""
    ens_step = n_models + 1
    print(f"\n{'=' * 50}")
    print_model_progress(ens_step, ENSEMBLE_HARD, phase="start")
    print("Ensemble: hard voting (majority)")
    print("=" * 50)

    hard_df = build_hard_ensemble(joined, pred_columns)
    hard_metrics = compute_metrics(
        hard_df.select("label_idx", F.col("prediction")),
        prep_model,
    )
    all_metrics[ENSEMBLE_HARD] = hard_metrics
    _print_metrics_row(ENSEMBLE_HARD, hard_metrics)
    _print_confusion_matrix(
        hard_df.select("label_idx", "prediction"),
        prep_model,
        ENSEMBLE_HARD,
    )
    print_model_progress(ens_step, ENSEMBLE_HARD, phase="done")

    ens_step = n_models + 2
    print(f"\n{'=' * 50}")
    print_model_progress(ens_step, ENSEMBLE_SOFT, phase="start")
    print("Ensemble: soft voting (averaged probabilities)")
    print("=" * 50)

    soft_df = build_soft_ensemble(joined, prob_columns)
    soft_metrics = compute_metrics(
        soft_df.select("label_idx", F.col("prediction")),
        prep_model,
    )
    all_metrics[ENSEMBLE_SOFT] = soft_metrics
    _print_metrics_row(ENSEMBLE_SOFT, soft_metrics)
    print_model_progress(ens_step, ENSEMBLE_SOFT, phase="done")


def train_and_evaluate(df: DataFrame, preprocessing_pipeline: Pipeline) -> dict:
    """
    Train four single classifiers plus hard/soft ensembles; compare all results.

    Args:
        df: Label-prepared DataFrame.
        preprocessing_pipeline: Unfitted feature pipeline.

    Returns:
        Dictionary with per-model metrics, comparison list, and prep_model.
    """
    train_df, test_df = split_train_test(df)

    if config.SAMPLE_FRACTION < 1.0:
        print(
            f"\nSampling {config.SAMPLE_FRACTION * 100:.0f}% of rows "
            f"(set NIDS_SAMPLE_FRACTION=1.0 for all data)"
        )
        train_df = train_df.sample(
            withReplacement=False,
            fraction=config.SAMPLE_FRACTION,
            seed=config.RANDOM_SEED,
        )
        test_df = test_df.sample(
            withReplacement=False,
            fraction=config.SAMPLE_FRACTION,
            seed=config.RANDOM_SEED + 1,
        )

    print_split_summary(train_df, test_df)

    # Cache splits so fit/transform do not re-read CSVs from disk on every Spark action.
    print("\nCaching train/test splits in memory (avoids re-reading CSV files)...")
    train_df = train_df.cache()
    test_df = test_df.cache()
    train_df.count()
    test_df.count()
    print("  Splits cached.")

    print("\nFitting preprocessing pipeline (StringIndexer + VectorAssembler)...")
    print("  On a weak VM this can take several minutes; check http://localhost:4040 for activity.")
    prep_model = preprocessing_pipeline.fit(train_df)
    print("  Preprocessing fit complete.")

    print("\nTransforming and sanitizing features (NaN/inf -> 0 for Naive Bayes)...")
    train_processed = sanitize_ml_features(prep_model.transform(train_df)).cache()
    test_processed = sanitize_ml_features(prep_model.transform(test_df)).cache()
    print("Materializing train/test data in memory...")
    print(f"  Train rows (processed): {train_processed.count():,}")
    print(f"  Test rows (processed):  {test_processed.count():,}")

    if config.USE_CLASS_WEIGHTS:
        print("\nApplying class weights (boost rare attack labels for better accuracy)...")
        train_processed = _add_class_weights(train_processed).cache()
        train_processed.count()

    test_processed = test_processed.withColumn(
        "row_id", F.monotonically_increasing_id()
    )

    all_metrics: dict[str, dict] = {}
    trained_models: dict = {}
    joined = test_processed.select("row_id", "label_idx")
    prob_columns: list[str] = []
    pred_columns: list[str] = []

    models_to_train = get_models_to_train()
    if config.FAST_MODE:
        print("\nFAST_MODE on: training Random Forest + Naive Bayes only (set NIDS_FAST=0 for all 4)")
    else:
        print(f"\nFull mode: training all {len(models_to_train)} classifiers + 2 ensembles")

    # Train each single classifier
    for step_idx, model_name in enumerate(models_to_train, start=1):
        print(f"\n{'=' * 50}")
        print_model_progress(step_idx, model_name, phase="start")
        print("=" * 50)

        classifier = build_classifier(model_name)
        start = time.time()
        fitted = classifier.fit(train_processed)
        elapsed = time.time() - start
        print_model_progress(step_idx, model_name, phase="done")
        trained_models[model_name] = fitted

        predictions = fitted.transform(test_processed)
        metrics = compute_metrics(predictions, prep_model)
        metrics["train_time_sec"] = elapsed
        all_metrics[model_name] = metrics

        pred_col = f"pred_{model_name}"
        prob_col = f"prob_{model_name}"

        pred_slice = predictions.select(
            "row_id",
            F.col("prediction").alias(pred_col),
            F.col("probability").alias(prob_col),
        )
        joined = joined.join(pred_slice, on="row_id", how="inner")
        pred_columns.append(pred_col)
        prob_columns.append(prob_col)

        _print_metrics_row(model_name, metrics)
        print(f"  Training time: {elapsed:.1f}s")

        if model_name == models_to_train[0]:
            _print_confusion_matrix(predictions, prep_model, model_name)
            if "per_class" in metrics:
                print_per_class_table(model_name, metrics["per_class"])

    # Ensembles need 2+ models; skip for single-model runs (e.g. NIDS_MODELS=gbt)
    if len(models_to_train) >= 2:
        _run_ensembles(
            joined,
            pred_columns,
            prob_columns,
            prep_model,
            all_metrics,
            len(models_to_train),
        )
    else:
        print("\nSkipping ensembles (single-model run).")

    # Comparison table
    comparison = _build_comparison_table(all_metrics)
    _print_comparison_table(comparison)

    best_single = max(
        models_to_train,
        key=lambda m: all_metrics[m]["f1_macro"],
    )
    best_overall = max(
        all_metrics.keys(),
        key=lambda k: all_metrics[k]["f1_macro"],
    )

    print(f"\nBest single model (macro-F1): {best_single}")
    print(f"Best overall (macro-F1):      {best_overall}")

    # Per-class metrics for all models (for export / paper tables)
    per_class_all = {
        name: m.get("per_class", [])
        for name, m in all_metrics.items()
        if m.get("per_class")
    }

    if per_class_all.get(best_overall):
        print_per_class_table(best_overall, per_class_all[best_overall])

    results = {
        "models": all_metrics,
        "comparison": comparison,
        "best_single": best_single,
        "best_overall": best_overall,
        "per_class": per_class_all,
        "prep_model": prep_model,
        "trained_models": trained_models,
        "split_mode": config.SPLIT_MODE,
        "accuracy": all_metrics[best_overall]["accuracy"],
        "precision": all_metrics[best_overall]["precision"],
        "recall": all_metrics[best_overall]["recall"],
        "f1_weighted": all_metrics[best_overall]["f1_weighted"],
        "f1_macro": all_metrics[best_overall]["f1_macro"],
    }

    _save_confusion_matrix_csv(
        trained_models[best_single].transform(test_processed),
        prep_model,
        best_single,
    )

    export_experiment_results(results)
    return results


def _save_confusion_matrix_csv(
    predictions: DataFrame,
    prep_model: Pipeline,
    model_name: str,
) -> None:
    """Export readable confusion matrix for the best single model."""
    label_indexer = _get_label_indexer(prep_model)
    if label_indexer is None:
        return

    readable = IndexToString(
        inputCol="label_idx",
        outputCol="actual_label",
        labels=label_indexer.labels,
    ).transform(predictions)
    readable = IndexToString(
        inputCol="prediction",
        outputCol="predicted_label",
        labels=label_indexer.labels,
    ).transform(readable)

    rows = (
        readable.groupBy("actual_label", "predicted_label")
        .count()
        .orderBy("actual_label", "predicted_label")
        .collect()
    )

    lines = ["actual_label,predicted_label,count\n"]
    for row in rows:
        lines.append(
            f"{row['actual_label']},{row['predicted_label']},{row['count']}\n"
        )

    config.CONFUSION_MATRIX_CSV.write_text("".join(lines), encoding="utf-8")
    print(f"Confusion matrix CSV saved to: {config.CONFUSION_MATRIX_CSV}")


def _build_comparison_table(all_metrics: dict[str, dict]) -> list[dict]:
    """Build sorted list of model results for reporting."""
    rows = []
    for name, m in all_metrics.items():
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_weighted": m["f1_weighted"],
                "f1_macro": m["f1_macro"],
                "train_time_sec": m.get("train_time_sec", 0.0),
            }
        )
    return sorted(rows, key=lambda r: r["f1_macro"], reverse=True)


def _print_comparison_table(comparison: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("MODEL COMPARISON (sorted by macro-F1)")
    print("=" * 90)
    print(
        f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
        f"{'F1-weight':>10} {'F1-macro':>10} {'Time(s)':>10}"
    )
    print("-" * 90)
    for row in comparison:
        print(
            f"{row['model']:<30} "
            f"{row['accuracy']:>10.4f} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['f1_weighted']:>10.4f} "
            f"{row['f1_macro']:>10.4f} "
            f"{row['train_time_sec']:>10.1f}"
        )
    print("=" * 90)
