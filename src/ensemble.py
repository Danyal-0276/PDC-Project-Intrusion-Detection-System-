"""
Ensemble methods: combine predictions from multiple classifiers.
"""

import numpy as np
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

# UDFs are created lazily after SparkSession starts (not at import time).
_majority_vote_udf = None
_average_prob_udf = None
_argmax_udf = None


def _majority_vote(*votes):
    """Return the most common class label (hard voting)."""
    valid = [int(v) for v in votes if v is not None]
    if not valid:
        return None
    counts = np.bincount(valid)
    return float(int(np.argmax(counts)))


def _average_probabilities(*vectors):
    """Element-wise average of probability vectors (soft voting input)."""
    arrays = []
    for vec in vectors:
        if vec is None:
            continue
        arrays.append(np.array(vec.toArray() if hasattr(vec, "toArray") else vec, dtype=float))
    if not arrays:
        return None
    # Models trained on Monday-only data can output different vector lengths (1 vs 2 classes).
    max_len = max(len(a) for a in arrays)
    padded = []
    for arr in arrays:
        if len(arr) < max_len:
            fill = np.zeros(max_len, dtype=float)
            fill[: len(arr)] = arr
            padded.append(fill)
        else:
            padded.append(arr)
    return Vectors.dense(np.mean(padded, axis=0))


def _vector_argmax(vec) -> float:
    """Index of maximum probability."""
    if vec is None:
        return None
    arr = vec.toArray() if hasattr(vec, "toArray") else vec
    return float(int(np.argmax(arr)))


def _ensure_udfs() -> None:
    """Register UDFs once SparkSession is active."""
    global _majority_vote_udf, _average_prob_udf, _argmax_udf
    if _majority_vote_udf is None:
        _majority_vote_udf = udf(_majority_vote, DoubleType())
        _average_prob_udf = udf(_average_probabilities, VectorUDT())
        _argmax_udf = udf(_vector_argmax, DoubleType())


def build_hard_ensemble(
    joined: DataFrame,
    prediction_columns: list[str],
    output_col: str = "prediction",
) -> DataFrame:
    """
    Majority-vote ensemble over classifier prediction columns.

    Args:
        joined: DataFrame with one prediction column per model.
        prediction_columns: e.g. ['pred_random_forest', 'pred_gradient_boosted_trees', ...]
        output_col: Name for the ensemble prediction column.

    Returns:
        DataFrame with label_idx and ensemble prediction.
    """
    _ensure_udfs()
    vote_cols = [F.col(c) for c in prediction_columns]
    return joined.withColumn(output_col, _majority_vote_udf(*vote_cols))


def build_soft_ensemble(
    joined: DataFrame,
    probability_columns: list[str],
    output_col: str = "prediction",
    prob_col: str = "ensemble_probability",
) -> DataFrame:
    """
    Soft-voting ensemble: average class probabilities, then argmax.

    Args:
        joined: DataFrame with one probability vector column per model.
        probability_columns: e.g. ['prob_random_forest', ...]
        output_col: Final predicted class index.
        prob_col: Averaged probability vector column name.

    Returns:
        DataFrame with label_idx and ensemble prediction.
    """
    _ensure_udfs()
    prob_cols = [F.col(c) for c in probability_columns]
    averaged = joined.withColumn(prob_col, _average_prob_udf(*prob_cols))
    return averaged.withColumn(output_col, _argmax_udf(F.col(prob_col)))
