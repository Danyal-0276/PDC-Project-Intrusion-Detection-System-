"""
Build the PySpark ML preprocessing pipeline for NIDS feature engineering.
"""

import math

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf

import config


def _sanitize_vector(vec) -> Vectors:
    """Replace NaN/inf/negative values — required by Spark Naive Bayes."""
    if vec is None:
        return Vectors.dense([0.0])
    cleaned = []
    for x in vec.toArray():
        try:
            v = float(x)
        except (TypeError, ValueError):
            v = 0.0
        if math.isnan(v) or math.isinf(v) or v < 0.0:
            v = 0.0
        cleaned.append(v)
    return Vectors.dense(cleaned)


_sanitize_udf = udf(_sanitize_vector, VectorUDT())


def sanitize_ml_features(df: DataFrame, col: str = "features") -> DataFrame:
    """Ensure feature vectors contain only finite, non-negative values."""
    return df.withColumn(col, _sanitize_udf(F.col(col)))


def prepare_labels(df: DataFrame) -> DataFrame:
    """
    Normalize labels for training and reporting.

    - KDD: trailing dots already stripped at load time.
    - Binary mode: map benign/normal to 'normal', all others to 'attack'.
    """
    df = df.withColumn(config.LABEL_COLUMN, F.trim(F.col(config.LABEL_COLUMN)))
    normal_label = config.get_normal_label()

    if config.CLASSIFICATION_MODE == "binary":
        df = df.withColumn(
            config.LABEL_COLUMN,
            F.when(
                F.upper(F.col(config.LABEL_COLUMN)) == normal_label.upper(),
                F.lit("normal"),
            ).otherwise(F.lit("attack")),
        )

    return df


def get_feature_column_names(df: DataFrame) -> list[str]:
    """Return numeric/categorical feature columns (everything except label)."""
    if config.DATASET == "kdd":
        return config.KDD_CATEGORICAL + config.KDD_NUMERIC

    skip = {config.LABEL_COLUMN, "source_file"}
    return [c for c in df.columns if c not in skip]


def build_preprocessing_pipeline(df: DataFrame) -> Pipeline:
    """
    Build an unfitted PySpark ML Pipeline for feature transformation.

    Stages:
      1. StringIndexer on categorical columns (KDD only)
      2. StringIndexer on label -> label_idx
      3. VectorAssembler -> features (no StandardScaler: it produces NaN/negatives
         that break Naive Bayes on CIC-IDS2017 constant-variance columns)

    The pipeline is fitted on training data inside detector.py to avoid leakage.
    """
    stages = []
    indexed_feature_cols = []

    categorical_cols = config.get_categorical_columns()
    for col_name in categorical_cols:
        out_col = f"{col_name}_idx"
        stages.append(
            StringIndexer(
                inputCol=col_name,
                outputCol=out_col,
                handleInvalid="keep",
            )
        )
        indexed_feature_cols.append(out_col)

    if config.DATASET == "kdd":
        indexed_feature_cols.extend(config.KDD_NUMERIC)
    else:
        feature_cols = get_feature_column_names(df)
        indexed_feature_cols.extend(feature_cols)

    stages.append(
        StringIndexer(
            inputCol=config.LABEL_COLUMN,
            outputCol="label_idx",
            handleInvalid="keep",
        )
    )

    stages.append(
        VectorAssembler(
            inputCols=indexed_feature_cols,
            outputCol="features",
            handleInvalid="skip",
        )
    )

    return Pipeline(stages=stages)


def preprocess_for_training(df: DataFrame) -> tuple[DataFrame, Pipeline]:
    """
    Apply label preparation and return the DataFrame plus preprocessing pipeline.

    Args:
        df: Raw loaded DataFrame.

    Returns:
        Tuple of (label-prepared DataFrame, unfitted Pipeline).
    """
    df = df.dropna()
    df = prepare_labels(df)
    pipeline = build_preprocessing_pipeline(df)
    return df, pipeline
