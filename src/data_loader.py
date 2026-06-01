"""
Load network intrusion datasets from Data/ into a Spark DataFrame.
"""

import csv
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

import config


def create_spark_session() -> SparkSession:
    """Create and return a SparkSession for the NIDS application."""
    return (
        SparkSession.builder.appName(config.SPARK_APP_NAME)
        .master("local[*]")
        # Weak laptops/VMs default to ~1g driver memory and 200 shuffle partitions,
        # which makes preprocessing look hung (GC thrash, disk spill, no log output).
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "4")
        .getOrCreate()
    )


def _trim_column_names(df: DataFrame) -> DataFrame:
    """Strip leading/trailing whitespace from all column names."""
    for col_name in df.columns:
        trimmed = col_name.strip()
        if trimmed != col_name:
            df = df.withColumnRenamed(col_name, trimmed)
    return df


def _normalize_label_column(df: DataFrame) -> DataFrame:
    """Rename raw label column (e.g. ' Label' or 'Label') to 'label'."""
    if config.LABEL_COLUMN in df.columns:
        return df
    for candidate in ("Label", config.CICIDS_LABEL_RAW.strip(), config.CICIDS_LABEL_RAW):
        if candidate in df.columns:
            return df.withColumnRenamed(candidate, config.LABEL_COLUMN)
    raise ValueError(
        f"Label column not found. Expected '{config.LABEL_COLUMN}' or 'Label'. "
        f"Columns: {df.columns}"
    )


_INF_STRINGS = ["inf", "Inf", "INF", "-inf", "-Inf", "-INF"]


def _clean_and_cast_features(df: DataFrame) -> DataFrame:
    """
    Replace inf values and cast features to double in ONE Spark pass.

    (Old code used 78 separate withColumn calls — that could take hours on a VM.)
    """
    skip = {config.LABEL_COLUMN, "source_file"}
    exprs = []
    for col_name in df.columns:
        if col_name in skip:
            exprs.append(F.col(col_name))
        else:
            raw = F.col(col_name)
            cleaned = F.when(raw.cast("string").isin(_INF_STRINGS), None).otherwise(raw)
            # Null/inf -> 0.0 so VectorAssembler and Naive Bayes never see NaN.
            exprs.append(F.coalesce(cleaned.cast(DoubleType()), F.lit(0.0)).alias(col_name))
    return df.select(exprs)


def _add_source_filename(df: DataFrame) -> DataFrame:
    """Add basename of source CSV for file-based train/test splitting."""
    return df.withColumn(
        "source_file",
        F.regexp_extract(F.input_file_name(), r"([^/\\\\]+)$", 1),
    )


def _read_unique_headers(csv_path: str) -> list[str]:
    """
    Read CSV header row and return unique column names.

    CIC-IDS2017 has duplicate 'Fwd Header Length'; suffix duplicates as _1, _2, ...
    """
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as handle:
        row = next(csv.reader(handle))
    headers = [cell.strip() for cell in row]
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in headers:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    return unique


def _load_one_cicids_file(spark: SparkSession, path: str) -> DataFrame:
    """Load one CIC-IDS CSV with consistent column names (avoids schema merge warnings)."""
    headers = _read_unique_headers(path)
    n_cols = len(headers)

    raw = spark.read.csv(path, header=False, inferSchema=False)
    spark_cols = raw.columns
    if len(spark_cols) != n_cols:
        raise ValueError(
            f"Column count mismatch in {path}: header has {n_cols}, data has {len(spark_cols)}"
        )

    df = raw.select([F.col(spark_cols[i]).alias(headers[i]) for i in range(n_cols)])

    # Drop the header row still present as the first data row
    first_feature = headers[0]
    df = df.filter(F.col(first_feature).cast(DoubleType()).isNotNull())

    df = df.withColumn("source_file", F.lit(Path(path).name))
    return df


def _load_cicids(spark: SparkSession, paths: list[str]) -> DataFrame:
    """Load and union CIC-IDS2017 CSV files from Data/ (one file at a time)."""
    if not paths:
        raise FileNotFoundError("No CIC-IDS CSV paths provided.")

    print("Loading CIC-IDS files individually (fixes duplicate column headers)...")
    parts = [_load_one_cicids_file(spark, path) for path in paths]
    df = parts[0]
    for part in parts[1:]:
        df = df.unionByName(part, allowMissingColumns=True)

    df = _normalize_label_column(df)
    print("Cleaning inf values and casting features (single pass)...")
    df = _clean_and_cast_features(df)
    return df


def split_train_test(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Split DataFrame into train and test sets.

    Uses file-based split for CIC-IDS when SPLIT_MODE=file; otherwise random 80/20.
    Drops the source_file column from both sets before returning.
    """
    if config.SPLIT_MODE == "file" and config.DATASET == "cicids":
        train_names = config.get_train_file_names()
        test_names = config.get_test_file_names()

        print("\n--- File-based train/test split ---")
        print("Train files:")
        for name in train_names:
            print(f"  - {name}")
        print("Test files:")
        for name in test_names:
            print(f"  - {name}")

        train_df = df.filter(F.col("source_file").isin(train_names)).drop("source_file")
        test_df = df.filter(F.col("source_file").isin(test_names)).drop("source_file")
        return train_df, test_df

    print(f"\n--- Random train/test split ({config.TRAIN_FRACTION:.0%}/{config.TEST_FRACTION:.0%}) ---")
    if "source_file" in df.columns:
        df = df.drop("source_file")
    train_df, test_df = df.randomSplit(
        [config.TRAIN_FRACTION, config.TEST_FRACTION],
        seed=config.RANDOM_SEED,
    )
    return train_df, test_df


def _load_kdd(spark: SparkSession, paths: list[str]) -> DataFrame:
    """Load KDD Cup 99 (headerless) and assign column names."""
    df = spark.read.csv(paths[0], header=False, inferSchema=False)
    df = df.toDF(*config.KDD_COLUMNS)

    for col_name in config.KDD_NUMERIC:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    # Strip trailing dot from KDD labels (e.g. 'normal.' -> 'normal')
    df = df.withColumn(
        config.LABEL_COLUMN,
        F.regexp_replace(F.col(config.LABEL_COLUMN), r"\.$", ""),
    )
    return df


def load_data(spark: SparkSession) -> DataFrame:
    """
    Load the configured dataset, print diagnostics, and return the DataFrame.

    Args:
        spark: Active SparkSession.

    Returns:
        Raw DataFrame ready for preprocessing.
    """
    paths = config.get_data_paths()
    print(f"Loading dataset '{config.DATASET}' from:")
    for path in paths:
        print(f"  - {path}")

    if config.DATASET == "kdd":
        df = _load_kdd(spark, paths)
        if config.SPLIT_MODE == "file":
            print("Note: file-based split not used for KDD; falling back to random split.")
    else:
        df = _load_cicids(spark, paths)

    print(f"\nSplit mode: {config.SPLIT_MODE}")

    print("\n--- Schema (first columns) ---")
    df.printSchema()

    from src.progress import print_dataset_summary

    print_dataset_summary(df, title="CIC-IDS2017 dataset loaded")

    print("\n--- Sample rows (first 5) ---")
    df.show(5, truncate=False)

    return df
