#!/usr/bin/env python3
"""Print dataset row counts only (no training). Fast check before full run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import create_spark_session, load_data
from src.preprocessor import prepare_labels
from pyspark.sql import functions as F

import config


def main() -> None:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = load_data(spark)
        df = df.dropna()
        df = prepare_labels(df)
        print("\n(After dropna + label prep)")
        from src.progress import print_dataset_summary

        print_dataset_summary(df)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
