#!/usr/bin/env python3
"""
Network Intrusion Detection System (NIDS) — main entry point.

Pipeline: load -> preprocess -> detect -> report

Run:
    cd ~/spark_project
    source env/bin/activate
    python main.py
"""

import sys
import traceback

import config
from src.data_loader import create_spark_session, load_data
from src.preprocessor import preprocess_for_training
from src.detector import train_and_evaluate
from src.reporter import generate_report


def main() -> int:
    """Run the full NIDS pipeline with step-by-step progress output."""
    spark = None

    try:
        print("=" * 60)
        print("Network Intrusion Detection System (PySpark)")
        print("=" * 60)
        print(f"Dataset:  {config.DATASET}")
        print(f"Data dir: {config.DATA_DIR}")
        print(f"Data size: {config.DATA_SIZE}", end="")
        if config.DATASET == "cicids":
            print(f" — {config.get_cicids_subset().get('description', '')}")
        else:
            print()
        print(f"Mode:     {config.CLASSIFICATION_MODE}")
        print(f"Split:    {config.SPLIT_MODE}")
        print(f"Sample:   {config.SAMPLE_FRACTION * 100:.0f}% of rows (NIDS_SAMPLE_FRACTION)")
        print(f"Fast mode: {config.FAST_MODE} (RF+NB only; NIDS_FAST=0 for all 4 models)")
        print(f"Output:   {config.OUTPUT_DIR}")
        print("=" * 60)

        # Step 1: Load data
        print("\n[1/4] Loading data...")
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("WARN")
        df = load_data(spark)

        # Step 2: Preprocess (label prep + build pipeline)
        print("\n[2/4] Preprocessing...")
        df, preprocessing_pipeline = preprocess_for_training(df)
        print("Preprocessing pipeline built (will be fit on training split).")

        # Step 3: Train 4 classifiers + ensembles, compare results
        print("\n[3/4] Training classifiers and ensembles...")
        print("  Models: see Model 1/N progress (fast mode = RF, NB, 2 ensembles)")
        from src.progress import total_model_steps

        total_steps = total_model_steps()
        print(f"  Progress: watch for 'Model 1/{total_steps}' ... 'Model {total_steps}/{total_steps}' in the log")
        print("  Note: first activity is preprocessing fit (before Model 1) — not a hang.")
        print("  Spark UI (optional): http://localhost:4040 — job stages while training")
        results = train_and_evaluate(df, preprocessing_pipeline)

        # Step 4: Generate report
        print("\n[4/4] Generating report...")
        generate_report(df, results)

        # Day 1: generate paper figures from exported results
        print("\n[5/5] Generating figures...")
        try:
            from scripts.generate_figures import main as generate_figures

            generate_figures()
        except Exception as fig_exc:
            print(f"Warning: figure generation failed: {fig_exc}")
            print("Run manually: python scripts/generate_figures.py")

        print("\nPipeline completed successfully.")
        return 0

    except FileNotFoundError as exc:
        print(f"\nERROR [File not found]: {exc}", file=sys.stderr)
        return 1

    except ValueError as exc:
        print(f"\nERROR [Configuration]: {exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        print(f"\nERROR [Pipeline failed]: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    finally:
        if spark is not None:
            spark.stop()
            print("\nSpark session stopped.")


if __name__ == "__main__":
    sys.exit(main())
