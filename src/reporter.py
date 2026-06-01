"""
Generate attack distribution summaries and save results to output/report.txt.
"""

from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config


def _format_model_name(name: str) -> str:
    return name.replace("_", " ").title()


def _metric_row(name: str, m: dict) -> str:
    return (
        f"| {_format_model_name(name):<28} | {m.get('accuracy', 0):>8.4f} | "
        f"{m.get('precision', 0):>9.4f} | {m.get('recall', 0):>8.4f} | "
        f"{m.get('f1_weighted', 0):>10.4f} | {m.get('f1_macro', 0):>8.4f} | "
        f"{m.get('train_time_sec', 0):>8.1f} |"
    )


def _table_header() -> list[str]:
    return [
        "| Model                        | Accuracy | Precision | Recall   | F1-weight | F1-macro | Time(s) |",
        "|------------------------------|----------|-----------|----------|-----------|----------|---------|",
    ]


def generate_report(df: DataFrame, results: dict) -> None:
    """
    Print and save a summary report: label distribution + model comparison.

    Args:
        df: Label-prepared DataFrame.
        results: Dictionary returned by detector.train_and_evaluate().
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.LATEX_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = results.get("models", results)
    comparison = results.get("comparison", [])
    per_class_all = results.get("per_class", {})

    total_rows = df.count()
    label_counts = (
        df.groupBy(config.LABEL_COLUMN)
        .count()
        .orderBy(F.desc("count"))
        .collect()
    )

    from src.models import SINGLE_MODELS

    single_rows = [r for r in comparison if r["model"] in SINGLE_MODELS]
    ensemble_rows = [
        r for r in comparison if r["model"] in ("ensemble_hard_vote", "ensemble_soft_vote")
    ]

    data_desc = (
        config.get_cicids_subset().get("description", "")
        if config.DATASET == "cicids"
        else config.KDD_FILENAME
    )

    lines = [
        "=" * 90,
        "Network Intrusion Detection System — Research Report",
        "=" * 90,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "TABLE I — Experimental configuration",
        "-" * 90,
        f"  Dataset:           {config.DATASET.upper()} (CIC-IDS2017)",
        f"  Data profile:      {config.DATA_SIZE} — {data_desc}",
        f"  Classification:    {config.CLASSIFICATION_MODE}",
        f"  Train/test split:  {config.SPLIT_MODE}"
        + (
            " (weekday train / Friday PM test)"
            if config.SPLIT_MODE == "file"
            else f" (random {config.TRAIN_FRACTION:.0%}/{config.TEST_FRACTION:.0%})"
        ),
        f"  Sample fraction:   {config.SAMPLE_FRACTION:.0%}",
        f"  Class weights:     {config.USE_CLASS_WEIGHTS}",
        f"  Fast mode:         {config.FAST_MODE}",
        f"  Total records:     {total_rows:,}",
        "",
        "TABLE II — Label distribution",
        "-" * 90,
        f"  {'Label':<30} {'Count':>12} {'Percent':>10}",
    ]

    for row in label_counts:
        label = row[config.LABEL_COLUMN]
        count = row["count"]
        pct = (count / total_rows * 100) if total_rows else 0
        lines.append(f"  {str(label):<30} {count:>12,} {pct:>9.2f}%")

    lines.extend(["", "TABLE III — Single classifier performance (test set)", "-" * 90])
    lines.extend(_table_header())
    for row in single_rows:
        lines.append(_metric_row(row["model"], row))

    lines.extend(["", "TABLE IV — Ensemble methods (test set)", "-" * 90])
    lines.extend(_table_header())
    for row in ensemble_rows:
        lines.append(_metric_row(row["model"], row))

    lines.extend(["", "TABLE V — Full model ranking (by macro-F1)", "-" * 90])
    lines.extend(_table_header())
    for row in comparison:
        lines.append(_metric_row(row["model"], row))

    best_single = results.get("best_single", "n/a")
    best_overall = results.get("best_overall", "n/a")
    lines.extend(
        [
            "",
            "KEY FINDINGS",
            "-" * 90,
            f"  Best single classifier (macro-F1): {_format_model_name(best_single)}",
            f"  Best overall method (macro-F1):    {_format_model_name(best_overall)}",
        ]
    )

    if per_class_all.get(best_overall):
        lines.extend(["", f"TABLE VI — Per-class metrics ({_format_model_name(best_overall)})", "-" * 90])
        lines.append(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        for row in per_class_all[best_overall]:
            lines.append(
                f"  {str(row['label']):<25} {row['precision']:>10.4f} "
                f"{row['recall']:>10.4f} {row['f1']:>10.4f} {row['support']:>10}"
            )

    lines.extend(
        [
            "",
            "OUTPUT FILES FOR PAPER",
            "-" * 90,
            f"  Report (this file):     {config.REPORT_PATH}",
            f"  Paper summary:          {config.PAPER_REPORT_PATH}",
            f"  All models CSV:         {config.COMPARISON_CSV}",
            f"  Single classifiers CSV: {config.SINGLE_MODELS_CSV}",
            f"  Ensembles CSV:          {config.ENSEMBLE_CSV}",
            f"  Per-class CSV:          {config.PER_CLASS_CSV}",
            f"  LaTeX tables:           {config.LATEX_TABLES_DIR}/",
            f"  Figures (PNG):          {config.FIGURES_DIR}/",
            "",
            "=" * 90,
        ]
    )

    report_text = "\n".join(lines)
    print("\n" + report_text)
    config.REPORT_PATH.write_text(report_text, encoding="utf-8")
    config.PAPER_REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to: {config.REPORT_PATH}")
    print(f"Paper report saved to: {config.PAPER_REPORT_PATH}")

    _save_label_distribution_csv(label_counts, total_rows)
    _save_comparison_csv(comparison)
    _save_single_and_ensemble_csvs(single_rows, ensemble_rows)
    _save_latex_tables(comparison, single_rows, ensemble_rows, label_counts, total_rows)


def _save_label_distribution_csv(label_counts, total_rows: int) -> None:
    """Export label counts for figure generation."""
    lines = ["label,count,percent\n"]
    for row in label_counts:
        label = row[config.LABEL_COLUMN]
        count = row["count"]
        pct = (count / total_rows * 100) if total_rows else 0
        lines.append(f"{label},{count},{pct:.4f}\n")
    config.LABEL_DIST_CSV.write_text("".join(lines), encoding="utf-8")


def _save_comparison_csv(comparison: list[dict]) -> None:
    """Export model comparison table to CSV for paper tables."""
    if not comparison:
        return

    header = "model,accuracy,precision,recall,f1_weighted,f1_macro,train_time_sec\n"
    rows = []
    for row in comparison:
        rows.append(
            f"{row['model']},{row['accuracy']:.6f},{row['precision']:.6f},"
            f"{row['recall']:.6f},{row['f1_weighted']:.6f},{row['f1_macro']:.6f},"
            f"{row['train_time_sec']:.2f}\n"
        )

    config.COMPARISON_CSV.write_text(header + "".join(rows), encoding="utf-8")
    print(f"Comparison CSV saved to: {config.COMPARISON_CSV}")


def _save_single_and_ensemble_csvs(
    single_rows: list[dict], ensemble_rows: list[dict]
) -> None:
    """Export separate CSV tables for single classifiers vs ensembles."""
    header = "model,accuracy,precision,recall,f1_weighted,f1_macro,train_time_sec\n"

    def _rows(data: list[dict]) -> str:
        return "".join(
            f"{r['model']},{r['accuracy']:.6f},{r['precision']:.6f},"
            f"{r['recall']:.6f},{r['f1_weighted']:.6f},{r['f1_macro']:.6f},"
            f"{r['train_time_sec']:.2f}\n"
            for r in data
        )

    if single_rows:
        config.SINGLE_MODELS_CSV.write_text(header + _rows(single_rows), encoding="utf-8")
        print(f"Single classifiers CSV saved to: {config.SINGLE_MODELS_CSV}")
    if ensemble_rows:
        config.ENSEMBLE_CSV.write_text(header + _rows(ensemble_rows), encoding="utf-8")
        print(f"Ensemble CSV saved to: {config.ENSEMBLE_CSV}")


def _save_latex_tables(
    comparison: list[dict],
    single_rows: list[dict],
    ensemble_rows: list[dict],
    label_counts,
    total_rows: int,
) -> None:
    """Export LaTeX table snippets for MIT-style research papers."""

    def _latex_table(caption: str, label: str, rows: list[dict]) -> str:
        body = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Model & Accuracy & Precision & Recall & F1-weighted & F1-macro & Time (s) \\\\",
            "\\midrule",
        ]
        for row in rows:
            name = _format_model_name(row["model"]).replace("_", "\\_")
            body.append(
                f"{name} & {row['accuracy']:.4f} & {row['precision']:.4f} & "
                f"{row['recall']:.4f} & {row['f1_weighted']:.4f} & "
                f"{row['f1_macro']:.4f} & {row['train_time_sec']:.1f} \\\\"
            )
        body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        return "\n".join(body)

    if single_rows:
        path = config.LATEX_TABLES_DIR / "table_single_classifiers.tex"
        path.write_text(
            _latex_table(
                "Single classifier performance on CIC-IDS2017 test set",
                "tab:single_classifiers",
                single_rows,
            ),
            encoding="utf-8",
        )

    if ensemble_rows:
        path = config.LATEX_TABLES_DIR / "table_ensembles.tex"
        path.write_text(
            _latex_table(
                "Ensemble method performance on CIC-IDS2017 test set",
                "tab:ensembles",
                ensemble_rows,
            ),
            encoding="utf-8",
        )

    if comparison:
        path = config.LATEX_TABLES_DIR / "table_full_comparison.tex"
        path.write_text(
            _latex_table(
                "Full model comparison ranked by macro-F1",
                "tab:full_comparison",
                comparison,
            ),
            encoding="utf-8",
        )

    label_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Dataset label distribution}",
        "\\label{tab:label_distribution}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Label & Count & Percent \\\\",
        "\\midrule",
    ]
    for row in label_counts:
        label = str(row[config.LABEL_COLUMN]).replace("_", "\\_")
        count = row["count"]
        pct = (count / total_rows * 100) if total_rows else 0
        label_lines.append(f"{label} & {count:,} & {pct:.2f}\\% \\\\")
    label_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (config.LATEX_TABLES_DIR / "table_label_distribution.tex").write_text(
        "\n".join(label_lines), encoding="utf-8"
    )
    print(f"LaTeX tables saved to: {config.LATEX_TABLES_DIR}/")
