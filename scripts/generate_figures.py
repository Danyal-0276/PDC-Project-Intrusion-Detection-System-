#!/usr/bin/env python3
"""
Generate publication-ready figures for NIDS research papers.

Reads CSV/JSON from output/results/ and saves PNGs to output/figures/.

Run after main.py:
    python scripts/generate_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

# Publication style (MIT / IEEE-friendly)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
    }
)

COLORS = {
    "single": "#2ecc71",
    "ensemble": "#e74c3c",
    "accent": "#3498db",
    "purple": "#9b59b6",
    "orange": "#e67e22",
}


def _ensure_dirs() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _pretty_model(name: str) -> str:
    return name.replace("_", " ").title()


def fig1_pipeline_diagram() -> None:
    """Block diagram of the NIDS pipeline."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    boxes = [
        "Data/*.csv",
        "Load & Clean",
        "Preprocess",
        "4 Classifiers",
        "Ensemble",
        "Report",
    ]
    x = 0.05
    for i, text in enumerate(boxes):
        ax.text(
            x + i * 0.16,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#d4e6f1", edgecolor="#2c3e50"),
        )
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x + (i + 1) * 0.16 - 0.02, 0.5),
                xytext=(x + i * 0.16 + 0.08, 0.5),
                arrowprops=dict(arrowstyle="->", color="#2c3e50"),
            )

    ax.set_title("Figure 1: PySpark NIDS Pipeline Architecture", fontweight="bold")
    out = config.FIGURES_DIR / "fig1_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig2_label_distribution() -> None:
    """Bar chart of class label counts."""
    path = config.LABEL_DIST_CSV
    if not path.exists():
        print(f"Skip fig2: {path} not found (run main.py first)")
        return

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        df["label"].astype(str),
        df["count"],
        color=COLORS["accent"],
        edgecolor="#2c3e50",
        linewidth=0.6,
    )
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    ax.set_xlabel("Class label")
    ax.set_ylabel("Number of flows")
    ax.set_title("Figure 2: CIC-IDS2017 Label Distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig2_label_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig3_model_comparison_f1() -> None:
    """Horizontal bar chart comparing macro-F1 across all models."""
    path = config.COMPARISON_CSV
    if not path.exists():
        print(f"Skip fig3: {path} not found (run main.py first)")
        return

    df = pd.read_csv(path).sort_values("f1_macro", ascending=True)
    df["display"] = df["model"].apply(_pretty_model)
    colors = [
        COLORS["ensemble"] if "ensemble" in m else COLORS["single"] for m in df["model"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(df["display"], df["f1_macro"], color=colors, edgecolor="#2c3e50", linewidth=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Macro-F1 score")
    ax.set_title("Figure 3: Model Comparison (Macro-F1)", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig3_model_comparison_f1macro.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig4_confusion_matrix() -> None:
    """Confusion matrix heatmap from confusion_matrix.csv."""
    path = config.CONFUSION_MATRIX_CSV
    if not path.exists():
        print(f"Skip fig4: {path} not found — using accuracy vs F1 fallback")
        _fig_accuracy_vs_f1_scatter()
        return

    df = pd.read_csv(path)
    labels = sorted(
        set(df["actual_label"].astype(str)) | set(df["predicted_label"].astype(str))
    )
    n = len(labels)
    idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=int)
    for _, row in df.iterrows():
        i = idx[str(row["actual_label"])]
        j = idx[str(row["predicted_label"])]
        matrix[i, j] = int(row["count"])

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Figure 4: Confusion Matrix (Best Single Classifier)", fontweight="bold")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig4_confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig5_multi_metric_grouped() -> None:
    """Grouped bar chart: accuracy, precision, recall, F1 for all models."""
    path = config.COMPARISON_CSV
    if not path.exists():
        return

    df = pd.read_csv(path).sort_values("f1_macro", ascending=False)
    df["display"] = df["model"].apply(_pretty_model)
    metrics = ["accuracy", "precision", "recall", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1-macro"]
    x = np.arange(len(df))
    width = 0.2
    palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, df[metric], width, label=label, color=palette[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(df["display"], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Figure 5: Multi-Metric Model Comparison", fontweight="bold")
    ax.legend(loc="upper right", ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig5_multi_metric_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig6_single_vs_ensemble() -> None:
    """Side-by-side comparison: single classifiers vs ensemble methods."""
    single_path = config.SINGLE_MODELS_CSV
    ens_path = config.ENSEMBLE_CSV
    if not single_path.exists() or not ens_path.exists():
        print("Skip fig6: single/ensemble CSV not found")
        return

    single = pd.read_csv(single_path)
    ensemble = pd.read_csv(ens_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, df, title, color in [
        (axes[0], single, "Single Classifiers", COLORS["single"]),
        (axes[1], ensemble, "Ensemble Methods", COLORS["ensemble"]),
    ]:
        df = df.sort_values("f1_macro", ascending=True)
        df["display"] = df["model"].apply(_pretty_model)
        bars = ax.barh(df["display"], df["f1_macro"], color=color, edgecolor="#2c3e50", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlabel("Macro-F1")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Figure 6: Single Classifiers vs Ensemble Methods", fontweight="bold", y=1.02)
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig6_single_vs_ensemble.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig7_per_class_heatmap() -> None:
    """Heatmap of per-class F1 scores across all models."""
    path = config.PER_CLASS_CSV
    if not path.exists():
        print(f"Skip fig7: {path} not found")
        return

    df = pd.read_csv(path)
    pivot = df.pivot(index="class_label", columns="model", values="f1")
    pivot.index = pivot.index.astype(str)
    pivot.columns = [_pretty_model(c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, max(4, len(pivot) * 0.6 + 2)))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Model")
    ax.set_ylabel("Class")
    ax.set_title("Figure 7: Per-Class F1 Score Heatmap", fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="F1 score")
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig7_per_class_f1_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig8_accuracy_recall_tradeoff() -> None:
    """Scatter plot: accuracy vs recall with model labels."""
    path = config.COMPARISON_CSV
    if not path.exists():
        return

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [
        COLORS["ensemble"] if "ensemble" in m else COLORS["single"] for m in df["model"]
    ]
    ax.scatter(df["recall"], df["accuracy"], s=120, c=colors, edgecolors="#2c3e50", linewidth=0.8, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(
            _pretty_model(row["model"]),
            (row["recall"], row["accuracy"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 8: Accuracy vs Recall by Model", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig8_accuracy_vs_recall.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig9_training_time() -> None:
    """Bar chart of training time per model."""
    path = config.COMPARISON_CSV
    if not path.exists():
        return

    df = pd.read_csv(path).sort_values("train_time_sec", ascending=True)
    df["display"] = df["model"].apply(_pretty_model)
    colors = [
        COLORS["ensemble"] if "ensemble" in m else COLORS["orange"] for m in df["model"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["display"], df["train_time_sec"], color=colors, edgecolor="#2c3e50", linewidth=0.6)
    ax.bar_label(bars, fmt="%.1fs", padding=3, fontsize=8)
    ax.set_xlabel("Training time (seconds)")
    ax.set_title("Figure 9: Model Training Time Comparison", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig9_training_time.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig10_ensemble_improvement() -> None:
    """Line chart showing if ensemble beats best single model on key metrics."""
    path = config.COMPARISON_CSV
    if not path.exists():
        return

    df = pd.read_csv(path)
    single = df[~df["model"].str.contains("ensemble")]
    if single.empty:
        return
    best_single = single.loc[single["f1_macro"].idxmax()]
    ensembles = df[df["model"].str.contains("ensemble")]
    if ensembles.empty:
        return

    metrics = ["accuracy", "precision", "recall", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1-macro"]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, [best_single[m] for m in metrics], "o-", color=COLORS["single"], linewidth=2, markersize=8,
            label=f"Best single ({_pretty_model(best_single['model'])})")
    for _, row in ensembles.iterrows():
        ax.plot(x, [row[m] for m in metrics], "s--", linewidth=1.5, markersize=7,
                label=_pretty_model(row["model"]))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Figure 10: Ensemble vs Best Single Classifier", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig10_ensemble_vs_best_single.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def _fig_accuracy_vs_f1_scatter() -> None:
    """Fallback scatter when confusion matrix unavailable."""
    path = config.COMPARISON_CSV
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["accuracy"], df["f1_macro"], s=80, c=COLORS["purple"], edgecolors="#2c3e50")
    for _, row in df.iterrows():
        ax.annotate(_pretty_model(row["model"]), (row["accuracy"], row["f1_macro"]), fontsize=7)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Accuracy vs Macro-F1 by Model")
    fig.tight_layout()
    out = config.FIGURES_DIR / "fig4_accuracy_vs_f1macro.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> int:
    _ensure_dirs()
    print("Generating publication figures...")
    fig1_pipeline_diagram()
    fig2_label_distribution()
    fig3_model_comparison_f1()
    fig4_confusion_matrix()
    fig5_multi_metric_grouped()
    fig6_single_vs_ensemble()
    fig7_per_class_heatmap()
    fig8_accuracy_recall_tradeoff()
    fig9_training_time()
    fig10_ensemble_improvement()
    print(f"\nAll figures saved to: {config.FIGURES_DIR}")
    print("Tables/CSVs for paper: output/results/")
    print("LaTeX tables: output/results/latex/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
