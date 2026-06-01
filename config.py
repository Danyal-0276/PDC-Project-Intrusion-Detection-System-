"""
Central configuration for the PySpark Network Intrusion Detection System (NIDS).

All CSV datasets live under Data/ (capital D). There is no data/ folder.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("NIDS_DATA_DIR", PROJECT_ROOT / "Data"))
# Use NIDS_OUTPUT_DIR=output2 for a second experiment (e.g. file-based vs random split)
OUTPUT_DIR = PROJECT_ROOT / os.getenv("NIDS_OUTPUT_DIR", "output")
REPORT_PATH = OUTPUT_DIR / "report.txt"

# ---------------------------------------------------------------------------
# Spark & dataset selection
# ---------------------------------------------------------------------------
SPARK_APP_NAME = "NetworkIntrusionDetection"
DATASET = os.getenv("NIDS_DATASET", "cicids").lower()  # "cicids" | "kdd"

# CIC-IDS2017 (default — matches files in Data/)
CICIDS_LABEL_RAW = " Label"
CICIDS_GLOB = str(DATA_DIR / "*.csv")

# KDD Cup 99 (optional — place kddcup.data_10_percent in Data/)
KDD_FILENAME = "kddcup.data_10_percent"
KDD_PATH = DATA_DIR / KDD_FILENAME

KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]

KDD_CATEGORICAL = ["protocol_type", "service", "flag"]
KDD_NUMERIC = [c for c in KDD_COLUMNS if c not in KDD_CATEGORICAL + ["label"]]

# Unified label column name after loading
LABEL_COLUMN = "label"

# ---------------------------------------------------------------------------
# ML settings
# ---------------------------------------------------------------------------
CLASSIFICATION_MODE = os.getenv("NIDS_MODE", "multiclass").lower()  # "multiclass" | "binary"
TRAIN_FRACTION = 0.8
TEST_FRACTION = 0.2
RANDOM_SEED = 42
RF_NUM_TREES = int(os.getenv("NIDS_RF_TREES", "15"))
GBT_MAX_ITER = int(os.getenv("NIDS_GBT_ITER", "50"))
LR_MAX_ITER = int(os.getenv("NIDS_LR_ITER", "100"))
LR_REG_PARAM = float(os.getenv("NIDS_LR_REG", "0.01"))

# Train/test split: "file" (day-based CSV files) or "random" (80/20 row split)
SPLIT_MODE = os.getenv("NIDS_SPLIT", "file").lower()

# Dataset size: use fewer CSV files on weak hardware (much faster)
#   small  = 2 files  (~750k rows)  — recommended for your VM
#   medium = 5 files  (~1.9M rows)
#   full   = 8 files  (~2.8M rows) — slow on 4GB RAM VM
DATA_SIZE = os.getenv("NIDS_DATA_SIZE", "small").lower()

# Use fraction of rows after train/test split (1.0 = all rows). 0.15–0.2 fits weak VMs.
SAMPLE_FRACTION = float(os.getenv("NIDS_SAMPLE_FRACTION", "0.2"))

# Fast mode: only RF + Naive Bayes + ensembles (skip slow GBT/LR)
FAST_MODE = os.getenv("NIDS_FAST", "1").lower() in ("1", "true", "yes")

# Balance rare classes (attack) during training — improves accuracy on imbalanced NIDS data
USE_CLASS_WEIGHTS = os.getenv("NIDS_CLASS_WEIGHTS", "1").lower() in ("1", "true", "yes")

# Train only specific model(s), comma-separated keys (e.g. gradient_boosted_trees)
# Empty = use FAST_MODE / full TRAINING_ORDER defaults.
MODELS_FILTER = os.getenv("NIDS_MODELS", "").strip().lower()

# CIC-IDS2017 file-based split — FULL set (all 8 files)
CICIDS_TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
]

CICIDS_TEST_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

# Smaller subsets (load only these files — do not delete originals from Data/)
CICIDS_SUBSETS = {
    "small": {
        "train": ["Monday-WorkingHours.pcap_ISCX.csv"],
        "test": ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
        "description": "~530k train + ~225k test (BENIGN + DDoS)",
    },
    "medium": {
        "train": [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
        ],
        "test": [
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        ],
        "description": "~1.7M train + ~510k test",
    },
    "full": {
        "train": CICIDS_TRAIN_FILES,
        "test": CICIDS_TEST_FILES,
        "description": "all 8 files (~2.8M rows)",
    },
}

# Results export
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
SINGLE_MODELS_CSV = RESULTS_DIR / "single_classifiers_comparison.csv"
ENSEMBLE_CSV = RESULTS_DIR / "ensemble_comparison.csv"
LABEL_DIST_CSV = RESULTS_DIR / "label_distribution.csv"
CONFUSION_MATRIX_CSV = RESULTS_DIR / "confusion_matrix.csv"
PER_CLASS_CSV = RESULTS_DIR / "per_class_metrics.csv"
LATEX_TABLES_DIR = RESULTS_DIR / "latex"
PAPER_REPORT_PATH = OUTPUT_DIR / "research_paper_report.txt"

# Normal-class identifiers used for binary mode
CICIDS_NORMAL_LABEL = "BENIGN"
KDD_NORMAL_LABEL = "normal"


def get_cicids_subset() -> dict:
    """Return train/test file lists for the active DATA_SIZE profile."""
    if DATA_SIZE not in CICIDS_SUBSETS:
        raise ValueError(
            f"Unknown NIDS_DATA_SIZE '{DATA_SIZE}'. Use: small, medium, full."
        )
    return CICIDS_SUBSETS[DATA_SIZE]


def get_data_paths() -> list[str]:
    """Return file path(s) to load for the active dataset profile."""
    if DATASET == "kdd":
        if not KDD_PATH.exists():
            raise FileNotFoundError(
                f"KDD dataset not found at {KDD_PATH}. "
                f"Place {KDD_FILENAME} in {DATA_DIR} or set NIDS_DATASET=cicids."
            )
        return [str(KDD_PATH)]

    if DATASET == "cicids":
        subset = get_cicids_subset()
        filenames = list(subset["train"]) + list(subset["test"])
        paths = [DATA_DIR / name for name in filenames]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"CSV file(s) missing for DATA_SIZE={DATA_SIZE}: {missing}"
            )
        return [str(p) for p in paths]

    raise ValueError(f"Unknown DATASET '{DATASET}'. Use 'cicids' or 'kdd'.")


def get_label_column() -> str:
    """Return the normalized label column name."""
    return LABEL_COLUMN


def get_categorical_columns() -> list[str]:
    """Return categorical feature columns for the active dataset."""
    if DATASET == "kdd":
        return list(KDD_CATEGORICAL)
    return []


def get_kdd_numeric_columns() -> list[str]:
    """Return numeric feature column names for KDD."""
    return list(KDD_NUMERIC)


def get_normal_label() -> str:
    """Return the string that represents benign/normal traffic for binary mode."""
    if DATASET == "kdd":
        return KDD_NORMAL_LABEL
    return CICIDS_NORMAL_LABEL


def get_train_file_names() -> list[str]:
    """Return CSV filenames used for training (file-based split)."""
    if DATASET == "cicids" and SPLIT_MODE == "file":
        return list(get_cicids_subset()["train"])
    return []


def get_test_file_names() -> list[str]:
    """Return CSV filenames used for testing (file-based split)."""
    if DATASET == "cicids" and SPLIT_MODE == "file":
        return list(get_cicids_subset()["test"])
    return []


def get_train_paths() -> list[str]:
    """Return absolute paths to training CSV files."""
    names = get_train_file_names()
    if not names:
        return []
    paths = [DATA_DIR / name for name in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Training files not found: {missing}")
    return [str(p) for p in paths]


def get_test_paths() -> list[str]:
    """Return absolute paths to test CSV files."""
    names = get_test_file_names()
    if not names:
        return []
    paths = [DATA_DIR / name for name in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Test files not found: {missing}")
    return [str(p) for p in paths]
