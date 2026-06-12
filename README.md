# Network Intrusion Detection System (NIDS)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.4-orange?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017-green)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey)](https://github.com/Danyal-0276/PDC-Project-Intrusion-Detection-System-)

Detect cyberattacks in network traffic using **Apache Spark MLlib** and machine learning on the **[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)** dataset. The pipeline loads raw CSV traffic captures, preprocesses features, trains multiple classifiers, compares them with ensemble voting, and exports publication-ready reports, tables, and charts.

> **PDC Project** — Parallel & Distributed Computing coursework demonstrating scalable ML for network security.

---

## Highlights

| | |
|---|---|
| **4 classifiers** | Naive Bayes, Logistic Regression, Gradient Boosted Trees, Random Forest |
| **2 ensemble methods** | Hard voting & soft voting |
| **2 evaluation splits** | Random 80/20 split & realistic file-based (day) split |
| **Research outputs** | LaTeX tables, CSV comparisons, PNG figures |
| **One-command runner** | `./run.sh both full 0.3` runs the full paper experiment |

---

## Pipeline overview

```mermaid
flowchart LR
    A[CIC-IDS2017 CSVs] --> B[Load & Clean]
    B --> C[Feature Engineering]
    C --> D[Train Classifiers]
    D --> E[Ensemble Voting]
    E --> F[Evaluate & Compare]
    F --> G[Reports & Figures]
```

1. **Load** network traffic data from CSV files  
2. **Clean** and preprocess features with Spark ML pipelines  
3. **Train** up to 4 ML models to detect attacks  
4. **Compare** models and build hard/soft ensemble voters  
5. **Export** reports, comparison tables, LaTeX, and charts  

On success you will see:

```text
Pipeline completed successfully.
All experiments completed successfully.
```

---

## Tech stack

| Component | Version | Role |
|-----------|---------|------|
| **Apache Spark / PySpark** | 3.5.4 | Distributed data processing & ML |
| **Java (OpenJDK)** | 11 (recommended) or 17 | Spark runtime |
| **Python** | 3.10 – 3.12 | Pipeline orchestration |
| **pandas / numpy** | ≥ 2.2 / ≥ 1.26 | Result analysis & charting |
| **scikit-learn** | ≥ 1.4 | ML utilities |
| **matplotlib** | ≥ 3.8 | Figure generation |

Full dependencies are pinned in [`requirements.txt`](requirements.txt).

---

## Quick start

### Prerequisites

- **Ubuntu Linux** 22.04 or 24.04 (VM is fine)
- **Java 11**, **Python 3.10+**, **Git**
- **CIC-IDS2017 CSV files** (~2.8 GB for full experiment) — [download here](https://www.unb.ca/cic/datasets/ids-2017.html)

### 1. Clone the repository

```bash
git clone https://github.com/Danyal-0276/PDC-Project-Intrusion-Detection-System-.git
cd PDC-Project-Intrusion-Detection-System-
```

### 2. Install system dependencies (Ubuntu)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y openjdk-11-jdk python3 python3-pip python3-venv git

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

### 3. Add dataset files

Place CIC-IDS2017 CSV files in the `Data/` folder:

```bash
mkdir -p Data
# Copy your downloaded CSVs into Data/
```

**Required file names (full experiment — 8 files):**

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

| Profile | CSV files | Approx. size |
|---------|-----------|--------------|
| **small** (quick test) | 2 | ~400 MB |
| **medium** | 5 | ~1.5 GB |
| **full** (paper results) | 8 | ~2.8 GB |

### 4. Install Python packages

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run

```bash
chmod +x run.sh
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Quick test (~15–20 min) — recommended first run
./run.sh random small 0.1 --fast

# Full paper experiment (~2–4 hours)
./run.sh both full 0.3
```

---

## Usage

### Command reference

| Command | Description |
|---------|-------------|
| `./run.sh random small 0.1 --fast` | Quick test — 2 CSVs, 10% sample, 2 models |
| `./run.sh both full 0.3` | Full paper run — both splits, all 8 CSVs, 30% sample |
| `./run.sh random full 0.3` | Random 80/20 split only → `output/` |
| `./run.sh file full 0.3` | File-based (day) split only → `output2/` |

**Positional arguments:** `[split] [size] [sample]`

| Argument | Options | Default |
|----------|---------|---------|
| `split` | `random`, `file`, `both` | `both` |
| `size` | `small`, `medium`, `full` | `full` |
| `sample` | `0.05` – `1.0` (fraction of rows) | `0.3` |

The `both` mode runs two experiments automatically:

1. **Random split** → saves to `output/`
2. **File-based split** → saves to `output2/`

### Expected runtime (4 GB RAM VM)

| Command | Approx. time |
|---------|--------------|
| `./run.sh random small 0.1 --fast` | 15–20 min |
| `./run.sh both full 0.3` | 2–4 hours |
| `./run.sh both full 1.0` | 4–8 hours |

---

## Results

During training, expect progress output like:

```text
[1/4] Loading data...
[2/4] Preprocessing...
[3/4] Training classifiers and ensembles...
>>> Model 1/6: naive_bayes — training started...
[4/4] Generating report...
Pipeline completed successfully.
```

Some steps pause for several minutes with no new output — this is normal.

### Output directories

| Folder | Experiment | Best for |
|--------|------------|----------|
| `output/` | Random 80/20 split | High accuracy comparison |
| `output2/` | File-based (day) split | Realistic / research evaluation |

### Key output files

| File | Contents |
|------|----------|
| `report.txt` | Summary report |
| `research_paper_report.txt` | Full report with Tables I–VI |
| `results/model_comparison.csv` | All models compared |
| `results/single_classifiers_comparison.csv` | NB, LR, GBT, RF only |
| `results/ensemble_comparison.csv` | Hard + soft vote results |
| `results/per_class_metrics.csv` | Per-class precision/recall |
| `results/latex/*.tex` | LaTeX tables for papers |
| `figures/*.png` | Charts for presentations |

### View results

```bash
cat output/research_paper_report.txt
cat output/results/model_comparison.csv
ls output/figures/
```

---

## Project structure

```
PDC-Project-Intrusion-Detection-System-/
├── Data/                    # CIC-IDS2017 CSV files (not included in repo)
├── env/                     # Python virtual environment (created locally)
├── output/                  # Results: random split experiment
├── output2/                 # Results: file-based split experiment
├── main.py                  # Main pipeline entry point
├── run.sh                   # One-command experiment runner
├── config.py                # Central configuration
├── requirements.txt         # Python dependencies
├── src/
│   ├── data_loader.py       # Spark session & data loading
│   ├── preprocessor.py      # Feature engineering pipeline
│   ├── detector.py          # Training & evaluation orchestration
│   ├── models.py            # Classifier factory
│   ├── ensemble.py          # Hard/soft voting ensembles
│   ├── evaluation.py        # Metrics & comparisons
│   ├── reporter.py          # Report generation
│   └── results_export.py    # CSV & LaTeX export
└── scripts/
    ├── run_experiment.py    # Multi-experiment runner (called by run.sh)
    └── generate_figures.py  # PNG chart generation
```

---

## Configuration

Environment variables for advanced tuning (see [`config.py`](config.py)):

| Variable | Default | Description |
|----------|---------|-------------|
| `NIDS_OUTPUT_DIR` | `output` | Result folder name |
| `NIDS_DATA_SIZE` | `small` | `small`, `medium`, or `full` |
| `NIDS_SAMPLE_FRACTION` | `0.2` | Row fraction (e.g. `0.1` = 10%) |
| `NIDS_SPLIT` | `file` | `file` or `random` |
| `NIDS_MODE` | `multiclass` | `binary` or `multiclass` |
| `NIDS_FAST` | `1` | `1` = 2 models, `0` = all 4 models |
| `NIDS_CLASS_WEIGHTS` | `1` | Balance rare attack classes |
| `NIDS_MODELS` | (all) | e.g. `gradient_boosted_trees` for one model |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `'JavaPackage' object is not callable` | Set `JAVA_HOME` and `PATH` (see Quick start step 2) |
| `No module named 'distutils'` | `pip install setuptools` |
| `requirements.txt not found` | Run commands from the project root directory |
| Missing CSV error | Place files in `Data/` with exact names listed above |
| Stuck at preprocessing | Wait 5–15 min; monitor Spark UI at http://localhost:4040 |
| Out of memory | Use `./run.sh both small 0.1 --fast` |

### Verify installation

```bash
source env/bin/activate
python -c "import pyspark; print('PySpark:', pyspark.__version__)"
python -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[1]').appName('test').getOrCreate()
print('Spark version:', spark.version)
spark.stop()
print('Spark + Java: OK')
"
```

Expected: `PySpark: 3.5.4` and `Spark + Java: OK`.

---

## Citation

If you use this project or its results in academic work, please cite the **CIC-IDS2017** dataset:

> Sharafaldin, I.; Lashkari, A. H.; Ghorbani, A. A. **Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.** ICISSP 2018.

- Dataset: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- Repository: [Danyal-0276/PDC-Project-Intrusion-Detection-System-](https://github.com/Danyal-0276/PDC-Project-Intrusion-Detection-System-)

---

## Contributors

Developed as part of a **Parallel & Distributed Computing (PDC)** course project.
