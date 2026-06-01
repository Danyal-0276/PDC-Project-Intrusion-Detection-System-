# Network Intrusion Detection System (NIDS)

Detect cyberattacks in network traffic using **Apache Spark** and machine learning on the **CIC-IDS2017** dataset.

This guide is written so **anyone can run the experiment** — copy each command, paste it in the terminal, and press Enter.

**GitHub:** https://github.com/allech01/Intrusion-Detection

---

## What this project does (in simple words)

1. **Loads** network traffic data (CSV files)
2. **Cleans** and prepares the data
3. **Trains** 4 AI models to detect attacks
4. **Compares** models and builds ensemble voting
5. **Saves** a report, tables, and charts you can use in a research paper

When finished, you will see messages like:

```text
Pipeline completed successfully.
All experiments completed successfully.
```

---

## Prerequisites (software you need)

Install these **before** running the project.

| Software | Required version | Why you need it |
|----------|------------------|-----------------|
| **Ubuntu Linux** | 22.04 or 24.04 (VM is OK) | Project runs on Linux |
| **Java (OpenJDK)** | **11** (recommended) or 17 | PySpark needs Java |
| **Python** | **3.10, 3.11, or 3.12** | Runs the pipeline |
| **Git** | Any recent version | Download project from GitHub |
| **Internet** | — | Install packages (~500 MB download first time) |

### Python packages (installed automatically from `requirements.txt`)

| Package | Pinned version | Purpose |
|---------|----------------|---------|
| pyspark | **3.5.4** | Apache Spark + machine learning |
| py4j | **0.10.9.7** | Python ↔ Java bridge for Spark |
| setuptools | ≥ 69.0.0 | Required on Python 3.12 |
| pandas | ≥ 2.2.0 | Read result tables |
| numpy | ≥ 1.26.0 | Math for charts |
| scikit-learn | ≥ 1.4.0 | ML utilities |
| matplotlib | ≥ 3.8.0 | Generate figures |

### Dataset files (you download separately — NOT on GitHub)

| Profile | CSV files needed | Approx. size |
|---------|------------------|--------------|
| **small** (quick test) | 2 files | ~400 MB |
| **medium** | 5 files | ~1.5 GB |
| **full** (paper results) | **8 files** | ~2.8 GB |

Download from: [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## Step 1 — Install system software (Ubuntu)

Open **Terminal** and run these commands **one at a time**.

### 1.1 Update system

```bash
sudo apt update
sudo apt upgrade -y
```

### 1.2 Install Java 11

```bash
sudo apt install -y openjdk-11-jdk
```

**Verify Java:**

```bash
java -version
```

**Expected output (similar to):**

```text
openjdk version "11.0.x"
OpenJDK Runtime Environment ...
OpenJDK 64-Bit Server VM ...
```

If you see `command not found`, Java is not installed — repeat step 1.2.

### 1.3 Set Java path (required every new terminal, or add to bashrc once)

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
echo $JAVA_HOME
```

**Expected output:**

```text
/usr/lib/jvm/java-11-openjdk-amd64
```

**Optional — make permanent:**

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 1.4 Install Python and Git

```bash
sudo apt install -y python3 python3-pip python3-venv git
```

**Verify Python:**

```bash
python3 --version
```

**Expected:** `Python 3.10.x` or `Python 3.12.x`

**Verify Git:**

```bash
git --version
```

**Expected:** `git version 2.x.x`

---

## Step 2 — Download the project

### Option A — Clone from GitHub (recommended)

```bash
cd ~/Desktop
git clone https://github.com/allech01/Intrusion-Detection.git spark_project
cd spark_project
```

### Option B — Already have the folder but no git?

```bash
cd ~/Desktop
mv spark_project spark_project_old
git clone https://github.com/allech01/Intrusion-Detection.git spark_project
mkdir -p spark_project/Data
cp spark_project_old/Data/*.csv spark_project/Data/
cd spark_project
```

**Verify project files:**

```bash
ls main.py requirements.txt run.sh
```

**Expected:** all three files listed.

---

## Step 3 — Add the dataset (CSV files)

1. Download CIC-IDS2017 CSV files from the [official website](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Copy all `.csv` files into the `Data/` folder:

```bash
mkdir -p ~/Desktop/spark_project/Data
# Copy your downloaded CSVs into Data/ (use file manager or cp command)
```

**Verify dataset (full run needs 8 files):**

```bash
ls ~/Desktop/spark_project/Data/*.csv | wc -l
```

**Expected for full experiment:** `8`

**File names must match exactly:**

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

---

## Step 4 — Install Python packages

Run once:

```bash
cd ~/Desktop/spark_project
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This downloads ~500 MB the first time. Wait until it finishes.

### Verify each Python package

Run these **after** `source env/bin/activate`:

```bash
python -c "import pyspark; print('PySpark:', pyspark.__version__)"
```

**Expected:** `PySpark: 3.5.4`

```bash
python -c "import py4j; print('py4j OK')"
```

**Expected:** `py4j OK`

```bash
python -c "import setuptools; print('setuptools OK')"
```

**Expected:** `setuptools OK`

```bash
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
```

Each should print a version number with no errors.

### Verify Spark + Java together

```bash
python -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[1]').appName('test').getOrCreate()
print('Spark version:', spark.version)
spark.stop()
print('Spark + Java: OK')
"
```

**Expected:**

```text
Spark version: 3.5.4
Spark + Java: OK
```

If you see `'JavaPackage' object is not callable` → Java is not set. Repeat Step 1.3.

If you see `No module named 'distutils'` → run: `pip install setuptools`

---

## Step 5 — Run the experiment (ONE command)

Make the runner executable (once):

```bash
cd ~/Desktop/spark_project
chmod +x run.sh
source env/bin/activate
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

### Quick test first (~15–20 minutes) — recommended for beginners

```bash
./run.sh random small 0.1 --fast
```

| Part | Meaning |
|------|---------|
| `random` | One experiment only (faster) |
| `small` | Uses 2 CSV files only |
| `0.1` | Uses 10% of rows |
| `--fast` | Trains 2 models instead of 4 (faster) |

### Full paper experiment (~2–4 hours)

```bash
./run.sh both full 0.3
```

| Part | Meaning |
|------|---------|
| `both` | Runs **two** experiments automatically |
| `full` | All 8 CSV files |
| `0.3` | Uses 30% of rows (good balance of speed vs accuracy) |

**What `both` runs internally:**

1. **Random split** → saves to `output/`
2. **File-based split** → saves to `output2/`

You do **not** need to run any other commands.

---

## Step 6 — How to know it worked

### During the run

You should see steps like:

```text
[1/4] Loading data...
[2/4] Preprocessing...
[3/4] Training classifiers and ensembles...
>>> Model 1/6: naive_bayes — training started...
[4/4] Generating report...
Pipeline completed successfully.
```

**Note:** Some steps pause for several minutes with no new text — this is normal. Do not close the terminal.

### After the run — check success

```bash
cat ~/Desktop/spark_project/output/report.txt
cat ~/Desktop/spark_project/output2/report.txt
```

If both files exist and show model accuracy numbers → **experiment succeeded**.

### View your results (easy)

```bash
# Text report
cat output/research_paper_report.txt

# Model comparison table
cat output/results/model_comparison.csv

# List chart images
ls output/figures/
```

Open the `figures/` folder in the file manager to see PNG charts.

---

## Where results are saved

| Folder | Experiment | Best for |
|--------|------------|----------|
| `output/` | Random 80/20 split | High accuracy comparison |
| `output2/` | File-based (day) split | Realistic / research paper |

| File | What it contains |
|------|------------------|
| `report.txt` | Summary report |
| `research_paper_report.txt` | Full report with Tables I–VI |
| `results/model_comparison.csv` | All models compared |
| `results/single_classifiers_comparison.csv` | NB, LR, GBT, RF only |
| `results/ensemble_comparison.csv` | Hard + soft vote |
| `results/per_class_metrics.csv` | Per-class precision/recall |
| `results/latex/*.tex` | Copy into LaTeX paper |
| `figures/*.png` | Charts for presentation |

---

## Command cheat sheet

```bash
# Activate environment (run this every time you open a new terminal)
cd ~/Desktop/spark_project
source env/bin/activate
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Quick test
./run.sh random small 0.1 --fast

# Full paper (both experiments)
./run.sh both full 0.3

# Random split only
./run.sh random full 0.3

# File-based split only
./run.sh file full 0.3

# Pull latest code from GitHub
git pull origin main
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `not a git repository` | Clone fresh: `git clone https://github.com/allech01/Intrusion-Detection.git spark_project` |
| `requirements.txt not found` | Run commands from inside `spark_project/` folder |
| `'JavaPackage' object is not callable` | Set JAVA_HOME (Step 1.3), use PySpark 3.5.4 |
| `No module named 'distutils'` | `pip install setuptools` |
| `Could not open requirements file` | Wrong folder — `cd ~/Desktop/spark_project` |
| Missing CSV error | Put files in `Data/` with exact names |
| Stuck at preprocessing | Wait 5–15 min; check http://localhost:4040 |
| Out of memory | Use smaller run: `./run.sh both small 0.1 --fast` |

---

## Project structure

```
spark_project/
├── Data/                    # Your CSV files (not on GitHub)
├── env/                     # Python virtual environment (created by you)
├── output/                  # Results: random split experiment
├── output2/                 # Results: file-based split experiment
├── run.sh                   # ONE command to run everything
├── main.py                  # Main pipeline
├── requirements.txt         # Python package versions
├── config.py                # Settings
├── src/                     # Source code modules
└── scripts/
    ├── run_experiment.py    # Experiment runner (called by run.sh)
    └── generate_figures.py  # Creates PNG charts
```

---

## Configuration reference (advanced)

| Variable | Default | Description |
|----------|---------|-------------|
| `NIDS_OUTPUT_DIR` | `output` | Result folder name |
| `NIDS_DATA_SIZE` | `small` | `small`, `medium`, or `full` |
| `NIDS_SAMPLE_FRACTION` | `0.2` | Fraction of rows (0.1 = 10%) |
| `NIDS_SPLIT` | `file` | `file` or `random` |
| `NIDS_MODE` | `multiclass` | `binary` or `multiclass` |
| `NIDS_FAST` | `1` | `1` = fast (2 models), `0` = all 4 models |
| `NIDS_CLASS_WEIGHTS` | `1` | Balance rare attack class |
| `NIDS_MODELS` | (all) | e.g. `gradient_boosted_trees` for one model |

---

## Expected runtime (VM with 4 GB RAM)

| Command | Approx. time |
|---------|--------------|
| `./run.sh random small 0.1 --fast` | 15–20 min |
| `./run.sh both full 0.3` | 2–4 hours |
| `./run.sh both full 1.0` | 4–8 hours |

---

## License & citation

Dataset: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — cite the original paper when using results in academic work.

Repository: https://github.com/allech01/Intrusion-Detection
