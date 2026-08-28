# CatBoost — Predicting Smartphone Addiction (Playground Series S6E8)

A CatBoost classifier for the Kaggle Playground Series S6E8 competition, which predicts whether a person is addicted to their smartphone based on lifestyle and demographic features. This version is configured to run headlessly as a batch job on **Snellius** (SURF's national supercomputer) using a GPU.

**Public leaderboard score:** 0.95909 (AUC)

## Overview

- **Target column:** `addicted_label` (binary classification)
- **Evaluation metric:** AUC-ROC
- **Model:** CatBoost, using its native categorical feature handling (no manual one-hot/label encoding needed), trained on GPU
- **Validation strategy:** Stratified **10-fold** cross-validation

## Features

| Type | Columns |
|---|---|
| Categorical | `gender`, `stress_level`, `academic_work_impact` |
| Numeric | All remaining columns (excluding the id column and target) |

CatBoost is given the categorical columns directly (cast to string, with missing values explicitly filled as `"missing"`) via their column indices, rather than requiring pre-encoding.

## Notebook Structure

1. **Imports** — CatBoost, scikit-learn, pandas, numpy, matplotlib (headless `Agg` backend for batch execution)
2. **Load data** — reads `train.csv` and `test.csv` from a configurable `DATA_DIR`
3. **Define features and target** — splits columns into categorical vs. numeric feature sets
4. **Quick EDA** — class balance plot (saved to `class_balance.png`) and summary statistics
5. **Prep for CatBoost** — fills missing categorical values with `"missing"`, casts to string, resolves column indices
6. **Stratified 10-fold CV training** — trains one CatBoost model per fold on GPU with early stopping, tracking out-of-fold (OOF) predictions and averaging test predictions across folds
7. **Overall CV score** — reports per-fold AUC and the overall OOF AUC
8. **Feature importance** — plots CatBoost's feature importances from the final fold's model (saved to `feature_importance.png`)
9. **Build submission** — writes fold-averaged test predictions to `submission.csv`, built directly from the test set's ID column (no dependency on a `sample_submission.csv` template)

## Model Configuration

```python
params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    early_stopping_rounds=200,
    verbose=200,
    task_type="GPU",
    devices="0",
)
```

## Requirements

- `catboost`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `jupyter` / `nbconvert` (for headless execution)

---

## Running on Snellius

### 1. Set up the project folder

On Snellius, create a project directory with a `data/` subfolder for your CSVs:

```bash
mkdir -p ~/smartphone-addiction-catboost/data
```

Upload `notebook_snellius.ipynb`, `run.job`, and `setup_env.sh` into `~/smartphone-addiction-catboost/`, and `train.csv` / `test.csv` into the `data/` subfolder (via `scp` from your local machine).

### 2. Create the environment (one-time setup)

```bash
module load 2025
module load Python/3.13.5-GCCcore-14.3.0

python -m venv ~/envs/catboost-env
source ~/envs/catboost-env/bin/activate

pip install catboost numpy pandas scikit-learn matplotlib jupyter nbconvert
```

Or simply run the included setup script:
```bash
bash setup_env.sh
```

Sanity check:
```bash
python -c "import catboost, pandas, sklearn, matplotlib; print('catboost', catboost.__version__)"
```

### 3. Submit the batch job

The included `run.job` SLURM script requests a GPU node:

```bash
#!/bin/bash
#SBATCH --job-name=catboost-run
#SBATCH --account=<your-account-name>
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

module load 2025
module load Python/3.13.5-GCCcore-14.3.0
source ~/envs/catboost-env/bin/activate

export DATA_DIR="$(pwd)/data"

jupyter nbconvert --to notebook --execute \
  --output executed_notebook.ipynb \
  notebook_snellius.ipynb
```

> **Note:** replace `--account` with your own SURF budget/account name (check with `accinfo`), and `--partition` with whichever partition your account's budget actually covers (check with `sinfo`).

Submit and monitor:
```bash
cd ~/smartphone-addiction-catboost
sbatch run.job
squeue -u $USER
```

### 4. Check results

```bash
# Live log while running
tail -f slurm-<jobid>.out

# Once finished, view the full log
ls -t slurm-*.out | head -1 | xargs cat
```

Outputs produced in the project folder:
- `executed_notebook.ipynb` — the executed notebook with all outputs (fold AUCs, plots) baked in
- `submission.csv` — the file to submit to Kaggle
- `class_balance.png`, `feature_importance.png` — saved plots

### 5. Retrieve the submission file

From your **local machine**:
```bash
scp yourusername@snellius.surf.nl:~/smartphone-addiction-catboost/submission.csv ./
```

Upload `submission.csv` to the Kaggle competition's "Submit Predictions" page.

## Repository Contents

| File | Purpose |
|---|---|
| `notebook_snellius.ipynb` | Main notebook — Snellius/headless-ready, 10-fold CV, GPU training |
| `run.job` | SLURM batch job script |
| `setup_env.sh` | One-time environment setup script |
| `data/` | `train.csv`, `test.csv` |
