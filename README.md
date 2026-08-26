# CatBoost — Predicting Smartphone Addiction (Playground Series S6E8)

A CatBoost classifier for the Kaggle Playground Series S6E8 competition, which predicts whether a person is addicted to their smartphone based on lifestyle and demographic features.

## Overview

- **Target column:** `addicted_label` (binary classification)
- **Evaluation metric:** AUC-ROC
- **Model:** CatBoost, using its native categorical feature handling (no manual one-hot/label encoding needed)
- **Validation strategy:** Stratified 5-fold cross-validation

## Features

| Type | Columns |
|---|---|
| Categorical | `gender`, `stress_level`, `academic_work_impact` |
| Numeric | All remaining columns (excluding the id column and target) |

CatBoost is given the categorical columns directly (cast to string) via their column indices, rather than requiring pre-encoding.

## Notebook Structure

1. **Imports** — installs/imports CatBoost, scikit-learn, pandas, numpy, matplotlib
2. **Load data** — reads `train.csv`, `test.csv`, and `sample_submission.csv` from the competition data directory
3. **Define features and target** — splits columns into categorical vs. numeric feature sets
4. **Quick EDA** — class balance plot and summary statistics of numeric features
5. **Prep for CatBoost** — casts categorical columns to string and resolves their column indices
6. **Stratified 5-fold CV training** — trains one CatBoost model per fold with early stopping, tracking out-of-fold (OOF) predictions and averaging test predictions across folds
7. **Overall CV score** — reports per-fold AUC and the overall OOF AUC
8. **Feature importance** — plots CatBoost's feature importances from the final fold's model
9. **Build submission** — writes fold-averaged test predictions to `submission.csv`

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
)
```

## Requirements

- `catboost`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

Install with:
```bash
pip install catboost numpy pandas scikit-learn matplotlib
```

## Usage

1. Update `TRAIN_PATH`, `TEST_PATH`, and `SAMPLE_SUB_PATH` at the top of the data-loading cell if your files are stored somewhere other than the default Kaggle competition path (`/kaggle/input/competitions/playground-series-s6e8/`).
2. Run all cells in order.
3. The final cell writes predictions to `submission.csv`, ready for upload to Kaggle.

## Output

- `submission.csv` — contains the id column and predicted probabilities for `addicted_label`, averaged across the 5 CV folds.
