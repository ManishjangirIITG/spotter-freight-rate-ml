# Spotter Freight Rate ML Pipeline

An end-to-end Machine Learning pipeline designed to predict spot market freight rates based on shipment parameters, spatial signals, and historical trends.

## Project Structure

```text
├── artifacts/              # Serialized models, evaluation metrics, logs, and prediction CSVs
├── data/                   # Input datasets (train-test.csv, validation.csv)
├── notebooks/              # Exploratory data analysis notebooks
├── scripts/                # CLI entry points for pipeline stages
├── src/freight_rate/       # Main Python package
│   ├── data/               # Ingestion, schema validation, and data utilities
│   ├── features/           # Feature transformations & engineering
│   ├── models/             # Baseline algorithms, LightGBM training, and evaluation
│   └── pipelines/          # High-level pipeline orchestrators
└── tests/                  # Pytest test suite
```

## Quickstart

1. Install Package & Dependencies:

```bash
pip install -e .
```

2. Run Pipeline Stages:

   * Train Model: ``python scripts/train.py``
   * Evaluate Metrics: ``python scripts/evaluate.py``
   * Generate Validation ``Predictions: python scripts/predict.py``
3. Or via Makefile:

```bash
make train
make evaluate
make predict
make test
```

## Model Architecture & Performance

* Primary Model: LightGBM Regressor
* Validation Strategy: Time-based split at 2025-09-01
* Validation MAE: ~$147.31
* Validation MAPE: ~7.25%

