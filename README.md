# Neural Network–Based Anomaly Detection & Correction in Financial Data

This project implements a contextual anomaly detection and correction pipeline for the Home Credit Default Risk dataset (`application_train.csv`). It uses a neural autoencoder as the primary detector and an Isolation Forest as a complementary detector, then corrects detected anomalies by projecting them onto the learned manifold. The corrected data is evaluated by downstream credit risk modeling.

## Roadmap Coverage
- **Contextual anomalies** via autoencoder reconstruction error.
- **Missingness signal** preserved using binary flags + median imputation.
- **Feature engineering** with interpretable ratios.
- **Hybrid scoring** with calibrated ensemble of autoencoder and Isolation Forest scores.
- **Correction** using reconstructed outputs with a confidence score.
- **Evaluation** using known anomalies and downstream ROC-AUC.
- **Visualization** for reconstruction error distribution, PCA feature space, and AUC impact.

## Project Structure
```
src/credit_anomaly/
  config.py
  preprocessing.py
  models.py
  evaluation.py
  visualization.py
  pipeline.py
scripts/
  run_pipeline.py
outputs/
```

## Setup
```bash
pip install -r requirements.txt
```

For a stable CLI across environments (recommended), install the project in editable mode:
```bash
pip install -e .
```

## Run
```bash
python scripts/run_pipeline.py --data /path/to/application_train.csv --output outputs --percentile 95
```

Or after editable install:
```bash
credit-anomaly-pipeline --data /path/to/application_train.csv --output outputs --percentile 95
```

## Outputs
- `outputs/reconstruction_error_hist.png`
- `outputs/feature_space_pca.png`
- `outputs/downstream_auc.png`

The CLI prints detection and downstream metrics along with plot paths.

## Notes
- `TARGET` is **not** used for anomaly detection training, only for evaluation.
- Known anomalies are validated with the `DAYS_EMPLOYED == 365243` rule.

## Troubleshooting
- If you still see `ModuleNotFoundError: No module named credit_anomaly`, make sure you are running the latest committed version of `scripts/run_pipeline.py` (it bootstraps `src` automatically).
- If working in a fresh environment, run `pip install -e .` once and use `credit-anomaly-pipeline ...` to avoid path issues.
