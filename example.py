"""
Quick Start Example
Complete pipeline demonstration in ~50 lines of code.
"""

import sys
sys.path.append('src')

from preprocessing import FinancialPreprocessor
from models import AnomalyDetectionSystem
from evaluation import Evaluator, quick_summary
from visualization import (
    plot_anomaly_distribution,
    plot_detection_metrics,
    plot_downstream_impact,
    create_evaluation_summary
)

# ═══════════════════════════════════════
# 1. Load & Preprocess
# ═══════════════════════════════════════
print("Loading data...")
preprocessor = FinancialPreprocessor()
X, y, features = preprocessor.fit_transform('data/application_train.csv')

print(f"\n✓ Loaded {len(X):,} samples with {len(features)} features")

# ═══════════════════════════════════════
# 2. Train Anomaly Detection System
# ═══════════════════════════════════════
print("\nTraining anomaly detection system...")
detector = AnomalyDetectionSystem(input_dim=len(features))

# Train on normal samples only (unsupervised)
X_normal = X[y == 0]
detector.fit(X_normal, X_all=X, epochs=30, batch_size=512, verbose=0)

# ═══════════════════════════════════════
# 3. Detect & Correct Anomalies
# ═══════════════════════════════════════
print("\nDetecting and correcting anomalies...")
results = detector.detect_and_correct(X)

print(f"✓ Detected {results['n_anomalies']:,} anomalies")
print(f"✓ Average correction confidence: {results['confidence'].mean():.3f}")

# ═══════════════════════════════════════
# 4. Evaluation
# ═══════════════════════════════════════
print("\nEvaluating...")
evaluator = Evaluator()

# Intrinsic (detection quality)
import pandas as pd
df = pd.read_csv('data/application_train.csv')
y_true_anomaly = (df['DAYS_EMPLOYED'] == 365243).astype(int).values

intrinsic = evaluator.evaluate_detection(
    y_true_anomaly,
    results['anomaly_mask'].astype(int),
    results['scores']
)

# Downstream (credit scoring impact)
downstream = evaluator.evaluate_downstream(
    X, results['corrected'], y
)

# ═══════════════════════════════════════
# 5. Visualize
# ═══════════════════════════════════════
print("\nGenerating visualizations...")

plot_anomaly_distribution(
    results['scores'],
    results['anomaly_mask'].astype(int),
    detector.ensemble_threshold,
    save_path='results/figures/anomaly_distribution.png'
)

plot_detection_metrics(
    intrinsic,
    save_path='results/figures/detection_metrics.png'
)

plot_downstream_impact(
    downstream,
    save_path='results/figures/downstream_impact.png'
)

create_evaluation_summary(
    intrinsic,
    downstream,
    save_path='results/figures/evaluation_summary.png'
)

# ═══════════════════════════════════════
# 6. Save Everything
# ═══════════════════════════════════════
print("\nSaving models and results...")
detector.save('models')
evaluator.save_results('results/metrics/evaluation_results.json')

print("\n" + "="*60)
print("✓ COMPLETE!")
print("="*60)
quick_summary(intrinsic, downstream)
