"""
Semi-Supervised Risk Detection & Data Cleaning
Complete pipeline for Home Credit Default Risk dataset.

Strategy:
1. Train on repayers (TARGET=0) to learn normal financial patterns
2. Detect defaulters (TARGET=1) as risky deviations
3. Clean data for downstream fuzzy credit scoring
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from preprocessing import FinancialPreprocessor
from models import AnomalyDetectionSystem
from evaluation import Evaluator
from visualization import (
    plot_anomaly_distribution,
    plot_detection_metrics,
    plot_downstream_impact,
    create_evaluation_summary
)

print("="*70)
print("SEMI-SUPERVISED RISK DETECTION & DATA CLEANING")
print("="*70)

# ═══════════════════════════════════════════════════════════
# 1. Load & Preprocess
# ═══════════════════════════════════════════════════════════
print("\n[1/6] Loading and preprocessing data...")
preprocessor = FinancialPreprocessor()
X, y, features = preprocessor.fit_transform('data/application_train.csv')

print(f"✓ Loaded {len(X):,} samples with {len(features)} features")
print(f"  - Repayers (y=0): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"  - Defaulters (y=1): {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════
# 2. Train Risk Detection System
# ═══════════════════════════════════════════════════════════
print("\n[2/6] Training risk detection system...")
print("Strategy: Learn patterns from REPAYERS only")

detector = AnomalyDetectionSystem(input_dim=len(features))

# CRITICAL: Train only on non-defaulters (repayers)
X_repayers = X[y == 0]
print(f"Training set: {len(X_repayers):,} repayers")

# Train with proper settings
detector.fit(
    X_normal=X_repayers,
    X_all=X,  # For IF and calibration
    epochs=50,
    batch_size=512,
    verbose=1
)

# ═══════════════════════════════════════════════════════════
# 3. Detect Risky Profiles & Clean Data
# ═══════════════════════════════════════════════════════════
print("\n[3/6] Detecting risky profiles and cleaning data...")

results = detector.detect_and_correct(X)

print(f"✓ Detected {results['n_anomalies']:,} risky profiles")
print(f"  Detection rate: {results['n_anomalies']/len(X)*100:.1f}%")
if len(results['confidence']) > 0:
    print(f"  Avg correction confidence: {results['confidence'].mean():.3f}")

# ═══════════════════════════════════════════════════════════
# 4. Evaluation - Risk Detection Quality
# ═══════════════════════════════════════════════════════════
print("\n[4/6] Evaluating risk detection quality...")
print("Ground truth: Defaulters (TARGET=1) as risky profiles")

evaluator = Evaluator()

# Use TARGET as ground truth (defaulters = risky)
y_true_risk = y  # 0=normal, 1=risky(defaulter)

intrinsic = evaluator.evaluate_detection(
    y_true_anomaly=y_true_risk,
    y_pred_flags=results['anomaly_mask'].astype(int),
    y_pred_scores=results['scores']
)

print("\nRisk Detection Metrics:")
print(f"  Precision: {intrinsic['precision']:.3f}")
print(f"  Recall:    {intrinsic['recall']:.3f}")
print(f"  F1-Score:  {intrinsic['f1_score']:.3f}")
print(f"  ROC-AUC:   {intrinsic['roc_auc']:.3f}")

# ═══════════════════════════════════════════════════════════
# 5. Downstream Impact (Credit Scoring)
# ═══════════════════════════════════════════════════════════
print("\n[5/6] Evaluating downstream impact on credit scoring...")

downstream = evaluator.evaluate_downstream(
    X, results['corrected'], y
)

print("\nDownstream Impact:")
for model_name, metrics in downstream.items():
    print(f"\n  {model_name.replace('_', ' ').title()}:")
    print(f"    Original AUC:  {metrics['original_auc']:.4f}")
    print(f"    Corrected AUC: {metrics['corrected_auc']:.4f}")
    print(f"    Improvement:   {metrics['improvement']:+.4f} ({metrics['improvement_pct']:+.2f}%)")

# ═══════════════════════════════════════════════════════════
# 6. Visualizations & Save Results
# ═══════════════════════════════════════════════════════════
print("\n[6/6] Generating visualizations...")

plot_anomaly_distribution(
    results['scores'],
    results['anomaly_mask'].astype(int),
    detector.ensemble_threshold,
    save_path='results/figures/risk_score_distribution.png'
)

plot_detection_metrics(
    intrinsic,
    save_path='results/figures/risk_detection_metrics.png'
)

plot_downstream_impact(
    downstream,
    save_path='results/figures/downstream_improvement.png'
)

create_evaluation_summary(
    intrinsic,
    downstream,
    save_path='results/figures/complete_evaluation.png'
)

# Save everything
print("\nSaving models and results...")
detector.save('models')
evaluator.save_results('results/metrics/evaluation_results.json')

# Save clean data for Fuzzy project
print("\nPreparing output for Fuzzy Credit Scoring...")
fuzzy_output = {
    'original_features': X,
    'corrected_features': results['corrected'],
    'feature_names': features,
    'risk_scores': results['scores'],
    'risk_flags': results['anomaly_mask'],
    'correction_confidence': results['confidence'],
    'target': y,
    'n_samples': len(X),
    'n_features': len(features)
}

import pickle
with open('results/fuzzy_input_data.pkl', 'wb') as f:
    pickle.dump(fuzzy_output, f)

print("✓ Clean data saved for Fuzzy project: results/fuzzy_input_data.pkl")

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("✓ PIPELINE COMPLETE")
print("="*70)

print("\nQUICK SUMMARY:")
print(f"  Risk Detection AUC:  {intrinsic['roc_auc']:.3f}")
print(f"  Detection F1-Score:  {intrinsic['f1_score']:.3f}")

for model, metrics in downstream.items():
    emoji = "🔥" if metrics['improvement'] > 0.01 else "✨" if metrics['improvement'] > 0 else "⚠️"
    print(f"  {emoji} {model.title()}: {metrics['improvement']:+.4f} ({metrics['improvement_pct']:+.2f}%)")

print(f"\n  Clean data ready: {len(results['corrected']):,} samples")
print(f"  Risky profiles flagged: {results['n_anomalies']:,}")
print("\n✓ Ready for Fuzzy Credit Scoring!")
print("="*70)
