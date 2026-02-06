"""
Evaluation Module
- Intrinsic: Detection quality metrics
- Extrinsic: Downstream impact on credit scoring
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import json


class Evaluator:
    """
    Two-level evaluation:
    1. Intrinsic: How well does it detect anomalies?
    2. Extrinsic: Does correction improve downstream tasks?
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_detection(self, y_true_anomaly, y_pred_flags, y_pred_scores):
        """
        Intrinsic evaluation of anomaly detection.
        
        Args:
            y_true_anomaly: Ground truth anomalies (e.g., DAYS_EMPLOYED_ANOM)
            y_pred_flags: Predicted binary flags
            y_pred_scores: Predicted anomaly scores
        
        Returns:
            Dict of metrics
        """
        # Classification metrics
        precision = precision_score(y_true_anomaly, y_pred_flags, zero_division=0)
        recall = recall_score(y_true_anomaly, y_pred_flags, zero_division=0)
        f1 = f1_score(y_true_anomaly, y_pred_flags, zero_division=0)
        
        # Ranking metrics
        try:
            roc_auc = roc_auc_score(y_true_anomaly, y_pred_scores)
            pr_auc = average_precision_score(y_true_anomaly, y_pred_scores)
        except:
            roc_auc, pr_auc = None, None
        
        # Confusion matrix
        cm = confusion_matrix(y_true_anomaly, y_pred_flags)
        
        results = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'pr_auc': float(pr_auc) if pr_auc else None,
            'confusion_matrix': cm.tolist(),
            'n_true_anomalies': int(y_true_anomaly.sum()),
            'n_detected': int(y_pred_flags.sum())
        }
        
        self.results['intrinsic'] = results
        return results
    
    def evaluate_downstream(self, X_original, X_corrected, y_target, 
                          test_size=0.2, random_state=42):
        """
        Extrinsic evaluation: Impact on credit scoring.
        
        Trains models on both original and corrected data,
        compares performance. This proves the value of correction!
        
        Args:
            X_original: Original feature matrix
            X_corrected: Corrected feature matrix
            y_target: Credit default target (TARGET column)
        
        Returns:
            Dict with improvements for each model
        """
        # Split data
        X_orig_train, X_orig_test, y_train, y_test = train_test_split(
            X_original, y_target, test_size=test_size, random_state=random_state,
            stratify=y_target
        )
        
        X_corr_train, X_corr_test, _, _ = train_test_split(
            X_corrected, y_target, test_size=test_size, random_state=random_state,
            stratify=y_target
        )
        
        results = {}
        
        # ══════════════════════════════════════════
        # Logistic Regression (Simple Baseline)
        # ══════════════════════════════════════════
        print("Evaluating Logistic Regression...")
        
        lr_orig = LogisticRegression(max_iter=1000, random_state=random_state)
        lr_orig.fit(X_orig_train, y_train)
        auc_orig_lr = roc_auc_score(y_test, lr_orig.predict_proba(X_orig_test)[:, 1])
        
        lr_corr = LogisticRegression(max_iter=1000, random_state=random_state)
        lr_corr.fit(X_corr_train, y_train)
        auc_corr_lr = roc_auc_score(y_test, lr_corr.predict_proba(X_corr_test)[:, 1])
        
        results['logistic_regression'] = {
            'original_auc': float(auc_orig_lr),
            'corrected_auc': float(auc_corr_lr),
            'improvement': float(auc_corr_lr - auc_orig_lr),
            'improvement_pct': float((auc_corr_lr - auc_orig_lr) / auc_orig_lr * 100)
        }
        
        # ══════════════════════════════════════════
        # LightGBM (Advanced Model)
        # ══════════════════════════════════════════
        print("Evaluating LightGBM...")
        
        lgbm_orig = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            verbose=-1
        )
        lgbm_orig.fit(X_orig_train, y_train)
        auc_orig_lgb = roc_auc_score(y_test, lgbm_orig.predict_proba(X_orig_test)[:, 1])
        
        lgbm_corr = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            verbose=-1
        )
        lgbm_corr.fit(X_corr_train, y_train)
        auc_corr_lgb = roc_auc_score(y_test, lgbm_corr.predict_proba(X_corr_test)[:, 1])
        
        results['lightgbm'] = {
            'original_auc': float(auc_orig_lgb),
            'corrected_auc': float(auc_corr_lgb),
            'improvement': float(auc_corr_lgb - auc_orig_lgb),
            'improvement_pct': float((auc_corr_lgb - auc_orig_lgb) / auc_orig_lgb * 100)
        }
        
        self.results['downstream'] = results
        return results
    
    def evaluate_all(self, X_original, X_corrected, y_target, 
                    y_true_anomaly, y_pred_flags, y_pred_scores):
        """
        Complete evaluation: intrinsic + downstream.
        
        Returns:
            Dict with all results
        """
        print("="*60)
        print("INTRINSIC EVALUATION (Detection Quality)")
        print("="*60)
        intrinsic = self.evaluate_detection(y_true_anomaly, y_pred_flags, y_pred_scores)
        self._print_intrinsic(intrinsic)
        
        print("\n" + "="*60)
        print("DOWNSTREAM EVALUATION (Credit Scoring Impact)")
        print("="*60)
        downstream = self.evaluate_downstream(X_original, X_corrected, y_target)
        self._print_downstream(downstream)
        
        return {
            'intrinsic': intrinsic,
            'downstream': downstream
        }
    
    def _print_intrinsic(self, results):
        """Pretty print intrinsic metrics."""
        print(f"\nDetection Metrics:")
        print(f"  Precision:  {results['precision']:.3f}")
        print(f"  Recall:     {results['recall']:.3f}")
        print(f"  F1-Score:   {results['f1_score']:.3f}")
        if results['roc_auc']:
            print(f"  ROC-AUC:    {results['roc_auc']:.3f}")
            print(f"  PR-AUC:     {results['pr_auc']:.3f}")
        print(f"\nDetected: {results['n_detected']} / {results['n_true_anomalies']} true anomalies")
    
    def _print_downstream(self, results):
        """Pretty print downstream metrics."""
        for model_name, metrics in results.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Original AUC:     {metrics['original_auc']:.4f}")
            print(f"  Corrected AUC:    {metrics['corrected_auc']:.4f}")
            print(f"  Improvement:      {metrics['improvement']:+.4f}")
            print(f"  Improvement %:    {metrics['improvement_pct']:+.2f}%")
    
    def save_results(self, filepath):
        """Save evaluation results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {filepath}")
    
    def generate_report(self):
        """Generate markdown report."""
        if not self.results:
            return "No results available. Run evaluation first."
        
        report = ["# Evaluation Report\n"]
        
        # Intrinsic
        if 'intrinsic' in self.results:
            report.append("## Intrinsic Performance (Anomaly Detection)\n")
            r = self.results['intrinsic']
            report.append(f"| Metric | Score |")
            report.append(f"|--------|-------|")
            report.append(f"| Precision | {r['precision']:.3f} |")
            report.append(f"| Recall | {r['recall']:.3f} |")
            report.append(f"| F1-Score | {r['f1_score']:.3f} |")
            if r['roc_auc']:
                report.append(f"| ROC-AUC | {r['roc_auc']:.3f} |")
            report.append(f"\nDetected: {r['n_detected']} anomalies\n")
        
        # Downstream
        if 'downstream' in self.results:
            report.append("## Downstream Impact (Credit Scoring)\n")
            report.append(f"| Model | Original AUC | Corrected AUC | Improvement |")
            report.append(f"|-------|--------------|---------------|-------------|")
            for model, metrics in self.results['downstream'].items():
                report.append(
                    f"| {model.replace('_', ' ').title()} | "
                    f"{metrics['original_auc']:.4f} | "
                    f"{metrics['corrected_auc']:.4f} | "
                    f"**{metrics['improvement_pct']:+.2f}%** |"
                )
        
        return "\n".join(report)


def quick_summary(intrinsic, downstream):
    """Quick summary for notebooks."""
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"\n✓ Detection F1: {intrinsic['f1_score']:.3f}")
    print(f"✓ Detection AUC: {intrinsic.get('roc_auc', 'N/A')}")
    
    for model, metrics in downstream.items():
        imp = metrics['improvement_pct']
        emoji = "🔥" if imp > 2 else "✨"
        print(f"{emoji} {model.title()}: {metrics['improvement']:+.4f} ({imp:+.2f}%)")
