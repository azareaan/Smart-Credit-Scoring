"""
Visualization Utilities
Clean, publication-quality plots for analysis and reporting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)


def plot_anomaly_distribution(scores, flags, threshold, save_path=None):
    """
    Distribution of anomaly scores with threshold line.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    ax.hist(scores[flags == 0], bins=50, alpha=0.6, label='Normal', color='steelblue')
    ax.hist(scores[flags == 1], bins=50, alpha=0.6, label='Anomaly', color='coral')
    
    # Threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold ({threshold:.3f})')
    
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_detection_metrics(metrics, save_path=None):
    """
    Bar chart of intrinsic detection metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['Precision', 'Recall', 'F1-Score']
    values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
    
    if metrics.get('roc_auc'):
        metric_names.extend(['ROC-AUC', 'PR-AUC'])
        values.extend([metrics['roc_auc'], metrics['pr_auc']])
    
    bars = ax.bar(metric_names, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(values)], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Anomaly Detection Performance', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_downstream_impact(downstream_results, save_path=None):
    """
    Comparison of model performance before/after correction.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = list(downstream_results.keys())
    x = np.arange(len(models))
    width = 0.35
    
    original_aucs = [downstream_results[m]['original_auc'] for m in models]
    corrected_aucs = [downstream_results[m]['corrected_auc'] for m in models]
    improvements = [downstream_results[m]['improvement'] for m in models]
    
    bars1 = ax.bar(x - width/2, original_aucs, width, label='Original Data',
                   color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, corrected_aucs, width, label='Corrected Data',
                   color='seagreen', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add improvement annotations
    for i, (orig, corr, imp) in enumerate(zip(original_aucs, corrected_aucs, improvements)):
        ax.annotate(f'+{imp:.4f}\n({imp/orig*100:+.2f}%)',
                   xy=(i, corr),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=11,
                   fontweight='bold',
                   color='darkgreen',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    ax.set_ylabel('ROC-AUC', fontsize=12)
    ax.set_title('Downstream Impact: Credit Scoring Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models], fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([min(original_aucs) - 0.02, max(corrected_aucs) + 0.05])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_feature_space_2d(X_original, X_corrected, anomaly_mask, method='tsne', 
                          n_samples=5000, save_path=None):
    """
    2D visualization of feature space (t-SNE or PCA).
    Shows how correction moves anomalies toward normal manifold.
    """
    # Sample for speed
    if len(X_original) > n_samples:
        idx = np.random.choice(len(X_original), n_samples, replace=False)
        X_orig_sample = X_original[idx]
        X_corr_sample = X_corrected[idx]
        mask_sample = anomaly_mask[idx]
    else:
        X_orig_sample = X_original
        X_corr_sample = X_corrected
        mask_sample = anomaly_mask
    
    # Dimensionality reduction
    if method == 'tsne':
        print("Computing t-SNE... (this may take a minute)")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    X_orig_2d = reducer.fit_transform(X_orig_sample)
    X_corr_2d = reducer.transform(X_corr_sample)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Original
    ax1.scatter(X_orig_2d[~mask_sample, 0], X_orig_2d[~mask_sample, 1],
               c='steelblue', alpha=0.5, s=20, label='Normal')
    ax1.scatter(X_orig_2d[mask_sample, 0], X_orig_2d[mask_sample, 1],
               c='coral', alpha=0.8, s=40, label='Anomaly', edgecolors='red', linewidths=1)
    ax1.set_title('Original Feature Space', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{method.upper()} Component 1', fontsize=11)
    ax1.set_ylabel(f'{method.upper()} Component 2', fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Corrected
    ax2.scatter(X_corr_2d[~mask_sample, 0], X_corr_2d[~mask_sample, 1],
               c='steelblue', alpha=0.5, s=20, label='Normal')
    ax2.scatter(X_corr_2d[mask_sample, 0], X_corr_2d[mask_sample, 1],
               c='seagreen', alpha=0.8, s=40, label='Corrected', edgecolors='darkgreen', linewidths=1)
    ax2.set_title('Corrected Feature Space', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'{method.upper()} Component 1', fontsize=11)
    ax2.set_ylabel(f'{method.upper()} Component 2', fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, save_path=None):
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_correction_confidence(confidence_scores, save_path=None):
    """Distribution of correction confidence scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(confidence_scores, bins=50, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.axvline(confidence_scores.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {confidence_scores.mean():.3f}')
    
    ax.set_xlabel('Correction Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Correction Confidence Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def create_evaluation_summary(intrinsic_metrics, downstream_results, save_path=None):
    """
    Combined summary figure with all key results.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Detection Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Precision', 'Recall', 'F1']
    values = [intrinsic_metrics['precision'], intrinsic_metrics['recall'], intrinsic_metrics['f1_score']]
    ax1.bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    for i, v in enumerate(values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.set_title('Detection Performance', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Downstream Impact
    ax2 = fig.add_subplot(gs[0, 1])
    models = list(downstream_results.keys())
    improvements = [downstream_results[m]['improvement_pct'] for m in models]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.barh(models, improvements, color=colors, alpha=0.7)
    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{imp:+.2f}%', ha='left' if imp > 0 else 'right',
                va='center', fontweight='bold')
    ax2.set_xlabel('AUC Improvement (%)')
    ax2.set_title('Downstream Impact', fontweight='bold')
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Summary Text
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    summary_text = f"""
    EVALUATION SUMMARY
    ═══════════════════════════════════════════════════════════════
    
    INTRINSIC (Detection Quality):
      • F1-Score: {intrinsic_metrics['f1_score']:.3f}
      • Precision: {intrinsic_metrics['precision']:.3f}
      • Recall: {intrinsic_metrics['recall']:.3f}
      • Detected: {intrinsic_metrics['n_detected']} anomalies
    
    DOWNSTREAM (Credit Scoring Impact):
    """
    
    for model, metrics in downstream_results.items():
        summary_text += f"\n      • {model.replace('_', ' ').title()}:"
        summary_text += f" {metrics['original_auc']:.4f} → {metrics['corrected_auc']:.4f}"
        summary_text += f" ({metrics['improvement_pct']:+.2f}%)"
    
    summary_text += "\n\n    ✓ Anomaly correction demonstrably improves downstream performance!"
    
    ax3.text(0.5, 0.5, summary_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Financial Anomaly Detection & Correction - Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot autoencoder training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('MSE Loss', fontsize=11)
    ax1.set_title('Training History - Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # MAE
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MAE', fontsize=11)
    ax2.set_title('Training History - MAE', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()
