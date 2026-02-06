"""
Financial Anomaly Detection & Correction System
"""

from .preprocessing import FinancialPreprocessor, get_feature_groups
from .models import AutoencoderModel, IsolationForestModel, AnomalyDetectionSystem
from .evaluation import Evaluator, quick_summary
from .visualization import (
    plot_anomaly_distribution,
    plot_detection_metrics,
    plot_downstream_impact,
    plot_feature_space_2d,
    plot_confusion_matrix,
    plot_correction_confidence,
    create_evaluation_summary,
    plot_training_history
)

__version__ = '1.0.0'
__author__ = 'Your Name'

__all__ = [
    'FinancialPreprocessor',
    'AutoencoderModel',
    'IsolationForestModel',
    'AnomalyDetectionSystem',
    'Evaluator',
    'quick_summary',
    'get_feature_groups',
    'plot_anomaly_distribution',
    'plot_detection_metrics',
    'plot_downstream_impact',
    'plot_feature_space_2d',
    'plot_confusion_matrix',
    'plot_correction_confidence',
    'create_evaluation_summary',
    'plot_training_history'
]
