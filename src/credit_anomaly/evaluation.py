"""Evaluation utilities for anomaly detection and downstream impact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_anomaly.config import KNOWN_ANOMALY_COLUMN, KNOWN_ANOMALY_VALUE


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float


@dataclass
class DownstreamMetrics:
    auc_raw: float
    auc_corrected: float


class Evaluator:
    """Compute detection metrics and downstream model impact."""

    def evaluate_known_anomalies(
        self, frame: pd.DataFrame, anomaly_mask: np.ndarray
    ) -> DetectionMetrics:
        known_anomalies = (frame[KNOWN_ANOMALY_COLUMN] == KNOWN_ANOMALY_VALUE).to_numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            known_anomalies, anomaly_mask, average="binary", zero_division=0
        )
        return DetectionMetrics(precision=precision, recall=recall, f1=f1)

    def compare_downstream(
        self,
        x_raw: np.ndarray,
        x_corrected: np.ndarray,
        target: np.ndarray,
        seed: int = 42,
    ) -> DownstreamMetrics:
        x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            x_raw, target, test_size=0.2, random_state=seed, stratify=target
        )
        x_train_corr, x_test_corr, _, _ = train_test_split(
            x_corrected, target, test_size=0.2, random_state=seed, stratify=target
        )
        model_raw = LGBMClassifier(random_state=seed)
        model_corr = LGBMClassifier(random_state=seed)
        model_raw.fit(x_train_raw, y_train)
        model_corr.fit(x_train_corr, y_train)
        prob_raw = model_raw.predict_proba(x_test_raw)[:, 1]
        prob_corr = model_corr.predict_proba(x_test_corr)[:, 1]
        auc_raw = roc_auc_score(y_test, prob_raw)
        auc_corr = roc_auc_score(y_test, prob_corr)
        return DownstreamMetrics(auc_raw=auc_raw, auc_corrected=auc_corr)

    def summarize(self, detection: DetectionMetrics, downstream: DownstreamMetrics) -> Dict[str, float]:
        return {
            "precision": detection.precision,
            "recall": detection.recall,
            "f1": detection.f1,
            "auc_raw": downstream.auc_raw,
            "auc_corrected": downstream.auc_corrected,
        }
