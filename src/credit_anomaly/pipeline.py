"""End-to-end pipeline for anomaly detection and correction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from credit_anomaly.evaluation import Evaluator
from credit_anomaly.models import (
    AnomalyEnsembler,
    AutoencoderModel,
    IsolationForestModel,
    apply_correction,
)
from credit_anomaly.preprocessing import Preprocessor
from credit_anomaly.visualization import (
    plot_downstream_auc,
    plot_feature_space,
    plot_reconstruction_error,
)


@dataclass
class PipelineArtifacts:
    anomaly_scores: np.ndarray
    corrected_features: np.ndarray
    confidence_scores: np.ndarray
    metrics: Dict[str, float]
    plots: Dict[str, Path]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    percentile_threshold: float = 95.0,
) -> PipelineArtifacts:
    set_seed()
    frame = pd.read_csv(data_path)
    if "TARGET" not in frame.columns:
        raise ValueError("TARGET column is required for downstream evaluation.")

    preprocessor = Preprocessor()
    processed = preprocessor.fit_transform(frame)

    autoencoder = AutoencoderModel(input_dim=processed.scaled.shape[1])
    autoencoder.fit(processed.scaled)

    isolation_forest = IsolationForestModel()
    isolation_forest.fit(processed.scaled)

    ae_scores = autoencoder.reconstruction_error(processed.scaled)
    if_scores = isolation_forest.score(processed.scaled)
    ensembler = AnomalyEnsembler()
    ensemble_scores = ensembler.fit_transform(ae_scores, if_scores)
    threshold = np.percentile(ensemble_scores, percentile_threshold)
    anomaly_mask = ensemble_scores >= threshold

    corrected_scaled, confidence = apply_correction(
        autoencoder, processed.scaled, anomaly_mask
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(processed.scaled, columns=processed.feature_names).to_csv(
        output_dir / "original_features.csv", index=False
    )
    pd.DataFrame(corrected_scaled, columns=processed.feature_names).to_csv(
        output_dir / "corrected_features.csv", index=False
    )
    pd.DataFrame({"anomaly_score": ensemble_scores}).to_csv(
        output_dir / "anomaly_scores.csv", index=False
    )
    pd.DataFrame({"confidence": confidence}).to_csv(
        output_dir / "confidence_scores.csv", index=False
    )

    evaluator = Evaluator()
    detection_metrics = evaluator.evaluate_known_anomalies(processed.raw_frame, anomaly_mask)
    downstream_metrics = evaluator.compare_downstream(
        processed.scaled, corrected_scaled, frame["TARGET"].to_numpy()
    )
    summary = evaluator.summarize(detection_metrics, downstream_metrics)

    plots = {
        "reconstruction_error": plot_reconstruction_error(ae_scores, output_dir),
        "feature_space": plot_feature_space(processed.scaled, corrected_scaled, output_dir),
        "downstream_auc": plot_downstream_auc(
            downstream_metrics.auc_raw, downstream_metrics.auc_corrected, output_dir
        ),
    }

    return PipelineArtifacts(
        anomaly_scores=ensemble_scores,
        corrected_features=corrected_scaled,
        confidence_scores=confidence,
        metrics=summary,
        plots=plots,
    )
