"""Models for anomaly detection and correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


@dataclass
class AnomalyScores:
    autoencoder: np.ndarray
    isolation_forest: np.ndarray
    ensemble: np.ndarray


class AutoencoderModel:
    """Dense autoencoder for contextual anomalies."""

    def __init__(self, input_dim: int, latent_dim: int = 16, seed: int = 42) -> None:
        tf.keras.utils.set_random_seed(seed)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.input_dim,))
        encoded = tf.keras.layers.Dense(64, activation="relu")(inputs)
        encoded = tf.keras.layers.Dense(32, activation="relu")(encoded)
        encoded = tf.keras.layers.Dense(self.latent_dim, activation="relu")(encoded)
        decoded = tf.keras.layers.Dense(32, activation="relu")(encoded)
        decoded = tf.keras.layers.Dense(64, activation="relu")(decoded)
        outputs = tf.keras.layers.Dense(self.input_dim, activation="linear")(decoded)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, x_train: np.ndarray, epochs: int = 50, batch_size: int = 256) -> None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True
            )
        ]
        self.model.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

    def reconstruct(self, x_data: np.ndarray) -> np.ndarray:
        return self.model.predict(x_data, verbose=0)

    def reconstruction_error(self, x_data: np.ndarray) -> np.ndarray:
        reconstructed = self.reconstruct(x_data)
        return np.mean(np.square(x_data - reconstructed), axis=1)


class IsolationForestModel:
    """Isolation Forest for distributional outliers."""

    def __init__(self, seed: int = 42) -> None:
        self.model = IsolationForest(
            n_estimators=300, contamination="auto", random_state=seed
        )

    def fit(self, x_data: np.ndarray) -> None:
        self.model.fit(x_data)

    def score(self, x_data: np.ndarray) -> np.ndarray:
        scores = -self.model.score_samples(x_data)
        return scores


class AnomalyEnsembler:
    """Combine autoencoder and isolation forest scores."""

    def __init__(self, weight_ae: float = 0.7, weight_if: float = 0.3) -> None:
        self.weight_ae = weight_ae
        self.weight_if = weight_if
        self.scaler_ae = MinMaxScaler()
        self.scaler_if = MinMaxScaler()

    def fit_transform(self, ae_scores: np.ndarray, if_scores: np.ndarray) -> np.ndarray:
        ae_scaled = self.scaler_ae.fit_transform(ae_scores.reshape(-1, 1)).ravel()
        if_scaled = self.scaler_if.fit_transform(if_scores.reshape(-1, 1)).ravel()
        return self.weight_ae * ae_scaled + self.weight_if * if_scaled

    def transform(self, ae_scores: np.ndarray, if_scores: np.ndarray) -> np.ndarray:
        ae_scaled = self.scaler_ae.transform(ae_scores.reshape(-1, 1)).ravel()
        if_scaled = self.scaler_if.transform(if_scores.reshape(-1, 1)).ravel()
        return self.weight_ae * ae_scaled + self.weight_if * if_scaled


def compute_anomaly_scores(
    autoencoder: AutoencoderModel, isolation_forest: IsolationForestModel, x_data: np.ndarray
) -> AnomalyScores:
    ae_scores = autoencoder.reconstruction_error(x_data)
    if_scores = isolation_forest.score(x_data)
    ensembler = AnomalyEnsembler()
    ensemble_scores = ensembler.fit_transform(ae_scores, if_scores)
    return AnomalyScores(autoencoder=ae_scores, isolation_forest=if_scores, ensemble=ensemble_scores)


def apply_correction(
    autoencoder: AutoencoderModel,
    x_data: np.ndarray,
    anomaly_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    reconstructed = autoencoder.reconstruct(x_data)
    corrected = x_data.copy()
    corrected[anomaly_mask] = reconstructed[anomaly_mask]
    residual_error = np.mean(np.square(corrected - reconstructed), axis=1)
    confidence = 1.0 / (1.0 + residual_error)
    return corrected, confidence
