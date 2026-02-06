"""
Anomaly Detection Models
- Autoencoder (primary, contextual)
- Isolation Forest (complementary, distributional)
- Ensemble (weighted combination)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib


class AutoencoderModel:
    """
    Autoencoder for contextual anomaly detection.
    Learns normal patterns and identifies deviations.
    """
    
    def __init__(self, input_dim=20, latent_dim=8):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
        
    def build(self):
        """Build symmetric autoencoder architecture."""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(64, activation='relu')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(16, activation='relu')(x)
        encoded = layers.Dense(self.latent_dim, activation='relu', name='bottleneck')(x)
        
        self.encoder = keras.Model(encoder_input, encoded, name='encoder')
        
        # Decoder
        decoder_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(16, activation='relu')(decoder_input)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        decoded = layers.Dense(self.input_dim, activation='linear', name='reconstruction')(x)
        
        self.decoder = keras.Model(decoder_input, decoded, name='decoder')
        
        # Full autoencoder
        autoencoder_input = layers.Input(shape=(self.input_dim,))
        encoded_repr = self.encoder(autoencoder_input)
        reconstructed = self.decoder(encoded_repr)
        
        self.model = keras.Model(autoencoder_input, reconstructed, name='autoencoder')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def fit(self, X_normal, epochs=50, batch_size=512, validation_split=0.2, verbose=1):
        """Train on normal samples only."""
        if self.model is None:
            self.build()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=verbose
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=verbose
            )
        ]
        
        history = self.model.fit(
            X_normal, X_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Set threshold on training data
        X_reconstructed = self.model.predict(X_normal, verbose=0)
        reconstruction_errors = np.mean((X_normal - X_reconstructed)**2, axis=1)
        self.threshold = np.percentile(reconstruction_errors, 95)
        
        return history
    
    def predict_scores(self, X):
        """Return anomaly scores (reconstruction errors)."""
        X_reconstructed = self.model.predict(X, batch_size=1024, verbose=0)
        scores = np.mean((X - X_reconstructed)**2, axis=1)
        return scores
    
    def predict_flags(self, X):
        """Return binary anomaly flags."""
        scores = self.predict_scores(X)
        return (scores > self.threshold).astype(int)
    
    def reconstruct(self, X):
        """Reconstruct (correct) anomalous samples."""
        return self.model.predict(X, batch_size=1024, verbose=0)
    
    def save(self, filepath):
        """Save model and threshold."""
        self.model.save(f'{filepath}_model.h5')
        joblib.dump({'threshold': self.threshold}, f'{filepath}_config.pkl')
    
    def load(self, filepath):
        """Load model and threshold."""
        self.model = keras.models.load_model(f'{filepath}_model.h5')
        config = joblib.load(f'{filepath}_config.pkl')
        self.threshold = config['threshold']


class IsolationForestModel:
    """
    Isolation Forest for distributional outlier detection.
    Complements autoencoder by detecting low-density samples.
    """
    
    def __init__(self, n_estimators=100, contamination=0.1, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples='auto',
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
    def fit(self, X):
        """Train on all data (not just normal)."""
        self.model.fit(X)
        return self
    
    def predict_scores(self, X):
        """Return anomaly scores (higher = more anomalous)."""
        return -self.model.score_samples(X)
    
    def predict_flags(self, X):
        """Return binary anomaly flags (-1 → 1, 1 → 0)."""
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)
    
    def save(self, filepath):
        """Save model."""
        joblib.dump(self.model, f'{filepath}.pkl')
    
    def load(self, filepath):
        """Load model."""
        self.model = joblib.load(f'{filepath}.pkl')


class AnomalyDetectionSystem:
    """
    Ensemble system: Autoencoder (70%) + Isolation Forest (30%).
    Provides contextual + distributional anomaly detection.
    """
    
    def __init__(self, input_dim=20, ae_weight=0.7, if_weight=0.3):
        self.autoencoder = AutoencoderModel(input_dim=input_dim)
        self.isolation_forest = IsolationForestModel()
        self.ae_weight = ae_weight
        self.if_weight = if_weight
        self.scaler = MinMaxScaler()
        self.ensemble_threshold = None
        
    def fit(self, X_normal, X_all=None, **ae_kwargs):
        """
        Fit both models.
        
        Args:
            X_normal: Normal samples for autoencoder
            X_all: All samples for isolation forest (default: same as X_normal)
        """
        print("Training Autoencoder...")
        self.autoencoder.fit(X_normal, **ae_kwargs)
        
        print("\nTraining Isolation Forest...")
        X_all = X_all if X_all is not None else X_normal
        self.isolation_forest.fit(X_all)
        
        print("\nCalibrating ensemble...")
        self._calibrate_ensemble(X_all)
        
        print("✓ Training complete")
        
    def _calibrate_ensemble(self, X):
        """
        Normalize and combine scores.
        For semi-supervised: threshold based on expected default rate (~8-10%)
        """
        ae_scores = self.autoencoder.predict_scores(X)
        if_scores = self.isolation_forest.predict_scores(X)
        
        # Fit scaler on training scores
        self.scaler.fit(np.column_stack([ae_scores, if_scores]))
        
        # Calculate ensemble scores
        ensemble_scores = self._get_ensemble_scores(ae_scores, if_scores)
        
        # Set threshold at 90-92th percentile (target ~8-10% as risky)
        self.ensemble_threshold = np.percentile(ensemble_scores, 90)
        
        print(f"Ensemble threshold: {self.ensemble_threshold:.4f} (90th percentile)")
        expected_anomalies = (ensemble_scores > self.ensemble_threshold).sum()
        print(f"Expected detections: {expected_anomalies:,} ({expected_anomalies/len(X)*100:.1f}%)")
    
    def _get_ensemble_scores(self, ae_scores, if_scores):
        """Weighted combination of normalized scores."""
        # Normalize both to [0, 1]
        normalized = self.scaler.transform(np.column_stack([ae_scores, if_scores]))
        ae_norm = normalized[:, 0]
        if_norm = normalized[:, 1]
        
        # Weighted average
        ensemble = self.ae_weight * ae_norm + self.if_weight * if_norm
        return ensemble
    
    def predict(self, X):
        """
        Complete anomaly detection.
        
        Returns dict with:
            - scores: Ensemble anomaly scores
            - flags: Binary anomaly flags
            - ae_scores: Individual AE scores
            - if_scores: Individual IF scores
        """
        ae_scores = self.autoencoder.predict_scores(X)
        if_scores = self.isolation_forest.predict_scores(X)
        ensemble_scores = self._get_ensemble_scores(ae_scores, if_scores)
        ensemble_flags = (ensemble_scores > self.ensemble_threshold).astype(int)
        
        return {
            'scores': ensemble_scores,
            'flags': ensemble_flags,
            'ae_scores': ae_scores,
            'if_scores': if_scores,
            'threshold': self.ensemble_threshold
        }
    
    def detect_and_correct(self, X):
        """
        Full pipeline: detect anomalies and correct them.
        
        Returns dict with:
            - corrected: Corrected feature matrix
            - original: Original features
            - anomaly_mask: Boolean mask
            - scores: Anomaly scores
            - confidence: Correction confidence
        """
        # Detect
        detection_results = self.predict(X)
        anomaly_mask = detection_results['flags'].astype(bool)
        
        # Correct anomalous samples using AE reconstruction
        X_corrected = X.copy()
        if anomaly_mask.sum() > 0:
            X_corrected[anomaly_mask] = self.autoencoder.reconstruct(X[anomaly_mask])
        
        # Calculate correction confidence
        pre_error = np.mean((X[anomaly_mask] - X_corrected[anomaly_mask])**2, axis=1) if anomaly_mask.sum() > 0 else np.array([])
        confidence = 1 / (1 + pre_error) if len(pre_error) > 0 else np.array([])
        
        return {
            'corrected': X_corrected,
            'original': X,
            'anomaly_mask': anomaly_mask,
            'scores': detection_results['scores'],
            'confidence': confidence,
            'n_anomalies': anomaly_mask.sum()
        }
    
    def save(self, dirpath):
        """Save all models."""
        self.autoencoder.save(f'{dirpath}/autoencoder')
        self.isolation_forest.save(f'{dirpath}/isolation_forest')
        joblib.dump({
            'scaler': self.scaler,
            'ensemble_threshold': self.ensemble_threshold,
            'weights': (self.ae_weight, self.if_weight)
        }, f'{dirpath}/ensemble_config.pkl')
    
    def load(self, dirpath):
        """Load all models."""
        self.autoencoder.load(f'{dirpath}/autoencoder')
        self.isolation_forest.load(f'{dirpath}/isolation_forest')
        config = joblib.load(f'{dirpath}/ensemble_config.pkl')
        self.scaler = config['scaler']
        self.ensemble_threshold = config['ensemble_threshold']
        self.ae_weight, self.if_weight = config['weights']
