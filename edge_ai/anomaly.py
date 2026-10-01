"""
Edge-Guardian Anomaly Detection Module (P2).

TinyML-compatible autoencoder for unsupervised anomaly detection on
the 7-value physics feature vector. The autoencoder is trained ONLY
on normal (non-fault) data so that it learns to reconstruct the
healthy operating distribution. Fault conditions produce higher
reconstruction error (MSE), flagging anomalies.

Architecture (lightweight, TFLite-deployable):
    Encoder:  7 → 16 → 4  (bottleneck)
    Decoder:  4 → 16 → 7

Anomaly score = MSE between input and reconstruction.
Threshold    = 97th percentile of training reconstruction errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy TensorFlow import to keep module importable without TF
_tf = None
_keras = None


def _ensure_tf():
    """Lazy-import TensorFlow and suppress info logs."""
    global _tf, _keras
    if _tf is None:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        tf.get_logger().setLevel("WARNING")
        _tf = tf
        _keras = tf.keras
    return _tf, _keras


class AnomalyDetector:
    """
    Autoencoder-based anomaly detector for transformer feature vectors.

    The autoencoder learns to reconstruct normal operating patterns.
    When a fault occurs, reconstruction error (MSE) spikes above the
    learned threshold, signaling an anomaly.

    Architecture:
        Input(7) → Dense(16, relu) → Dense(4, relu) [bottleneck]
                 → Dense(16, relu) → Dense(7, sigmoid)

    Training:
        - Fit on scaled normal-operation data ONLY
        - Threshold set at the configured percentile of training MSEs

    Inference:
        - Compute reconstruction error (MSE)
        - Compare to threshold → is_anomaly boolean
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 16,
        encoding_dim: int = 4,
    ) -> None:
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._encoding_dim = encoding_dim
        self._model = None
        self._threshold: Optional[float] = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    def _build_model(self):
        """Build the autoencoder architecture."""
        tf, keras = _ensure_tf()

        encoder_input = keras.Input(shape=(self._input_dim,), name="encoder_input")

        # Encoder
        x = keras.layers.Dense(
            self._hidden_dim, activation="relu", name="encoder_hidden"
        )(encoder_input)
        bottleneck = keras.layers.Dense(
            self._encoding_dim, activation="relu", name="bottleneck"
        )(x)

        # Decoder
        x = keras.layers.Dense(
            self._hidden_dim, activation="relu", name="decoder_hidden"
        )(bottleneck)
        decoder_output = keras.layers.Dense(
            self._input_dim, activation="sigmoid", name="decoder_output"
        )(x)

        self._model = keras.Model(
            inputs=encoder_input,
            outputs=decoder_output,
            name="edge_guardian_autoencoder",
        )
        return self._model

    def train(
        self,
        X_normal: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        threshold_percentile: float = 97.0,
        validation_split: float = 0.1,
    ) -> dict:
        """
        Train the autoencoder on normal-operation data and set threshold.

        Args:
            X_normal:              Scaled feature array (N, 7) — normal data ONLY.
            epochs:                Training epochs.
            batch_size:            Batch size.
            learning_rate:         Adam learning rate.
            threshold_percentile:  Percentile for anomaly threshold.
            validation_split:      Fraction for validation.

        Returns:
            Dict with training metrics and threshold value.

        Raises:
            ValueError: If input dimensions are wrong.
        """
        tf, keras = _ensure_tf()

        if X_normal.ndim != 2 or X_normal.shape[1] != self._input_dim:
            raise ValueError(
                f"Expected shape (N, {self._input_dim}), got {X_normal.shape}"
            )

        logger.info(
            f"Training autoencoder on {X_normal.shape[0]} normal samples "
            f"[{epochs} epochs, batch={batch_size}, lr={learning_rate}]"
        )

        # Build model
        self._build_model()

        # Compile
        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )

        # Train (autoencoder: input == target)
        history = self._model.fit(
            X_normal, X_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
        )

        # Compute reconstruction errors on full training set
        reconstructed = self._model.predict(X_normal, verbose=0)
        mse_errors = np.mean((X_normal - reconstructed) ** 2, axis=1)

        # Set threshold at configured percentile
        self._threshold = float(np.percentile(mse_errors, threshold_percentile))
        self._is_trained = True

        metrics = {
            "final_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history.get("val_loss", [0.0])[-1]),
            "threshold": self._threshold,
            "threshold_percentile": threshold_percentile,
            "training_mse_mean": float(np.mean(mse_errors)),
            "training_mse_std": float(np.std(mse_errors)),
            "training_mse_max": float(np.max(mse_errors)),
            "training_samples": X_normal.shape[0],
            "epochs": epochs,
        }

        logger.info(
            f"Autoencoder trained — threshold={self._threshold:.6f} "
            f"(P{threshold_percentile}), "
            f"mean_mse={metrics['training_mse_mean']:.6f}"
        )

        return metrics

    def predict(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores and binary anomaly flags.

        Args:
            X_scaled: Scaled feature array, shape (N, 7) or (7,).

        Returns:
            Tuple of (anomaly_scores, is_anomaly) arrays.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("AnomalyDetector not trained — call train() first")

        # Handle single sample
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)

        reconstructed = self._model.predict(X_scaled, verbose=0)
        mse_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        is_anomaly = mse_errors > self._threshold

        return mse_errors.astype(np.float64), is_anomaly

    def predict_single(self, x_scaled: np.ndarray) -> Tuple[float, bool]:
        """
        Compute anomaly score for a single scaled feature vector.

        Args:
            x_scaled: Scaled feature array of shape (7,).

        Returns:
            Tuple of (anomaly_score, is_anomaly).
        """
        scores, flags = self.predict(x_scaled.reshape(1, -1))
        return float(scores[0]), bool(flags[0])

    def save(self, directory: Optional[Path] = None) -> Path:
        """Save model and threshold to disk."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        from config.settings import MODEL_DIR
        save_dir = directory or MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        model_path = save_dir / "autoencoder.keras"
        self._model.save(model_path)

        # Save threshold
        threshold_path = save_dir / "anomaly_threshold.npy"
        np.save(threshold_path, np.array([self._threshold]))

        logger.info(f"Autoencoder saved to {save_dir}")
        return save_dir

    def load(self, directory: Optional[Path] = None) -> "AnomalyDetector":
        """Load model and threshold from disk."""
        tf, keras = _ensure_tf()
        from config.settings import MODEL_DIR

        load_dir = directory or MODEL_DIR
        model_path = load_dir / "autoencoder.keras"
        threshold_path = load_dir / "anomaly_threshold.npy"

        self._model = keras.models.load_model(model_path)
        self._threshold = float(np.load(threshold_path)[0])
        self._is_trained = True

        logger.info(f"Autoencoder loaded from {load_dir}")
        return self

    def convert_to_tflite(self, directory: Optional[Path] = None) -> Path:
        """
        Convert the trained autoencoder to TFLite format.

        This produces a lightweight model suitable for edge deployment
        on microcontrollers and resource-constrained devices.

        Args:
            directory: Target directory (defaults to MODEL_DIR).

        Returns:
            Path to the .tflite file.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot convert untrained model to TFLite")

        tf, _ = _ensure_tf()
        from config.settings import MODEL_DIR

        save_dir = directory or MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(self._model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_path = save_dir / "autoencoder.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        logger.info(
            f"TFLite model saved to {tflite_path} "
            f"({len(tflite_model)} bytes)"
        )
        return tflite_path

    def get_architecture_summary(self) -> str:
        """Return a text summary of the model architecture."""
        if self._model is None:
            return "Model not built yet"

        lines = []
        self._model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)

