"""
Edge-Guardian Preprocessing Module (P2).

StandardScaler-based feature preprocessing for the 7-value physics
feature vector. The scaler is fit ONLY on normal (non-fault) training
data to establish the healthy operating baseline.

Design:
    - Wraps sklearn.preprocessing.StandardScaler
    - Preserves the canonical 7-feature order from P1
    - Persists to disk via joblib for deployment
    - TinyML-compatible: scaling parameters (mean, std) can be
      exported as fixed arrays for on-device inference
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from data.models import FeatureVector
from config.settings import FEATURE_ORDER, FEATURE_COUNT, MODEL_DIR

logger = logging.getLogger(__name__)

# Persistence paths
_SCALER_FILENAME = "scaler.joblib"


class Preprocessor:
    """
    Feature preprocessing for the Edge AI pipeline.

    Wraps a StandardScaler that is fit on normal-operation feature
    vectors only. This ensures the scaler captures the healthy
    operating distribution, making anomalies stand out after scaling.

    Usage:
        preprocessor = Preprocessor()
        preprocessor.fit(normal_feature_vectors)
        scaled = preprocessor.transform(feature_vector)
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the scaler has been fit on training data."""
        return self._is_fitted

    def fit(self, normal_features: List[FeatureVector]) -> "Preprocessor":
        """
        Fit the scaler on normal-operation feature vectors.

        MUST be called with ONLY normal (non-fault) data to establish
        the healthy operating distribution baseline.

        Args:
            normal_features: List of FeatureVector from normal operation.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If no features are provided or feature count is wrong.
        """
        if not normal_features:
            raise ValueError("Cannot fit scaler with empty feature list")

        # Convert FeatureVectors to a 2D numpy array (N × 7)
        X = np.array([fv.to_array() for fv in normal_features], dtype=np.float64)

        if X.shape[1] != FEATURE_COUNT:
            raise ValueError(
                f"Expected {FEATURE_COUNT} features, got {X.shape[1]}"
            )

        self._scaler.fit(X)
        self._is_fitted = True

        logger.info(
            f"Preprocessor fitted on {X.shape[0]} normal samples "
            f"({FEATURE_COUNT} features)"
        )
        logger.debug(f"  Feature means: {self._scaler.mean_}")
        logger.debug(f"  Feature stds:  {self._scaler.scale_}")

        return self

    def transform(self, feature_vector: FeatureVector) -> np.ndarray:
        """
        Scale a single FeatureVector using the fitted scaler.

        Args:
            feature_vector: A FeatureVector from the P1 physics pipeline.

        Returns:
            Scaled numpy array of shape (7,).

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted — call fit() first")

        arr = feature_vector.to_array().reshape(1, -1)
        scaled = self._scaler.transform(arr)
        return scaled.flatten().astype(np.float32)

    def transform_batch(self, feature_vectors: List[FeatureVector]) -> np.ndarray:
        """
        Scale a batch of FeatureVectors.

        Args:
            feature_vectors: List of FeatureVector objects.

        Returns:
            Scaled numpy array of shape (N, 7).

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted — call fit() first")

        X = np.array([fv.to_array() for fv in feature_vectors], dtype=np.float64)
        return self._scaler.transform(X).astype(np.float32)

    def save(self, directory: Optional[Path] = None) -> Path:
        """
        Persist the fitted scaler to disk.

        Args:
            directory: Target directory (defaults to MODEL_DIR).

        Returns:
            Path to the saved scaler file.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        save_dir = directory or MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / _SCALER_FILENAME
        joblib.dump(self._scaler, path)
        logger.info(f"Scaler saved to {path}")
        return path

    def load(self, directory: Optional[Path] = None) -> "Preprocessor":
        """
        Load a previously fitted scaler from disk.

        Args:
            directory: Source directory (defaults to MODEL_DIR).

        Returns:
            self (for chaining).
        """
        load_dir = directory or MODEL_DIR
        path = load_dir / _SCALER_FILENAME
        self._scaler = joblib.load(path)
        self._is_fitted = True
        logger.info(f"Scaler loaded from {path}")
        return self

    def get_params(self) -> dict:
        """
        Return scaler parameters for TinyML export.

        Returns:
            Dict with 'mean' and 'scale' arrays (can be hard-coded
            into a C header for on-device inference).
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted")

        return {
            "mean": self._scaler.mean_.tolist(),
            "scale": self._scaler.scale_.tolist(),
            "feature_order": FEATURE_ORDER,
        }

