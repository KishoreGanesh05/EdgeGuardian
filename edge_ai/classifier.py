"""
Edge-Guardian Fault Classifier Module (P2).

RandomForest-based fault classification for transformer anomalies.
The classifier is trained on labeled fault data from the P1
generate_training_dataset() pipeline.

Fault classes:
    FEEDER_RESISTANCE_INCREASE
    OVERLOAD
    THERMAL_STRESS
    EFFICIENCY_DEGRADATION

When classifier confidence falls below the configured threshold
(CLASSIFIER_UNKNOWN_CONFIDENCE = 0.60), the fault class is set
to "UNKNOWN" to avoid false certainty.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config.settings import (
    CLASSIFIER_N_ESTIMATORS,
    CLASSIFIER_RANDOM_STATE,
    CLASSIFIER_UNKNOWN_CONFIDENCE,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)

_CLASSIFIER_FILENAME = "fault_classifier.joblib"


class FaultClassifier:
    """
    RandomForest-based fault type classifier.

    Classifies detected anomalies into specific fault categories.
    When the classification confidence is below the threshold,
    returns "UNKNOWN" to avoid unreliable diagnoses.

    Training:
        - Uses labeled fault data from P1 scenario generation
        - Each sample is a scaled 7-element feature vector
        - Labels are FaultScenario enum values (strings)

    Inference:
        - Returns (fault_class, confidence) tuple
        - fault_class = "UNKNOWN" if confidence < threshold
    """

    def __init__(
        self,
        n_estimators: int = CLASSIFIER_N_ESTIMATORS,
        random_state: int = CLASSIFIER_RANDOM_STATE,
        unknown_threshold: float = CLASSIFIER_UNKNOWN_CONFIDENCE,
    ) -> None:
        self._classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._unknown_threshold = unknown_threshold
        self._is_trained = False
        self._classes: List[str] = []

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def classes(self) -> List[str]:
        """Return the list of known fault classes."""
        return self._classes

    def train(
        self,
        X_faults: np.ndarray,
        y_labels: np.ndarray,
    ) -> dict:
        """
        Train the classifier on labeled fault data.

        Args:
            X_faults: Scaled feature array (N, 7) from fault scenarios.
            y_labels: String labels array (N,) with fault scenario names.

        Returns:
            Dict with training metrics.

        Raises:
            ValueError: If inputs are empty or mismatched.
        """
        if X_faults.shape[0] == 0:
            raise ValueError("Cannot train classifier with empty data")
        if X_faults.shape[0] != y_labels.shape[0]:
            raise ValueError(
                f"Feature/label count mismatch: {X_faults.shape[0]} vs {y_labels.shape[0]}"
            )

        self._classifier.fit(X_faults, y_labels)
        self._classes = list(self._classifier.classes_)
        self._is_trained = True

        # Training accuracy
        train_predictions = self._classifier.predict(X_faults)
        accuracy = float(np.mean(train_predictions == y_labels))

        # Class distribution
        unique, counts = np.unique(y_labels, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))

        metrics = {
            "training_accuracy": accuracy,
            "n_classes": len(self._classes),
            "classes": self._classes,
            "class_distribution": class_dist,
            "n_estimators": self._classifier.n_estimators,
            "training_samples": X_faults.shape[0],
        }

        logger.info(
            f"Classifier trained — accuracy={accuracy:.4f}, "
            f"classes={self._classes}, "
            f"samples={X_faults.shape[0]}"
        )

        return metrics

    def predict(self, x_scaled: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single scaled feature vector.

        Args:
            x_scaled: Scaled feature array of shape (7,).

        Returns:
            Tuple of (fault_class, confidence).
            fault_class is "UNKNOWN" if confidence < threshold.

        Raises:
            RuntimeError: If classifier has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("FaultClassifier not trained — call train() first")

        x = x_scaled.reshape(1, -1)
        probabilities = self._classifier.predict_proba(x)[0]
        max_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[max_idx])
        predicted_class = self._classes[max_idx]

        # Apply UNKNOWN threshold
        if confidence < self._unknown_threshold:
            return "UNKNOWN", confidence

        return predicted_class, confidence

    def predict_batch(
        self, X_scaled: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Classify a batch of scaled feature vectors.

        Args:
            X_scaled: Scaled feature array (N, 7).

        Returns:
            List of (fault_class, confidence) tuples.
        """
        if not self._is_trained:
            raise RuntimeError("FaultClassifier not trained — call train() first")

        probabilities = self._classifier.predict_proba(X_scaled)
        results = []
        for probs in probabilities:
            max_idx = int(np.argmax(probs))
            confidence = float(probs[max_idx])
            predicted_class = self._classes[max_idx]

            if confidence < self._unknown_threshold:
                results.append(("UNKNOWN", confidence))
            else:
                results.append((predicted_class, confidence))

        return results

    def save(self, directory: Optional[Path] = None) -> Path:
        """Persist the trained classifier to disk."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained classifier")

        save_dir = directory or MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / _CLASSIFIER_FILENAME
        joblib.dump(self._classifier, path)
        logger.info(f"Classifier saved to {path}")
        return path

    def load(self, directory: Optional[Path] = None) -> "FaultClassifier":
        """Load a previously trained classifier from disk."""
        load_dir = directory or MODEL_DIR
        path = load_dir / _CLASSIFIER_FILENAME
        self._classifier = joblib.load(path)
        self._classes = list(self._classifier.classes_)
        self._is_trained = True
        logger.info(f"Classifier loaded from {path}")
        return self

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importances keyed by feature name."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained")

        from config.settings import FEATURE_ORDER
        importances = self._classifier.feature_importances_
        return dict(zip(FEATURE_ORDER, importances.tolist()))

