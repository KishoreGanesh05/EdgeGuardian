"""
Edge-Guardian Features / Orchestrator Module (P2).

This is the main orchestrator that ties together:
    Preprocessing → Anomaly Detection → Fault Classification

It provides the high-level train() and infer() API that downstream
modules (Health Index, Digital Shadow, Dashboard) will consume.

IMPORTANT:
    - This module does NOT duplicate P1 physics/feature extraction logic.
    - FeatureVectors come from the P1 pipeline (generator.extract_feature_vector).
    - Training data comes from the P1 generate_training_dataset() pipeline.
    - This module only handles ML preprocessing, detection, and classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data.models import AIResult, FeatureVector
from config.settings import (
    AUTOENCODER_BATCH_SIZE,
    AUTOENCODER_ENCODING_DIM,
    AUTOENCODER_EPOCHS,
    AUTOENCODER_HIDDEN_DIM,
    AUTOENCODER_LEARNING_RATE,
    ANOMALY_THRESHOLD_PERCENTILE,
    FEATURE_COUNT,
    MODEL_DIR,
)
from edge_ai.preprocessing import Preprocessor
from edge_ai.anomaly import AnomalyDetector
from edge_ai.classifier import FaultClassifier

logger = logging.getLogger(__name__)


class EdgeAIEngine:
    """
    Main Edge AI engine — orchestrates the full ML pipeline.

    Pipeline:
        FeatureVector (from P1)
            ↓  Preprocessor.transform()
        Scaled features (7,)
            ↓  AnomalyDetector.predict_single()
        anomaly_score, is_anomaly
            ↓  FaultClassifier.predict()  [if anomaly]
        fault_class, confidence
            ↓
        AIResult (returned to caller)

    Training:
        1. Fit Preprocessor on normal features
        2. Scale normal features → train Autoencoder
        3. Scale fault features → train Classifier

    This class does NOT perform feature extraction from telemetry.
    That responsibility belongs to P1 (generator.extract_feature_vector).
    """

    def __init__(self) -> None:
        self._preprocessor = Preprocessor()
        self._anomaly_detector = AnomalyDetector(
            input_dim=FEATURE_COUNT,
            hidden_dim=AUTOENCODER_HIDDEN_DIM,
            encoding_dim=AUTOENCODER_ENCODING_DIM,
        )
        self._classifier = FaultClassifier()
        self._is_trained = False
        self._training_metrics: Dict = {}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_metrics(self) -> Dict:
        return self._training_metrics

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    @property
    def anomaly_detector(self) -> AnomalyDetector:
        return self._anomaly_detector

    @property
    def classifier(self) -> FaultClassifier:
        return self._classifier

    def train(self, dataset: Dict[str, List[Dict]]) -> Dict:
        """
        Train the complete Edge AI pipeline from a P1 training dataset.

        This expects the output of generate_training_dataset():
            dataset["normal"]  — list of dicts with "features" key (FeatureVector)
            dataset["faults"]  — list of dicts with "features" and "label" keys

        Steps:
            1. Extract FeatureVectors from dataset
            2. Fit preprocessor on normal features
            3. Scale normal features → train autoencoder
            4. Scale fault features → train classifier

        Args:
            dataset: Output of generate_training_dataset().

        Returns:
            Dict with training metrics from all components.

        Raises:
            ValueError: If dataset is empty or malformed.
        """
        # ── 1. Extract features ────────────────────────────────────────────
        normal_features: List[FeatureVector] = [
            record["features"] for record in dataset["normal"]
        ]
        fault_features: List[FeatureVector] = [
            record["features"] for record in dataset["faults"]
        ]
        fault_labels = np.array([
            record["label"] for record in dataset["faults"]
        ])

        if not normal_features:
            raise ValueError("No normal training data provided")
        if not fault_features:
            raise ValueError("No fault training data provided")

        logger.info(
            f"Training Edge AI pipeline: "
            f"{len(normal_features)} normal, {len(fault_features)} fault samples"
        )

        # ── 2. Fit preprocessor on normal data ONLY ────────────────────────
        self._preprocessor.fit(normal_features)

        # ── 3. Scale and train autoencoder on normal data ──────────────────
        X_normal_scaled = self._preprocessor.transform_batch(normal_features)

        ae_metrics = self._anomaly_detector.train(
            X_normal=X_normal_scaled,
            epochs=AUTOENCODER_EPOCHS,
            batch_size=AUTOENCODER_BATCH_SIZE,
            learning_rate=AUTOENCODER_LEARNING_RATE,
            threshold_percentile=ANOMALY_THRESHOLD_PERCENTILE,
        )

        # ── 4. Scale fault data and train classifier ───────────────────────
        X_fault_scaled = self._preprocessor.transform_batch(fault_features)

        clf_metrics = self._classifier.train(
            X_faults=X_fault_scaled,
            y_labels=fault_labels,
        )

        # ── Done ───────────────────────────────────────────────────────────
        self._is_trained = True
        self._training_metrics = {
            "preprocessor": self._preprocessor.get_params(),
            "autoencoder": ae_metrics,
            "classifier": clf_metrics,
        }

        logger.info("Edge AI pipeline training complete")
        return self._training_metrics

    def infer(self, feature_vector: FeatureVector) -> AIResult:
        """
        Run inference on a single FeatureVector.

        Pipeline:
            1. Scale the feature vector
            2. Run anomaly detection (autoencoder MSE)
            3. If anomaly → run fault classification
            4. Return AIResult

        Args:
            feature_vector: A FeatureVector from the P1 physics pipeline.

        Returns:
            AIResult with anomaly_score, is_anomaly, fault_class, and confidence.

        Raises:
            RuntimeError: If the engine has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("EdgeAIEngine not trained — call train() first")

        # 1. Preprocess
        x_scaled = self._preprocessor.transform(feature_vector)

        # 2. Anomaly detection
        anomaly_score, is_anomaly = self._anomaly_detector.predict_single(x_scaled)

        # 3. Fault classification (only if anomaly detected)
        fault_class: Optional[str] = None
        fault_confidence: Optional[float] = None

        if is_anomaly:
            fault_class, fault_confidence = self._classifier.predict(x_scaled)

        # 4. Build AIResult using the existing P1 data contract
        return AIResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            fault_class=fault_class,
            fault_confidence=fault_confidence,
            model_version="v1.0-stage2",
        )

    def infer_batch(self, feature_vectors: List[FeatureVector]) -> List[AIResult]:
        """
        Run inference on a batch of FeatureVectors.

        Args:
            feature_vectors: List of FeatureVector objects.

        Returns:
            List of AIResult objects.
        """
        return [self.infer(fv) for fv in feature_vectors]

    def save(self, directory: Optional[Path] = None) -> Path:
        """Save all model artifacts to disk."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained engine")

        save_dir = directory or MODEL_DIR
        self._preprocessor.save(save_dir)
        self._anomaly_detector.save(save_dir)
        self._classifier.save(save_dir)
        logger.info(f"Edge AI engine saved to {save_dir}")
        return save_dir

    def load(self, directory: Optional[Path] = None) -> "EdgeAIEngine":
        """Load all model artifacts from disk."""
        load_dir = directory or MODEL_DIR
        self._preprocessor.load(load_dir)
        self._anomaly_detector.load(load_dir)
        self._classifier.load(load_dir)
        self._is_trained = True
        logger.info(f"Edge AI engine loaded from {load_dir}")
        return self

    def convert_to_tflite(self, directory: Optional[Path] = None) -> Path:
        """
        Convert the autoencoder to TFLite for edge deployment.

        Args:
            directory: Target directory (defaults to MODEL_DIR).

        Returns:
            Path to the .tflite file.
        """
        return self._anomaly_detector.convert_to_tflite(directory)

