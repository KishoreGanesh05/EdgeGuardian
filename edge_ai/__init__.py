"""
Edge-Guardian Edge AI Module (P2).

Provides lightweight ML-based anomaly detection and fault classification
for transformer feature vectors produced by the P1 physics engine.

Pipeline:
    FeatureVector (P1) → Preprocessing → Autoencoder → Anomaly Score
                                                     → Classifier → AIResult

Public API:
    EdgeAIEngine    — Main orchestrator (train + infer)
    Preprocessor    — Feature scaling (StandardScaler)
    AnomalyDetector — Autoencoder-based anomaly detection
    FaultClassifier — RandomForest fault classification
"""

from edge_ai.features import EdgeAIEngine
from edge_ai.preprocessing import Preprocessor
from edge_ai.anomaly import AnomalyDetector
from edge_ai.classifier import FaultClassifier

__all__ = [
    "EdgeAIEngine",
    "Preprocessor",
    "AnomalyDetector",
    "FaultClassifier",
]

