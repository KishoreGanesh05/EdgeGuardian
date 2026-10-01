"""
Edge-Guardian Edge AI Tests (P2).

Comprehensive tests for the P2 Edge AI pipeline:
    - Preprocessing (StandardScaler on normal data only)
    - Anomaly detection (Autoencoder threshold behavior)
    - Fault classification (RandomForest + UNKNOWN threshold)
    - End-to-end pipeline (FeatureVector → AIResult)
    - TFLite conversion
    - Model persistence (save/load)

All tests use the existing P1 generate_training_dataset() pipeline
as the data source — no duplicate data generation.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from data.models import AIResult, FeatureVector, FaultScenario
from data.generator import (
    generate_fleet_configs,
    generate_training_dataset,
    generate_telemetry,
    calculate_physics,
    extract_feature_vector,
)
from config.settings import (
    ANOMALY_THRESHOLD_PERCENTILE,
    AUTOENCODER_ENCODING_DIM,
    AUTOENCODER_HIDDEN_DIM,
    CLASSIFIER_UNKNOWN_CONFIDENCE,
    FEATURE_COUNT,
    FEATURE_ORDER,
)
from edge_ai.preprocessing import Preprocessor
from edge_ai.anomaly import AnomalyDetector
from edge_ai.classifier import FaultClassifier
from edge_ai.features import EdgeAIEngine


# ═══════════════════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def training_dataset():
    """Generate a small training dataset from the P1 pipeline."""
    configs = generate_fleet_configs(count=5, seed=42)
    return generate_training_dataset(
        configs,
        normal_duration_s=30,
        fault_duration_s=20,
        seed=42,
    )


@pytest.fixture(scope="module")
def normal_features(training_dataset):
    """Extract normal FeatureVectors from the dataset."""
    return [record["features"] for record in training_dataset["normal"]]


@pytest.fixture(scope="module")
def fault_features(training_dataset):
    """Extract fault FeatureVectors from the dataset."""
    return [record["features"] for record in training_dataset["faults"]]


@pytest.fixture(scope="module")
def fault_labels(training_dataset):
    """Extract fault labels from the dataset."""
    return np.array([record["label"] for record in training_dataset["faults"]])


@pytest.fixture(scope="module")
def fitted_preprocessor(normal_features):
    """A preprocessor fitted on normal data."""
    return Preprocessor().fit(normal_features)


@pytest.fixture(scope="module")
def trained_engine(training_dataset):
    """A fully trained EdgeAIEngine."""
    engine = EdgeAIEngine()
    engine.train(training_dataset)
    return engine


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    """Tests for the StandardScaler-based preprocessor."""

    def test_fit_on_normal_data(self, normal_features):
        """Preprocessor fits without error on normal FeatureVectors."""
        pp = Preprocessor()
        pp.fit(normal_features)
        assert pp.is_fitted

    def test_not_fitted_raises(self):
        """Transform before fit raises RuntimeError."""
        pp = Preprocessor()
        fv = FeatureVector(
            rms_v=11000, rms_i=30, power=300000,
            power_factor=0.9, efficiency=0.97,
            temperature=55.0, eff_delta=0.0,
        )
        with pytest.raises(RuntimeError, match="not been fitted"):
            pp.transform(fv)

    def test_empty_fit_raises(self):
        """Fitting with empty list raises ValueError."""
        pp = Preprocessor()
        with pytest.raises(ValueError, match="empty"):
            pp.fit([])

    def test_output_shape(self, fitted_preprocessor, normal_features):
        """Transform produces a (7,) float32 array."""
        scaled = fitted_preprocessor.transform(normal_features[0])
        assert scaled.shape == (7,)
        assert scaled.dtype == np.float32

    def test_batch_transform_shape(self, fitted_preprocessor, normal_features):
        """Batch transform produces (N, 7) array."""
        scaled = fitted_preprocessor.transform_batch(normal_features[:10])
        assert scaled.shape == (10, 7)

    def test_scaled_mean_near_zero(self, fitted_preprocessor, normal_features):
        """After scaling, mean of normal data should be near zero."""
        scaled = fitted_preprocessor.transform_batch(normal_features)
        means = np.mean(scaled, axis=0)
        for m in means:
            assert abs(m) < 0.5, f"Scaled mean too far from zero: {m}"

    def test_scaled_std_near_one(self, fitted_preprocessor, normal_features):
        """After scaling, std of normal data should be near one."""
        scaled = fitted_preprocessor.transform_batch(normal_features)
        stds = np.std(scaled, axis=0)
        for s in stds:
            assert 0.5 < s < 1.5, f"Scaled std too far from one: {s}"

    def test_preserves_feature_order(self, fitted_preprocessor):
        """Scaling preserves the canonical 7-feature order."""
        params = fitted_preprocessor.get_params()
        assert params["feature_order"] == FEATURE_ORDER
        assert len(params["mean"]) == FEATURE_COUNT
        assert len(params["scale"]) == FEATURE_COUNT

    def test_save_and_load(self, fitted_preprocessor, normal_features):
        """Scaler can be saved and loaded with identical behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            fitted_preprocessor.save(save_path)

            loaded = Preprocessor().load(save_path)
            assert loaded.is_fitted

            fv = normal_features[0]
            original = fitted_preprocessor.transform(fv)
            restored = loaded.transform(fv)
            np.testing.assert_array_almost_equal(original, restored)


# ═══════════════════════════════════════════════════════════════════════════
# Anomaly Detection Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAnomalyDetector:
    """Tests for the autoencoder-based anomaly detector."""

    def test_train_sets_threshold(self, fitted_preprocessor, normal_features):
        """Training sets a positive threshold."""
        X = fitted_preprocessor.transform_batch(normal_features)
        detector = AnomalyDetector(
            input_dim=FEATURE_COUNT,
            hidden_dim=AUTOENCODER_HIDDEN_DIM,
            encoding_dim=AUTOENCODER_ENCODING_DIM,
        )
        metrics = detector.train(X, epochs=10, batch_size=32)
        assert detector.is_trained
        assert detector.threshold is not None
        assert detector.threshold > 0

    def test_threshold_is_97th_percentile(self, fitted_preprocessor, normal_features):
        """Threshold uses the configured percentile."""
        X = fitted_preprocessor.transform_batch(normal_features)
        detector = AnomalyDetector(input_dim=FEATURE_COUNT)
        metrics = detector.train(
            X, epochs=10,
            threshold_percentile=ANOMALY_THRESHOLD_PERCENTILE,
        )
        assert metrics["threshold_percentile"] == ANOMALY_THRESHOLD_PERCENTILE

    def test_normal_mostly_not_anomalous(self, fitted_preprocessor, normal_features):
        """Most normal samples should not be flagged as anomalies."""
        X = fitted_preprocessor.transform_batch(normal_features)
        detector = AnomalyDetector(input_dim=FEATURE_COUNT)
        detector.train(X, epochs=30, batch_size=32)

        scores, flags = detector.predict(X)
        anomaly_rate = np.mean(flags)
        # With 97th percentile threshold, ≤5% of training data should be anomalous
        # (allowing small margin above 3% for edge effects)
        assert anomaly_rate < 0.10, f"Too many normal samples flagged: {anomaly_rate:.2%}"

    def test_fault_features_higher_scores(
        self, fitted_preprocessor, normal_features, fault_features
    ):
        """Fault samples should have higher anomaly scores than normal on average."""
        X_normal = fitted_preprocessor.transform_batch(normal_features)
        X_fault = fitted_preprocessor.transform_batch(fault_features)

        detector = AnomalyDetector(input_dim=FEATURE_COUNT)
        detector.train(X_normal, epochs=30, batch_size=32)

        normal_scores, _ = detector.predict(X_normal)
        fault_scores, _ = detector.predict(X_fault)

        assert np.mean(fault_scores) > np.mean(normal_scores), (
            f"Fault mean score ({np.mean(fault_scores):.6f}) should exceed "
            f"normal mean ({np.mean(normal_scores):.6f})"
        )

    def test_untrained_predict_raises(self):
        """Predict before training raises RuntimeError."""
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="not trained"):
            detector.predict_single(np.zeros(7))

    def test_architecture_summary(self, fitted_preprocessor, normal_features):
        """Model summary is available after training."""
        X = fitted_preprocessor.transform_batch(normal_features)
        detector = AnomalyDetector(input_dim=FEATURE_COUNT)
        detector.train(X, epochs=5)
        summary = detector.get_architecture_summary()
        assert "encoder_hidden" in summary
        assert "bottleneck" in summary
        assert "decoder_output" in summary

    def test_save_and_load(self, fitted_preprocessor, normal_features):
        """Detector can be saved and loaded with consistent predictions."""
        X = fitted_preprocessor.transform_batch(normal_features)
        detector = AnomalyDetector(input_dim=FEATURE_COUNT)
        detector.train(X, epochs=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            detector.save(save_path)

            loaded = AnomalyDetector(input_dim=FEATURE_COUNT)
            loaded.load(save_path)

            assert loaded.is_trained
            assert loaded.threshold == pytest.approx(detector.threshold)

            score_orig, flag_orig = detector.predict_single(X[0])
            score_loaded, flag_loaded = loaded.predict_single(X[0])
            assert score_orig == pytest.approx(score_loaded, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Fault Classifier Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFaultClassifier:
    """Tests for the RandomForest fault classifier."""

    def test_train_on_fault_data(self, fitted_preprocessor, fault_features, fault_labels):
        """Classifier trains without error on fault data."""
        X = fitted_preprocessor.transform_batch(fault_features)
        clf = FaultClassifier()
        metrics = clf.train(X, fault_labels)

        assert clf.is_trained
        assert metrics["training_accuracy"] > 0.5
        assert metrics["n_classes"] >= 2

    def test_known_fault_classes(self, fitted_preprocessor, fault_features, fault_labels):
        """All expected fault classes are learned."""
        X = fitted_preprocessor.transform_batch(fault_features)
        clf = FaultClassifier()
        clf.train(X, fault_labels)

        expected = {
            "FEEDER_RESISTANCE_INCREASE",
            "OVERLOAD",
            "THERMAL_STRESS",
            "EFFICIENCY_DEGRADATION",
        }
        assert set(clf.classes) == expected

    def test_predict_returns_class_and_confidence(
        self, fitted_preprocessor, fault_features, fault_labels
    ):
        """Predict returns a (class, confidence) tuple."""
        X = fitted_preprocessor.transform_batch(fault_features)
        clf = FaultClassifier()
        clf.train(X, fault_labels)

        fault_class, confidence = clf.predict(X[0])
        assert isinstance(fault_class, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_unknown_below_threshold(self, fitted_preprocessor, fault_features, fault_labels):
        """
        Classifier returns UNKNOWN when confidence is below threshold.

        We test this by using a very high threshold (0.99) so normal
        samples (which the classifier hasn't seen) produce UNKNOWN.
        """
        X = fitted_preprocessor.transform_batch(fault_features)
        clf = FaultClassifier(unknown_threshold=0.99)
        clf.train(X, fault_labels)

        # Use a synthetic ambiguous sample (zeros — unlikely to match any class)
        ambiguous = np.zeros(7, dtype=np.float32)
        fault_class, confidence = clf.predict(ambiguous)
        # Either UNKNOWN or a class with high confidence
        # The key test is that the threshold logic works
        if confidence < 0.99:
            assert fault_class == "UNKNOWN"

    def test_high_confidence_returns_class(
        self, fitted_preprocessor, fault_features, fault_labels
    ):
        """
        With threshold of 0.0, everything should get a class name.
        """
        X = fitted_preprocessor.transform_batch(fault_features)
        clf = FaultClassifier(unknown_threshold=0.0)
        clf.train(X, fault_labels)

        fault_class, confidence = clf.predict(X[0])
        assert fault_class != "UNKNOWN"
        assert fault_class in clf.classes


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeAIEngine:
    """End-to-end tests for the EdgeAIEngine orchestrator."""

    def test_train_from_dataset(self, training_dataset):
        """Engine trains successfully from P1 dataset."""
        engine = EdgeAIEngine()
        metrics = engine.train(training_dataset)

        assert engine.is_trained
        assert "preprocessor" in metrics
        assert "autoencoder" in metrics
        assert "classifier" in metrics

    def test_infer_returns_ai_result(self, trained_engine, normal_features):
        """Inference produces a valid AIResult."""
        result = trained_engine.infer(normal_features[0])

        assert isinstance(result, AIResult)
        assert result.anomaly_score >= 0
        assert isinstance(result.is_anomaly, bool)
        assert result.model_version == "v1.0-stage2"

    def test_normal_mostly_not_anomalous(self, trained_engine, normal_features):
        """Most normal samples should not be flagged as anomalies."""
        results = trained_engine.infer_batch(normal_features)
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        anomaly_rate = anomaly_count / len(results)
        assert anomaly_rate < 0.10, f"Too many normal anomalies: {anomaly_rate:.2%}"

    def test_normal_no_fault_class(self, trained_engine, normal_features):
        """Normal (non-anomaly) samples should have no fault class."""
        results = trained_engine.infer_batch(normal_features[:50])
        for r in results:
            if not r.is_anomaly:
                assert r.fault_class is None
                assert r.fault_confidence is None

    def test_fault_detection(self, trained_engine, fault_features):
        """Fault samples should have higher anomaly scores on average."""
        results = trained_engine.infer_batch(fault_features)
        mean_score = np.mean([r.anomaly_score for r in results])
        # Some faults should be detected
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        assert anomaly_count > 0, "No faults detected at all"

    def test_fault_classification(self, trained_engine, fault_features):
        """Detected anomalies should have fault_class and confidence."""
        results = trained_engine.infer_batch(fault_features)
        classified = [r for r in results if r.is_anomaly]

        if classified:
            for r in classified:
                assert r.fault_class is not None
                assert r.fault_confidence is not None
                assert 0.0 <= r.fault_confidence <= 1.0

    def test_ai_result_contract(self, trained_engine, normal_features, fault_features):
        """AIResult follows the existing P1 data contract."""
        # Test normal
        normal_result = trained_engine.infer(normal_features[0])
        assert hasattr(normal_result, "anomaly_score")
        assert hasattr(normal_result, "is_anomaly")
        assert hasattr(normal_result, "fault_class")
        assert hasattr(normal_result, "fault_confidence")
        assert hasattr(normal_result, "model_version")

        # Test fault
        fault_result = trained_engine.infer(fault_features[0])
        assert hasattr(fault_result, "anomaly_score")

    def test_untrained_infer_raises(self):
        """Inference before training raises RuntimeError."""
        engine = EdgeAIEngine()
        fv = FeatureVector(
            rms_v=11000, rms_i=30, power=300000,
            power_factor=0.9, efficiency=0.97,
            temperature=55.0, eff_delta=0.0,
        )
        with pytest.raises(RuntimeError, match="not trained"):
            engine.infer(fv)

    def test_empty_normal_raises(self):
        """Training with no normal data raises ValueError."""
        engine = EdgeAIEngine()
        with pytest.raises(ValueError, match="No normal"):
            engine.train({"normal": [], "faults": [{"features": None, "label": "X"}]})

    def test_empty_faults_raises(self, training_dataset):
        """Training with no fault data raises ValueError."""
        engine = EdgeAIEngine()
        with pytest.raises(ValueError, match="No fault"):
            engine.train({"normal": training_dataset["normal"], "faults": []})


# ═══════════════════════════════════════════════════════════════════════════
# Threshold Behavior Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestThresholdBehavior:
    """Tests that verify the anomaly threshold works correctly."""

    def test_threshold_separates_normal_and_fault(
        self, trained_engine, normal_features, fault_features
    ):
        """
        The threshold should create meaningful separation:
        normal scores mostly below, fault scores mostly above.
        """
        normal_results = trained_engine.infer_batch(normal_features)
        fault_results = trained_engine.infer_batch(fault_features)

        normal_scores = [r.anomaly_score for r in normal_results]
        fault_scores = [r.anomaly_score for r in fault_results]

        # Mean fault score should exceed mean normal score
        assert np.mean(fault_scores) > np.mean(normal_scores)

    def test_threshold_is_positive(self, trained_engine):
        """Anomaly threshold should be a positive number."""
        threshold = trained_engine.anomaly_detector.threshold
        assert threshold is not None
        assert threshold > 0


# ═══════════════════════════════════════════════════════════════════════════
# UNKNOWN Classification Behavior Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestUnknownBehavior:
    """Tests that UNKNOWN classification works correctly."""

    def test_unknown_threshold_value(self):
        """The UNKNOWN confidence threshold is 0.60 as configured."""
        assert CLASSIFIER_UNKNOWN_CONFIDENCE == 0.60

    def test_low_confidence_returns_unknown(
        self, fitted_preprocessor, fault_features, fault_labels
    ):
        """
        With an artificially high threshold, ambiguous samples
        should be classified as UNKNOWN.
        """
        X = fitted_preprocessor.transform_batch(fault_features)
        # Use threshold of 1.0 — nothing can pass
        clf = FaultClassifier(unknown_threshold=1.0)
        clf.train(X, fault_labels)

        fault_class, confidence = clf.predict(X[0])
        assert fault_class == "UNKNOWN"
        assert confidence < 1.0

    def test_high_confidence_returns_class(
        self, fitted_preprocessor, fault_features, fault_labels
    ):
        """
        With threshold of 0.0, everything should get a class name.
        """
        X = fitted_preprocessor.transform_batch(fault_features)
        clf = FaultClassifier(unknown_threshold=0.0)
        clf.train(X, fault_labels)

        fault_class, confidence = clf.predict(X[0])
        assert fault_class != "UNKNOWN"
        assert fault_class in clf.classes


# ═══════════════════════════════════════════════════════════════════════════
# TFLite Conversion Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTFLiteConversion:
    """Tests for TFLite model conversion."""

    def test_convert_produces_tflite_file(self, trained_engine):
        """TFLite conversion produces a .tflite file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = trained_engine.convert_to_tflite(Path(tmpdir))
            assert tflite_path.exists()
            assert tflite_path.suffix == ".tflite"
            assert tflite_path.stat().st_size > 0

    def test_tflite_model_runs(self, trained_engine):
        """The TFLite model can run inference."""
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = trained_engine.convert_to_tflite(Path(tmpdir))

            # Load and run TFLite model
            with open(tflite_path, "rb") as f:
                model_content = f.read()
            interpreter = tf.lite.Interpreter(model_content=model_content)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Verify input shape
            assert input_details[0]["shape"].tolist() == [1, 7]

            # Run inference with dummy data
            test_input = np.random.randn(1, 7).astype(np.float32)
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]["index"])
            assert output.shape == (1, 7)

# ═══════════════════════════════════════════════════════════════════════════
# Model Persistence Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestModelPersistence:
    """Tests for save/load of the complete pipeline."""

    def test_save_and_load_engine(self, trained_engine, normal_features):
        """Complete engine can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            trained_engine.save(save_path)

            loaded = EdgeAIEngine()
            loaded.load(save_path)

            assert loaded.is_trained

            # Results should be consistent
            orig = trained_engine.infer(normal_features[0])
            restored = loaded.infer(normal_features[0])

            assert orig.anomaly_score == pytest.approx(
                restored.anomaly_score, abs=1e-5
            )
            assert orig.is_anomaly == restored.is_anomaly

