"""
Edge-Guardian Telemetry Generator Tests.

Tests for synthetic telemetry generation, fleet configuration,
fault injection, and the physics pipeline. Verifies determinism,
physical plausibility, and that fault scenarios produce expected
downstream changes in the feature vector.
"""

import pytest
import numpy as np
from datetime import datetime

from data.models import (
    TransformerConfig,
    TelemetrySample,
    PhysicsResult,
    FeatureVector,
    FaultScenario,
)
from data.generator import (
    generate_fleet_configs,
    generate_telemetry,
    calculate_physics,
    extract_feature_vector,
    generate_training_dataset,
    SimulatedIndustrialDataSource,
)
from data.scenarios import apply_scenario, ScenarioEffect, get_available_scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Fleet Configuration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFleetGeneration:
    """Tests for fleet configuration generation."""

    def test_generates_correct_count(self):
        """Generates the requested number of transformers."""
        configs = generate_fleet_configs(count=20, seed=42)
        assert len(configs) == 20

    def test_asset_ids_format(self):
        """Asset IDs follow T001, T002, ... format."""
        configs = generate_fleet_configs(count=5, seed=42)
        assert configs[0].asset_id == "T001"
        assert configs[4].asset_id == "T005"

    def test_deterministic_with_seed(self):
        """Same seed produces identical fleet."""
        fleet_a = generate_fleet_configs(count=10, seed=42)
        fleet_b = generate_fleet_configs(count=10, seed=42)
        for a, b in zip(fleet_a, fleet_b):
            assert a.asset_id == b.asset_id
            assert a.rated_power_kva == b.rated_power_kva
            assert a.baseline_efficiency == b.baseline_efficiency

    def test_different_seed_different_fleet(self):
        """Different seeds produce different fleets."""
        fleet_a = generate_fleet_configs(count=10, seed=42)
        fleet_b = generate_fleet_configs(count=10, seed=99)
        # At least some parameters should differ
        diffs = sum(1 for a, b in zip(fleet_a, fleet_b)
                    if a.rated_power_kva != b.rated_power_kva)
        assert diffs > 0

    def test_all_configs_valid(self):
        """All generated configs have valid parameter ranges."""
        configs = generate_fleet_configs(count=50, seed=42)
        for c in configs:
            assert c.rated_power_kva > 0
            assert 0.5 <= c.baseline_efficiency <= 1.0
            assert c.base_resistance_ohm >= 0
            assert 0.1 <= c.age_factor <= 5.0
            assert 0 <= c.initial_health <= 100

    def test_t023_exists_in_fleet(self):
        """Fleet of 50 contains T023 (important for demo scenario)."""
        configs = generate_fleet_configs(count=50, seed=42)
        ids = [c.asset_id for c in configs]
        assert "T023" in ids


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry Generation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTelemetryGeneration:
    """Tests for telemetry sample generation."""

    @pytest.fixture
    def config(self):
        return TransformerConfig(
            asset_id="T001",
            rated_power_kva=500.0,
            nominal_voltage_v=11000.0,
            baseline_efficiency=0.97,
            base_resistance_ohm=0.15,
            ambient_temperature_c=35.0,
        )

    def test_generates_correct_count(self, config):
        """Generates expected number of samples."""
        samples = generate_telemetry(config, duration_seconds=60, seed=42)
        assert len(samples) == 60  # 1-second intervals

    def test_deterministic_with_seed(self, config):
        """Same seed → same output."""
        a = generate_telemetry(config, duration_seconds=10, seed=42)
        b = generate_telemetry(config, duration_seconds=10, seed=42)
        for sa, sb in zip(a, b):
            assert sa.voltage_feeder_v == sb.voltage_feeder_v
            assert sa.current_feeder_a == sb.current_feeder_a
            assert sa.transformer_temperature_c == sb.transformer_temperature_c

    def test_different_seed_different_noise(self, config):
        """Different seeds produce different noise patterns."""
        a = generate_telemetry(config, duration_seconds=10, seed=42)
        b = generate_telemetry(config, duration_seconds=10, seed=99)
        # At least some values should differ due to noise
        diffs = sum(1 for sa, sb in zip(a, b)
                    if sa.voltage_feeder_v != sb.voltage_feeder_v)
        assert diffs > 0

    def test_samples_have_valid_values(self, config):
        """All samples have physically plausible values."""
        samples = generate_telemetry(config, duration_seconds=60, seed=42)
        for s in samples:
            assert s.asset_id == "T001"
            assert s.voltage_feeder_v > 0
            assert s.current_feeder_a >= 0
            assert s.voltage_secondary_v >= 0
            assert s.current_secondary_a >= 0
            assert s.frequency_hz > 0
            assert s.load_percent > 0
            assert s.transformer_temperature_c > s.ambient_temperature_c or s.load_percent < 1

    def test_normal_scenario_no_label(self, config):
        """Normal samples have no scenario label."""
        samples = generate_telemetry(config, duration_seconds=10, seed=42)
        for s in samples:
            assert s.scenario_id is None
            assert s.scenario_severity is None

    def test_fault_scenario_has_label(self, config):
        """Fault samples carry scenario metadata."""
        samples = generate_telemetry(
            config, duration_seconds=10, seed=42,
            scenario=FaultScenario.FEEDER_RESISTANCE_INCREASE,
        )
        for s in samples:
            assert s.scenario_id == "FEEDER_RESISTANCE_INCREASE"
            assert s.scenario_severity is not None
            assert s.scenario_severity > 0


# ═══════════════════════════════════════════════════════════════════════════
# Physics Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicsPipeline:
    """Tests for the calculate_physics function."""

    @pytest.fixture
    def config(self):
        return TransformerConfig(
            asset_id="T001",
            rated_power_kva=500.0,
            nominal_voltage_v=11000.0,
            baseline_efficiency=0.97,
            base_resistance_ohm=0.15,
            ambient_temperature_c=35.0,
        )

    @pytest.fixture
    def normal_sample(self, config):
        samples = generate_telemetry(config, duration_seconds=5, seed=42)
        return samples[0]

    def test_physics_result_complete(self, normal_sample, config):
        """PhysicsResult has all required fields populated."""
        physics = calculate_physics(normal_sample, config)
        assert physics.rms_v > 0
        assert physics.rms_i >= 0
        assert physics.power_w >= 0
        assert physics.apparent_power_va >= 0
        assert 0.0 <= physics.power_factor <= 1.0
        assert 0.0 <= physics.efficiency <= 1.0
        assert physics.loss_w >= 0
        assert physics.thermal_stress >= 0

    def test_power_factor_bounded(self, normal_sample, config):
        """Power factor must always be in [0, 1]."""
        physics = calculate_physics(normal_sample, config)
        assert 0.0 <= physics.power_factor <= 1.0

    def test_efficiency_bounded(self, normal_sample, config):
        """Efficiency must always be in [0, 1]."""
        physics = calculate_physics(normal_sample, config)
        assert 0.0 <= physics.efficiency <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Feature Vector Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureExtraction:
    """Tests for feature vector extraction — the Edge AI interface."""

    @pytest.fixture
    def config(self):
        return TransformerConfig(
            asset_id="T001",
            rated_power_kva=500.0,
            nominal_voltage_v=11000.0,
            baseline_efficiency=0.97,
            base_resistance_ohm=0.15,
            ambient_temperature_c=35.0,
        )

    def test_exactly_seven_features(self, config):
        """Feature vector must have exactly 7 elements."""
        samples = generate_telemetry(config, duration_seconds=5, seed=42)
        for s in samples:
            physics = calculate_physics(s, config)
            fv = extract_feature_vector(s, physics)
            arr = fv.to_array()
            assert len(arr) == 7

    def test_correct_feature_order(self, config):
        """Features must be in the canonical order."""
        samples = generate_telemetry(config, duration_seconds=5, seed=42)
        s = samples[0]
        physics = calculate_physics(s, config)
        fv = extract_feature_vector(s, physics)
        arr = fv.to_array()
        # Verify order: [rms_v, rms_i, power, pf, efficiency, temperature, eff_delta]
        assert arr[0] == fv.rms_v
        assert arr[1] == fv.rms_i
        assert arr[2] == fv.power
        assert arr[3] == fv.power_factor
        assert arr[4] == fv.efficiency
        assert arr[5] == fv.temperature
        assert arr[6] == fv.eff_delta

    def test_no_nan_features(self, config):
        """No feature should be NaN."""
        samples = generate_telemetry(config, duration_seconds=30, seed=42)
        for s in samples:
            physics = calculate_physics(s, config)
            fv = extract_feature_vector(s, physics)
            arr = fv.to_array()
            assert not np.any(np.isnan(arr)), f"NaN in features: {arr}"

    def test_roundtrip_array(self, config):
        """to_array() and from_array() roundtrip correctly."""
        samples = generate_telemetry(config, duration_seconds=5, seed=42)
        physics = calculate_physics(samples[0], config)
        fv = extract_feature_vector(samples[0], physics)
        arr = fv.to_array()
        fv2 = FeatureVector.from_array(arr)
        assert fv2.rms_v == pytest.approx(fv.rms_v)
        assert fv2.eff_delta == pytest.approx(fv.eff_delta)


# ═══════════════════════════════════════════════════════════════════════════
# Fault Scenario Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScenarios:
    """Tests for fault scenario definitions."""

    def test_available_scenarios(self):
        """All required scenarios are available."""
        available = get_available_scenarios()
        assert "NORMAL" in available
        assert "FEEDER_RESISTANCE_INCREASE" in available
        assert "OVERLOAD" in available
        assert "THERMAL_STRESS" in available
        assert "EFFICIENCY_DEGRADATION" in available

    def test_normal_scenario_no_effect(self):
        """Normal scenario produces no modifications."""
        config = TransformerConfig(asset_id="T001")
        effect = apply_scenario(FaultScenario.NORMAL, config)
        assert effect.resistance_multiplier == 1.0
        assert effect.load_adjustment_percent == 0.0
        assert effect.ambient_temp_adjustment_c == 0.0
        assert effect.efficiency_multiplier == 1.0

    def test_resistance_increase_multiplier(self):
        """Feeder resistance scenario increases resistance multiplier."""
        config = TransformerConfig(asset_id="T001")
        effect = apply_scenario(
            FaultScenario.FEEDER_RESISTANCE_INCREASE, config,
            parameters={"resistance_increase_percent": 25.0},
        )
        assert effect.resistance_multiplier == pytest.approx(1.25)

    def test_overload_increases_load(self):
        """Overload scenario adds load adjustment."""
        config = TransformerConfig(asset_id="T001")
        effect = apply_scenario(
            FaultScenario.OVERLOAD, config,
            parameters={"overload_percent": 130.0},
        )
        assert effect.load_adjustment_percent > 0

    def test_thermal_stress_increases_ambient(self):
        """Thermal stress scenario raises ambient temperature."""
        config = TransformerConfig(asset_id="T001")
        effect = apply_scenario(
            FaultScenario.THERMAL_STRESS, config,
            parameters={"ambient_increase_c": 15.0},
        )
        assert effect.ambient_temp_adjustment_c == pytest.approx(15.0)

    def test_efficiency_degradation_reduces(self):
        """Efficiency degradation reduces efficiency multiplier."""
        config = TransformerConfig(asset_id="T001")
        effect = apply_scenario(
            FaultScenario.EFFICIENCY_DEGRADATION, config,
            parameters={"efficiency_reduction_percent": 5.0},
        )
        assert effect.efficiency_multiplier < 1.0
        assert effect.efficiency_multiplier == pytest.approx(0.95)


# ═══════════════════════════════════════════════════════════════════════════
# Fault Impact Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFaultImpactOnFeatures:
    """
    THE KEY INTEGRATION TEST:
    Verify that injecting a fault produces detectable changes
    in the feature vector that the Edge AI will learn from.
    """

    @pytest.fixture
    def config(self):
        return TransformerConfig(
            asset_id="T023",
            rated_power_kva=500.0,
            nominal_voltage_v=11000.0,
            baseline_efficiency=0.97,
            base_resistance_ohm=0.15,
            ambient_temperature_c=35.0,
        )

    def _get_mean_features(self, config, scenario, seed=42, **params):
        """Generate telemetry and compute mean feature vector."""
        samples = generate_telemetry(
            config, duration_seconds=60, seed=seed,
            scenario=scenario, scenario_params=params,
        )
        features = []
        for s in samples:
            physics = calculate_physics(s, config)
            fv = extract_feature_vector(s, physics)
            features.append(fv.to_array())
        return np.mean(features, axis=0)

    def test_feeder_resistance_changes_features(self, config):
        """
        CRITICAL: +25% feeder resistance must change the feature vector.

        Expected chain:
            Higher R → Higher loss → Lower efficiency → Negative eff_delta
        """
        normal = self._get_mean_features(config, FaultScenario.NORMAL)
        fault = self._get_mean_features(
            config, FaultScenario.FEEDER_RESISTANCE_INCREASE,
            resistance_increase_percent=25.0,
        )

        # Efficiency should decrease (index 4)
        assert fault[4] < normal[4], "Efficiency should decrease with higher resistance"

        # Eff_delta should be more negative (index 6)
        assert fault[6] < normal[6], "Eff_delta should be more negative"

    def test_overload_changes_features(self, config):
        """Overload should increase current, power, and temperature."""
        normal = self._get_mean_features(config, FaultScenario.NORMAL)
        fault = self._get_mean_features(
            config, FaultScenario.OVERLOAD,
            overload_percent=130.0,
        )

        # Current should increase (index 1)
        assert fault[1] > normal[1], "Current should increase with overload"
        # Power should increase (index 2)
        assert fault[2] > normal[2], "Power should increase with overload"
        # Temperature should increase (index 5)
        assert fault[5] > normal[5], "Temperature should increase with overload"

    def test_thermal_stress_changes_temperature(self, config):
        """Thermal stress should increase temperature."""
        normal = self._get_mean_features(config, FaultScenario.NORMAL)
        fault = self._get_mean_features(
            config, FaultScenario.THERMAL_STRESS,
            ambient_increase_c=15.0,
        )

        # Temperature should increase (index 5)
        assert fault[5] > normal[5], "Temperature should increase with thermal stress"

    def test_efficiency_degradation_changes_efficiency(self, config):
        """Efficiency degradation should reduce efficiency and eff_delta."""
        normal = self._get_mean_features(config, FaultScenario.NORMAL)
        fault = self._get_mean_features(
            config, FaultScenario.EFFICIENCY_DEGRADATION,
            efficiency_reduction_percent=5.0,
        )

        # Efficiency should decrease (index 4)
        assert fault[4] < normal[4], "Efficiency should decrease with degradation"


# ═══════════════════════════════════════════════════════════════════════════
# Industrial Data Source Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSimulatedDataSource:
    """Tests for the IndustrialDataSource abstraction."""

    def test_returns_telemetry(self):
        config = TransformerConfig(asset_id="T001")
        source = SimulatedIndustrialDataSource({"T001": config}, seed=42)
        sample = source.get_telemetry("T001")
        assert sample is not None
        assert sample.asset_id == "T001"

    def test_returns_none_for_unknown_asset(self):
        source = SimulatedIndustrialDataSource({}, seed=42)
        assert source.get_telemetry("UNKNOWN") is None

    def test_batch_returns_correct_count(self):
        config = TransformerConfig(asset_id="T001")
        source = SimulatedIndustrialDataSource({"T001": config}, seed=42)
        batch = source.get_batch("T001", 10)
        assert len(batch) == 10


# ═══════════════════════════════════════════════════════════════════════════
# Training Dataset Generation Test
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainingDataset:
    """Tests for training dataset generation (used by Edge AI team)."""

    def test_dataset_has_normal_and_faults(self):
        """Dataset contains both normal and fault-labeled samples."""
        configs = generate_fleet_configs(count=5, seed=42)
        dataset = generate_training_dataset(
            configs, normal_duration_s=10, fault_duration_s=10, seed=42,
        )
        assert len(dataset["normal"]) > 0
        assert len(dataset["faults"]) > 0

    def test_normal_samples_labeled_normal(self):
        """All normal samples have label NORMAL."""
        configs = generate_fleet_configs(count=3, seed=42)
        dataset = generate_training_dataset(
            configs, normal_duration_s=10, fault_duration_s=10, seed=42,
        )
        for record in dataset["normal"]:
            assert record["label"] == "NORMAL"

    def test_fault_samples_have_correct_labels(self):
        """Fault samples carry their scenario label."""
        configs = generate_fleet_configs(count=3, seed=42)
        dataset = generate_training_dataset(
            configs, normal_duration_s=5, fault_duration_s=5, seed=42,
        )
        labels = set(r["label"] for r in dataset["faults"])
        assert "FEEDER_RESISTANCE_INCREASE" in labels
        assert "OVERLOAD" in labels

    def test_each_record_has_feature_vector(self):
        """Every record contains a valid 7-element feature vector."""
        configs = generate_fleet_configs(count=3, seed=42)
        dataset = generate_training_dataset(
            configs, normal_duration_s=5, fault_duration_s=5, seed=42,
        )
        for record in dataset["normal"][:10]:
            fv = record["features"]
            assert len(fv.to_array()) == 7
            assert not np.any(np.isnan(fv.to_array()))
