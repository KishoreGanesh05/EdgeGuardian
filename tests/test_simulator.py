"""
Edge-Guardian Simulator Tests (P4).
"""

import pytest
from data.models import TransformerConfig, FaultScenario, HealthState
from digital_shadow.shadow import DigitalShadow
from simulator.what_if import WhatIfSimulator
from edge_ai.features import EdgeAIEngine
from health.calculator import HealthCalculator
from data.generator import _generate_single_sample, calculate_physics, extract_feature_vector
import numpy as np

@pytest.fixture
def base_config():
    return TransformerConfig(asset_id="T001")

@pytest.fixture
def mock_shadow(base_config):
    shadow = DigitalShadow(base_config)
    rng = np.random.default_rng(42)
    sample = _generate_single_sample(base_config, load_pct=60.0, rng=rng)
    physics = calculate_physics(sample, base_config)

    class MockAIResult:
        anomaly_score = 0.05
        is_anomaly = False
        fault_class = None
        fault_confidence = None
        model_version = "v1.0-stage2"

    class MockHealthResult:
        health_index = 95.0
        health_state = HealthState.NORMAL
        degradation_score = 0.05

    class MockRULResult:
        rul_hours = 150000.0
        critical_threshold = 20.0
        current_degradation = 0.05
        model_based = True

    shadow.update(sample, physics, MockAIResult(), MockHealthResult(), MockRULResult())
    return shadow

@pytest.fixture
def simulator():
    # Use mock engine or real initialized engine?
    # Actually, we can just instantiate EdgeAIEngine. It won't have the model trained, so infer will raise.
    # But wait, test_edge_ai tests train from dataset. We can mock infer.
    class MockEngine:
        def infer(self, features):
            from data.models import AIResult
            # Mock fault detection if load or temperature is high
            is_anomaly = features.temperature > 80.0
            return AIResult(
                anomaly_score=10.0 if is_anomaly else 0.5,
                is_anomaly=is_anomaly,
                fault_class="OVERLOAD" if is_anomaly else None,
                fault_confidence=0.9 if is_anomaly else None
            )

    return WhatIfSimulator(MockEngine(), HealthCalculator())

def test_simulator_does_not_mutate_shadow(simulator, mock_shadow):
    """Verify the original DigitalShadow is unmodified after simulation."""
    import copy

    # Deep copy the current state dictionary and lists for comparison
    original_state_dump = copy.deepcopy(mock_shadow.state.model_dump())

    # Snapshot histories
    original_rul_history = copy.deepcopy(mock_shadow.rul_history)
    original_health_history = copy.deepcopy(mock_shadow.state.health_history)
    original_anomaly_history = copy.deepcopy(mock_shadow.state.anomaly_history)
    original_efficiency_history = copy.deepcopy(mock_shadow.state.efficiency_history)
    original_temperature_history = copy.deepcopy(mock_shadow.state.temperature_history)

    # Run simulation
    result = simulator.run_scenario(
        mock_shadow,
        FaultScenario.OVERLOAD,
        severity=1.5
    )

    # Assert shadow state is identical to original
    current_state_dump = mock_shadow.state.model_dump()
    assert current_state_dump == original_state_dump

    # Assert all histories are completely unchanged
    assert mock_shadow.rul_history == original_rul_history
    assert mock_shadow.state.health_history == original_health_history
    assert mock_shadow.state.anomaly_history == original_anomaly_history
    assert mock_shadow.state.efficiency_history == original_efficiency_history
    assert mock_shadow.state.temperature_history == original_temperature_history

def test_simulator_produces_structured_result(simulator, mock_shadow):
    """Verify the scenario result contains expected structure and values."""
    result = simulator.run_scenario(
        mock_shadow,
        FaultScenario.FEEDER_RESISTANCE_INCREASE,
        severity=1.0
    )

    assert result.asset_id == "T001"
    assert result.scenario == FaultScenario.FEEDER_RESISTANCE_INCREASE.value
    assert "severity" in result.parameters

    assert "health_index" in result.current_state
    assert "health_index" in result.projected_state

    assert "health_index" in result.deltas
    assert "rul_hours" in result.deltas

def test_simulator_multiple_runs_no_mutation(simulator, mock_shadow):
    """Verify repeated simulations don't mutate shadow."""
    import copy
    original_state_dump = copy.deepcopy(mock_shadow.state.model_dump())

    simulator.run_scenario(mock_shadow, FaultScenario.OVERLOAD)
    simulator.run_scenario(mock_shadow, FaultScenario.THERMAL_STRESS)
    simulator.run_scenario(mock_shadow, FaultScenario.EFFICIENCY_DEGRADATION)

    assert mock_shadow.state.model_dump() == original_state_dump
