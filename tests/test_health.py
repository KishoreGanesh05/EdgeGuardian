"""
Edge-Guardian Health Module Tests (P3).
"""

import pytest
from datetime import datetime, timezone

from data.models import TelemetrySample, PhysicsResult, AIResult, HealthState, HealthResult
from config.settings import HEALTH_WARNING_THRESHOLD, HEALTH_CRITICAL_THRESHOLD, RUL_CRITICAL_THRESHOLD
from health.calculator import HealthCalculator

@pytest.fixture
def calculator():
    return HealthCalculator()

@pytest.fixture
def base_telemetry():
    return TelemetrySample(
        asset_id="T001",
        timestamp=datetime.now(timezone.utc),
        voltage_feeder_v=11000,
        current_feeder_a=30,
        voltage_secondary_v=400,
        current_secondary_a=800,
        ambient_temperature_c=35.0,
        transformer_temperature_c=60.0,
        load_percent=80.0
    )

@pytest.fixture
def base_physics():
    return PhysicsResult(
        rms_v=11000.0,
        rms_i=30.0,
        power_w=320000.0,
        apparent_power_va=330000.0,
        power_factor=0.97,
        input_power_w=320000.0,
        output_power_w=315000.0,
        loss_w=5000.0,
        efficiency=0.98,
        eff_delta=0.0,
        thermal_stress=0.8
    )

@pytest.fixture
def base_ai():
    return AIResult(
        anomaly_score=0.0,
        is_anomaly=False,
        fault_class=None,
        fault_confidence=None,
        model_version="v1.0-stage2"
    )

def test_health_perfect_conditions(calculator, base_telemetry, base_physics, base_ai):
    """Test health calculation under perfect conditions."""
    result = calculator.calculate_health(base_telemetry, base_physics, base_ai)
    
    assert 0 <= result.health_index <= 100
    assert result.health_index == 100.0
    assert result.health_state == HealthState.NORMAL
    assert result.degradation_score == 0.0

def test_health_bounds_and_degradation(calculator, base_telemetry, base_physics, base_ai):
    """Test health index degrades and stays bounded."""
    
    # Simulate extreme degradation
    base_ai.is_anomaly = True
    base_ai.anomaly_score = 15.0
    base_physics.thermal_stress = 2.5
    base_telemetry.load_percent = 200.0
    base_physics.eff_delta = -0.10
    
    result = calculator.calculate_health(base_telemetry, base_physics, base_ai, previous_degradation=1.0)
    
    # Must remain bounded 0-100
    assert 0 <= result.health_index <= 100
    assert result.health_index == 0.0
    assert result.health_state == HealthState.CRITICAL
    assert result.degradation_score == 1.0

def test_health_state_classification(calculator, base_telemetry, base_physics, base_ai):
    """Test health state classification at configured boundaries."""
    
    # Adjust total penalty to target specific health scores
    # Health = 100 * (1 - penalty)
    
    # Force health to just above warning threshold
    target_health_warning = HEALTH_WARNING_THRESHOLD + 1.0
    history_penalty = 1.0 - (target_health_warning / 100.0)
    # We'll put all penalty into history for simplicity
    # Wait, history_penalty is multiplied by a weight (0.15). We can't reach arbitrary values this way easily.
    # Instead, we just mock the total penalty output, but `calculate_health` doesn't allow that.
    # Let's adjust anomaly_score (weight 0.3) and history to hit the targets.
    
    # Let's just override HEALTH_WARNING_THRESHOLD logic by creating a state that results in a specific health.
    # We can pass an AI result with a specific anomaly score to tune the health index.
    
    base_ai.is_anomaly = False
    
    # Calculate exactly
    # total_penalty = anomaly_penalty * 0.3
    # penalty = 1 - (health / 100)
    
    def get_health_for_penalty(penalty):
        # We can simulate penalty via anomaly_score if it's <= 0.3
        if penalty <= 0.3:
            base_ai.anomaly_score = (penalty / 0.3) * 10.0
            return calculator.calculate_health(base_telemetry, base_physics, base_ai).health_index
        return None

    # Normal: health >= HEALTH_WARNING_THRESHOLD
    base_ai.anomaly_score = 0.0
    assert calculator.calculate_health(base_telemetry, base_physics, base_ai).health_state == HealthState.NORMAL

    # Warning: HEALTH_CRITICAL_THRESHOLD <= health < HEALTH_WARNING_THRESHOLD
    # Let's use a combination of penalties to get health into Warning zone
    base_ai.is_anomaly = True # 0.3 penalty
    base_physics.thermal_stress = 1.5 # 0.5 * 0.2 = 0.1 penalty
    # Total penalty = 0.4 -> Health = 60
    # Assuming HEALTH_WARNING_THRESHOLD = 70 and CRITICAL = 40
    res = calculator.calculate_health(base_telemetry, base_physics, base_ai)
    assert HEALTH_CRITICAL_THRESHOLD <= res.health_index < HEALTH_WARNING_THRESHOLD
    assert res.health_state == HealthState.WARNING

def test_rul_estimation(calculator, base_telemetry, base_physics, base_ai):
    """Test RUL estimation is produced and model-based."""
    
    health = calculator.calculate_health(base_telemetry, base_physics, base_ai)
    rul = calculator.estimate_rul(health)
    
    assert rul.rul_hours > 0
    assert rul.model_based is True
    assert rul.critical_threshold == RUL_CRITICAL_THRESHOLD

def test_rul_zero_at_critical(calculator):
    """Test RUL is zero when health is at or below critical threshold."""
    health = HealthResult(
        health_index=RUL_CRITICAL_THRESHOLD - 1.0,
        health_state=HealthState.CRITICAL,
        degradation_score=0.9
    )
    rul = calculator.estimate_rul(health)
    assert rul.rul_hours == 0.0
    assert rul.model_based is True
