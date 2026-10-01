"""
Edge-Guardian Digital Shadow Module Tests (P3).
"""

import pytest
from datetime import datetime, timezone

from data.models import (
    TransformerConfig, TelemetrySample, PhysicsResult, 
    AIResult, HealthResult, HealthState, RULResult
)
from digital_shadow.shadow import DigitalShadow
from digital_shadow.fleet import FleetManager

@pytest.fixture
def config():
    return TransformerConfig(asset_id="T001")

@pytest.fixture
def shadow(config):
    return DigitalShadow(config)

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
        anomaly_score=0.1,
        is_anomaly=False,
        fault_class=None,
        fault_confidence=None,
        model_version="v1.0-stage2"
    )

@pytest.fixture
def base_health():
    return HealthResult(
        health_index=95.0,
        health_state=HealthState.NORMAL,
        degradation_score=0.05
    )

@pytest.fixture
def base_rul():
    return RULResult(
        rul_hours=150000.0,
        critical_threshold=20.0,
        current_degradation=0.05,
        model_based=True
    )

def test_shadow_initialization(shadow, config):
    """Test DigitalShadow initializes correctly."""
    assert shadow.state.asset_id == "T001"
    assert shadow.state.config == config
    assert len(shadow.state.health_history) == 0
    assert len(shadow.rul_history) == 0

def test_shadow_update(shadow, base_telemetry, base_physics, base_ai, base_health, base_rul):
    """Test DigitalShadow updates current state and histories."""
    
    shadow.update(base_telemetry, base_physics, base_ai, base_health, base_rul)
    
    state = shadow.state
    
    # Assert current state updated
    assert state.load_percent == 80.0
    assert state.temperature == 60.0
    assert state.efficiency == 0.98
    assert state.anomaly_score == 0.1
    assert state.health_index == 95.0
    assert state.rul_hours == 150000.0
    
    # Assert histories retained
    assert len(state.health_history) == 1
    assert state.health_history[0] == 95.0
    
    assert len(state.anomaly_history) == 1
    assert state.anomaly_history[0] == 0.1
    
    assert len(state.efficiency_history) == 1
    assert state.efficiency_history[0] == 0.98
    
    assert len(state.temperature_history) == 1
    assert state.temperature_history[0] == 60.0
    
    assert len(shadow.rul_history) == 1
    assert shadow.rul_history[0] == 150000.0

def test_rul_history_retention(shadow, base_telemetry, base_physics, base_ai, base_health, base_rul):
    """Test RUL history accumulates and is bounded."""
    from config.settings import HISTORY_MAX_LENGTH
    
    # 1. Update once
    shadow.update(base_telemetry, base_physics, base_ai, base_health, base_rul)
    assert len(shadow.rul_history) == 1
    assert shadow.rul_history[0] == 150000.0
    
    # 2. Update multiple times
    new_rul = base_rul.model_copy(update={'rul_hours': 140000.0})
    shadow.update(base_telemetry, base_physics, base_ai, base_health, new_rul)
    assert len(shadow.rul_history) == 2
    assert shadow.rul_history[0] == 150000.0
    assert shadow.rul_history[1] == 140000.0
    
    # 3. Exceed HISTORY_MAX_LENGTH
    # We already have 2, so we add HISTORY_MAX_LENGTH - 1 more to push one out
    for _ in range(HISTORY_MAX_LENGTH - 1):
        shadow.update(base_telemetry, base_physics, base_ai, base_health, new_rul)
        
    assert len(shadow.rul_history) == HISTORY_MAX_LENGTH
    # The first element (150000.0) should have been popped out
    assert shadow.rul_history[0] == 140000.0


def test_fleet_aggregation_empty():
    """Test empty fleet behavior."""
    fleet = FleetManager()
    metrics = fleet.get_fleet_metrics()
    
    assert metrics["asset_count"] == 0
    assert metrics["healthy_count"] == 0
    assert metrics["average_health"] == 0.0

def test_fleet_aggregation_multiple_assets(base_telemetry, base_physics, base_ai, base_health, base_rul):
    """Test fleet aggregation works with multiple assets."""
    fleet = FleetManager()
    
    # Add Asset 1 (Normal)
    config1 = TransformerConfig(asset_id="T001")
    shadow1 = DigitalShadow(config1)
    shadow1.update(base_telemetry, base_physics, base_ai, base_health, base_rul)
    fleet.add_shadow(shadow1)
    
    # Add Asset 2 (Warning)
    config2 = TransformerConfig(asset_id="T002")
    shadow2 = DigitalShadow(config2)
    
    health2 = HealthResult(
        health_index=60.0,
        health_state=HealthState.WARNING,
        degradation_score=0.4
    )
    shadow2.update(base_telemetry, base_physics, base_ai, health2, base_rul)
    fleet.add_shadow(shadow2)
    
    metrics = fleet.get_fleet_metrics()
    
    assert metrics["asset_count"] == 2
    assert metrics["healthy_count"] == 1
    assert metrics["warning_count"] == 1
    assert metrics["critical_count"] == 0
    assert metrics["average_health"] == (95.0 + 60.0) / 2
    assert metrics["minimum_health"] == 60.0

