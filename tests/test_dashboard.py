"""
Edge-Guardian Dashboard Tests (P4).
"""

import pytest
from dashboard.data_source import initialize_demo_backend, get_fleet_manager, get_simulator
from dashboard.components import render_fleet_overview, render_asset_detail, render_what_if_lab

def test_dashboard_initialization():
    """Verify demo backend initializes correctly without streamlit errors."""
    fleet, simulator = initialize_demo_backend()
    assert fleet is not None
    assert simulator is not None
    assert fleet.get_fleet_metrics()["asset_count"] == 20

def test_dashboard_imports():
    """Verify Streamlit modules can be imported."""
    try:
        import dashboard.app
        import dashboard.components
        import dashboard.data_source
    except ImportError as e:
        pytest.fail(f"Failed to import dashboard modules: {e}")

def test_dashboard_data_contracts():
    """Verify the dashboard data source provides correct historical and metric data."""
    fleet = get_fleet_manager()
    assets = list(fleet._shadows.keys())

    # 1. Fleet overview data exists
    metrics = fleet.get_fleet_metrics()
    assert metrics["asset_count"] == len(assets)
    assert "healthy_count" in metrics
    assert "average_health" in metrics
    assert "average_rul_hours" in metrics

    # 2. Selected asset detail data is available
    shadow = fleet.get_shadow(assets[0])
    s = shadow.state

    # Operational/Physics
    assert s.load_percent > 0
    assert s.temperature > 0
    assert s.efficiency > 0

    # Edge AI / Anomaly Data
    assert hasattr(s, "anomaly_score")
    assert hasattr(s, "is_anomaly")

    # Health and RUL data
    assert 0 <= s.health_index <= 100
    assert s.rul_hours > 0

    # Trend Data
    assert len(s.health_history) > 0
    assert len(s.temperature_history) > 0
    assert len(s.efficiency_history) > 0

def test_dashboard_what_if_data_contracts():
    """Verify what-if simulation data contracts used by the dashboard."""
    fleet = get_fleet_manager()
    simulator = get_simulator()

    shadow = fleet.get_shadow("T001")
    from data.models import FaultScenario

    result = simulator.run_scenario(shadow, FaultScenario.OVERLOAD, severity=1.5)

    # What-if comparison data
    assert "health_index" in result.current_state
    assert "health_index" in result.projected_state
    assert "rul_hours" in result.deltas
    assert "efficiency" in result.deltas
