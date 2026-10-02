"""
Mandatory end-to-end integration test (plan §29).

Flow:
    create fleet
      -> generate normal telemetry -> process
      -> inject fault into T023 -> process
      -> T023 anomaly increases
      -> T023 health decreases
      -> Digital Shadow updates
      -> fleet metrics update
      -> scenario engine works (projection, no mutation)

Fault scenario note: the plan narrative names FEEDER_RESISTANCE_INCREASE, but in
the Stage-2 P1 physics that fault is numerically weak at 11 kV (efficiency is
insensitive to feeder resistance; only temperature responds, weakly) and does
not reliably trip the autoencoder. Per the P5 decision, the end-to-end fault is
OVERLOAD on T023, which genuinely exercises the full cascade. A separate test
documents the feeder-resistance behavior.
"""

import pytest

from data.models import FaultScenario, HealthState
from main import run_operation, run_demo, DEMO_FAULT_SCENARIO, DEMO_FAULT_PARAMS, DEMO_FAULT_SEVERITY
from tests.conftest import TARGET, SEED


class TestEndToEnd:
    def test_full_pipeline_cascade(self, normal_backend):
        backend = normal_backend
        shadow = backend.fleet.get_shadow(TARGET)
        assert shadow is not None, "Fleet must contain the target asset"

        # ── Snapshot T023 after normal operation ─────────────────────────
        before_anomaly = shadow.state.anomaly_score
        before_health = shadow.state.health_index
        before_ts = shadow.state.timestamp
        before_hist_len = len(shadow.state.health_history)

        initial_metrics = backend.fleet.get_fleet_metrics()

        # ── Inject the fault into T023 and process ───────────────────────
        target_cfg = next(c for c in backend.configs if c.asset_id == TARGET)
        run_operation(
            backend, target_cfg, steps=20,
            scenario=DEMO_FAULT_SCENARIO, severity=DEMO_FAULT_SEVERITY,
            params=DEMO_FAULT_PARAMS, seed=SEED + 999,
        )

        after_anomaly = shadow.state.anomaly_score
        after_health = shadow.state.health_index
        after_ts = shadow.state.timestamp

        # ── Assertions: the required cascade ─────────────────────────────
        assert after_anomaly > before_anomaly, "T023 anomaly score should increase"
        assert shadow.state.is_anomaly is True, "T023 should be flagged anomalous"
        assert after_health < before_health, "T023 health should decrease"

        # Digital Shadow updated (timestamp advanced, history grew)
        assert after_ts >= before_ts
        assert len(shadow.state.health_history) > before_hist_len

        # Fleet metrics update (average health drops OR a state transition occurs)
        final_metrics = backend.fleet.get_fleet_metrics()
        degraded = (
            final_metrics["average_health"] < initial_metrics["average_health"]
            or final_metrics["warning_count"] > initial_metrics["warning_count"]
            or final_metrics["critical_count"] > initial_metrics["critical_count"]
        )
        assert degraded, "Fleet metrics should reflect T023 degradation"

    def test_scenario_engine_no_mutation(self, normal_backend):
        backend = normal_backend
        shadow = backend.fleet.get_shadow(TARGET)
        before = shadow.state.health_index

        result = backend.tools.run_scenario(
            TARGET, "LOAD_INCREASE", {"load_increase_percent": 20, "severity": 1.0}
        )
        after = shadow.state.health_index

        assert after == before, "What-if must NOT mutate the real Digital Shadow"
        assert "projected" in result and "current" in result
        # Projection should move health downward for a load increase.
        assert result["projected"]["health_index"] <= result["current"]["health_index"]

    def test_agent_investigation_uses_tools_only(self, normal_backend):
        """The agent explains T023 using tool evidence, inventing nothing."""
        backend = normal_backend
        # Degrade T023 first so there is something to explain.
        target_cfg = next(c for c in backend.configs if c.asset_id == TARGET)
        run_operation(backend, target_cfg, steps=20,
                      scenario=DEMO_FAULT_SCENARIO, severity=DEMO_FAULT_SEVERITY,
                      params=DEMO_FAULT_PARAMS, seed=SEED + 7)

        out = backend.agent.ask(f"Why is {TARGET} unhealthy?")
        assert out["backend"].startswith("deterministic")
        assert out["tool_calls"], "Agent must call tools for evidence"
        # The cited health index must match the tool value exactly (no fabrication).
        status = backend.tools.get_transformer_status(TARGET)
        assert str(status["health_index"]) in out["answer"]


class TestFeederResistanceBehavior:
    """
    Documents the honest behavior of FEEDER_RESISTANCE_INCREASE in the P1
    physics: it runs through the full pipeline and raises temperature, but is a
    weak anomaly trigger at realistic severities (efficiency is insensitive to
    feeder resistance at 11 kV). This is why the demo uses OVERLOAD instead.
    """

    def test_feeder_resistance_runs_and_updates_shadow(self, normal_backend):
        backend = normal_backend
        shadow = backend.fleet.get_shadow(TARGET)
        before_temp = shadow.state.temperature
        before_hist = len(shadow.state.temperature_history)

        target_cfg = next(c for c in backend.configs if c.asset_id == TARGET)
        run_operation(
            backend, target_cfg, steps=15,
            scenario=FaultScenario.FEEDER_RESISTANCE_INCREASE,
            severity=1.0, params={"resistance_increase_percent": 300.0},
            seed=SEED + 11,
        )

        # The pipeline runs end-to-end and the shadow updates.
        assert len(shadow.state.temperature_history) > before_hist
        # Temperature is the responsive channel — it should rise.
        assert shadow.state.temperature >= before_temp


class TestDemoSmoke:
    def test_run_demo_small_fleet(self):
        """run_demo wires everything together and returns a populated backend."""
        backend = run_demo(asset_count=23, target_asset=TARGET, seed=SEED,
                           run_agent=False, force_fallback=True)
        metrics = backend.fleet.get_fleet_metrics()
        assert metrics["asset_count"] == 23
        target = backend.fleet.get_shadow(TARGET).state
        assert target.is_anomaly is True
        assert backend.sink.total_sent > 0
